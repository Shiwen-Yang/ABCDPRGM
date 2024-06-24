import torch
from time import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("mps") if torch.backends.mps.is_available() else device

class fit:
    """

    Fit a Dirichlet GLM with a log link. MLE obtained using Fisher's Scoring Algorithm.

    Args:
        predictor (torch.tensor of shape n by (3p-2)): the design matrix.
        response (torch.tensor of shape n by p): the response matrix.
        constrained (bool): whether estimate beta as a 4 dimensional vector or (3p-2) x p matrix. 
        beta_guess (torch.tensor of shape 4): a vector to initiate the gradient descent
        beta_0_guess (float): estimate of the intercept term but negative. 
        tol (float): a stopping criterion for the gradient descent algorithm.

    Attributes:
        settings (class): a class object with attributes "constrained", "beta_guess", "beta_0_guess", and "tol"
        est_result (dictionary): estimated beta, fisher's information, and percent of rows of data with negative entries
    
    """

    def __init__(self, predictor, response, constrained = False, beta_guess = None, tol = 10, max_iter = 20000, verbose = False):
        self.predictor = predictor
        self.response = response
        self.settings = self.reg_settings(constrained, beta_guess, tol, max_iter, verbose)
        self.est_result = None
        self.NGD_result = None

    class reg_settings:
        def __init__(self, constrained, beta_guess, tol, max_iter, verbose):
            self.constrained = constrained
            self.beta_guess = beta_guess
            self.tol = tol
            self.max_iter = max_iter
            self.verbose = verbose
    
    def update_settings(self, constrained = None, beta_guess = None, tol = None):

        """  
        Update the settings of the gradient descent, then rerun the gradient descent.

        Args:
        constrained (bool): whether estimate beta as a 4 dimensional vector or (3p-2) x p matrix. 
        beta_guess (torch.tensor of shape 4): a vector to initiate the gradient descent
        beta_0_guess (float): estimate of the intercept term but negative. 
        tol (float): a stopping criterion for the gradient descent algorithm.

        """
        updated_settings = {
            "constrained": constrained if constrained is not None else self.settings.constrained,
            "beta_guess": beta_guess if beta_guess is not None else self.settings.beta_guess,
            "tol": tol if tol is not None else self.settings.tol,
        }
        self.settings = self.reg_settings(**updated_settings)
        self.est_result = self.Dir_NGD()

    @staticmethod
    def proj_beta(B, constraint):

        """  
        Given a (3p-2) by p matrix B, project it to the column space of a constraint matrix to get beta, the desired 4 dimensional estimator

        Args:
            B (torch.Tensor of shape (3p-2) by p): the estimated matrix B
            constraint (torch.tensor of shape (3p-2)p by 4): the constraint matrix, C, such that vec(B) = C * beta

        Returns:
            beta_tilde (torch.tensor of shape 4): the desired 4 dimensional estimator
        """
        A = constraint
        return(torch.linalg.solve(A.T @ A, A.T) @ B.reshape(-1))
    
    @staticmethod
    def gen_constraint(lat_dim, constrained = False, core_only = False):


        """ 
        Let beta be the m dimensional parameter of interest. Let the (3p-2) by p matrix B be the parameter that we estimate.
        This generate a constraint matrix, C, such that vec(B) = C * beta

        Args: 
            lat_dim (int): the dimension of the latent space
            constrained (bool): if true, beta is assumped to be 4 dimensional, otherwise, beta is assumed to be vec(B)

        Returns:
            constraint (torch.tensor of shape (3p-2)*p by m): when constrained is true, return a (3p-2)p by 4 matrix, C, such that vec(B) = C * beta
            when constrained is false, return the (3p-2)p identity
        
        """

        p = lat_dim
        if constrained == False:
            A = torch.eye((3*(p-1)+1)*p)

        else:
            temp = torch.cat([torch.eye(p-1), -torch.ones(p-1).unsqueeze(0).T], dim = 1).reshape(-1)
            core = torch.kron(torch.eye(3), temp).T
            right = torch.zeros(core.shape[0], 1)
            bot = torch.cat([torch.tensor([0,0,0,1.0]).unsqueeze(0)]*(p-1), dim = 0)
            bot = torch.cat([bot, torch.ones(4).unsqueeze(0)])
            A = torch.cat((torch.cat((core, right), dim = 1), bot), dim = 0)
            
        if core_only: 
            return(core)

        return(A)
    
    @staticmethod
    def exclu_neg_res_pred(response, predictor):

        """ 

        Exclude all response-predictor pair that contains negative entries from ASE

        Args:
            response (torch.tensor of shape n by p)
            predictor (torch.tensor of shape n by (3p-2))
        
        Returns: 
            predictor and response with rows that have negative entries removed.

        """
        
        n, p = response.shape
        temp = torch.cat((response, predictor), dim = 1)
        indicator = (temp >= 0) & (temp <= 1)
        filtered = temp[indicator.all(dim = 1)]
        
        # Further filter rows where the sum of the row is less than or equal to 1
        final_filtered = filtered[(filtered[:, :p-1].sum(dim = 1) <= 1) & (filtered[:, p:(2*p - 1)].sum(dim = 1) <= 1)]

        # Split the final filtered tensor back into response and predictor
        response, pred = final_filtered[:, :p], final_filtered[:, p:]
        n_new = response.shape[0]

        result_dict = {"pred": pred, "response": response, "n_new": n_new}

        return(result_dict)
    
    @staticmethod
    def linear_init(predictor, response, beta_0_guess, tol = 10e-3):

        """ 
        Returns an initialization for the gradient descent algorithm based on some method of moment heuristic. 

        Args: 
            response (torch.tensor of shape n by p)
            predictor (torch.Tensor of shape n by (3p-2))
            beta_0_guess (float, negative): the intercept that the heuristic cannot estimate. Use smaller guess, e.g.
            -10 -> -20, when trouble is encountered

        Returns:
            (torch.tensor of shape 4): A seemingly reliable matrix to initialize the gradient descent. 
         
        """

        n, p = response.shape
        pred = predictor[:, :(3*(p-1))]

        split = fit.exclu_neg_res_pred(response, pred)
        response, pred, n_new = split["response"], split["pred"], split["n_new"]

        H = torch.linalg.solve((pred.T @ pred), pred.T)

        constraint_no_int = fit.gen_constraint(p, True, True)
        
        first_estimate_no_int = H.matmul(torch.log(response))

        current_estimate = first_estimate_no_int
        
        result = torch.cat((fit.proj_beta(current_estimate, constraint_no_int), 
                            torch.tensor([beta_0_guess])))

        return(result)
    
    @staticmethod
    def grad_dir_reg(predictor, response, reg_parameter, constrained = False):

        """
        Computes the gradient of the loss function for a Dirichlet Generalized Linear Model (GLM)
        with a logarithmic link function, optionally including a regularization term.

        Args:
            response (torch.Tensor of shape n by p)
            predictor (torch.Tensor of shape n by (3p-2))
            reg_parameter (float): beta, the parameter that derivative is taken with respect to.
            constrained (bool, optional): whether beta is estimated as a 4  or (3p-2)p dimensional vector

        Returns:
            torch.Tensor: The gradient of the loss function with respect to the model parameters.

        """

        n, p = response.shape
        constraint = fit.gen_constraint(p, constrained).to(device)

        predictor = predictor.to(device)
        alpha = torch.exp(torch.matmul(predictor, reg_parameter.to(device))).to(device)
        full_alpha = torch.stack([alpha.reshape(-1)]*(constraint.shape[1]), dim = 1)
        
        predictor = torch.kron(predictor.to(device), torch.eye(p).to(device))
        if device != torch.device("mps"):
            predictor = predictor.to_sparse()
        response = response.to(device)
        predictor_mod = torch.sparse.mm(predictor, constraint)

        digamma_alpha_sum = torch.digamma(torch.sum(alpha, axis = 1))
        mean_log = torch.digamma(alpha) - torch.stack([digamma_alpha_sum]*p).T

        bias_log = (torch.log(response) - mean_log.to(device)).reshape(-1) 

        grad = bias_log.matmul(predictor_mod * full_alpha)

        grad = grad.to("cpu")

        del(predictor, response, constraint, predictor_mod, alpha, full_alpha, digamma_alpha_sum, mean_log, bias_log)
        torch.cuda.empty_cache()

        return(grad)
    
    
    
    @staticmethod
    def fish_dir_reg(predictor, response, reg_parameter, constrained = False):

        
        """
        Computes the Fisher's Information Matrix of the loss function for a Dirichlet GLM with a log link.

        Args:
            response (torch.Tensor of shape n by p)
            predictor (torch.Tensor of shape n by p)
            reg_parameter (torch.Tensor of shape (3p - 2) by p): beta, the parameter of interest.
            constrained (bool, optional): whether beta is estimated as a 4  or (3p-2)p dimensional vector

        Returns:
            (torch.Tensor of shape 4 by 4 or (3p-2)p by (3p-2)p): The Fisher's Info of the MLE of the model parameters.

        """

        n, p = response.shape
        constraint = fit.gen_constraint(p, constrained).to(device)
        
        predictor = predictor.to(device)
        alpha = torch.exp(torch.matmul(predictor, reg_parameter.to(device))).to(device)
        full_alpha = torch.stack([alpha.reshape(-1)]*(constraint.shape[1]), dim = 1)

        predictor = torch.kron(predictor, torch.eye(p).to(device))
        if device != torch.device("mps"):
            predictor = predictor.to_sparse()
        response = response.to(device)
        predictor_mod = torch.sparse.mm(predictor, constraint.to(device))

        p1 = predictor_mod * full_alpha
    
        var_p1 = torch.diag(torch.polygamma(1, alpha).reshape(-1))

        var_p2 = torch.kron(torch.diag(torch.polygamma(1, torch.sum(alpha, axis = 1))), torch.ones(p, p).to(device))

        if device != torch.device("mps"):
            var_p1 = var_p1.to_sparse()
            var_p2 = var_p2.to_sparse()
    
        var_dir = var_p1 - var_p2

        fish = p1.T.matmul(torch.sparse.mm(var_dir, p1))

        fish = fish.to("cpu")

        del(predictor, response, constraint, predictor_mod, alpha, full_alpha, p1, var_p1, var_p2, var_dir)
        torch.cuda.empty_cache()

        return(fish)
    


    def Dir_NGD_unconstrained(self, tol = 1):
        """ 
        Fisher's Scoring algorithm. When the step size exceeds the tolerance after 100 iterations, returns the 0 matrix.
        
        Returns:
            estimate (torch.tensor of shape (3p -2) by p): the MLE
            fisher_info (torch.Tensor of shape 4 by 4 or (3p-2)p by (3p-2)p): The Fisher's Info of the MLE of the model parameters.
            info_lost (float): percentage of rows in design matrix and response that contains negative entries
        """
        if self.est_result is None:
            self.Dir_GD_unconstrained()
            
        start_time = time()
        
        predictor, response = self.predictor.to(device), self.response.to(device)
        n, p = response.shape
        good_df = fit.exclu_neg_res_pred(response, predictor)
        predictor, response, n_new = good_df.values()

        constrained = self.settings.constrained
        C = fit.gen_constraint(p, constrained).to(device)
        
        C_mat = fit.gen_constraint(p, True).to(device)
        
        B = self.est_result["estimate"].to(device)
        B = torch.linalg.solve(C_mat.T @ C_mat , C_mat.T @ B.reshape(-1)).reshape(-1)
    
        B = (C_mat @ B).reshape(3*p-2, p)
        
        i, max_iter = 1, self.settings.max_iter   
        go = True

        while go:
            
            current_gradient = fit.grad_dir_reg(predictor, response, B, constrained)

            current_fisher_info = fit.fish_dir_reg(predictor, response, B, constrained)
        
            step = torch.linalg.solve(current_fisher_info, current_gradient).to(device)
            B = B + (C @ step).reshape(3*p-2, p)

            step_size_now = torch.norm(step, p = "fro")
            
            i += 1
            go = (step_size_now > tol) & (i < max_iter) 

        current_fisher_info = current_fisher_info.to("cpu")
        end_time = time()
        result_dic = {"estimate" : B.to("cpu"), 
                      "fisher_info": current_fisher_info, 
                      "info_lost": (1 - n_new/n), 
                      "num_iter": i - 1, 
                      "max_iter": self.settings.max_iter,
                      "time_elapsed": end_time - start_time}

        del(current_gradient, current_fisher_info, step, B, C, C_mat)
        torch.cuda.empty_cache()
        
        self.NGD_result = result_dic
    
    
    def Dir_GD_unconstrained(self):
        
        start_time = time()
        
        predictor, response = self.predictor.to(device), self.response.to(device)
        n, p = response.shape
        good_df = fit.exclu_neg_res_pred(response, predictor)
        predictor, response, n_new = good_df.values()
        
        constrained = self.settings.constrained

        C = fit.gen_constraint(p, True).to(device)
        
        B_guess = self.settings.beta_guess.to(torch.float32).to(device)
        
        B = (C @ B_guess).reshape(3*p-2, p).to(device)
        B.requires_grad_(True)
        Adam = torch.optim.Adam([B], lr = 0.01)
        
        i, max_iter, tol = 1, self.settings.max_iter, self.settings.tol
        go = True

        while go:
            
            Adam.zero_grad()
            manual_gradient = fit.grad_dir_reg(predictor, response, B, constrained).to(device)
            
            if constrained:
                manual_gradient = (C @ manual_gradient).reshape(3*p-2, p)
            
            B.grad = -manual_gradient.reshape(3*p-2, p)
            Adam.step()
            
            grad_size = torch.norm(B.grad, p = "fro")
            go = (i < max_iter) & (grad_size > tol)
            
            if self.settings.verbose:
                if i % 1000 == 0:
                    current_est = (torch.linalg.solve(C.T @ C, C.T) @ B.reshape(-1)).cpu().detach().numpy().tolist()
                    print(f"i = {i}, \ncurrent estimate of \u03B2 is {[round(i, 2) for i in current_est]}, \nthe size of the current gradient is {grad_size.item(): 2f}")
            
            i += 1

        end_time = time()
        
        fish = fit.fish_dir_reg(predictor, response, B)
        
        result_dic = {"estimate" : B.cpu().detach(), 
                      "fisher_info": fish.cpu(), 
                      "info_lost": (1 - n_new/n), 
                      "num_iter": i - 1, 
                      "max_iter": self.settings.max_iter,
                      "time_elapsed": end_time - start_time}
        
        del(predictor, response, C, B)
        torch.cuda.empty_cache()
        
        self.est_result = result_dic
        return()
    
    
    
#Given observations a Dirichlet random variable, estimate the parameter alpha using MOME and MLE
class est_alpha:
    """
    To estimate the parameter of a Dirichlet random variable using MLE

    Args:
        observation (torch.Tensor): Tensor of shape (n, p) containing i.i.d. samples of a Dirichlet random variable.
        tol (float): Tolerance for the gradient descent method (default is 10e-5).

    Attributes:
        observation (torch.Tensor): Tensor of shape (n, p) containing i.i.d. samples of a Dirichlet random variable.
        tolerance (float): Tolerance for the gradient descent method.
        MOME (torch.Tensor): Method of moment estimator for Dirichlet random variables.
        MLE (torch.Tensor): Maximum likelihood estimator for Dirichlet random variables.
    """
    def __init__(self, observation, tol = 10e-5):
        
        self.observation = observation
        self.tolerance = tol
        self.MOME = self.method_of_moment()
        self.MLE = self.MLE()

    
    def method_of_moment(self):

        """
        Computes the method of moment estimator (MOME) for Dirichlet random variables.

        Returns:
            torch.Tensor: The MOME.
        """

        E = self.observation.mean(dim = 0)
        V = torch.var(self.observation, dim = 0)

        mome = E * (E*(1 - E)/V - 1)

        return(mome)
    
    def likelihood(self, alpha, log_likelihood = False):

        """  
        Given observations, compute the likelihood or log-likelihood, L(X; alpha)

        Args:
            alpha(torch.Tensor): a component-wise positive vector
            log_likelihood(boolean): when true, the output becomes the log likelihood

        Returns:
            float: the likelihood or log-likelihood
        """

        alpha.to(device)
        alphap = alpha - 1

        c = torch.exp(torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum())
        likelihood = c * (self.observation ** alphap).prod(axis=1)
        likelihood.to("cpu")

        del(alpha, alphap, c)
        torch.cuda.empty_cache()
    
        if log_likelihood:
            return(torch.log(likelihood).sum())
        else:
            return(likelihood.sum())
    
    @staticmethod
    def mean_log(alpha):

        """  
        Expectation of the log(Dir(alpha)) distribution

        Args:
            alpha(torch.Tensor): a component-wise positive vector

        Returns:
            torch.Tensor of length p: the expected value of the log(Dir(alpha)) distribution
        """

        mean_of_log = (torch.digamma(alpha) - torch.digamma(alpha.sum()))

        return(mean_of_log)
    
    @staticmethod
    def var_log(alpha, inverse = False):

        """  
        Variance matrix of the log(Dir(alpha)) distribution

        Args:
            alpha(torch.Tensor): a component-wise positive vector
            inverse(boolean): when true, the output becomes the inverse of the variance matrix

        Returns:
            torch.Tensor: the variance matrix of the log(Dir(alpha)) distribution
        """
 
        p = alpha.shape[0]
        one_p = torch.ones(p).unsqueeze(1)
        c = torch.polygamma(1, alpha.sum())
        Q = -torch.polygamma(1, alpha)
        Q_inv = torch.diag(1/Q)

       #When inverse is true, the Sherman-Morrison formula is used to compute the inverse of the variance matrix
        if inverse:

            numerator = (Q_inv @ (c*one_p @ one_p.T) @ Q_inv)

            denominator = (1 + c * one_p.T @ Q_inv @ one_p)

            var_inv = Q_inv - numerator/denominator

            return(var_inv)
        else:
            
            return(torch.diag(Q) + c)

    def MLE(self):

        """  
        Given observations of a Dirichlet random variable, compute the maximum likelihood estimator

        Returns:
            torch.Tensor: the estimated parameter vector
        """
        #Given observations, compute the maximum likelihood estimator using newton gradient descent
        #The gradient descent is initialized on the computed MOME
        
        empirical_avg_log = torch.log(self.observation).mean(dim = 0)

        initialization = self.MOME

        tol = self.tolerance

        next_par = initialization

        go = True
        i = 1
        while go:
            var_inv = self.var_log(next_par, inverse = True)
            
            log_mean = empirical_avg_log - self.mean_log(next_par)

            step = var_inv @ (log_mean)

            next_par = next_par - step

            go = (i <= 1000) & (torch.norm(step) > tol)

            i += 1

        return(next_par)

