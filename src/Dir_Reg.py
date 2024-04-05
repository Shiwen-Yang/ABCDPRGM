import torch
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

    def __init__(self, predictor, response, constrained = False, beta_guess = None, beta_0_guess = -10, tol = 10e-3):
        self.predictor = predictor
        self.response = response
        self.settings = self.reg_settings(constrained, beta_guess, beta_0_guess, tol)
        self.est_result = self.Dir_NGD()

    class reg_settings:
        def __init__(self, constrained, beta_guess, beta_0_guess, tol):
            self.constrained = constrained
            self.beta_guess = beta_guess
            self.beta_0_guess = beta_0_guess
            self.tol = tol
    
    def update_settings(self, constrained = None, beta_guess = None, beta_0_guess = None, tol = None):

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
            "beta_0_guess": beta_0_guess if beta_0_guess is not None else self.settings.beta_0_guess,
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
    def gen_constraint(lat_dim, constrained = False):


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
        response, pred = filtered[:,:p], filtered[:,p:]
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

        constraint_no_int = torch.kron(torch.eye(3), torch.tensor([[1,0,-1,0,1,-1]])).T

        first_estimate_no_int = H.matmul(torch.log(response))

        def new_response(estimate_no_int):
            estimate_core = fit.proj_beta(estimate_no_int, constraint_no_int)

            L = torch.tensor([0, 0, torch.sum(estimate_core)])
            exp_lambda = torch.exp(L).repeat(p,1).T

            new_response = torch.log(response) + torch.log( torch.exp(pred @ estimate_no_int) @ exp_lambda ) - L.repeat(n_new, 1)
            return(new_response)

        current_estimate = first_estimate_no_int

        current_response = response
        go = True
        i = 1
        while go and i < 100:
            current_response = new_response(current_estimate)

            new_estimate = H @ current_response

            change = torch.norm(fit.proj_beta((new_estimate - current_estimate), constraint_no_int), p = "fro")
            current_estimate = new_estimate
            go = change > tol
            i += 1
        
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

        # some things in here are done on every iteration but could be done in advance
        # could set up an internal function to call and if it hasn't been done already then 
        # you can run it otherwise just return what you have
        # already computed
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
    
    def Dir_NGD(self):
        """ 
        Fisher's Scoring algorithm. When the step size exceeds the tolerance after 100 iterations, returns the 0 matrix.
        
        Returns:
            estimate (torch.tensor of shape (3p -2) by p): the MLE
            fisher_info (torch.Tensor of shape 4 by 4 or (3p-2)p by (3p-2)p): The Fisher's Info of the MLE of the model parameters.
            info_lost (float): percentage of rows in design matrix and response that contains negative entries
        """

        constrained = self.settings.constrained
        predictor, response = self.predictor, self.response

        n, p = response.shape
        B = fit.gen_constraint(p, True).to(device)
        constraint = fit.gen_constraint(p, constrained).to(device)

        if self.settings.beta_guess is not None:
            init_est_vec = self.settings.beta_guess.to(device)
        else: 
            init_est_vec = fit.linear_init(predictor, response, self.settings.beta_0_guess).to(device)
        
        init_est_mat = (B @ init_est_vec).reshape(3*(p-1)+1, p)


        split = fit.exclu_neg_res_pred(response, predictor)
        response, predictor, n_new = split["response"], split["pred"], split["n_new"]
     
        next_estimate = init_est_mat

        go = True
        i = 1
        while go:
        
            current_gradient = fit.grad_dir_reg(predictor, response, next_estimate, constrained)
        
            current_fisher_info = fit.fish_dir_reg(predictor, response, next_estimate, constrained)
        
            step = torch.linalg.solve(current_fisher_info, current_gradient).to(device)
            
            next_estimate = next_estimate + (constraint.matmul(step)).reshape(3*(p-1)+1, p)

            go = (torch.norm(step, p = "fro") > self.settings.tol)
            
            i += 1
            if i > 100:
                return(init_est_mat*0)
        
        
        next_estimate = next_estimate.to("cpu")
        current_fisher_info = current_fisher_info.to("cpu")
        result_dic = {"estimate" : next_estimate, "fisher_info": current_fisher_info, "info_lost": (1 - n_new/n)}

        del(current_gradient, current_fisher_info, step, next_estimate, B, constraint, init_est_vec)
        torch.cuda.empty_cache()

        return(result_dic)