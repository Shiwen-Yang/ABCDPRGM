import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class fit:

    def __init__(self, predictor, response, constrained = False, beta_guess = None, beta_0_guess = -10, tol = 10e-3):
        self.predictor = predictor
        self.response = response
        self.GD_tolerance = tol
        self.constrained = constrained
        self.beta_0_guess = beta_0_guess
        self.beta_guess = beta_guess
        self.est_result = self.Dir_NGD()

    @staticmethod
    def proj_beta(B, constraint):
        A = constraint
        return(torch.linalg.solve(A.T @ A, A.T) @ B.reshape(-1))
    
    @staticmethod
    def gen_constraint(lat_dim, constrained = False):

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
    def linear_init(predictor, response, beta_0_guess, tol = 10e-3):

        n, p = response.shape
        pred = predictor[:, :(3*(p-1))]


        temp = torch.cat((response, pred), dim = 1)
        indicator = (temp > 0) & (temp <= 1)
        filtered = temp[indicator.all(dim = 1)]
        response, pred = filtered[:,:p], filtered[:,p:]
        n_new = response.shape[0]

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
        n, p = response.shape
        constraint = fit.gen_constraint(p, constrained).to(device)

        predictor = predictor.to(device)
        alpha = torch.exp(torch.matmul(predictor, reg_parameter.to(device))).to(device)
        full_alpha = torch.stack([alpha.reshape(-1)]*(constraint.shape[1]), dim = 1)

        predictor = torch.kron(predictor.to(device), torch.eye(p).to(device)).to_sparse()
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

        n, p = response.shape
        constraint = fit.gen_constraint(p, constrained).to(device)
        
        predictor = predictor.to(device)
        alpha = torch.exp(torch.matmul(predictor, reg_parameter.to(device))).to(device)
        full_alpha = torch.stack([alpha.reshape(-1)]*(constraint.shape[1]), dim = 1)

        predictor = torch.kron(predictor, torch.eye(p).to(device)).to_sparse()
        response = response.to(device)
        predictor_mod = torch.sparse.mm(predictor, constraint.to(device))

        p1 = predictor_mod * full_alpha
    
        var_p1 = torch.diag(torch.polygamma(1, alpha).reshape(-1)).to_sparse()

        var_p2 = torch.kron(torch.diag(torch.polygamma(1, torch.sum(alpha, axis = 1))), torch.ones(p, p).to(device)).to_sparse()
    
        var_dir = var_p1 - var_p2

        fish = p1.T.matmul(torch.sparse.mm(var_dir, p1))

        fish = fish.to("cpu")

        del(predictor, response, constraint, predictor_mod, alpha, full_alpha, p1, var_p1, var_p2, var_dir)
        torch.cuda.empty_cache()

        return(fish)
    
    def Dir_NGD(self):

        constrained = self.constrained
        predictor = self.predictor
        response = self.response
        n, p = response.shape
        B = fit.gen_constraint(p, True).to(device)
        constraint = fit.gen_constraint(p, constrained).to(device)

        if self.beta_guess is not None:
            init_est_vec = self.beta_guess.to(device)
        else: 
            init_est_vec = fit.linear_init(predictor, response, self.beta_0_guess).to(device)
        
        init_est_mat = (B @ init_est_vec).reshape(3*(p-1)+1, p)

        temp = torch.cat((response, predictor), dim = 1)
        indicator = (temp >= 0) & (temp <= 1)
        filtered = temp[indicator.all(dim = 1)]
        
        response, predictor = filtered[:,:p], filtered[:,p:]
        n_new = response.shape[0]
     
        next_estimate = init_est_mat

        go = True
        i = 1
        while go:
        
            current_gradient = fit.grad_dir_reg(predictor, response, next_estimate, constrained)
        
            current_fisher_info = fit.fish_dir_reg(predictor, response, next_estimate, constrained)
        
            step = torch.linalg.solve(current_fisher_info, current_gradient).to(device)
            
            next_estimate = next_estimate + (constraint.matmul(step)).reshape(3*(p-1)+1, p)

            go = (torch.norm(step, p = "fro") > self.GD_tolerance)
            
            i += 1
            if i > 100:
                return(init_est_mat*0)
            
        result_dic = {"final_estimate" : next_estimate, "final_fisher_info": current_fisher_info, "ASE_info_lost": (1 - n_new/n)}

        return(result_dic)