import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Dirichlet_GLM_log:
    def __init__(self, predictor, response, tol = 10e-3, intercept_est = -10, init_est = False):
        self.predictor = predictor
        self.response = response
        self.GD_tolerance = tol
        self.int_est = intercept_est
        self.init_est = init_est
        self.GD_init = self.linear_init()

    @staticmethod
    def proj_beta(B, constraint):
        A = constraint
        return(torch.linalg.solve(A.T @ A, A.T) @ B.reshape(-1))
    
    def gen_constraint(self):
        n, p = self.response.shape

        temp = torch.cat([torch.eye(p-1), -torch.ones(p-1).unsqueeze(0).T], dim = 1)
        temp = temp.reshape(-1)
        core = torch.kron(torch.eye(3), temp).T
        right = torch.zeros(core.shape[0], 1)
        bot = torch.cat([torch.tensor([0,0,0,1.0]).unsqueeze(0)]*(p-1), dim = 0)
        bot = torch.cat([bot, torch.ones(4).unsqueeze(0)])
        A = torch.cat((torch.cat((core, right), dim = 1), bot), dim = 0)

        return(A)
    

    def linear_init(self):

        if self.init_est  != False:
            return(self.init_est)
        else:
            response = self.response
            n, p = response.shape
            pred = self.predictor[:, :(3*(p-1))]


            temp = torch.cat((response, pred), dim = 1)
            indicator = (temp > 0) & (temp <= 1)
            filtered = temp[indicator.all(dim = 1)]
            response, pred = filtered[:,:p], filtered[:,p:]
            n_new = response.shape[0]

            H = torch.linalg.solve((pred.T @ pred), pred.T)

            constraint_no_int = torch.kron(torch.eye(3), torch.tensor([[1,0,-1,0,1,-1]])).T

            first_estimate_no_int = H.matmul(torch.log(response))

            def new_response(estimate_no_int):
                estimate_core = Dirichlet_GLM_log.proj_beta(estimate_no_int, constraint_no_int)

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

                change = torch.norm(Dirichlet_GLM_log.proj_beta((new_estimate - current_estimate), constraint_no_int), p = "fro")
                current_estimate = new_estimate
                go = change > self.GD_tolerance
                i += 1
            
            result = torch.cat((Dirichlet_GLM_log.proj_beta(current_estimate, constraint_no_int), 
                                torch.tensor([self.int_est])))
        return(result)



# def grad_dir_reg(predictor, response, reg_parameter, cs = current_settings):
#     n, p = response.shape

#     predictor = predictor.to(device)
#     alpha = torch.exp(torch.matmul(predictor, reg_parameter.to(device))).to(device)
#     full_alpha = torch.stack([alpha.reshape(-1)]*(cs.constraint.shape[1]), dim = 1)
#     # check the K parameter here, might just be a 4

#     predictor = torch.kron(predictor.to(device), torch.eye(p).to(device)).to_sparse()
#     response = response.to(device)
#     predictor_mod = torch.sparse.mm(predictor, cs.constraint.to(device))

#     digamma_alpha_sum = torch.digamma(torch.sum(alpha, axis = 1))
#     mean_log = torch.digamma(alpha) - torch.stack([digamma_alpha_sum]*p).T

#     bias_log = (torch.log(response) - mean_log.to(device)).reshape(-1) 

#     grad = bias_log.matmul(predictor_mod * full_alpha)

#     grad = grad.to("cpu")

#     del(predictor, response, predictor_mod, alpha, full_alpha, digamma_alpha_sum, mean_log, bias_log)
#     torch.cuda.empty_cache()
#     return(grad)

# def fish_dir_reg(predictor, response, reg_parameter, cs = current_settings):

#     n, p = response.shape
    
#     predictor = predictor.to(device)
#     alpha = torch.exp(torch.matmul(predictor, reg_parameter.to(device))).to(device)
#     full_alpha = torch.stack([alpha.reshape(-1)]*(cs.constraint.shape[1]), dim = 1)

#     predictor = torch.kron(predictor, torch.eye(p).to(device)).to_sparse()
#     response = response.to(device)
#     predictor_mod = torch.sparse.mm(predictor, cs.constraint.to(device))

#     p1 = predictor_mod * full_alpha
   
#     var_p1 = torch.diag(torch.polygamma(1, alpha).reshape(-1)).to_sparse()

#     var_p2 = torch.kron(torch.diag(torch.polygamma(1, torch.sum(alpha, axis = 1))), torch.ones(p, p).to(device)).to_sparse()
   
#     var_dir = var_p1 - var_p2

#     fish = p1.T.matmul(torch.sparse.mm(var_dir, p1))

#     fish = fish.to("cpu")


#     del(predictor, response, predictor_mod, alpha, full_alpha, p1, var_p1, var_p2, var_dir)
#     torch.cuda.empty_cache()
#     return(fish)

# def Dir_NGD(predictor, response, initial_estimate_mat, cs = current_settings, tolerance = 0.001):

#     n, p = response.shape

#     temp = torch.cat((response, predictor), dim = 1)
#     indicator = (temp > 0) & (temp <= 1)
#     filtered = temp[indicator.all(dim = 1)]
#     response, predictor = filtered[:,:p], filtered[:,p:]
#     n_new = response.shape[0]

#     next_estimate = initial_estimate_mat
#     go = True
#     i = 1
#     while go:
    
#         current_gradient = grad_dir_reg(predictor, response, next_estimate, cs)
     
#         current_fisher_info = fish_dir_reg(predictor, response, next_estimate, cs)
    
#         step = torch.linalg.solve(current_fisher_info, current_gradient)
        
#         next_estimate = next_estimate + (cs.constraint.matmul(step)).reshape(3*(p-1)+1, p)

#         go = (torch.norm(step, p = "fro") > tolerance)
        
#         i += 1
#         if i > 100:
#             return(initial_estimate_mat*0)
        
#     result_dic = {"final_estimate" : next_estimate, "final_fisher_info": current_fisher_info, "ASE_info_lost": (1 - n_new/n)}

#     return(result_dic)

# def Dir_Linear_Initialization(predictor, response, added_est_int = -10,  tolerance = 0.1):

#     n, p = response.shape
#     pred = predictor[:, :(3*(p-1))]


#     temp = torch.cat((response, pred), dim = 1)
#     indicator = (temp > 0) & (temp <= 1)
#     filtered = temp[indicator.all(dim = 1)]
#     response, pred = filtered[:,:p], filtered[:,p:]
#     n_new = response.shape[0]

#     H = torch.linalg.solve((pred.T @ pred), pred.T)

#     constraint_no_int = torch.kron(torch.eye(3), torch.tensor([[1,0,-1,0,1,-1]])).T

#     first_estimate_no_int = H.matmul(torch.log(response))

#     def new_response(estimate_no_int):
#         estimate_core = proj_beta(estimate_no_int, constraint_no_int)

#         L = torch.tensor([0, 0, torch.sum(estimate_core)])
#         exp_lambda = torch.exp(L).repeat(p,1).T

#         new_response = torch.log(response) + torch.log( torch.exp(pred @ estimate_no_int) @ exp_lambda ) - L.repeat(n_new, 1)
#         return(new_response)

#     current_estimate = first_estimate_no_int

#     current_response = response
#     go = True
#     i = 1
#     while go and i < 100:
#         current_response = new_response(current_estimate)

#         new_estimate = H @ current_response

#         change = torch.norm(proj_beta((new_estimate - current_estimate), constraint_no_int), p = "fro")
#         current_estimate = new_estimate
#         go = change > tolerance
#         i += 1
    
#     result = torch.cat((proj_beta(current_estimate, constraint_no_int), torch.tensor([added_est_int])))
        
#     return(result)