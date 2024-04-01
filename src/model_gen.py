#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#To generate
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
import torch
from torch.distributions import Dirichlet, Bernoulli, Uniform
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Settings:
    def __init__(self, time, nodes, latent_space_dim, groups, parameter_vec, starting_lat_parameter_mat, constraint):
        self.time = time
        self.nodes = nodes
        self.latent_space_dim = latent_space_dim
        self.groups = groups
        self.parameter_vec = parameter_vec
        self.starting_lat_parameter_mat = starting_lat_parameter_mat
        self.constraint = self.gen_constraint(constraint)
        self.parameter_mat = self.gen_beta_mat()
        self.rand_initialization = self.gen_init()

    def gen_constraint(self, constraint):
        p = self.latent_space_dim
        q = 3 * (p - 1) + 1
        if constraint:
            temp = torch.cat([torch.eye(p-1), -torch.ones(p-1).unsqueeze(0).T], dim = 1)
            temp = temp.reshape(-1)
            core = torch.kron(torch.eye(3), temp).T
            right = torch.zeros(core.shape[0], 1)
            bot = torch.cat([torch.tensor([0,0,0,1.0]).unsqueeze(0)]*(p-1), dim = 0)
            bot = torch.cat([bot, torch.ones(4).unsqueeze(0)])
            A = torch.cat((torch.cat((core, right), dim = 1), bot), dim = 0)
        else:
            A = torch.eye(q * p)
        return(A)
    
    def gen_beta_mat(self):
        p = self.latent_space_dim
        parameter_mat = (self.constraint @ self.parameter_vec).reshape(3 * (p-1)+1, p)
        return(parameter_mat)
    
    def gen_init(self):
        n = self.nodes
        K = self.groups
        prev_state =  torch.random.get_rng_state()
        torch.manual_seed(42069)
        init_dist = Uniform(0.2, 0.8)
        init = init_dist.sample((n*K,)).reshape(n, 3)
        torch.random.set_rng_state(prev_state)
        return(init)


    def change_node(self, new_nodes):
        self.nodes = new_nodes
        self.rand_initialization = self.gen_init()

    def change_init_distr(self, new_parameter):
        self.starting_lat_parameter_mat = new_parameter
        
    def change_parameter_vec(self, new_vec):
        self.parameter_vec = new_vec
        self.parameter_mat = self.gen_beta_mat()
    
    def change_constraint(self, new_constraint):
        self.constraint = self.gen_constraint(new_constraint)

def proj_beta(B, constraint):
    A = constraint
    return(torch.linalg.solve(A.T @ A, A.T) @ B.reshape(-1))

#default settings
N, n, p, K = 2, 3000, 3, 3
current_settings = Settings(time = N,
                nodes = n,
                latent_space_dim = p,
                groups = K, 
                parameter_vec = torch.tensor([1,1,-4,5.0]),
                starting_lat_parameter_mat = torch.tensor([[1, 1, 10], [1, 10, 1], [10, 1, 1]], dtype= torch.float).reshape(K, p),
                constraint = True)


def init_Z(cs = current_settings):
    nk = int(cs.nodes/cs.groups)
    init_dist = Dirichlet(cs.starting_lat_parameter_mat)
    init_samp = init_dist.sample((nk,)).transpose(0,1).reshape(cs.groups*nk, cs.latent_space_dim)
    return(init_samp)

def gen_Y(Z):
    n, p = Z.shape
    #take out the last dimension to deal with linear dependency
    Z_s = Z[:, :(p-1)].to(device)      

    #generate the parameter probability matrix from the latent positions, since the proceeding adj matrix has to be symmetric, we only keep the upper triangular portiion.
    P = torch.matmul(Z_s, Z_s.T).triu(diagonal = 1)     

    #sampling form the Bernoulli distributions as defined by P. we choose the 0th element because sampling returns a [1 x n x n] tensor
    Y_half = Bernoulli(P).sample((1,))[0]
    #symmetricize the adjacency matrix       
    Y = Y_half + Y_half.T
    
    del(Z_s, P, Y_half)

    Y = Y.to("cpu")
    torch.cuda.empty_cache() 

    return(Y)

def gen_X(Y, Z, cs = current_settings):
    Y = Y.to(device)
    n, p = Z.shape
    nk = int(n/cs.groups)
    
    Z_s = Z[:, :(p-1)].to(device)

    #within-group indicator matrix -- jth element of ith row is 1 IF j and i has the same group membership
    wthin_g = torch.kron(torch.eye(cs.groups), torch.ones(nk, nk)).to(device)
    #between-group indicator matrix       
    btwn_g  = (-1)*(wthin_g - 1)        

    Y1 = (Y * wthin_g)
    Y2 = (Y * btwn_g)

    all_ones = torch.ones(n,1).to(device)

    #take row/col sums of within/between group adjacency matrix -- this will be the scaling factor
    S1 = torch.max(all_ones, torch.sum(Y1, 1, True)).repeat(1, (p-1))       
    S2 = torch.max(all_ones, torch.sum(Y2, 1, True)).repeat(1, (p-1))

    A1 = Y1.matmul(Z_s)/S1
    A2 = Y2.matmul(Z_s)/S2

    X = torch.cat((Z_s, A1, A2, all_ones), 1)

    
    X = X.to("cpu")
    del(Z_s, wthin_g, btwn_g, Y1, Y2, all_ones, S1, S2, A1, A2)
    torch.cuda.empty_cache()

    return(X)

def next_Z(Y, Z, cs = current_settings):

    parameter_mat = cs.parameter_mat.to(device)

    Y = Y.to(device)
    Z = Z.to(device)
    X = gen_X(Y, Z, cs).to(device)

    alpha = torch.exp(X.matmul(parameter_mat))
    samp = Dirichlet(alpha).sample((1,))[0]

    samp = samp.to("cpu")
    del(Y, Z, X, alpha)
    torch.cuda.empty_cache()
    
    return(samp)




def grad_dir_reg(predictor, response, reg_parameter, cs = current_settings):
    n, p = response.shape

    predictor = predictor.to(device)
    alpha = torch.exp(torch.matmul(predictor, reg_parameter.to(device))).to(device)
    full_alpha = torch.stack([alpha.reshape(-1)]*(cs.constraint.shape[1]), dim = 1)
    # check the K parameter here, might just be a 4

    predictor = torch.kron(predictor.to(device), torch.eye(p).to(device)).to_sparse()
    response = response.to(device)
    predictor_mod = torch.sparse.mm(predictor, cs.constraint.to(device))

    digamma_alpha_sum = torch.digamma(torch.sum(alpha, axis = 1))
    mean_log = torch.digamma(alpha) - torch.stack([digamma_alpha_sum]*p).T

    bias_log = (torch.log(response) - mean_log.to(device)).reshape(-1) 

    grad = bias_log.matmul(predictor_mod * full_alpha)

    grad = grad.to("cpu")

    del(predictor, response, predictor_mod, alpha, full_alpha, digamma_alpha_sum, mean_log, bias_log)
    torch.cuda.empty_cache()
    return(grad)

def fish_dir_reg(predictor, response, reg_parameter, cs = current_settings):

    n, p = response.shape
    
    predictor = predictor.to(device)
    alpha = torch.exp(torch.matmul(predictor, reg_parameter.to(device))).to(device)
    full_alpha = torch.stack([alpha.reshape(-1)]*(cs.constraint.shape[1]), dim = 1)

    predictor = torch.kron(predictor, torch.eye(p).to(device)).to_sparse()
    response = response.to(device)
    predictor_mod = torch.sparse.mm(predictor, cs.constraint.to(device))

    p1 = predictor_mod * full_alpha
   
    var_p1 = torch.diag(torch.polygamma(1, alpha).reshape(-1)).to_sparse()

    var_p2 = torch.kron(torch.diag(torch.polygamma(1, torch.sum(alpha, axis = 1))), torch.ones(p, p).to(device)).to_sparse()
   
    var_dir = var_p1 - var_p2

    fish = p1.T.matmul(torch.sparse.mm(var_dir, p1))

    fish = fish.to("cpu")


    del(predictor, response, predictor_mod, alpha, full_alpha, p1, var_p1, var_p2, var_dir)
    torch.cuda.empty_cache()
    return(fish)

def Dir_NGD(predictor, response, initial_estimate_mat, cs = current_settings, tolerance = 0.001):

    n, p = response.shape

    temp = torch.cat((response, predictor), dim = 1)
    indicator = (temp > 0) & (temp <= 1)
    filtered = temp[indicator.all(dim = 1)]
    response, predictor = filtered[:,:p], filtered[:,p:]
    n_new = response.shape[0]

    next_estimate = initial_estimate_mat
    go = True
    i = 1
    while go:
    
        current_gradient = grad_dir_reg(predictor, response, next_estimate, cs)
     
        current_fisher_info = fish_dir_reg(predictor, response, next_estimate, cs)
    
        step = torch.linalg.solve(current_fisher_info, current_gradient)
        
        next_estimate = next_estimate + (cs.constraint.matmul(step)).reshape(3*(p-1)+1, p)

        go = (torch.norm(step, p = "fro") > tolerance)
        
        i += 1
        if i > 100:
            return(initial_estimate_mat*0)
        
    result_dic = {"final_estimate" : next_estimate, "final_fisher_info": current_fisher_info, "ASE_info_lost": (1 - n_new/n)}

    return(result_dic)

def Dir_Linear_Initialization(predictor, response, added_est_int = -10,  tolerance = 0.1):

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
        estimate_core = proj_beta(estimate_no_int, constraint_no_int)

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

        change = torch.norm(proj_beta((new_estimate - current_estimate), constraint_no_int), p = "fro")
        current_estimate = new_estimate
        go = change > tolerance
        i += 1
    
    result = torch.cat((proj_beta(current_estimate, constraint_no_int), torch.tensor([added_est_int])))
        
    return(result)