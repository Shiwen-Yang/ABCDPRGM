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

class ABC_sim:
    def __init__(self, time, nodes, latent_space_dim, groups, parameter_vec, starting_lat_parameter_mat):
        self.time = time
        self.nodes = nodes
        self.latent_space_dim = latent_space_dim
        self.groups = groups
        self.parameter_vec = parameter_vec
        self.starting_lat_parameter_mat = starting_lat_parameter_mat
        self.constraint = self.gen_constraint()
        self.parameter_mat = self.gen_beta_mat()
        self.synth_data = self.simulate()

    def gen_constraint(self):

        p = self.latent_space_dim
        temp = torch.cat([torch.eye(p-1), -torch.ones(p-1).unsqueeze(0).T], dim = 1)
        temp = temp.reshape(-1)
        core = torch.kron(torch.eye(3), temp).T
        right = torch.zeros(core.shape[0], 1)
        bot = torch.cat([torch.tensor([0,0,0,1.0]).unsqueeze(0)]*(p-1), dim = 0)
        bot = torch.cat([bot, torch.ones(4).unsqueeze(0)])
        A = torch.cat((torch.cat((core, right), dim = 1), bot), dim = 0)

        return(A)
    
    def gen_beta_mat(self):
        p = self.latent_space_dim
        parameter_mat = (self.constraint @ self.parameter_vec).reshape(3 * (p-1)+1, p)
        return(parameter_mat)
    
    # def gen_init(self):
    #     n, p, K = self.nodes, self.latent_space_dim, self.groups
        
    #     prev_state =  torch.random.get_rng_state()
    #     torch.manual_seed(42069)
    #     init_dist = Uniform(0.2, 0.8)
    #     init = init_dist.sample((n*K,)).reshape(n, p)
    #     torch.random.set_rng_state(prev_state)
    #     return(init)


    def change_node(self, new_nodes):
        self.nodes = new_nodes
        # self.rand_initialization = self.gen_init()
        self.synth_data = self.simulate()

    def change_time(self, new_time):
        self.time = new_time
        self.synth_data = self.simulate()

    def change_init_distr(self, new_parameter):
        K, p = new_parameter.shape
        self.starting_lat_parameter_mat = new_parameter
        self.groups = K
        self.latent_space_dim = p
        self.constraint = self.gen_constraint()
        self.parameter_mat = self.gen_beta_mat()
        self.synth_data = self.simulate()
        
    def change_parameter_vec(self, new_vec):
        self.parameter_vec = new_vec
        self.parameter_mat = self.gen_beta_mat()
        self.synth_data = self.simulate()

    @staticmethod
    def init_Z(n_nodes, init_distr_mat):
        K, p = init_distr_mat.shape
        n = n_nodes
        init_dist = Dirichlet(init_distr_mat)
        init_samp = init_dist.sample((int(n/K),)).transpose(0, 1).reshape(n, p)
        return(init_samp)
    @staticmethod
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
    
    @staticmethod
    def gen_X(Y, Z, K_groups):
        Y = Y.to(device)
        n, p = Z.shape
        nk = int(n/K_groups)
        
        Z_s = Z[:, :(p-1)].to(device)

        #within-group indicator matrix -- jth element of ith row is 1 IF j and i has the same group membership
        wthin_g = torch.kron(torch.eye(K_groups), torch.ones(nk, nk)).to(device)
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
    
    @staticmethod
    def next_Z(Y, Z, para_mat, K_groups):

        para_mat = para_mat.to(device)

        Y = Y.to(device)
        Z = Z.to(device)
        X = ABC_sim.gen_X(Y, Z, K_groups).to(device)

        alpha = torch.exp(X.matmul(para_mat))
        samp = Dirichlet(alpha).sample((1,))[0]

        samp = samp.to("cpu")
        del(Y, Z, X, alpha, para_mat)
        torch.cuda.empty_cache()
        
        return(samp)
    
    def simulate(self):
        n = self.nodes
        p = self.latent_space_dim
        K = self.groups
        N = self.time
        init_distr_mat = self.starting_lat_parameter_mat
        para_mat = self.parameter_mat

        Z0 = self.init_Z(n, init_distr_mat)
        Y0 = self.gen_Y(Z0)

        Z_next = self.next_Z(Y0, Z0, para_mat, K)
        Y_next = self.gen_Y(Z_next)

        lat_pos = torch.cat([Z0.unsqueeze(dim = 0), Z_next.unsqueeze(dim = 0)], dim = 0)
        obs_adj = torch.cat([Y0.unsqueeze(dim = 0), Y_next.unsqueeze(dim = 0)], dim = 0)
        
        t = 3
        while t <= N:
            Z_next = self.next_Z(Y_next, Z_next, para_mat, K)
            Y_next = self.gen_Y(Z_next)
            lat_pos = torch.cat([lat_pos, Z_next.unsqueeze(dim = 0)], dim = 0)
            obs_adj = torch.cat([obs_adj, Y_next.unsqueeze(dim = 0)], dim = 0)
            t += 1
        
        sim_model = {"lat_pos": lat_pos, 
                     "obs_adj": obs_adj}
        return(sim_model)
        


""" #default settings
N, n, p, K = 2, 3000, 3, 3
current_settings = ABC_sim(time = N,
                nodes = n,
                latent_space_dim = p,
                groups = K, 
                parameter_vec = torch.tensor([1,1,-4,5.0]),
                starting_lat_parameter_mat = torch.tensor([[1, 1, 10], [1, 10, 1], [10, 1, 1]], dtype= torch.float).reshape(K, p),
                constraint = True) """
""" 
def proj_beta(B, constraint):
    A = constraint
    return(torch.linalg.solve(A.T @ A, A.T) @ B.reshape(-1))

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
 """