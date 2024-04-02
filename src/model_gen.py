#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#To generate
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
import torch
from torch.distributions import Dirichlet, Bernoulli, Uniform
from src.Dirichlet_Reg import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ABC_sim:

    def __init__(self, time, nodes, beta, alpha_0):

        self.settings = self.ABC_settings(time, nodes, beta, alpha_0)
        self.synth_data = self.simulate()


    class ABC_settings:
        def __init__(self, time, nodes, beta, alpha_0):
            self.T = time
            self.n = nodes
            self.beta = torch.as_tensor(beta, dtype = torch.float)
            self.alpha_0 = torch.as_tensor(alpha_0, dtype = torch.float)
            self.K, self.p = self.alpha_0.shape
            self.C = Dirichlet_GLM_log.gen_constraint(self.p)
            self.B = (self.C @ self.beta).reshape(3 * (self.p - 1) + 1, self.p)
            self.dict = {"T": self.T,
                         "n": self.n,
                         "K": self.K,
                         "p": self.p,
                         "beta": self.beta,
                         "alpha_0": self.alpha_0}

    def update_settings(self, time = None, nodes = None, beta = None, alpha_0 = None):

        if time is None:
            time = self.settings.T

        if nodes is None:
            nodes = self.settings.n
        
        if beta is None:
            beta = self.settings.beta
        
        if alpha_0 is None:
            alpha_0 = self.settings.alpha_0
        
        self.settings = self.ABC_settings(time, nodes, beta, alpha_0)

        self.synth_data = self.simulate()

    @staticmethod
    def init_Z(n_nodes, alpha_0):
        K, p = alpha_0.shape
        n = n_nodes
        init_dist = Dirichlet(alpha_0)
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
    def next_Z(Y, Z, B, K_groups):

        B = B.to(device)

        Y = Y.to(device)
        Z = Z.to(device)
        X = ABC_sim.gen_X(Y, Z, K_groups).to(device)

        alpha = torch.exp(X.matmul(B))
        samp = Dirichlet(alpha).sample((1,))[0]

        samp = samp.to("cpu")
        del(Y, Z, X, alpha, B)
        torch.cuda.empty_cache()
        
        return(samp)
    
    def simulate(self):
        n = self.settings.n
        K = self.settings.K
        N = self.settings.T
        alpha_0 = self.settings.alpha_0
        B = self.settings.B

        Z0 = self.init_Z(n, alpha_0)
        Y0 = self.gen_Y(Z0)

        Z_next = self.next_Z(Y0, Z0, B, K)
        Y_next = self.gen_Y(Z_next)

        lat_pos = torch.cat([Z0.unsqueeze(dim = 0), Z_next.unsqueeze(dim = 0)], dim = 0)
        obs_adj = torch.cat([Y0.unsqueeze(dim = 0), Y_next.unsqueeze(dim = 0)], dim = 0)
        
        t = 3
        while t <= N:
            Z_next = self.next_Z(Y_next, Z_next, B, K)
            Y_next = self.gen_Y(Z_next)
            lat_pos = torch.cat([lat_pos, Z_next.unsqueeze(dim = 0)], dim = 0)
            obs_adj = torch.cat([obs_adj, Y_next.unsqueeze(dim = 0)], dim = 0)
            t += 1
        
        sim_model = {"lat_pos": lat_pos, 
                     "obs_adj": obs_adj}
        return(sim_model)
        