import torch
from src import Align
from src import Simulation as sim
from src import Dir_Reg

class est:
    """ 
    
    This is a wrapper for the Dirichlet Regression method. Taking in raw data, it processes it and then run the Dirichlet Regression.
    TO USE: 
    1. est(two_lat_pos = synth_data['lat_pos'], two_adj_mat = synth_data['obs_adj'])
    2. specify_model(new_mode)
    if new mode is not "OL", then the embed_dim argument needs to be specified
    
    Args:
        embed_dimension (int): the embedding dimension when true latent position is unknown
        two_lat_pos (torch.tensor of shape 2 by n by (3p-2)): the latent positions, two_lat_pos[0,:,:] is the latent position at t = 0, the other slice is the latent position at t = 1
        two_adj_mat (torch.tensor of shape 2 by n by n): the adjacency matrices with similar construction to two_adj_mat
        group (int): the number of groups in the network 
        beta_guess (torch.tensor of shape 4): a vector to initiate the gradient descent
        beta_0_guess (float): estimate of the intercept term but negative. 
        max_iter_est (int): maximum number of iterations for the gradient descent
        RGD_mode (string: softplus or relu): specifies the type of penalty function for RGD
        RGD_softplus_parameter (float): bigger parameter means that the softplus penalty looks more relu.
        tol_(est/RGD) (float): stopping criterions

    Attributes:
        specify_model (method): use this to specify the method to be used to obtain the latent postions. It can be "OL", "OA", or "NO". this method automatically calls the process method that process the data. When fit = True, a regression is fitted as well. 
        data (dictionary): has two components, "predictor" and "response". It's the output of the 'process' method
        fitted (Dir_Reg.fit): it's the result object from the Dirichlet regression
    
    """
    
    def __init__(self, embed_dimension = None, two_lat_pos = None, two_adj_mat = None, groups = 3, constrained = False, beta_guess = None, 
                 beta_0_guess = -10, max_iter_est = 200,  RGD_mode = "softplus", RGD_softplus_parameter = 5, tol_est = 10e-2, tol_RGD = 10e-2):
        
        self.settings = self.settings(embed_dimension, groups, constrained, beta_guess, beta_0_guess, max_iter_est,
                                      RGD_mode, RGD_softplus_parameter, tol_est, tol_RGD)
        
        self.input = {"latpos": two_lat_pos, "adjmat": two_adj_mat}
        self.raw_data = None
        self.data = None
        self.fitted = None

    def specify_mode(self, new_mode, fit = True, embed_dim = None):
        self.settings.EST.specify_mode(new_mode)
        
        if embed_dim is not None:
            self.settings.EST.embed_dim = embed_dim
            
        self.raw_data = self.data_raw(self.input["latpos"], self.input["adjmat"], self.settings.EST.embed_dim)
        self.data = self.process(self.settings.EST.mode)
        
        if fit:
            self.fitted = self.fit()

    class settings:
        def __init__(self, embed_dimension, groups, constrained, beta_guess, beta_0_guess, max_iter_est, 
                        RGD_mode, RGD_softplus_parameter, tol_est, tol_RGD):
            self.EST = self.EST(embed_dimension, groups, constrained, beta_guess, beta_0_guess, max_iter_est, tol_est)
            self.RGD = self.RGD(RGD_mode, RGD_softplus_parameter, tol_RGD)

        class EST:
            def __init__(self, embed_dimension, groups, constrained, beta_guess, beta_0_guess, max_iter_est, tol_est):
                self.mode = None
                self.embed_dim = embed_dimension
                self.K = groups
                self.constrained = constrained
                self.beta_guess = beta_guess
                self.beta_0_guess = beta_0_guess
                self.max_iter = max_iter_est
                self.tol = tol_est
            
            def specify_mode(self, mode):
                self.mode = mode

        class RGD:
            def __init__(self, RGD_mode, RGD_softplus_parameter, tol_RGD):
                self.mode = RGD_mode
                self.softplus_beta = RGD_softplus_parameter
                self.tol = tol_RGD
    
    class data_raw:
        def __init__(self, two_lat_pos, two_adj_mat, embed_dim):
            if two_lat_pos is not None:
                self.Z0 = two_lat_pos[0,]
                self.Z1 = two_lat_pos[1,]
            else:
                self.Z0 = self.Z1 = None

            if two_adj_mat is not None:
                self.Y0 = two_adj_mat[0,]
                self.Y1 = two_adj_mat[1,]

                Z0_ase_core = Align.Oracle.ASE(self.Y0, embed_dim)
                Z1_ase_core = Align.Oracle.ASE(self.Y1, embed_dim)

                Z0_last_dim = 1 - torch.sum(Z0_ase_core, dim = 1).unsqueeze(dim = 1)
                Z1_last_dim = 1 - torch.sum(Z1_ase_core, dim = 1).unsqueeze(dim = 1)

                self.Z0_ASE = torch.cat([Z0_ase_core, Z0_last_dim], dim = 1)
                self.Z1_ASE = torch.cat([Z1_ase_core, Z1_last_dim], dim = 1)
            else: 
                self.Y0 = self.Y1 = self.Z0_ASE = self.Z1_ASE = None

    @staticmethod
    def mult_W(Z, W):
        p = W.shape[0]
        temp = Z[:,:p] @ W
        temp_last_dim = 1 - temp.sum(dim = 1)
        temp_full = torch.cat([temp, temp_last_dim.unsqueeze(dim = 1)], dim = 1)
        return(temp_full)

    def process(self, mode):
        if mode == "OL":
            predictor_Z = self.raw_data.Z0
            response = self.raw_data.Z1

        if mode == "OA":
            predictor_Z = Align.Oracle(self.raw_data.Z0, self.raw_data.Y0, self.settings.EST.embed_dim).embed_aligned
            response = Align.Oracle(self.raw_data.Z1, self.raw_data.Y1, self.settings.EST.embed_dim).embed_aligned

        if mode == "NO":

            n, p = self.raw_data.Z0_ASE.shape
            Z0_NO_align_mat = Align.Op_Riemannian_GD(data = self.raw_data.Z0_ASE[:,:(p-1)], 
                                                    initialization = None, 
                                                    mode = self.settings.RGD.mode, 
                                                    softplus_parameter = self.settings.RGD.softplus_beta, 
                                                    tolerance = self.settings.RGD.tol).align_mat
            Z0_NO = est.mult_W(self.raw_data.Z0_ASE, Z0_NO_align_mat)

            Z1_NO_init = Align.Oracle.ortho_proc(Z0_NO[:,:(p-1)], self.raw_data.Z1_ASE[:,:(p-1)])
            Z1_NO_align_mat = Align.Op_Riemannian_GD(data = self.raw_data.Z1_ASE[:,:(p-1)], 
                                                    initialization = Z1_NO_init, 
                                                    mode = self.settings.RGD.mode, 
                                                    softplus_parameter = self.settings.RGD.softplus_beta, 
                                                    tolerance = self.settings.RGD.tol).align_mat
            Z1_NO = est.mult_W(self.raw_data.Z1_ASE, Z1_NO_align_mat)
            
            predictor_Z, response = Z0_NO, Z1_NO

        predictor = sim.ABC.gen_X(self.raw_data.Y0, predictor_Z, self.settings.EST.K)

        
        return({"predictor": predictor, "response": response})
    
    def fit(self):
        result = Dir_Reg.fit(predictor = self.data["predictor"], 
                               response = self.data["response"], 
                               constrained = self.settings.EST.constrained, 
                               beta_guess = self.settings.EST.beta_guess, 
                               beta_0_guess = self.settings.EST.beta_0_guess, 
                               tol = self.settings.EST.tol, 
                               max_iter = self.settings.EST.max_iter)
        return(result)

      



                


                
