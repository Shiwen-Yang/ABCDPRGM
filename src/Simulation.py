#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#To generate
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
import torch
import pandas as pd
from torch.distributions import Dirichlet, Bernoulli, Uniform
from src import Dir_Reg
from src import Align

from src import visualize_latent_space as vls #delete afterwards

from tqdm import tqdm as tm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ABC:

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
            self.C = Dir_Reg.fit.gen_constraint(self.p, True)
            self.B = (self.C @ self.beta).reshape(3 * (self.p - 1) + 1, self.p)
            self.dict = {"T": self.T,
                         "n": self.n,
                         "K": self.K,
                         "p": self.p,
                         "beta": self.beta,
                         "alpha_0": self.alpha_0}

    def update_settings(self, time = None, nodes = None, beta = None, alpha_0 = None):

        updated_settings = {
            "time": time if time is not None else self.settings.T,
            "nodes": nodes if nodes is not None else self.settings.n,
            "beta": beta if beta is not None else self.settings.beta,
            "alpha_0": alpha_0 if alpha_0 is not None else self.settings.alpha_0,
        }

        # Assuming ABC_settings is a method that updates the settings
        # Or if ABC_settings is a class, instantiate it with the updated settings
        self.settings = self.ABC_settings(**updated_settings)
        
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
        X = ABC.gen_X(Y, Z, K_groups).to(device)

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
    

class ABC_Monte_Carlo:
    def __init__(self) -> None:
        pass

    
    @staticmethod
    def lat_pos(synth_data, K_groups):
        K = K_groups
        T, n, p = synth_data.shape
        group_membership = torch.arange(K).repeat_interleave(int(n/K)).repeat(T).reshape(T, n, 1)
        time = torch.arange(T).repeat_interleave(n).reshape(T, n, 1)
        new_tensor = torch.cat([time, group_membership, synth_data], dim = 2).reshape(-1, T * n, p + 2).squeeze(dim = 0)

        dim_col = ["dim_" + str(i) for i in range(1, p + 1)]
        all_cols = ["time", "group"] + dim_col

        new_df = pd.DataFrame(new_tensor)
        new_df.columns = all_cols
        new_df = new_df.astype({"time": int, "group": int})
        return(new_df)
    
    @staticmethod
    class consistency_T2:
        """ keep no_oracle False for now... RGD fucks up the permutation bad, need to minimize wrt permuation in addition """
        def __init__(self, number_of_iterations, nodes_set, beta, alpha_0, seeded = True, constrained = False, oracle_lat_pos = True, oracle_align = False, no_oracle = False):
            self.settings = self.settings(number_of_iterations, nodes_set, beta, alpha_0, seeded, constrained, oracle_lat_pos, oracle_align, no_oracle)
            self.MC_result = self.many_rounds()


        class settings:
            def __init__(self, number_of_iterations, nodes_set, beta, alpha_0, seeded, constrained, oracle_lat_pos, oracle_align, no_oracle):
                self.n_iter = number_of_iterations
                self.n_set = nodes_set
                self.beta = beta
                self.alpha_0 = alpha_0
                self.seeded = seeded
                self.constrained = constrained
                self.OL = oracle_lat_pos
                self.OA = oracle_align
                self.NO = no_oracle

                self.init_model = ABC(time = 2,
                                    nodes = 3,
                                    beta = self.beta,
                                    alpha_0 = self.alpha_0)
                p, q = self.init_model.settings.B.shape
                self.constraint = Dir_Reg.fit.gen_constraint(p , True)
                self.B_size = 4 if self.constrained else q*p

        def one_round(self, nodes, seed = None):

            if seed is not None: 
                torch.manual_seed(seed)
            
            n_type = self.settings.OL + self.settings.OA + self.settings.NO
            q, p = self.settings.init_model.settings.B.shape

            model = self.settings.init_model
            model.update_settings(nodes = nodes)
            constrained = self.settings.constrained

            Z0 = model.synth_data["lat_pos"][0,]
            Z1 = model.synth_data["lat_pos"][1,]
            Y0 = model.synth_data["obs_adj"][0,]
            Y1 = model.synth_data["obs_adj"][1,]

            est_OL= est_OA = est_NO = fish_OL = fish_OA = fish_NO = None
            method_OL= method_OA = method_NO = None
    
            if self.settings.OL:
                X0_ora_lat_pos = ABC.gen_X(Y0, Z0, model.settings.K)
                est_ora_lat_pos = Dir_Reg.fit(predictor = X0_ora_lat_pos, 
                                            response = Z1, 
                                            constrained = constrained, 
                                            beta_guess = model.settings.beta)
                est_OL = est_ora_lat_pos.est_result["estimate"].reshape(-1)
                fish_OL = est_ora_lat_pos.est_result["fisher_info"].reshape(1, -1)
                method_OL = torch.tensor([[1,0,0]])

            if self.settings.OA:
                Z0_ora_align = Align.Oracle(Z0, Y0, (p-1)).embed_aligned
                Z1_ora_align = Align.Oracle(Z1, Y1, (p-1)).embed_aligned
                X0_ora_align = ABC.gen_X(Y0, Z0_ora_align, model.settings.K)
                est_ora_align = Dir_Reg.fit(predictor = X0_ora_align, 
                                            response = Z1_ora_align, 
                                            constrained = constrained, 
                                            beta_guess = model.settings.beta)
                est_OA = est_ora_align.est_result["estimate"].reshape(-1)
                fish_OA = est_ora_align.est_result["fisher_info"].reshape(1, -1)
                method_OA = torch.tensor([[0,1,0]])

            if self.settings.NO:
                Z0_no_oracle = Align.No_Oracle(Y0, (p-1)).aligned

                # init_guess = Align.Oracle(Z0_no_oracle, Y1, (p-1)).align_mat
                # Z1_no_oracle = Align.No_Oracle(Y1, (p-1), init_guess).aligned

                Z1_no_oracle = Align.No_Oracle(Y1, (p-1)).aligned

                X0_no_oracle = ABC.gen_X(Y0, Z0_no_oracle, model.settings.K)

                est_no_oracle = Dir_Reg.fit(predictor = X0_no_oracle, 
                                            response = Z1_no_oracle, 
                                            constrained = constrained, 
                                            beta_guess = model.settings.beta)

                est_NO = est_no_oracle.est_result["estimate"].reshape(-1)
                fish_NO = est_no_oracle.est_result["fisher_info"].reshape(1, -1)
                method_NO = torch.tensor([[0,0,1]])
            
            est_list = [est_OL, est_OA, est_NO]
            est_to_concat = [item for item in est_list if item is not None]

            method_list = [method_OL, method_OA, method_NO]
            method_to_concat = [item for item in method_list if item is not None]

            est = torch.cat(est_to_concat, dim = 0).unsqueeze(dim = 1)


            method_core = torch.cat(method_to_concat, dim = 0)
            est_constrained = torch.tensor([constrained]).repeat(n_type * p * q).unsqueeze(dim = 1)
            est_nodes = torch.ones(n_type * q * p, 1) * nodes
            est_component = torch.arange(q*p).repeat(n_type).unsqueeze(dim = 1)
            est_method = torch.kron(method_core, torch.ones(q*p, 1))
            real = self.settings.init_model.settings.B.reshape(-1).repeat(n_type).unsqueeze(dim = 1)
            
            if seed is not None:
                seed_list = torch.ones(n_type * q * p, 1) * seed
                est_full = torch.cat([seed_list, est_nodes, est_constrained, est_method, est_component, est, real], dim = 1)
            else:
                est_full = torch.cat([est_nodes, est_constrained, est_method, est_component, est, real], dim = 1)

            fish_list = [fish_OL, fish_OA, fish_NO]
            fish_to_concat = [item for item in fish_list if item is not None]
            fish = torch.cat(fish_to_concat, dim = 0)
            fish_nodes = torch.ones(n_type, 1) * nodes
            fish_full = torch.cat([fish_nodes, method_core, fish], dim = 1)

            result_dic = {"est": est_full, "fish": fish_full}

            return(result_dic)
        
        class sim_result:
            def __init__(self, est, fish):
                self.est = est
                self.fish = fish
        
        def many_rounds(self):
            est_result_list = []
            fish_result_list = []
            n_set = self.settings.n_set
            n_iter = self.settings.n_iter

            for i in n_set:
                for j in tm(range(n_iter), desc = str(i)):

                    if self.settings.seeded:
                        temp = self.one_round(i, seed = j)
                    else:
                        temp = self.one_round(i)
                
                    est, fish = temp["est"], temp["fish"]
                    est_result_list.append(est)
                    fish_result_list.append(fish)
            
            est_result = torch.cat(est_result_list, dim = 0)
            if self.settings.seeded:
                est_result = pd.DataFrame(est_result, 
                                          columns = ["seed", "nodes", "constrained", "method_OL", "method_OA", "method_NO", "component", "B_est", "B_real"])
            else:
                est_result = pd.DataFrame(est_result, 
                                          columns = ["nodes", "constrained", "method_OL", "method_OA", "method_NO", "component", "B_est", "B_real"])
            
            for column in est_result.columns:
                if not column.startswith('B_'):
                    est_result[column] = est_result[column].astype(int)


            fish_result = torch.cat(fish_result_list, dim = 0)

            first_columns = ["nodes", "method_1", "method_2", "method_3"]
            fisher_info_columns = [f"fisher_info_{i+1}" for i in range(fish_result.size(1) - len(first_columns))]
            all_columns = first_columns + fisher_info_columns
            df_full = pd.DataFrame(fish_result.numpy(), columns=all_columns)

            for column in df_full.columns:
                if not column.startswith('fisher_'):
                    df_full[column] = df_full[column].astype(int)

            result = self.sim_result(est_result, df_full)

            return(result)
