#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#Alignment Non-negative Constraint
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################;
import torch
import pandas as pd
from src import Dir_Reg
from src import Simulation as sim
from tqdm import tqdm as tm
from functools import partial

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
"""
Solves the problem  min ||(A-XX^T)*M||_F^2 without any constraint on X,
by classical gradient descent.
Here * is the entry-wise product.

Parameters
----------
A : matrix nxn
X : initialization
M : mask matrix nxn
tol: tolerance used in the stop criterion  
    
Returns
-------
Matrix X
    solution of the embedding problem
"""
def GD_RDPG(A,X, L, tol=1e-3):

    n = A.shape[0]
    M = torch.ones(n, n) - torch.diag(torch.ones(n))
    M, A, X = M.to(device), A.to(device), X.to(device)

    X = X[:, :2]

    def gradient(A,X,M):

        gd = 2 * torch.matmul(( -torch.mul((M.T+M), (A)) + torch.mul((M.T+M), torch.matmul(X, X.T))), X)
        gd = gd + L*(-1)*(X < 0)
        # rowsum_X_ind = (torch.sum(X, axis = 1) > 1) * 1.0
        # gd = gd + 50*torch.cat([rowsum_X_ind.unsqueeze(0)] * 2, dim=0).T

        return gd
    
    def cost_function(A,X,M):

        cost = 0.5 * torch.norm((A - torch.matmul(X, X.T)) * M, p='fro')**2 
        cost = cost + L*torch.sum(-X[X<0])

        # rowsum_X = torch.sum(X, axis = 1)
        # cost = cost + 50*torch.sum(rowsum_X[rowsum_X > 1])

        cost = cost.to("cpu")
        torch.cuda.empty_cache()
        return cost


    b=0.3; sigma=0.1 # Armijo parameters
    rank = X.shape[1]
    max_iter = 200*rank
    t = 0.1
    Xd=X
    k=0
    last_jump=1
    d = -gradient(A,Xd,M)
    tol = tol*(torch.norm(d))
    while (torch.norm(d) > tol) & (last_jump > 1e-16) & (k<max_iter):

        # Armijo
        while (cost_function(A, Xd+t*d, M) > cost_function(A, Xd, M) - sigma*t*torch.norm(d)**2):
            t=b*t

        Xd = Xd+t*d
        last_jump = sigma*t*torch.norm(d)**2
        t=t/(b)
        k=k+1
        d = -gradient(A,Xd,M)

    Xd = Xd.to("cpu")
    del(M, A, X, d)
    torch.cuda.empty_cache()

    last_dim = 1 - torch.sum(Xd, axis = 1).unsqueeze(1)
    aligned_ASE_full = torch.cat((Xd, last_dim), dim = 1)

    return(aligned_ASE_full)



def fix_nodes_n_iter(model, nodes, n_iter, constrained):

    """ run the model with n nodes n_iter number of times, record the estimated beta and fisher's information """

    model.update_settings(nodes = nodes)
    
    B = model.settings.B
    beta = model.settings.beta

    q, p = B.shape
    constraint = Dir_Reg.fit.gen_constraint(p, True)

    B_hat = torch.zeros(n_iter, q*p)

    if constrained: 
        qp = 4
    else:
        qp = q*p

    fish_est = torch.zeros(n_iter, (qp)**2)

    for i in tm(range(n_iter), desc = str(nodes)):

        torch.manual_seed(i)
        model.update_settings()
        
        Z0 = model.synth_data["lat_pos"][0,]
        Z1 = model.synth_data["lat_pos"][1,]
        Y0 = model.synth_data["obs_adj"][0,]
        X0 = sim.ABC.gen_X(Y0, Z0, model.settings.K)

        est = Dir_Reg.fit(predictor = X0, response = Z1, constrained = constrained, beta_guess = model.settings.beta)

        B_hat[i,] = est.est_result["estimate"].reshape(-1)
        fish_est[i,] = est.est_result["fisher_info"].reshape(-1)

    def to_pd_df(n, mat, b_real, name):
        """ so that the experiment result can live in the DLS data wonderland """
        n_iter, qp = mat.shape
        vec = mat.reshape(1, -1)
        comp = torch.arange(1, qp+1).repeat(n_iter).unsqueeze(dim = 0)
        seed_id = torch.arange(n_iter).repeat_interleave(qp).unsqueeze(dim = 0)
        node_id = n * torch.ones(qp*n_iter).unsqueeze(dim = 0)
        b_real_stack = torch.stack([b_real.reshape(-1)]* n_iter).reshape(1, -1)
        b_df = torch.cat([seed_id, node_id, comp, vec, b_real_stack], dim = 0).T
        b_df = pd.DataFrame(b_df)
        b_df.columns = ["Seed", "Nodes", "Comp", name + "_hat", name + "_real"]
        for column in b_df.columns:
            if "_" not in column:
                try:
                    b_df[column] = b_df[column].astype(int)
                except ValueError:
                    pass

        return(b_df)

    B_df = to_pd_df(nodes, B_hat, B, "B")
    beta_tilde = torch.linalg.solve(constraint.T @ constraint, constraint.T) @ B_hat.T
    beta_df = to_pd_df(nodes, beta_tilde.T, beta, "beta")

    result_dict = {"B_hat": B_df, "beta_tilde": beta_df, "fish_est": fish_est.mean(dim = 0).reshape(qp, qp)}
    return(result_dict)



#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#Riemannian Gradient Descent on O(p)
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

class Op_Riemannian_GD:
    
    def __init__(self, data, mode, initialization = None, softplus_parameter = 25, tolerance = 0.01):

        self.data = data
        self.tolerance = tolerance
        self.mode = mode
        self.initialization = initialization
        self.smoothing = softplus_parameter
        self.relu_loss = self.simplex_loss_relu(self.data)
        self.softplus_loss = self.simplex_loss_softplus(self.data, self.smoothing)
        self.align_mat = self.GD_Armijo()

    def update_parameter(self, smoothing, tolerance):

        self.smoothing = smoothing
        self.tolerance = tolerance

    @staticmethod
    def simplex_loss_relu(data_set):

        X = data_set
        relu = torch.nn.ReLU()

        negativity_loss = torch.sum(relu(-X))

        row_sum_minus_1 = torch.sum(X, dim = 1) - 1
        simp_loss = torch.sum(relu(row_sum_minus_1))

        return(negativity_loss + simp_loss)
    
    @staticmethod
    def simplex_loss_softplus(data_set, smoothing):

        X = data_set
        mu = smoothing

        softplus = torch.nn.Softplus(beta = mu)

        negativity_loss = torch.sum(softplus(-X))

        row_sum_minus_1 = torch.sum(X, dim = 1) - 1
        simp_loss = torch.sum(softplus(row_sum_minus_1))

        return(negativity_loss + simp_loss)
    
    def deriv_W_relu(self, W):

        X = self.data
        n, p = X.shape

        signs_1 = (-X @ W > 0) * 1.
        deriv_neg = -X.T @ torch.relu(signs_1)

        sign_2 = ((torch.sum(X @ W,  dim = 1) - 1) > 0) * 1.
        deriv_simp = X.T @ torch.relu(sign_2.unsqueeze(dim = 1)) @ torch.ones((1,p))

        return(deriv_neg + deriv_simp)
    
    def deriv_W_softplus(self, W):

        X = self.data
        n, p = X.shape
        
        mu = self.smoothing

        T0 = torch.exp(-mu* X @ W)
        deriv_neg = -X.T @ (T0/(1 + T0))

        row_sum_minus_1 = torch.sum(X @ W, dim = 1) - 1

        T1 = torch.exp(mu * row_sum_minus_1.unsqueeze(dim = 1))
        deriv_simp = X.T @ (T1/(1 + T1)) @ torch.ones((1, p))

        return(deriv_neg + deriv_simp)
    
    def proj_skew_sym_at_W(self, M, W):

        projection = W @ (W.T @ M - M.T @ W)/2

        return(projection)

    def matrix_exp_at_W(self, xi, W):

        Exp_w_xi = W @ torch.matrix_exp(W.T @ xi)

        return(Exp_w_xi)
    
    def GD_one_step(self, prev_position, step):
        
        W_old = prev_position

        W_old = W_old * torch.sqrt(1/(W_old @ W_old.T)[0,0])
        
        if self.mode == "relu":
            euclid_deriv = self.deriv_W_relu(W_old)
        else:
            euclid_deriv = self.deriv_W_softplus(W_old)
        
        tangent_deriv = self.proj_skew_sym_at_W(euclid_deriv, W_old)

        W_new = self.matrix_exp_at_W(-step*tangent_deriv, W_old)
        
        return(W_new)
    
    def GD_Armijo(self):

        X = self.data
        n, p = X.shape

        if self.initialization is not None:
            W = self.initialization
        else: 
            W = torch.eye(p)

        if self.mode == "relu":
            grad = self.deriv_W_relu
            cost = self.simplex_loss_relu
        else:
            grad = self.deriv_W_softplus
            cost = partial(self.simplex_loss_softplus, smoothing = self.smoothing)
        
        b = 0.1; sigma = 0.1
        max_iter = 200 * p

        iter = 1
        go = True
        while go:
            
            t = 0.001
            k = 1
            while (cost(X @ self.GD_one_step(W, t)) > cost(X @ W) - sigma * t * torch.norm(grad(W))):
                t = t * (b**k)
                k += 1
                if k > 10:
                    break


            W = self.GD_one_step(W, t)
            jump = sigma * t * torch.norm(grad(W))

            go = (torch.norm(grad(W)) > self.tolerance) & (jump > 10e-8) & (iter < max_iter)
            iter += 1

        return(W)
    


class Op_Riemannian_GD:
    
    def __init__(self, data, reference = None, softplus_parameter = 5, tolerance = 0.01):

        self.data = data
        self.reference = reference
        self.smoothing = softplus_parameter
        self.tolerance = tolerance
        self.align_mat = self.GD_Armijo()

    
    @staticmethod
    def simplex_loss_softplus(W, data_set, smoothing, reference):
        n, p = data_set.shape
        X = data_set @ W
        mu = smoothing

        softplus = torch.nn.Softplus(beta = mu)

        negativity_loss = torch.sum(softplus(-X))

        row_sum_minus_1 = torch.sum(X, dim = 1) - 1
        simp_loss = torch.sum(softplus(row_sum_minus_1))

        if reference is not None:
            off_target_loss = torch.norm(X - reference, p = "fro")
            result = negativity_loss + simp_loss + off_target_loss
            print(negativity_loss, simp_loss, off_target_loss)

        else: 
            result = negativity_loss + simp_loss

        return(result)
    
    def deriv_W_softplus(self, W):

        X = self.data
        n, p = X.shape
        R = self.reference
        mu = self.smoothing

        T0 = torch.exp(-mu* X @ W)
        deriv_neg = -X.T @ (T0/(1 + T0))

        row_sum_minus_1 = torch.sum(X @ W, dim = 1) - 1

        T1 = torch.exp(mu * row_sum_minus_1.unsqueeze(dim = 1))
        deriv_simp = X.T @ (T1/(1 + T1)) @ torch.ones((1, p))

        if self.reference is not None:
            T2 = X @ W - R
            deriv_off_target = (1/torch.norm(T2, p = "fro")) * X.T @ T2
            result = deriv_neg + deriv_simp + 10 * deriv_off_target

        else:
            result = deriv_neg + deriv_simp

        return(result)

    def proj_skew_sym_at_W(self, M, W):

        projection = W @ (W.T @ M - M.T @ W)/2

        return(projection)

    def matrix_exp_at_W(self, xi, W):

        Exp_w_xi = W @ torch.matrix_exp(W.T @ xi)

        return(Exp_w_xi)
    
    def GD_one_step(self, prev_position, step):
        
        W_old = prev_position

        #rescale W_old so that W^TW = WW^T = I_p.
        #without this, the product slowly deviates from I_p due to numeric reasons.
        W_old = W_old * torch.sqrt(1/(W_old @ W_old.T)[0,0])
        
        euclid_deriv = self.deriv_W_softplus(W_old)

        tangent_deriv = self.proj_skew_sym_at_W(euclid_deriv, W_old)

        W_new = self.matrix_exp_at_W(-step*tangent_deriv, W_old)

        return(W_new)
    
    def GD_Armijo(self):

        X = self.data
        n, p = X.shape

        W = torch.eye(p)

        grad = self.deriv_W_softplus
        cost = partial(self.simplex_loss_softplus, smoothing = self.smoothing, reference = self.reference)
        
        b = 0.1; sigma = 0.1
        max_iter = 200 * p

        iter = 1
        go = True
        while go:
            
            t = 0.001
            k = 1
            next_cost = cost(self.GD_one_step(W, t), X)
            current_cost = cost(W, X)
            while (next_cost > current_cost - sigma * t * torch.norm(grad(W))):

                t = t * (b**k)
                k += 1
                if k > 10:
                    break

          
            W = self.GD_one_step(W, t)
            jump = sigma * t * torch.norm(grad(W))

            go = (torch.norm(grad(W)) > self.tolerance) & (jump > 10e-8) & (iter < max_iter)
            iter += 1

        return(W)
    

class No_Oracle:
    def __init__(self, need_align_adj, embed_dim, reference = None, softplus_parameter = 5, tol = 10e-2):
        self.data = Oracle.ASE(need_align_adj, embed_dim)
        self.align_mat = Op_Riemannian_GD(self.data, reference, softplus_parameter, tol).align_mat
        self.aligned = self.aligned()

    def aligned(self):
        aligned_core = self.data @ self.align_mat
        aligned_last = (1 - aligned_core.sum(dim = 1)).unsqueeze(dim = 1)
        aligned = torch.cat([aligned_core, aligned_last], dim = 1)
        return(aligned)