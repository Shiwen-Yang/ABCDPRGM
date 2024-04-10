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