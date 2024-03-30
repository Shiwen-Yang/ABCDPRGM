#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#Alignment
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from functools import partial

#regular ASE
def ASE(A, embed_dim):
    temp_svd = torch.svd_lowrank(A, q = embed_dim)
    temp_ASE = temp_svd[0] @ torch.diag(torch.sqrt(temp_svd[1]))
    return(temp_ASE)

#orthogonal procrustes used to align the ASE with the real latent position (or a reasonably estimated one)
def orthogonal_procrustes(reference, need_embed_align, dim, ortho_mat = False):
    n, p = reference.shape
    reference = reference[:, : (p - 1)]

    t1, t2 = need_embed_align.shape
    #when t1 = t2, then need_embed_align is an adjacency matrix, embed it first
    if t1 == t2:
        ASE_need_align = ASE(need_embed_align, dim)  # Assuming ASE is defined elsewhere
    #otherwise, we have some latent positions that needs to be aligned
    else: 
        ASE_need_align = need_embed_align[:, :(p-1)]
        

    # Perform partial SVD (dim = (p-1)) on the product of the matrices, since latent position has rank (p-1)
    LRsvd_ASE = torch.svd_lowrank(ASE_need_align.T @ reference, q = (p-1))

    # Calculate the rotation matrix
    procrustes_sol = LRsvd_ASE[2] @ LRsvd_ASE[0].T

    # # same code but uses full_svd
    # svd_ASE = torch.svd(ASE_need_align.T @ reference)
    # procrustes_sol = svd_ASE.V @ svd_ASE.U.T
    
    # Apply the rotation to the aligned matrix
    aligned_ASE = ASE_need_align @ procrustes_sol.T
    last_dim = 1 - torch.sum(aligned_ASE, axis = 1).unsqueeze(1)
    aligned_ASE_full = torch.cat((aligned_ASE, last_dim), dim = 1)

    if ortho_mat:
        return procrustes_sol.T
    
    return aligned_ASE_full 



#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#Alignment Non-negative Constraint
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################;
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





#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#Riemannian Gradient Descent on O(p)
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

class Op_Riemannian_GD:
    
    def __init__(self, data, mode, softplus_parameter = 25,tolerance = 0.01):

        self.data = data
        self.tolerance = tolerance
        self.mode = mode
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
            
