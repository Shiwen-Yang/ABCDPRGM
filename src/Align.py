#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#Alignment
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("mps") if torch.backends.mps.is_available() else device
from functools import partial


class Oracle:
    
    """ 
    
    Given a reference latent position (typically, the true latent position), embed a adjacency matrix in some 
    dimension using ASE, and find an orthogonal transformation that aligns the ASE to the reference
    
    Args:
        reference (torch.tensor of dimension n by p): under our settings, it will be the true latent position
        need_align_adj (torch.tensor of dimension n by n): the adjacency matrix for the observed network
        embed_dimension (int): the number of dimensions to embed the adjacency matrix in

    Attributes:
        embed_raw (torch.tensor of dimension n by embed_dimension): the ASE of the adjacency matrix in embed_dimension-dimension
        align_mat (torch.tensor of dimension embed_dimension by embed_dimension): the orthogonal transformation that aligns the ASE to the reference
        embed_aligned (torch.tensor of dimension n by embed_dimension): the aligned ASE
            
    """
    
    
    def __init__(self, reference, need_align_adj, embed_dimension):
        self.reference = reference
        self.need_align = need_align_adj
        self.embed_dim = embed_dimension
        self.embed_raw = self.ASE(need_align_adj, embed_dimension)
        self.align_mat = self.ortho_proc(reference[:,:embed_dimension], self.embed_raw[:,:embed_dimension])
        self.embed_aligned = self.align()
        
    
    @staticmethod
    def ASE(A, embed_dim):
        
        """ 
        Given an adjacency matrix, embed it to embed_dim dimension using ASE 
        """
        
        temp_svd = torch.svd_lowrank(A, q = embed_dim)
        temp_ASE = temp_svd[0] @ torch.diag(torch.sqrt(temp_svd[1]))
        return(temp_ASE)
    
    @staticmethod
    def ortho_proc(reference, need_align):
        
        """ 
        Given a reference, and a matrix, find the orthogonal transformation that aligns the matrix to the reference.
        """
        
        n, p = reference.shape

        # Perform partial SVD (dim = (p-1)) on the product of the matrices, since latent position has rank (p-1)
        LRsvd_ASE = torch.svd_lowrank(need_align.T @ reference, q = p)

        # Calculate the rotation matrix
        align_mat = LRsvd_ASE[2] @ LRsvd_ASE[0].T

        return(align_mat.T)
    
    def align(self):
        need_align = self.embed_raw
        align_mat = self.align_mat

        aligned_ASE = need_align @ align_mat
        last_dim = 1 - torch.sum(aligned_ASE, axis = 1).unsqueeze(1)
        aligned_ASE_full = torch.cat((aligned_ASE, last_dim), dim = 1)
        return(aligned_ASE_full)




#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#Riemannian Gradient Descent on O(p)
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

class Op_Riemannian_GD:
    
    """ 
    
    Given some data, find an orthogonal transforamtion that put it into the standard simplex using Riemannian gradient descent on Op, the orthogonal group of dimension p. 
    
    Args:
        data (torch.tensor of dimension n by p): under our settings, it is the raw ASE that we cannot use to estimate the model parameter
        tolerace (float): stopping criterion for the gradient descent
        mode (string): either "softplus" or "relu", it defines the type of penalty function used
        softplus_parameter (float): bigger parameter means that the softplus function looks more like the relu function, i.e. less smooth penalty
    
    Attributes:
        relu_loss: loss under the relu penalty
        softplus_loss: loss under the softplus penalty
        align_mat: the desired orthogonal transformation
    
    """
    
    def __init__(self, data, initialization = None, mode = "softplus", softplus_parameter = 5, tolerance = 0.01):

        self.data = data
        self.tolerance = tolerance
        self.mode = mode
        self.initialization = initialization
        self.smoothing = softplus_parameter
        self.relu_loss = self.simplex_loss_relu(self.data)
        self.softplus_loss = self.simplex_loss_softplus(self.data, self.smoothing)
        self.align_mat = self.GD_Armijo()

    def update(self, mode = None, smoothing = None, tolerance = None):
        if mode is not None:
            self.mode = mode
        if smoothing is not None:
            self.smoothing = smoothing
        if tolerance is not None:
            self.tolerance = tolerance
        self.align_mat = self.GD_Armijo

    @staticmethod
    def simplex_loss_relu(data_set):
        """
        Computes the simplex loss using the ReLU function on the input dataset.

        The loss consists of two components:
        1. Negativity Loss: Penalizes negative values in the dataset.
        2. Simplex Constraint Loss: Penalizes rows whose sum deviates from 1.

        Args:
        -----
        data_set : torch.Tensor
            A 2D tensor where each row represents a vector that is expected to lie on a simplex.
            The function assumes `data_set` has dimensions (n, p), where `n` is the number of vectors and `p` is the dimension of each vector.

        Returns:
        --------
        torch.Tensor
            A scalar tensor representing the total simplex loss, which is the sum of the negativity loss and the simplex constraint loss.
        """

        X = data_set
        relu = torch.nn.ReLU()

        negativity_loss = torch.sum(relu(-X))

        row_sum_minus_1 = torch.sum(X, dim = 1) - 1
        simp_loss = torch.sum(relu(row_sum_minus_1))

        return(negativity_loss + simp_loss)
    
    @staticmethod
    def simplex_loss_softplus(data_set, smoothing):
        
        """ 
        
        Same thing as the simplex_loss_relu function, but replacing the ReLU function with a softplus function with parameter smoothing. 
        The softplus function with parameter beta is defined to be:
        
        softplus(x, beta) = log(1 + exp(x * beta))/beta
        
        """

        X = data_set
        mu = smoothing

        softplus = torch.nn.Softplus(beta = mu)

        negativity_loss = torch.sum(softplus(-X))

        row_sum_minus_1 = torch.sum(X, dim = 1) - 1
        simp_loss = torch.sum(softplus(row_sum_minus_1))

        return(negativity_loss + simp_loss)
    
    def deriv_W_relu(self, W):
        
        """ 
        The derivative of the ReLU penalty function, L(X*W) with respect to W, the orthogonal transformation 
        """

        X = self.data
        n, p = X.shape

        signs_1 = (-X @ W > 0) * 1.
        deriv_neg = -X.T @ torch.relu(signs_1)

        sign_2 = ((torch.sum(X @ W,  dim = 1) - 1) > 0) * 1.
        deriv_simp = X.T @ torch.relu(sign_2.unsqueeze(dim = 1)) @ torch.ones((1,p))

        return(deriv_neg + deriv_simp)
    
    def deriv_W_softplus(self, W):
        
        """ 
        The derivative of the softplus function, L(X*W, mu) with respect to W, the orthogonal transformation 
        """

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
        
        """ 
        Projection of M to the tangent space of Op at W
        """

        projection = W @ (W.T @ M - M.T @ W)/2

        return(projection)

    def matrix_exp_at_W(self, xi, W):
        
        """ 
        The retractiom, it takes xi, the computed gradient step, and map it onto Op along a geodesic that starts at W
        """

        Exp_w_xi = W @ torch.matrix_exp(W.T @ xi)

        return(Exp_w_xi)
    
    def GD_one_step(self, prev_position, step):
        
        """ 
        Given the current orthogonal transformation, and a step size, take a graident descent step, and get another orthogonal transformation
        """
        
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
        
        """ 
        Backtracking line search but uses the riemannian gradient instead
        """

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

#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
# Penalized ASE by Gradient Descent
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

class GD_RDPG:
    
    """
    
    Given an adjacency matrix, A, find latent positions, X, such that it minimizes the following two losses:
    1. AL, alignment loss: ||A - X * X.T||_F
    2. SL, smplex loss, same as the one defined in Op_Riemannian_GD.simplex_loss_softplus
    
    Args: 
        adj_mat (torch.tensor of dimension n by n): an adjacency matrix that we are trying to embed into the simplex
        regularization (float): the weight of SL in the combined penalty
        lat_pos (torch.tensor of dimension n by p): a guess of the latent position, typically, it will be the ASE. it is the starting position for the gradient descent
        smoothing (float): the parameter for the softplus function, bigger = less smooth penalty
        tol (float): stopping criterion for the gradient descent
        verbose (boolean): if the gradient progress should be printed
        
    Attributes:
        settings (class): the settings...
        fitted (torch.tensor of dimension n by p): the desired estimated latent position
        loss_align (float): the loss from AL of the final estimate
        loss_simples (float): the loss from SL of the final estimate
        loss_data (float): the amount of estimated latent position that is outside of the simplex
    
    """
    def __init__(self, adj_mat, regularization, lat_pos, smoothing = 5, tol = 1e-1, verbose = False):
        self.settings = self.Settings(adj_mat, regularization, lat_pos, smoothing, tol, verbose)
        self.fitted, self.loss_align = self.Adam_GD()
        self.loss_simplex = Op_Riemannian_GD.simplex_loss_relu(self.fitted)
        self.loss_data = GD_RDPG.data_loss(self.fitted)
        
        
    class Settings:
        def __init__(self, adj_mat, regularization, lat_pos, smoothing, tol, verbose):
            self.A = adj_mat
            self.L = regularization
            self.Z = lat_pos
            self.mu = smoothing
            self.tol = tol
            self.verbose = verbose
            
    @staticmethod
    def align_simplex_loss(A, est_X, smoothing):
        
        """ 
        loss from AL + loss from SL
        """
        
        est_X.requires_grad_(True)
        
        n, p = est_X.shape
        mu = smoothing
        M = torch.ones(n, n) - torch.diag(torch.ones(n))
        A, est_X, M = A.to(device), est_X.to(device), M.to(device)
            
        softplus = torch.nn.Softplus(beta = mu, threshold = 50)
        
        align_loss = torch.norm((A - torch.matmul(est_X, est_X.T)) * M, p = "fro")**2
        
        negativity_loss = torch.sum(softplus(-est_X))

        row_sum_minus_1 = torch.sum(est_X, dim = 1) - 1
        simp_loss = torch.sum(softplus(row_sum_minus_1))
        
        simplex_loss = negativity_loss + simp_loss

        return(align_loss, simplex_loss)
    
    @staticmethod
    def data_loss(tensor):
        
        """ 
        counting the amount of data that is outside of the simplex
        """
        
        # Condition 1: Row sum greater than 1
        row_sum_greater_than_1 = torch.sum(tensor, dim=1) > 1
        
        # Condition 2: Entries in the row less than 0
        entries_less_than_0 = (tensor < 0).any(dim=1)
        
        # Count rows satisfying each condition
        count_sum_greater_than_1 = torch.sum(row_sum_greater_than_1).item()
        count_entries_less_than_0 = torch.sum(entries_less_than_0).item()
        
        return count_sum_greater_than_1 + count_entries_less_than_0
    
    def Adam_GD(self):
        
        """ 
        minimizing the penalty function with respect to Z, the latent position, using Adam 
        """
        
        A, Z, L, mu, tol = self.settings.A, self.settings.Z, self.settings.L, self.settings.mu, self.settings.tol
        A, Z = A.to(device), Z.to(device)
        optimizer = torch.optim.Adam([Z], lr=0.01)
        
        i = 1
        go = True
        
        while go:
            
            optimizer.zero_grad()
            align_loss, simplex_loss = GD_RDPG.align_simplex_loss(A, Z, mu)
            simplex_loss = L * simplex_loss
            align_loss.backward()
            simplex_loss.backward()
            optimizer.step()
            
            if self.settings.verbose & (i % 100 == 0):
                print(f'Epoch {i}, Align Loss: {align_loss.item()}, simplex Loss: {simplex_loss.item()}')

            go = (torch.norm(Z.grad, p = "fro") > tol) & (i <= 1500)
            i += 1
            
        return(Z, align_loss.item())