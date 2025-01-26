# Stochastic gradient descent-based inference for dynamic network models with attractors
# This script defines helper functions and classes for use to simulate a dynamic network with changing membership and fit the CLSNA model to the simulated data for inference

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import laplacian
from scipy.stats import truncnorm
from scipy import linalg
import torch
from torch.nn import Parameter
from scipy import sparse
from torch import from_numpy
import matplotlib.pyplot as plt


def simulate_clsna(N=200, d=2, T=10, alpha=1, delta=3, sigma=2, tau = 0.2, gammaw=0.5, gammab=-0.5):
    """
    Generate a synthetic dynamic network with attractor.
    """
    assert N%2 == 0
    s = np.concatenate((np.ones(N//2),np.zeros(N//2)))
    z_t = np.random.randn(N,d)*sigma
#     z_t = np.concatenate((np.random.randn(N//2,d)*sigma+0.5, np.random.randn(N//2,d)*sigma-0.5), axis=0)
#     z_t = truncnorm.rvs(-4, 4, size=N*d).reshape((N,d))*sigma
    eta_t = alpha-pdist(z_t)
    p_t = 1/(1+np.exp(-eta_t))
    y_t = np.random.binomial(1, p_t)
    #save
    z = [z_t]
    y = [y_t]
    Aw = []
    Ab = []
    for i in range(1,T):
        #normalized graph laplalcian by row
        Aw_t = squareform(y_t)*(np.outer(s, s)+np.outer(1-s, 1-s))
        Aw_t = -2*normalize(laplacian(Aw_t), axis=1, norm='l1')
        Ab_t = squareform(y_t)*(np.outer(1-s, s)+np.outer(s, 1-s))
        Ab_t = -2*normalize(laplacian(Ab_t), axis=1, norm='l1')
        mu_t = z_t+gammaw*(Aw_t@z_t)+gammab*(Ab_t@z_t)
        z_t = mu_t+np.random.randn(N,d)*tau
#         z_t = mu_t+truncnorm.rvs(-4, 4, size=N*d).reshape((N,d))*tau
        eta_t = alpha-pdist(z_t)+delta*y_t
        p_t = 1/(1+np.exp(-eta_t))
        y_t = np.random.binomial(1, p_t)
        #save
        z.append(z_t)
        y.append(y_t)
        Aw.append(Aw_t)
        Ab.append(Ab_t)
    return z, y, Aw, Ab

def convert(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def preprocess(y, Aw, Ab, N, T):
    """
    Preprocess data for model training.
    """
    
    #label
    label = np.concatenate(y)
    #persist
    persist = np.concatenate(y[:(T-1)])
    persist = np.concatenate([np.zeros(y[0].shape[0]),persist])
    assert persist.shape[0] == label.shape[0]
    #Aw
    sparse_Aw = []
    for aw in Aw:
        sparse_Aw.append(sparse.csr_matrix(aw))
    sparse_Aw = sparse.block_diag(sparse_Aw)
    #Ab
    sparse_Ab = []
    for ab in Ab:
        sparse_Ab.append(sparse.csr_matrix(ab))
    sparse_Ab = sparse.block_diag(sparse_Ab)
    #combination_N
    combination_N = torch.combinations(torch.arange(N))
    _combination_N = torch.combinations(torch.arange(N))
    for i in range(1,T):
        start = i*N
        combination_N = torch.cat((combination_N, _combination_N+start), dim = 0)
    return from_numpy(label), from_numpy(persist), convert(sparse_Aw), convert(sparse_Ab), combination_N

def soft_align(z_hat,z_true):
    R, sca = linalg.orthogonal_procrustes(z_hat,z_true)
    return z_hat@R

def visualize(z_hat,z_true,start,end,caption=None):
    """
    Visualize and compare embedding
    """
    #rotate as a whole
    z_hat = soft_align(z_hat,z_true)
    #keep time t
    z_true = z_true[start:end]
    z_hat = z_hat[start:end]
    x_hat = z_hat[:,0]
    y_hat = z_hat[:,1]
    x_true = z_true[:,0]
    y_true = z_true[:,1]
    #subplot
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(9, 4))
    #figure on the left
    ax1.scatter(x_hat, y_hat, c=x_true, s=10, cmap="Set2")
    x_lim = ((z_hat**2)**0.5).max()+1
    ax1.set_xlim((-x_lim, x_lim))
    ax1.set_ylim((-x_lim, x_lim))
    ax1.set_title("Estimation")
    #figure on the right
    ax2.scatter(x_true, y_true, c=x_true, s=10, cmap="Set2")
    x_lim = ((z_true**2)**0.5).max()+1
    ax2.set_xlim((-x_lim, x_lim))
    ax2.set_ylim((-x_lim, x_lim))
    ax2.set_title("Gound truth")
    #add label at the bottom of the plot
    if caption is not None:
        fig.supxlabel(caption, fontsize=12)    
    plt.show()
#     return np.mean(z_hat-z_true), np.std(z_hat-z_true)
    return z_hat-z_true

    
def visualize_membership(z,membership,start,end,caption=None):  
    z = z[start:end]
    x_lim = ((z**2)**0.5).max()+1
    membership = membership[start:end]
    plt.figure(figsize=(7,7))
    plt.scatter(z[:,0], z[:,1], c=membership, s=20, cmap="Set2")
    plt.xlim(-x_lim, x_lim) 
    plt.ylim(-x_lim, x_lim) 
    if caption is not None:
        fig.supxlabel(caption, fontsize=12)
    plt.show()

class ClsnaModel(torch.nn.Module):
    def __init__(self,device,N,T,ar_pair,Aw,Ab,D=2):
        """
        A CLSNA model for training on dynamic networks using PyTorch.

        Parameters:
        -----------
        device : torch.device
            The device (CPU or GPU) to perform the computations on.
        N : int
            The number of nodes in the network.
        T : int
            The number of time steps in the network.
        ar_pair : torch.Tensor
            A pair of indices of nodes which are modeled by an AR(1) process (autoregressive of order 1).
        Aw : torch.sparse
            A sparse adjacency matrix representing within-group connections.
        Ab : torch.sparse
            A sparse adjacency matrix representing between-group connections.
        D : int, optional
            Dimensionality of the latent space (default is 2).
        """
        super().__init__()
        self.N = N
        self.T = T
        self.D = D
        self.z = Parameter(torch.randn((N*T,D)), requires_grad = True)
        self.para = Parameter(torch.zeros((3, 2)), requires_grad = True)
        #ar_pair: a pair of indices of nodes which are modeled by AR(1)
        self.ar_pair = ar_pair.detach().clone().to(device)
        self.Aw = Aw.to(device).coalesce()
        self.Ab = Ab.to(device).coalesce()

    def forward(self):
        return self.z, self.para
            
    def loss(self, device,label,persist,sample_edge,T_index,ss=1,tt=1):
        #label: minibatch y
        #sample_edge: minibatch edge
        #persist: minibatch y_(t-1)
        #logsigma2 = self.para[0,0]
        logsigma2 = 2*torch.log(torch.tensor([ss], device = device, requires_grad=False))
        #logtau2 = self.para[1,0]
        logtau2 = 2*torch.log(torch.tensor([tt], device = device, requires_grad=False))
        alpha = self.para[0,1]
        gw = self.para[1,1]
        gb = self.para[2,0]
        delta = self.para[2,1]
        tau2 = torch.exp(logtau2)
        sigma2 = torch.exp(logsigma2)
        #calculate loss related to edges
        target = self.z[sample_edge[:,0]]
        source = self.z[sample_edge[:,1]]
        distance = torch.sum((target-source)**2,dim = 1)**0.5
        eta = (alpha-distance+delta*persist)
        eta2 = eta[eta>15]
        eta3 = eta[eta<-90]
        eta4 = eta[(eta<15) & (eta>-90)]
        y2 = label[eta>15]
        y3 = label[eta<-90]
        y4 = label[(eta<15) & (eta>-90)]
        log_p2 = (1-y2)*(-eta2)
        log_p3 = y3*(eta3)
        log_p4 = y4*torch.log(torch.sigmoid(eta4))+(1-y4)*torch.log(1-torch.sigmoid(eta4))
        p1 = torch.sum(log_p2)+torch.sum(log_p3)+torch.sum(log_p4)
        #T=1 attractor
        p2 = -self.z[:self.N]**2/2/sigma2-logsigma2/2
        #T>1 attractor    
        att_w = gw*torch.sparse.mm(self.Aw,self.z[:(self.T-1)*self.N])
        att_b = gb*torch.sparse.mm(self.Ab,self.z[:(self.T-1)*self.N])
        _p3 = self.z[self.ar_pair[:,1]]-self.z[self.ar_pair[:,0]]-(att_w+att_b)
        p3 = -_p3**2/2/tau2-logtau2/2
        #combine p2 p3
        pt = torch.cat((p2,p3),dim = 0)[T_index]
        #adjust*T_index/edge_index = 2/(N-1)
        adjust = 2*sample_edge.size(0)/T_index.size(0)/(self.N-1)
        #return -(p1+adjust*torch.sum(pt)-(gw-0)**2/2/100-(gb-0)**2/2/100)
        return -(p1+adjust*torch.sum(pt))