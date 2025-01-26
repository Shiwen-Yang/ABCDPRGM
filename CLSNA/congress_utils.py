# Stochastic gradient descent-based inference for dynamic network models with attractors
# This script defines helper functions and classes for use to simulate a dynamic network with changing membership and fit the extended CLSNA model to the simulated data for inference

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
from utils import convert
from numpy.random import default_rng

def congress_clsna(N=200, d=2, T=10, alpha=1, delta=3, sigma=2, tau = 0.2 , phi=1, gammaw=0.5, gammab=-0.5, n_leave=20):
    """
    Generate a synthetic extended CLSNA network with attractor. Similar to Congress, at each time step, a random subset of nodes are dropped out, while an equal number of new nodes are randomly initialized according to the model's prior distribution.
    """
    assert N%2 == 0
    N_each = N//2
    s = np.concatenate((np.ones(N//2),np.zeros(N//2)))
    z_t = np.random.randn(N,d)*sigma
    z_t = np.concatenate((np.random.randn(N//2,d)*sigma+0.5, np.random.randn(N//2,d)*sigma-0.5), axis=0)
#     z_t = truncnorm.rvs(-4, 4, size=N*d).reshape((N,d))*sigma
    eta_t = alpha-pdist(z_t)
    p_t = 1/(1+np.exp(-eta_t))
    y_t = np.random.binomial(1, p_t)
    #save
    z = [z_t]
    y = [y_t]
    Aw = []
    Ab = []
    persist = []
    leaves = {}
    for i in range(1,T):
        #normalized graph laplalcian by row
        mu_s1 = np.mean(z_t[:N_each],axis=0)
        mu_s2 = np.mean(z_t[N_each:],axis=0)
        Aw_t = squareform(y_t)*(np.outer(s, s)+np.outer(1-s, 1-s))
        Aw_t = -2*normalize(laplacian(Aw_t), axis=1, norm='l1')
        Ab_t = squareform(y_t)*(np.outer(1-s, s)+np.outer(s, 1-s))
        Ab_t = -2*normalize(laplacian(Ab_t), axis=1, norm='l1')
        #from here, mu_t, z_t are stats of time i
        mu_t = z_t+gammaw*(Aw_t@z_t)+gammab*(Ab_t@z_t)
        z_t = mu_t+np.random.randn(N,d)*tau
        rng = default_rng()
        s1_leave = rng.choice(N_each, size=n_leave//2, replace=False)
        s2_leave = rng.choice(N_each, size=n_leave//2, replace=False)+N_each
        z_t[s1_leave] = mu_s1+np.random.randn(n_leave//2,d)*phi
        z_t[s2_leave] = mu_s2+np.random.randn(n_leave//2,d)*phi
        leave_t = np.concatenate((s1_leave,s2_leave))
#         z_t = mu_t+truncnorm.rvs(-4, 4, size=N*d).reshape((N,d))*tau
        _c = np.full((N,), 1)
        _c[leave_t] = 0
        correction = squareform(np.outer(_c,_c),checks=False)
        eta_t = alpha-pdist(z_t)+delta*y_t*correction
        persist.append(y_t*correction)
        p_t = 1/(1+np.exp(-eta_t))
        y_t = np.random.binomial(1, p_t)
        #save
        z.append(z_t)
        y.append(y_t)
        Aw.append(Aw_t)
        Ab.append(Ab_t)
        leaves[i] = leave_t
    return z, y, persist, Aw, Ab, leaves

def get_always_in(N, T, z, y, persist, Aw, Ab, leaves):
    index_z = np.full((N,), True)
    for key in leaves:
        index_z[leaves[key]] = False
    index_y = squareform(np.outer(index_z,index_z),checks=False)
    s = np.concatenate((np.ones(N//2),np.zeros(N//2)))
    s = s[index_z]
    y_t = y[0][index_y]
    z_t = z[0][index_z]
    nz = [z_t]
    ny = [y_t]
    nAw = []
    nAb = []
    npersist = []
    for i in range(1,T):
        Aw_t = squareform(y_t)*(np.outer(s, s)+np.outer(1-s, 1-s))
        Aw_t = -2*normalize(laplacian(Aw_t), axis=1, norm='l1')
        Ab_t = squareform(y_t)*(np.outer(1-s, s)+np.outer(s, 1-s))
        Ab_t = -2*normalize(laplacian(Ab_t), axis=1, norm='l1')
        #save
        y_t = y[i][index_y]
        z_t = z[i][index_z]
        npersist.append(persist[i-1][index_y])
        nz.append(z_t)
        ny.append(y_t)
        nAw.append(Aw_t)
        nAb.append(Ab_t)
    return nz, ny, npersist, nAw, nAb

def make_ar_pair(device,leaves,N,T):
    ar_pair = []
    for i in range(1,T):
        mask = torch.ones((N,),dtype=torch.bool)
        mask[from_numpy(leaves[i])]=False
        _s = torch.arange(0,N, requires_grad = False)
        _ar_pair = torch.stack((_s,_s+N), dim = 1)
        _ar_pair = _ar_pair[mask]+N*(i-1)
        ar_pair.append(_ar_pair)
    return (torch.cat(ar_pair,dim=0)).to(device)

def member_dict(device,leaves,N,T):
    new_at_t = {}
    n_each = N//2
    for i in range(1,T):
        leave_t = from_numpy(leaves[i]).to(device)
        #current_new
        current_new = {}
        current_new[0] = leave_t[leave_t<n_each]
        current_new[1] = leave_t[leave_t>=n_each]
        #store
        new_at_t[i] = current_new
    return new_at_t

def preprocess(y, Aw, Ab, N, T, persist):
    #label
    label = np.concatenate(y)
    #persist
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

class ClsnaModelCongress(torch.nn.Module):
    """
        A PyTorch torch.nn class for fitting a extended CLSNA model.

        Attributes:
        ----------
        device : torch.device
            The device to perform computations on (CPU or GPU).
        n_nodes : int
            The number of nodes (congress members) in the network at each time step.
        T : int
            The total number of time steps.
        D : int, optional
            The dimensionality of the latent space (default is 2).
        z : torch.nn.Parameter
            Latent positions of all nodes in the network.
        para : torch.nn.Parameter
            Model global parameters such as attraction parameters.
        ar_pair : torch.Tensor
            Pairs of nodes modeled by an AR(1) process, indicating temporal dependencies.
        Aw, Aw2, Ab : torch.sparse
            Sparse adjacency matrices representing network interactions (within and between groups).
        new_at_t, member_at_t : dict
            Dictionary tracking nodes that are newly elected or remain members at each time step.
    """ 
    def __init__(self,device,N,T,ar_pair,Aw,Ab,new_at_t,D=2):
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
        self.new_at_t = new_at_t

    def forward(self):
        return self.z, self.para
            
    def loss(self,device,label,persist,sample_edge,T_index,ss=1,tt=1,pp=1):
        #label: minibatch y
        #sample_edge: minibatch edge
        #persist: minibatch y_(t-1)
        #logsigma2 = self.para[0,0]
        logsigma2 = 2*torch.log(torch.tensor([ss], device = device, requires_grad=False))
        #logtau2 = self.para[1,0]
        logtau2 = 2*torch.log(torch.tensor([tt], device = device, requires_grad=False))
        logphi2 = 2*torch.log(torch.tensor([pp], device = device, requires_grad=False))
        alpha = self.para[0,1]
        gw = self.para[1,1]
        gb = self.para[2,0]
        delta = self.para[2,1]
        tau2 = torch.exp(logtau2)
        sigma2 = torch.exp(logsigma2)
        phi2 = torch.exp(logphi2)
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
        #T>1 and get reelected attractor
        att_w = gw*torch.sparse.mm(self.Aw,self.z[:(self.T-1)*self.N])
        att_b = gb*torch.sparse.mm(self.Ab,self.z[:(self.T-1)*self.N])
        _p3 = self.z[self.ar_pair[:,1]]-self.z[self.ar_pair[:,0]]-(att_w+att_b)[self.ar_pair[:,0]]
        
        p3 = -_p3**2/2/tau2-logtau2/2
        #T>1 and get newly elected attractor
        for time in range(1,self.T):
            for party in range(2):
                _start = (time-1)*self.N+party*self.N//2
                _end = _start+self.N//2
                party_mean = torch.mean(self.z[_start:_end],dim=0)
                _p4 = self.z[self.new_at_t[time][party]]-party_mean
                p4 = -_p4**2/2/phi2-logphi2/2
                p3 = torch.cat((p3,p4),dim = 0)
        #combine p2 p3
        pt = torch.cat((p2,p3),dim = 0)
        #adjust*T_index/sample_edge = #nodes/#edges
#         adjust = 2*sample_edge.size(0)/T_index.size(0)/(self.N-1)
        adjust = 1
        #return -(p1+adjust*torch.sum(pt)-(gw-0)**2/2/100-(gb-0)**2/2/100)
        return -(p1+adjust*torch.sum(pt))