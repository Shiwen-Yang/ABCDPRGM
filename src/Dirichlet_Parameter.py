import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Given observations a Dirichlet random variable, estimate the parameter alpha using MOME and MLE
class Estimate:
    """
    To estimate the parameter of a Dirichlet random variable using MLE

    Args:
        observation (torch.Tensor): Tensor of shape (n, p) containing i.i.d. samples of a Dirichlet random variable.
        tol (float): Tolerance for the gradient descent method (default is 10e-5).

    Attributes:
        observation (torch.Tensor): Tensor of shape (n, p) containing i.i.d. samples of a Dirichlet random variable.
        tolerance (float): Tolerance for the gradient descent method.
        MOME (torch.Tensor): Method of moment estimator for Dirichlet random variables.
        MLE (torch.Tensor): Maximum likelihood estimator for Dirichlet random variables.
    """
    def __init__(self, observation, tol = 10e-5):
        
        self.observation = observation
        self.tolerance = tol
        self.MOME = self.method_of_moment()
        self.MLE = self.MLE()

    
    def method_of_moment(self):

        """
        Computes the method of moment estimator (MOME) for Dirichlet random variables.

        Returns:
            torch.Tensor: The MOME.
        """

        E = self.observation.mean(dim = 0)
        V = torch.var(self.observation, dim = 0)

        mome = E * (E*(1 - E)/V - 1)

        return(mome)
    
    def likelihood(self, alpha, log_likelihood = False):

        """  
        Given observations, compute the likelihood or log-likelihood, L(X; alpha)

        Args:
            alpha(torch.Tensor): a component-wise positive vector
            log_likelihood(boolean): when true, the output becomes the log likelihood

        Returns:
            float: the likelihood or log-likelihood
        """

        alpha.to(device)
        alphap = alpha - 1

        c = torch.exp(torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum())
        likelihood = c * (self.observation ** alphap).prod(axis=1)
        likelihood.to("cpu")

        del(alpha, alphap, c)
        torch.cuda.empty_cache()
    
        if log_likelihood:
            return(torch.log(likelihood).sum())
        else:
            return(likelihood.sum())
    
    @staticmethod
    def mean_log(alpha):

        """  
        Expectation of the log(Dir(alpha)) distribution

        Args:
            alpha(torch.Tensor): a component-wise positive vector

        Returns:
            torch.Tensor of length p: the expected value of the log(Dir(alpha)) distribution
        """

        mean_of_log = (torch.digamma(alpha) - torch.digamma(alpha.sum()))

        return(mean_of_log)
    
    @staticmethod
    def var_log(alpha, inverse = False):

        """  
        Variance matrix of the log(Dir(alpha)) distribution

        Args:
            alpha(torch.Tensor): a component-wise positive vector
            inverse(boolean): when true, the output becomes the inverse of the variance matrix

        Returns:
            torch.Tensor: the variance matrix of the log(Dir(alpha)) distribution
        """
 
        p = alpha.shape[0]
        one_p = torch.ones(p).unsqueeze(1)
        c = torch.polygamma(1, alpha.sum())
        Q = -torch.polygamma(1, alpha)
        Q_inv = torch.diag(1/Q)

       #When inverse is true, the Sherman-Morrison formula is used to compute the inverse of the variance matrix
        if inverse:

            numerator = (Q_inv @ (c*one_p @ one_p.T) @ Q_inv)

            denominator = (1 + c * one_p.T @ Q_inv @ one_p)

            var_inv = Q_inv - numerator/denominator

            return(var_inv)
        else:
            
            return(torch.diag(Q) + c)

    def MLE(self):

        """  
        Given observations of a Dirichlet random variable, compute the maximum likelihood estimator

        Returns:
            torch.Tensor: the estimated parameter vector
        """
        #Given observations, compute the maximum likelihood estimator using newton gradient descent
        #The gradient descent is initialized on the computed MOME
        
        empirical_avg_log = torch.log(self.observation).mean(dim = 0)

        initialization = self.MOME

        tol = self.tolerance

        next_par = initialization

        go = True
        i = 1
        while go:
            var_inv = self.var_log(next_par, inverse = True)
            
            log_mean = empirical_avg_log - self.mean_log(next_par)

            step = var_inv @ (log_mean)

            next_par = next_par - step

            go = (i <= 1000) & (torch.norm(step) > tol)

            i += 1

        return(next_par)
