import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import gammaln, digamma

import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import gammaln
import scipy.stats as st

def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# Custom score for Normal-Inverse-Gamma.
class NIGLogScore(LogScore):

    def kl_regularizer(self, params, prior_alpha=1.0, prior_beta=1.0, prior_lam=1.0):
        mu, lam, alpha, beta = np.stack(params, axis=-1).T

        kl = (alpha - prior_alpha) * digamma(alpha) - gammaln(alpha) + gammaln(prior_alpha) \
             + prior_alpha * (np.log(beta) - np.log(prior_beta)) \
             + (prior_beta / beta) - prior_alpha

        # Include KL divergence on lam (optional)
        kl_lam = lam - np.log(lam / prior_lam) - 1

        return np.mean(kl + kl_lam)

    def score(self, Y, params=None, reg_strength=0.01):
        if params is None:
            params = [self.mu, self.lam, self.alpha, self.beta]

        
        params = np.stack(params, axis=-1)
        epsilon = 1e-8
        mu, lam, alpha, beta = params.T
        lam = np.maximum(lam, epsilon)
        beta = np.maximum(beta, epsilon)

        term1 = np.maximum(np.pi / lam, epsilon)
        term2 = np.maximum(2 * beta, epsilon)
        term3 = np.maximum(beta + 0.5 * lam * (Y - mu)**2, epsilon)

        nll = (0.5 * np.log(term1)
               - alpha * np.log(term2)
               + gammaln(alpha)
               - gammaln(alpha + 0.5)
               + (alpha + 0.5) * np.log(term3))

        reg = self.kl_regularizer([mu, lam, alpha, beta]) if reg_strength > 0 else 0.0

        return -nll + reg_strength * reg


    def d_score(self, Y, params=None):
        """
        Compute the gradients of the NLL with respect to the parameters.
        """
        # If params is None, use the class attributes
        if params is None:
            params = [self.mu, self.lam, self.alpha, self.beta]

        # Stack the parameters into a 2D array (shape: (N, 4)) where each row is [mu, lam, alpha, beta] for a sample
        params = np.stack(params, axis=-1)  # Shape: (N, 4)

        # Extract parameters
        mu, lam, alpha, beta = params.T  # Transpose to get shape (N,) for each parameter

        # Stabilize the calculation by ensuring no term inside the logarithms becomes invalid
        epsilon = 1e-8
        term1 = np.clip(beta + 0.5 * lam * (Y - mu)**2, 1e-6, 1e6)  # Prevent extremes


        # Compute gradients with respect to each parameter
        grad_mu = (alpha + 0.5) * lam * (Y - mu) / term1
        grad_lambda = (alpha + 0.5) * (Y - mu)**2 / term1
        grad_alpha = -np.log(2 * beta) + digamma(alpha) - digamma(alpha + 0.5) \
                     + np.log(term1)
        grad_beta = np.clip(-alpha / beta + (alpha + 0.5) / term1, -1e3, 1e3) 

        # Return gradients for each parameter (mu, lambda, alpha, beta)
        return np.stack([grad_mu, grad_lambda, grad_alpha, grad_beta], axis=1) 

    def metric(self):
        #params = np.stack(params, axis=-1)
        #mu, lam, alpha, beta = params.T
        #I_mu = lam * (alpha + 0.5) / beta  # Diagonal term for μ
        #I_lam = (alpha + 0.5) / (2 * lam**2)  # Diagonal term for λ
        #I_alpha = digamma(alpha) - digamma(alpha + 0.5)  # Diagonal term for α
        #I_beta = alpha / beta**2  # Diagonal term for β
        return np.identity(4)
        return np.diag([I_mu, I_lam, I_alpha, I_beta])
    
    

class NormalInverseGamma(RegressionDistn):
    """
    Implements the Normal-Inverse-Gamma (NIG) distribution for NGBoost.
    
    Parameters:
      μ   : the mean (location) of the Normal component,
      λ   : a positive scaling factor for the Normal variance,
      α   : the shape parameter (α > 1) for the Inverse Gamma,
      β   : the scale parameter (β > 0) for the Inverse Gamma.
      
    Predictive (aleatoric) variance:  E[σ²] = β/(α - 1).
    Epistemic uncertainty approximation:  Var(μ) ≈ β²/(λ*(α - 1)²*(α - 2)).
    """
    n_params = 4  # Four parameters: μ, λ, α, β.
    scores = [NIGLogScore]

    def __init__(self, params):
        """
        params: an array-like with 4 elements.
            params[0]: μ (location)
            params[1]: raw parameter for λ (we use exp to enforce λ > 0)
            params[2]: raw parameter for α (we use exp and add 1 to enforce α > 1)
            params[3]: raw parameter for β (we use exp to enforce β > 0)
        """
        super().__init__(params)
        self.mu    = params[0]
        self.lam   = softplus(params[1]) + 1e-6      # Avoid zero
        self.alpha = softplus(params[2]) + 2.0       # Enforce α > 1
        self.beta  = softplus(params[3]) + 1e-6      # Avoid zero
        

    @staticmethod
    def fit(Y):
        """
        Provides initial parameter estimates from the data Y.
        
        Returns an np.array of 4 elements:
          - raw_mu: the mean of Y,
          - raw_lam: initialized to 0 so that exp(0)=1,
          - raw_alpha: initialized to 0 so that alpha = exp(0)+1 = 2,
          - raw_beta: the log of the variance of Y (so that beta = variance of Y).
        """
        m = np.mean(Y)
        s = np.std(Y)
        return np.array([m, 0.0, np.log(3.0), np.log(s**2)])  

    def sample(self, m):
        """
        Draw m samples per prediction.
        
        Sampling:
          1. Sample σ² from an Inverse Gamma with shape α and scale β.
             (Implemented here by sampling from Gamma and taking the reciprocal.)
          2. Sample Y from a Normal with mean μ and variance σ²/λ.
        """
        shape = self.mu.shape  # assuming vectorized μ
        # Sample sigma^2 from Inverse Gamma(α, β).
        sigma2 = 1 / np.random.gamma(self.alpha, 1 / self.beta, size=(m, *shape))
        samples = np.random.normal(self.mu, np.sqrt(sigma2 / self.lam))
        return samples

    def pred_dist(self):
        """
        Computes predictive statistics and returns them as a dictionary.
          - "mean": μ,
          - "aleatoric": β/(α - 1),
          - "epistemic": β²/(λ*(α - 1)²*(α - 2)).
        """
        aleatoric = self.beta / (self.alpha - 1)
        epistemic = self.beta**2 / (self.lam * (self.alpha - 1)**2 * (self.alpha - 2))
        return {"mean": self.mu, "aleatoric": aleatoric, "epistemic": epistemic}

    def score(self, Y):
        """
        Calculate the negative log-likelihood and return it.
        """
        params = [self.mu, self.lam, self.alpha, self.beta]
        return NIGLogScore().score(Y, params=params)

    def d_score(self, Y):
        """
        Calculate the gradients of the NLL with respect to parameters.
        """
        params = [self.mu, self.lam, self.alpha, self.beta]
        return NIGLogScore().d_score(Y, params=params)
    
    def metric(self):
        """
        Calculate the metric for the parameters.
        """
        params = [self.mu, self.lam, self.alpha, self.beta]
        return NIGLogScore().metric(params=params)
    
    @property
    def is_regression(self):
        return True

    @property
    def params_dict(self):
        return {"mu": self.mu, "lam": self.lam, "alpha": self.alpha, "beta": self.beta}

    def mean(self):
        return self.mu
    
    def var(self):
        """
        For the Normal-Inverse Gamma, the variance can be derived from the parameters.
        """
        return self.lam * (self.alpha + 0.5)  # This might need adjustment depending on the definition

    def logpdf(self, Y):
        """
        Compute the log of the probability density function (logpdf) for Normal-Inverse Gamma.
        """
        mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        logpdf = (
            -0.5 * np.log(np.pi / lam) 
            - alpha * np.log(2 * beta) 
            + gammaln(alpha) 
            - gammaln(alpha + 0.5)
            + (alpha + 0.5) * np.log(beta + 0.5 * lam * (Y - mu) ** 2)
        )
        return logpdf
