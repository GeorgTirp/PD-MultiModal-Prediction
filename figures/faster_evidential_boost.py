import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import gammaln, digamma, polygamma, psi
import scipy.stats as st
from scipy.optimize import approx_fprime
from numba import njit, prange
from ngb_jit import d_score_numba, full_score_numba, compute_diag_fim#, digamma, trigamma, psi, gammaln
#import line_profiler

def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def positive(x, eps=1e-3):
    return np.log1p(np.exp(x)) + eps

# Custom score for Normal-Inverse-Gamma.
class NIGLogScore(LogScore):

    
    lower_bound = None
    upper_bound = None

    #def __init__(self, lower_bound=None, upper_bound=None):
    #    
    #    if lower_bound is not None:
    #        self.lower_bound = np.asarray(lower_bound, float)
    #    if upper_bound is not None:
    #        self.upper_bound = np.asarray(upper_bound, float)

    @classmethod
    def set_bounds(cls, lower, upper):
        cls.lower_bound = lower
        cls.upper_bound = upper

    def kl_divergence_nig(self, mu, lam, alpha, beta,
                      mu0=0.0, lam0=1.0, alpha0=2.0, beta0=1.0,
                      eps=1e-8):
        

        # Clip to avoid division by zero or log of zero
        lam = np.clip(lam, eps, 1e6)
        alpha = np.clip(alpha, 1.0 + eps, 1e6)
        beta = np.clip(beta, eps, 1e6)

        # Individual terms
        term1 = 0.5 * np.log(lam0 / lam)
        term2 = alpha0 * np.log(beta / beta0)
        term3 = -gammaln(alpha) + gammaln(alpha0)
        term4 = (alpha - alpha0) * digamma(alpha)
        term5 = alpha0 * (lam * (mu - mu0) ** 2 / (2 * beta))
        term6 = alpha0 * (lam / lam0 - 1)
        term7 = alpha * (beta0 / beta - 1)

        kl = term1 + term2 + term3 + term4 + term5 + term6 + term7
        return np.mean(kl)

    def evidential_regularizer(self, Y, mu, lam, alpha):
        
        error = np.abs(Y - mu)
        penalty = error * (2 * alpha + lam)
        return np.mean(penalty)
    
    #@line_profiler.profile
    def score(self, Y, params=None, evid_strength=0.1, kl_strength=0.05):
        # 1) unpack parameters into arrays
        self._last_Y = Y
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
            mu, lam, alpha, beta = mu.ravel(), lam.ravel(), alpha.ravel(), beta.ravel()
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T

        # 2) (optional) pre-clip to avoid NaNs
        eps = 1e-8
        lam   = np.clip(lam,   eps, None)
        alpha = np.clip(alpha, 1.0 + eps, None)
        beta  = np.clip(beta,  eps, None)

        # 3) call the Numba ufunc — this returns an (n,) array of per-sample losses
        per_sample_losses = full_score_numba(
            Y.astype(np.float64),
            mu.astype(np.float64),
            lam.astype(np.float64),
            alpha.astype(np.float64),
            evid_strength,
            kl_strength
        )

        # 4) return the vector directly
        return per_sample_losses
    

    def old_score(self, Y, params=None, evid_strength=0.1, kl_strength=0.05):
       
        self._last_Y = Y
        if params is None:
            params = [self.mu, self.lam, self.alpha, self.beta]

        mu, lam, alpha, beta = np.stack(params, axis=-1).T

        Omega = 2 * beta * (1 + lam)
        term1 = 0.5*( np.log(np.pi) - np.log(lam) )
        term2 = -alpha * np.log( Omega )
        term3 = (alpha + 0.5) * np.log( lam*(Y-mu)**2 + Omega )
        term4 = gammaln(alpha) - gammaln(alpha + 0.5)
        nll = term1 + term2 + term3 + term4

        evidential_reg = self.evidential_regularizer(Y, mu, lam, alpha)
        kl_reg = self.kl_divergence_nig(mu, lam, alpha, beta)
        
        return nll + evid_strength * evidential_reg + kl_strength * kl_reg
    


    #@line_profiler.profile
    def d_score(self, Y, params=None, evid_strength=0.1, kl_strength=0.05):
        # Unpack or use stored
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T
        
        # Stabilize
        grads = d_score_numba(Y.astype(np.float64),
                                 mu.astype(np.float64),
                                 lam.astype(np.float64),
                                 alpha.astype(np.float64),
                                 beta.astype(np.float64),
                                 evid_strength,
                                 kl_strength)
        self.current_grads = grads
        return grads
    
    #@line_profiler.profile
    def metric(self, Y=None, params=None, diagonal: bool = False):
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
            params = [mu, lam, alpha, beta]
        else:
            params = np.stack(params, axis=-1).T
        if Y is None:
            Y = self._last_Y

        grads = self.current_grads

        if diagonal:
            return compute_diag_fim(grads)
        else:
            # Full FIM
            return np.array([np.outer(g, g) + 1e-5*np.eye(g.shape[0]) for g in grads])
        #a11 = ((2 * self.alpha + 1) * 0.5) * (self.lam / self.beta)
        #a22 = (1+ (1/ self.alpha -1)) * 0.5
        #a33 = (self.alpha -1)**2 * (polygamma(self.alpha + 0.5) - polygamma(self.alpha )) 
        #a44 = -0.5
        #a34 = a43 = (self.alpha -1) * (polygamma(self.alpha + 0.5) - polygamma(self.alpha ))
        #FIM = np.zeros([4, 4])
        #FIM[0, 0] = a11
        #FIM[1, 1] = a22
        #FIM[2, 2] = a33
        #FIM[3, 3] = a44
        #FIM[3, 2] = a34
        #FIM[2, 3] = a43
        #return FIM
   



    
    

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
        self.mu    = params[0]
        self.lam   = np.exp(params[1])     # Avoid zero
        self.alpha = np.exp(params[2]) + 1      # Enforce α > 1
        self.beta  = np.exp(params[3])     # Avoid zero
        print(f"Initialized NIG with params: {self.mu}, {self.lam}, {self.alpha}, {self.beta}")


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
        #m = np.mean(Y)
        #s = np.std(Y)
        #return np.array([m, 0.0, np.log(1.0), np.log(s**2)])  
        Y = np.asarray(Y).ravel()
        n   = len(Y)
        mu_y  = np.mean(Y)
        var_y = np.var(Y)  # important: variance, not std!

        # 1) Define a weak prior centered at mu_y
        mu0, lam0 = mu_y, 1.0
        alpha0, beta0 = 1.0, 1e-6

        # 2) Compute posterior parameters (Normal‐Inverse‐Gamma conjugacy)
        lam_n   = lam0 + n
        mu_n    = (lam0 * mu0 + n * mu_y) / lam_n
        alpha_n = alpha0 + n/2.0

        # Sum of squares about the sample mean
        sse = np.sum((Y - mu_y)**2)
        # Additional term for shift of means
        mean_diff_term = (lam0 * n * (mu_y - mu0)**2) / (2.0 * lam_n)
        beta_n  = beta0 + 0.5 * sse + mean_diff_term

        # 3) Clip to safe ranges
        lam_n   = np.clip(lam_n,   1e-3, 1e3)
        alpha_n = np.clip(alpha_n, 1+1e-3, 1e3)   # enforce α>1
        beta_n  = np.clip(beta_n,  1e-6, 1e3)

        # 4) Return in raw‐param form:
        #    [ mu_n,  log(lambda_n),  log(alpha_n - 1),  log(beta_n) ]
        return np.array([
            mu_n,
            np.log(lam_n),
            np.log(alpha_n - 1.0),
            np.log(beta_n)
        ])

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

    def pred_uncertainty(self):
        """
        Computes predictive statistics and returns them as a dictionary.
          - "mean": μ,
          - "aleatoric": β/(α - 1),
          - "epistemic": β²/(λ*(α - 1)²*(α - 2)).
        """
        aleatoric = self.beta / (self.alpha - 1)
        epistemic = self.beta**2 / (self.lam * (self.alpha - 1)**2 * (self.alpha - 2))
        return {"mean": self.mu, "aleatoric": aleatoric, "epistemic": epistemic}

    def pred_dist(self):
        """
        Computes predictive statistics and returns them as a dictionary.
          - "mean": μ,
          - "aleatoric": β/(α - 1),
          - "epistemic": β²/(λ*(α - 1)²*(α - 2)).
        """
        return self.mu, self.lam, self.alpha, self.beta
    
    #def score(self, Y):
    #    """
    #    Calculate the negative log-likelihood and return it.
    #    """
    #    params = [self.mu, self.lam, self.alpha, self.beta]
    #    return NIGLogScore().score(Y, params=params)
#
    #def d_score(self, Y):
    #    """
    #    Calculate the gradients of the NLL with respect to parameters.
    #    """
    #    params = [self.mu, self.lam, self.alpha, self.beta]
    #    return NIGLogScore().d_score(Y, params=params)
    
    def metric(self, Y):
        """
        Calculate the metric for the parameters.
        """
        params = [self.mu, self.lam, self.alpha, self.beta]
        return NIGLogScore().metric(Y, params=params)
    
    @property
    def is_regression(self):
        return True

    @property
    def params(self):
        return {"mu": self.mu, "lam": self.lam, "alpha": self.alpha, "beta": self.beta}

    def mean(self):
        return self.mu
    
    def var(self):
        """
        For the Normal-Inverse Gamma, the variance can be derived from the parameters.
        """
        # Aleatoric variance (σ²) = β / (α - 1)
        aleatoric = self.beta / (self.alpha - 1)
        epistemic = self.beta / (self.lam * (self.alpha - 1))
        predictive = aleatoric + epistemic
        return predictive
        #return aleatoric
    
    def predict_variance(self, X):
        # predict_dist returns a list/array of NormalInverseGamma instances
        dists = self.predict_dist(X)
        # call .var() on each to get aleatoric+epistemic
        return np.array([dist.var() for dist in dists])

    def logpdf(self, Y):
        mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta

        # Degrees of freedom
        nu = 2 * alpha
        # Scale (variance of Student-t)
        var = beta / (lam * alpha)

        # Compute the log-pdf of the Student-t distribution
        coeff = gammaln((nu + 1) / 2) - gammaln(nu / 2)
        norm = -0.5 * np.log(nu * np.pi * var)
        sq_term = (Y - mu) ** 2 / (nu * var)
        log_prob = coeff + norm - 0.5 * (nu + 1) * np.log1p(sq_term)

        return log_prob


