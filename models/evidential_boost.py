import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import gammaln, digamma
from scipy.special import polygamma

import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import gammaln
import scipy.stats as st
from scipy.special import psi  # digamma
from scipy.optimize import approx_fprime

def grad_logcdf_bound(Yi, mu, lam, alpha, beta, bound, which='lower'):
    eps = np.sqrt(np.finfo(float).eps)
    def loss_fn(x):
        m, l, a, b = x
        lam_nat = np.exp(l)
        alpha_nat = np.exp(a) + 1
        beta_nat = np.exp(b)
        nu = 2 * alpha_nat
        Omega = 2 * beta_nat * (1 + lam_nat)
        scale = np.sqrt(Omega / (lam_nat * nu))
        td = st.t(df=nu, loc=m, scale=scale)
        if which == 'lower':
            return -np.log(np.clip(td.cdf(bound), 1e-8, 1.0))
        else:
            return -np.log(np.clip(td.sf(bound), 1e-8, 1.0))
    x0 = np.array([mu, np.log(lam), np.log(alpha - 1), np.log(beta)])
    return approx_fprime(x0, loss_fn, epsilon=eps)


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
    

    def score(self, Y, params=None, evid_strength=0.1, kl_strength=0.05):
       
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

        if self.lower_bound is not None or self.upper_bound is not None:
            nu = 2*alpha
            scale = np.sqrt(Omega/(lam*nu))
            td = st.t(df=nu, loc=mu, scale=scale)
            if self.lower_bound is not None:
                mask_low = (Y <= self.lower_bound)
                if mask_low.any():
                    cdf_l = np.clip(td.cdf(self.lower_bound), 1e-8, 1.0)
                    nll[mask_low] = -np.log(cdf_l[mask_low])
            if self.upper_bound is not None:
                mask_up = (Y >= self.upper_bound)
                if mask_up.any():
                    sf_u = np.clip(td.sf(self.upper_bound), 1e-8, 1.0)
                    nll[mask_up] = -np.log(sf_u[mask_up])

        evidential_reg = self.evidential_regularizer(Y, mu, lam, alpha)
        kl_reg = self.kl_divergence_nig(mu, lam, alpha, beta)
        return nll + evid_strength * evidential_reg + kl_strength * kl_reg
   
    
    def d_score(self, Y, params=None, evid_strength=0.1, kl_strength=0.05):
        # Unpack or use stored
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T
        
        # Stabilize
        eps = 1e-8
        lam   = np.clip(lam,   eps, None)
        alpha = np.clip(alpha, 1.0+eps, None)
        beta  = np.clip(beta,  eps, None)
        nu    = 2*alpha
        Omega = 2*beta*(1+lam)

        # --- corrected helper for the uncensored part ---
        def comps(r):
            t = lam*r*r + Omega
            g_mu    = lam*(nu+1)*(-r)/t
            g_lam = (
            -0.5/lam
            - alpha * (2*beta)/Omega
            + (alpha+0.5)*(resid**2) / t
            )
            g_alpha = -np.log(Omega) + np.log(t) + psi(alpha) - psi(alpha+0.5)
            g_beta  = -alpha/beta + (alpha+0.5)*(2*(1+lam))/t
            return g_mu, g_lam, g_alpha, g_beta

        # Uncensored gradients
        resid = Y - mu
        d_mu, d_lam, d_alpha, d_beta = comps(resid)

        # Tobit override using numerical derivatives
        if self.lower_bound is not None or self.upper_bound is not None:
            for i in range(len(Y)):
                if self.lower_bound is not None and Y[i] <= self.lower_bound:
                    grad = grad_logcdf_bound(Y[i], mu[i], lam[i], alpha[i], beta[i], self.lower_bound, which='lower')
                    d_mu[i], d_lam[i], d_alpha[i], d_beta[i] = grad
                elif self.upper_bound is not None and Y[i] >= self.upper_bound:
                    grad = grad_logcdf_bound(Y[i], mu[i], lam[i], alpha[i], beta[i], self.upper_bound, which='upper')
                    d_mu[i], d_lam[i], d_alpha[i], d_beta[i] = grad
                # uncensored points keep their original values

        # Regularizer gradients (unchanged)
        sign = np.sign(Y - mu)
        gev_mu    = np.mean(-sign*(2*alpha+lam))
        gev_lam   = np.mean(np.abs(Y-mu))
        gev_alpha = np.mean(2*np.abs(Y-mu))
        gev_beta  = 0.0
        mu0, lam0, alpha0, beta0 = 0,1,2,1
        gk_mu    = alpha0*lam*(mu-mu0)/beta
        gk_lam   = -0.5/lam + alpha0*((mu-mu0)**2/(2*beta)+1/lam0)
        gk_alpha = (alpha-alpha0)*polygamma(1,alpha)+(beta0/beta-1)
        gk_beta  = alpha0*(1/beta-beta0/beta**2) - alpha/beta**2 - alpha0*lam*(mu-mu0)**2/(2*beta**2)

        d_mu    += evid_strength*gev_mu    + kl_strength*gk_mu
        d_lam   += evid_strength*gev_lam   + kl_strength*gk_lam
        d_alpha += evid_strength*gev_alpha + kl_strength*gk_alpha
        d_beta  += evid_strength*gev_beta  + kl_strength*gk_beta

        # Chain‐rule to raw space
        raw_mu    = d_mu
        raw_lam   = d_lam*lam
        raw_alpha = d_alpha*(alpha-1)
        raw_beta  = d_beta*beta
        return np.stack([raw_mu, raw_lam, raw_alpha, raw_beta], axis=1)
    

    def old_d_score(self, Y, params=None, evid_strength=0.1, kl_strength=0.05):
    # Unpack or use stored
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T

        # Stabilize
        eps = 1e-8
        lam   = np.clip(lam, eps, None)
        alpha = np.clip(alpha, 1.0+eps, None)
        beta  = np.clip(beta, eps, None)

        # Shorthands
        nu      = 2 * alpha
        Omega   = 2 * beta * (1 + lam)
        resid   = Y - mu
        term    = lam * resid**2 + Omega

        # 1) ∂ℓ/∂μ
        d_mu = lam*(nu+1)*(-resid) / term

        # 2) ∂ℓ/∂λ (corrected)
        d_lam = (
            -0.5/lam
            - alpha * (2*beta)/Omega
            + (alpha+0.5)*((resid**2)) / term
        )

        # 3) ∂ℓ/∂α
        d_alpha = (
            -np.log(Omega)
            + np.log(term)
            + psi(alpha)
            - psi(alpha + 0.5)
        )
        # 4) ∂ℓ/∂β
        d_beta = (
            -alpha/beta
            + (alpha+0.5)*(2*(1+lam)) / term
        )

        # --- evidential reg grads ---
        sign = np.sign(Y - mu)                       # vector
        grad_ev_mu    = np.mean(-sign * (2*alpha + lam))
        grad_ev_lam   = np.mean( np.abs(Y - mu) )
        grad_ev_alpha = np.mean( 2 * np.abs(Y - mu) )
        grad_ev_beta  = 0.0

        # --- KL regularization grads ---
        mu0, lam0, alpha0, beta0 = (0, 1, 2, 1)
        # d_mu_KL
        gk_mu = alpha0 * lam * (mu - mu0) / beta
        # d_lam_KL
        gk_lam = -0.5 / lam + alpha0 * ((mu - mu0) ** 2 / (2 * beta) + 1 / lam0)
        # d_alpha_KL
        gk_alpha = (alpha - alpha0) * polygamma(1, alpha) + (beta0 / beta - 1)
        # d_beta_KL
        gk_beta = alpha0 * (1 / beta - beta0 / beta**2) - alpha / beta**2 - alpha0 * lam * (mu - mu0) ** 2 / (2 * beta**2)


        # combine
        d_mu   += evid_strength * grad_ev_mu     + kl_strength * gk_mu
        d_lam  += evid_strength * grad_ev_lam    + kl_strength * gk_lam
        d_alpha+= evid_strength * grad_ev_alpha  + kl_strength * gk_alpha
        d_beta += evid_strength * grad_ev_beta   + kl_strength * gk_beta

        # chain‐rule back to raw‐space
        raw_mu_grad   = d_mu
        raw_lam_grad   = d_lam   * lam
        raw_alpha_grad = d_alpha * (alpha-1)
        raw_beta_grad = d_beta  * beta
        
        return np.stack([
            raw_mu_grad,
            raw_lam_grad,
            raw_alpha_grad,
            raw_beta_grad
        ], axis=1)

    
    def metric(self, Y=None, params=None,
               evid_strength: float = 0.2,
               kl_strength:    float = 0.01):
        """ Empirical FIM from full‐loss gradients """
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
            params = [mu, lam, alpha, beta]
        else:
            params = np.stack(params, axis=-1).T
        if Y is None:
            Y = self._last_Y
        grads = self.d_score(Y, params=params)
        FIM = np.array([np.outer(g, g) + 1e-5*np.eye(g.shape[0]) for g in grads])
        return FIM



    
    

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
        return np.array([m, 0.0, np.log(1.0), np.log(s**2)])  

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
