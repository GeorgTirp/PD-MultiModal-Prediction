import numpy as np
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
from scipy.special import gammaln, digamma, polygamma, psi
import scipy.stats as st
from scipy.optimize import approx_fprime
from numba import njit, prange
from ngb_jit import d_score_numba, full_score_numba, compute_diag_fim#, digamma, trigamma, psi, gammaln
import line_profiler

def softplus(x):
    """
    Numerically stable implementation of the softplus function.

    Parameters:
        x (float or np.ndarray): Input value(s).

    Returns:
        np.ndarray: Softplus-transformed values.
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def positive(x, eps=1e-3):
    """
    Maps input to positive values using softplus, with a minimum epsilon shift.

    Parameters:
        x (float or np.ndarray): Input value(s).
        eps (float): Minimum value shift to ensure strict positivity.

    Returns:
        np.ndarray: Positive-transformed values.
    """
    return np.log1p(np.exp(x)) + eps

# Custom score for Normal-Inverse-Gamma.
class NIGLogScore(LogScore):
    """
    Custom scoring function for the Normal-Inverse-Gamma (NIG) distribution in NGBoost.

    Attributes:
        evid_strength (float): Coefficient for the evidential regularization term.
        kl_strength (float): Coefficient for KL divergence regularization.
    """
    
    lower_bound = None
    upper_bound = None
    evid_strength = 0.1
    kl_strength = 0.01

    @classmethod
    def set_params(cls, evid_strength=None, kl_strength=None):
        """
        Sets the regularization strengths for the NIG log score.

        Parameters:
            evid_strength (float): Evidential regularization strength.
            kl_strength (float): KL divergence regularization strength.
        """
        if evid_strength is not None:
            cls.evid_strength = evid_strength
        if kl_strength is not None:
            cls.kl_strength = kl_strength
    
    @classmethod
    def set_bounds(cls, lower, upper):
        """
        Set lower and upper bounds for predictions.

        Parameters:
            lower (float): Lower bound.
            upper (float): Upper bound.
        """
        cls.lower_bound = lower
        cls.upper_bound = upper

    def kl_divergence_nig(
            self, 
            mu, 
            lam, 
            alpha, 
            beta,
            mu0=0.0, 
            lam0=1.0, 
            alpha0=2.0, 
            beta0=1.0,
            eps=1e-8):
        """
        Compute the KL divergence between two NIG distributions.

        Parameters:
            mu, lam, alpha, beta: Current parameters.
            mu0, lam0, alpha0, beta0: Prior parameters.
            eps (float): Numerical stability threshold.

        Returns:
            float: Average KL divergence.
        """

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
        """
        Computes an evidential penalty encouraging predictive uncertainty to match residual error.

        Parameters:
            Y (np.ndarray): Target values.
            mu, lam, alpha (np.ndarray): Model parameters.

        Returns:
            float: Average evidential penalty.
        """
        error = np.abs(Y - mu)
        penalty = error * (2 * alpha + lam)
        return np.mean(penalty)
    
    @line_profiler.profile
    def score(self, Y, params=None):
        """
        Computes the custom NIG loss for each data point using JIT-compiled scoring.

        Parameters:
            Y (np.ndarray): Target values.
            params (list or None): Optional list of model parameters.

        Returns:
            np.ndarray: Per-sample NIG loss values.
        """
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

        evid_strength = self.__class__.evid_strength
        kl_strength = self.__class__.kl_strength
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
    


    @line_profiler.profile
    def d_score(self, Y, params=None):
        """
        Computes the gradient of the NIG loss with respect to the model parameters.

        Parameters:
            Y (np.ndarray): Target values.
            params (list or None): Optional list of model parameters.

        Returns:
            np.ndarray: Per-sample gradient vectors.
        """
        # Unpack or use stored
        if params is None:
            mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        else:
            mu, lam, alpha, beta = np.stack(params, axis=-1).T
        
        evid_strength = self.__class__.evid_strength
        kl_strength = self.__class__.kl_strength
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
    
    @line_profiler.profile
    def metric(self, Y=None, params=None, diagonal: bool = False):
        """
        Estimates the Fisher Information Matrix (FIM) from gradients.

        Parameters:
            Y (np.ndarray): Target values.
            params (list): Model parameters.
            diagonal (bool): If True, compute only the diagonal of FIM.

        Returns:
            np.ndarray: Empirical FIM matrix or diagonals.
        """
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

    
    
class NormalInverseGamma(RegressionDistn):
    """
    Normal-Inverse-Gamma distribution implementation for NGBoost.

    Used to model both aleatoric and epistemic uncertainty.

    Attributes:
        mu (np.ndarray): Mean of the Normal component.
        lam (np.ndarray): Precision of the Normal component.
        alpha (np.ndarray): Shape parameter of Inverse Gamma.
        beta (np.ndarray): Scale parameter of Inverse Gamma.
    """
    n_params = 4  # Four parameters: μ, λ, α, β.
    scores = [NIGLogScore]

    def __init__(self, params):
        """
        Initializes the NIG distribution from parameter array.

        Parameters:
            params (np.ndarray): Raw parameters [mu, raw_lam, raw_alpha, raw_beta].
        """
        self.mu    = params[0]
        self.lam   = np.exp(params[1])     # Avoid zero
        self.alpha = np.exp(params[2]) + 1      # Enforce α > 1
        self.beta  = np.exp(params[3])     # Avoid zero
        #print(f"Initialized NIG with params: mu={self.mu}, lam={self.lam}, alpha={self.alpha}, beta={self.beta}")
        print(f"Mean beta: {np.mean(self.beta)}, Min: {np.min(self.beta)}, Max: {np.max(self.beta)}")

    @staticmethod
    def fit(Y):
        """
        Estimates initial parameters for NIG from observed data Y.

        Parameters:
            Y (np.ndarray): Target values.

        Returns:
            np.ndarray: Initial parameter estimates.
        """
        m = np.mean(Y)
        s = np.std(Y)
        return np.array([m, 0.0, np.log(1.0), np.log(s**2)])  

    def sample(self, m):
        """
        Samples m values from the predictive NIG distribution.

        Parameters:
            m (int): Number of samples per prediction.

        Returns:
            np.ndarray: Sampled values from the NIG distribution.
        """
        shape = self.mu.shape  # assuming vectorized μ
        # Sample sigma^2 from Inverse Gamma(α, β).
        sigma2 = 1 / np.random.gamma(self.alpha, 1 / self.beta, size=(m, *shape))
        samples = np.random.normal(self.mu, np.sqrt(sigma2 / self.lam))
        return samples

    def pred_uncertainty(self):
        """
        Computes predictive uncertainty (aleatoric + epistemic).

        Returns:
            dict: Dictionary containing:
                - "mean": predictive mean (mu),
                - "aleatoric": β / (α - 1),
                - "epistemic": β² / (λ(α - 1)²(α - 2)).
        """
        aleatoric = self.beta / (self.alpha - 1)
        epistemic = self.beta**2 / (self.lam * (self.alpha - 1)**2 * (self.alpha - 2))
        return {"mean": self.mu, "aleatoric": aleatoric, "epistemic": epistemic}

    def pred_dist(self):
        """
        Returns the predictive parameters.

        Returns:
            tuple: (mu, lam, alpha, beta)
        """
        return self.mu, self.lam, self.alpha, self.beta
    
    
    def metric(self, Y):
        """
        Computes the Fisher Information Matrix for parameter estimates.

        Parameters:
            Y (np.ndarray): Target values.

        Returns:
            np.ndarray: FIM matrix.
        """
        params = [self.mu, self.lam, self.alpha, self.beta]
        return NIGLogScore().metric(Y, params=params)
    
    @property
    def is_regression(self):
        """
        Indicates that this is a regression model.

        Returns:
            bool: Always True.
        """
        return True

    @property
    def params(self):
        """
        Returns a dictionary of NIG parameters.

        Returns:
            dict: {mu, lam, alpha, beta}
        """
        return {"mu": self.mu, "lam": self.lam, "alpha": self.alpha, "beta": self.beta}

    def mean(self):
        """
        Returns the predictive mean.

        Returns:
            np.ndarray: mu
        """
        return self.mu
    
    def var(self):
        """
        Computes total predictive variance.

        Returns:
            np.ndarray: Predictive variance (aleatoric + epistemic).
        """
        # Aleatoric variance (σ²) = β / (α - 1)
        aleatoric = self.beta / (self.alpha - 1)
        epistemic = self.beta / (self.lam * (self.alpha - 1))
        predictive = aleatoric + epistemic
        return predictive
        #return aleatoric
    
    def predict_variance(self, X):
        """
        Returns the predicted total variance per test input.

        Parameters:
            X (array-like): Input data.

        Returns:
            np.ndarray: Predictive variances.
        """
        dists = self.predict_dist(X)
        # call .var() on each to get aleatoric+epistemic
        return np.array([dist.var() for dist in dists])

    def logpdf(self, Y):
        """
        Computes the log probability density function using the implied Student-t marginal.

        Parameters:
            Y (np.ndarray): Target values.

        Returns:
            np.ndarray: Log-pdf values.
        """
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