import numpy as np
import scipy.stats as st
from scipy.special import gammaln, psi, polygamma
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

# Activation helpers

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def positive(x, eps=1e-3):
    return np.log1p(np.exp(x)) + eps

# --- Custom Score: Normal-Inverse-Gamma (no params argument) ---
class NIGLogScore(LogScore):
    """
    Score for Normal-Inverse-Gamma predictive distribution.
    Computes NLL + evidential & KL regularizers.
    """
    def kl_divergence_nig(self, mu, lam, alpha, beta,
                          mu0=0.0, lam0=1.0, alpha0=2.0, beta0=1.0,
                          eps=1e-8):
        lam = np.clip(lam, eps, 1e6)
        alpha = np.clip(alpha, 1.0+eps, 1e6)
        beta = np.clip(beta, eps, 1e6)

        term1 = 0.5 * np.log(lam0/lam)
        term2 = alpha0 * np.log(beta/beta0)
        term3 = -gammaln(alpha) + gammaln(alpha0)
        term4 = (alpha - alpha0) * psi(alpha)
        term5 = alpha0 * (lam*(self.mu-mu0)**2 / (2*beta))
        term6 = alpha0 * (lam/lam0 - 1)
        term7 = alpha * (beta0/beta - 1)
        return np.mean(term1+term2+term3+term4+term5+term6+term7)

    def evidential_regularizer(self, Y, mu, lam, alpha):
        err = np.abs(Y - mu)
        return np.mean(err * (2*alpha + lam))

    def score(self, Y, evid_strength=0.1, kl_strength=0.05):
        """
        Elementwise NLL + evidential & KL penalties.
        """
        mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        Omega = 2*beta*(1+lam)
        t1 = 0.5*(np.log(np.pi) - np.log(lam))
        t2 = -alpha * np.log(Omega)
        t3 = (alpha+0.5) * np.log(lam*(Y-mu)**2 + Omega)
        t4 = gammaln(alpha) - gammaln(alpha+0.5)
        nll = t1 + t2 + t3 + t4

        evid = self.evidential_regularizer(Y, mu, lam, alpha)
        kl = self.kl_divergence_nig(mu, lam, alpha, beta)
        return nll + evid_strength*evid + kl_strength*kl

    def d_score(self, Y, evid_strength=0.1, kl_strength=0.05):
        """
        Gradients of NLL + evidential & KL penalties.
        """
        mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        eps = 1e-8
        lam = np.clip(lam, eps, None)
        alpha = np.clip(alpha, 1.0+eps, None)
        beta = np.clip(beta, eps, None)
        nu = 2*alpha
        Omega = 2*beta*(1+lam)

        def comps(r):
            t = lam*r*r + Omega
            g_mu = lam*(nu+1)*(-r)/t
            g_lam = -0.5/lam - alpha*(2*beta)/Omega + (alpha+0.5)*(r*r)/t
            g_alpha = -np.log(Omega) + np.log(t) + psi(alpha) - psi(alpha+0.5)
            g_beta = -alpha/beta + (alpha+0.5)*(2*(1+lam))/t
            return g_mu, g_lam, g_alpha, g_beta

        resid = Y - mu
        d_mu, d_lam, d_alpha, d_beta = comps(resid)

        # regularizers
        sign = np.sign(Y - mu)
        gev_mu = np.mean(-sign*(2*alpha+lam))
        gev_lam = np.mean(np.abs(Y-mu))
        gev_alpha = np.mean(2*np.abs(Y-mu))
        gev_beta = 0.0

        mu0, lam0, alpha0, beta0 = 0,1,2,1
        gk_mu = alpha0*lam*(mu-mu0)/beta
        gk_lam = -0.5/lam + alpha0*((mu-mu0)**2/(2*beta) + 1/lam0)
        gk_alpha = (alpha-alpha0)*polygamma(1,alpha) + (beta0/beta-1)
        gk_beta = alpha0*(1/beta - beta0/beta**2) - alpha/beta**2 - alpha0*lam*(mu-mu0)**2/(2*beta**2)

        d_mu    += evid_strength*gev_mu    + kl_strength*gk_mu
        d_lam   += evid_strength*gev_lam   + kl_strength*gk_lam
        d_alpha += evid_strength*gev_alpha + kl_strength*gk_alpha
        d_beta  += evid_strength*gev_beta  + kl_strength*gk_beta

        # raw-space
        raw = [d_mu, d_lam*lam, d_alpha*(alpha-1), d_beta*beta]
        return np.stack(raw, axis=1)

# --- Custom Score with Tobit censoring ---
class NIGTobitScore(NIGLogScore):
    """
    Extends NIGLogScore by adding per-sample Tobit censoring bounds.
    """
    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = np.asarray(lower_bound, float)
        self.upper_bound = np.asarray(upper_bound, float)

    def score(self, Y, evid_strength=0.1, kl_strength=0.05):
        base = super().score(Y, evid_strength, kl_strength)
        mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        Omega = 2*beta*(1+lam)
        nu = 2*alpha
        scale = np.sqrt(Omega/(lam*nu))
        td = st.t(df=nu, loc=mu, scale=scale)

        low_m = (Y <= self.lower_bound)
        if low_m.any():
            cdf_l = np.clip(td.cdf(self.lower_bound),1e-8,1)
            base[low_m] = -np.log(cdf_l[low_m])
        up_m = (Y >= self.upper_bound)
        if up_m.any():
            sf_u = np.clip(td.sf(self.upper_bound),1e-8,1)
            base[up_m] = -np.log(sf_u[up_m])
        return base + 0  # regs already included

    def d_score(self, Y, evid_strength=0.1, kl_strength=0.05):
        d = super().d_score(Y, evid_strength, kl_strength)
        mu, lam, alpha, beta = self.mu, self.lam, self.alpha, self.beta
        eps = 1e-8
        nu = 2*alpha
        Omega = 2*beta*(1+lam)
        scale = np.sqrt(Omega/(lam*nu))
        td = st.t(df=nu, loc=mu, scale=scale)

        low_m = (Y <= self.lower_bound)
        if low_m.any():
            pdf_l = np.clip(td.pdf(self.lower_bound),eps,None)
            cdf_l = np.clip(td.cdf(self.lower_bound),eps,1)
            factor = pdf_l/cdf_l
            g = super(NIGLogScore,self).d_score(self.lower_bound, evid_strength, kl_strength)  # grads at bound
            d[low_m] = g[low_m]*factor[low_m]
        up_m = (Y >= self.upper_bound)
        if up_m.any():
            pdf_u = np.clip(td.pdf(self.upper_bound),eps,None)
            sf_u  = np.clip(td.sf(self.upper_bound),eps,1)
            factor = pdf_u/sf_u
            g = super(NIGLogScore,self).d_score(self.upper_bound, evid_strength, kl_strength)
            d[up_m] = g[up_m]*factor[up_m]
        return d

# --- Distribution ---
class NormalInverseGamma(RegressionDistn):
    n_params = 4
    scores = [NIGLogScore, NIGTobitScore]

    def __init__(self, params):
        super().__init__(params)
        self.mu    = params[0]
        self.lam   = np.exp(params[1])
        self.alpha = np.exp(params[2]) + 1
        self.beta  = np.exp(params[3])

    @staticmethod
    def fit(Y):
        m, s = np.mean(Y), np.std(Y)
        return np.array([m, 0.0, np.log(1.0), np.log(s**2)])

    def sample(self, m):
        shape = self.mu.shape
        sigma2 = 1/np.random.gamma(self.alpha,1/self.beta,size=(m,*shape))
        return np.random.normal(self.mu, np.sqrt(sigma2/self.lam))

    def pred_dist(self):
        return {"mean":self.mu,
                "aleatoric":self.beta/(self.alpha-1),
                "epistemic":self.beta**2/(self.lam*(self.alpha-1)**2*(self.alpha-2))}

    @property
    def is_regression(self): return True
    @property
    def params(self): return {"mu":self.mu,"lam":self.lam,"alpha":self.alpha,"beta":self.beta}

    def logpdf(self, Y):
        nu = 2*self.alpha
        var = self.beta/(self.lam*self.alpha)
        coeff = gammaln((nu+1)/2)-gammaln(nu/2)
        norm  = -0.5*np.log(nu*np.pi*var)
        sq    = (Y-self.mu)**2/(nu*var)
        return coeff+norm -0.5*(nu+1)*np.log1p(sq)
