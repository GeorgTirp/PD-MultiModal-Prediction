
import numpy as np
import math
from numba import njit, prange, float64, vectorize


@njit
def digamma(x):
    result = 0.0
    while x < 6.0:
        result -= 1.0/x
        x += 1.0
    inv = 1.0/x
    inv2 = inv*inv
    result += math.log(x) - 0.5*inv - inv2*(1/12 - inv2*(1/120 - inv2*(1/252)))
    return result

@njit
def trigamma(x):
    result = 0.0
    while x < 6.0:
        result += 1.0/(x*x)
        x += 1.0
    inv = 1.0/x
    inv2 = inv*inv
    result += inv + 0.5*inv2 + inv2*inv2*(1/6 - inv2*(1/30 + inv2/42))
    return result


@njit(parallel=True, fastmath=True)
def d_score_numba(Y, mu, lam, alpha, beta,
                  evid_strength, kl_strength):
    n = Y.shape[0]
    grads = np.empty((n, 4), np.float64)
    eps = 1e-8
    mu0, lam0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0

    for i in prange(n):
        yi, mui = Y[i], mu[i]
        lami, alphai, betai = lam[i], alpha[i], beta[i]

        # ── clip & common terms ───────────────────────
        if lami   < eps:     lami   = eps
        if alphai < 1.0+eps: alphai = 1.0+eps
        if betai  < eps:     betai  = eps

        nu    = 2.0 * alphai
        two_b = 2.0 * betai
        Om    = two_b * (1.0 + lami)
        resid = yi - mui
        r2    = resid * resid
        term  = lami*r2 + Om
        inv_l = 1.0 / lami
        inv_O = 1.0 / Om

        # ── NLL gradients ─────────────────────────────
        d_mu    = lami*(nu+1.0)*(-resid)/term
        d_lam   = -0.5*inv_l \
                  - alphai*two_b*inv_O \
                  + (alphai+0.5)*r2/term
        d_alpha = -math.log(Om) + math.log(term) \
                  + digamma(alphai) - digamma(alphai+0.5)
        d_beta  = -alphai/betai \
                  + (alphai+0.5)*(two_b*(1.0+lami)/two_b)/term

        # ── evidential reg (per‐sample) ───────────────
        sgn       = 1.0 if resid >= 0.0 else -1.0
        ev_mu     = -sgn*(2.0*alphai + lami)
        ev_lam    = abs(resid)
        ev_alpha  = 2.0*abs(resid)

        # ── KL reg (per‐sample) ───────────────────────
        gk_mu    = alpha0*lami*(mui - mu0)/betai
        gk_lam   = -0.5*inv_l + alpha0*((r2/(2.0*betai)) + 1.0/lam0)
        gk_alpha = (alphai-alpha0)*trigamma(alphai) + (beta0/betai - 1.0)
        gk_beta  = alpha0*(1.0/betai - beta0/(betai*betai)) \
                   - alphai/(betai*betai) \
                   - alpha0*lami*r2/(2.0*betai*betai)

        # ── combine & regs ────────────────────────────
        d_mu    += evid_strength*ev_mu   + kl_strength*gk_mu
        d_lam   += evid_strength*ev_lam  + kl_strength*gk_lam
        d_alpha += evid_strength*ev_alpha+ kl_strength*gk_alpha
        d_beta  +=                  kl_strength*gk_beta

        # ── chain‐rule back to raw parameters ────────
        grads[i, 0] = d_mu
        grads[i, 1] = d_lam   * lami
        grads[i, 2] = d_alpha*(alphai-1.0)
        grads[i, 3] = d_beta  * betai

    return grads
