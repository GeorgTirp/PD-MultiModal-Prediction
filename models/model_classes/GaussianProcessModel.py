# Standard Libraries
import os
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Dict, Optional, Tuple, Union

# Data handling and numerical computation
import numpy as np
import pandas as pd

# JAX for accelerated Gaussian Processes
import jax
import jax.numpy as jnp
from jax import value_and_grad
import jax.scipy.linalg as jsp_linalg
import jax.scipy.special as jsp_special
import jax.random as jrandom

# Visualization / Explainability
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import (
    KFold, LeaveOneOut,
    GroupKFold, LeaveOneGroupOut,
)

# NEW: univariate scoring helpers
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

# Custom Base Model
from model_classes.BaseRegressionModel import BaseRegressionModel


def _bayes_cg(A: np.ndarray,
              b: np.ndarray,
              *,
              maxiter: Optional[int] = None,
              tol: float = 1e-8) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Bayesian CG with calibrated prior V0 = A^{-1}.
    Returns:
      x_n : CG mean iterate (solution estimate)
      info:
        iterations          : number of iterations
        residual_norm       : ||r_n||
        S                   : np.ndarray of shape (n, m) with CG search dirs p_i
        d                   : np.ndarray of shape (m,) with d_i = p_i^T A p_i
        qform_fn(v)         : callable computing v^T V_n v = Σ (p_i^T v)^2 / d_i
    """
    if maxiter is None:
        maxiter = A.shape[0]

    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()

    b_norm = float(np.linalg.norm(b)) + 1e-12
    rs_old = float(r @ r)

    S_cols = []
    d_vals = []

    iters = 0
    for iters in range(1, maxiter + 1):
        Ap = A @ p
        denom = float(p @ Ap)                # d_i = p^T A p  (A-orthogonality diag)
        if denom <= 0:
            warnings.warn("CG encountered non-positive curvature; aborting early.", RuntimeWarning)
            break

        # save calibrated ingredients
        S_cols.append(p.copy())
        d_vals.append(denom)

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) <= tol * b_norm:
            rs_old = rs_new
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    residual_norm = np.sqrt(rs_old)

    # stack S and d
    if len(S_cols) > 0:
        S = np.column_stack(S_cols)                       # (n, m)
        d = np.array(d_vals, dtype=b.dtype)               # (m,)
    else:
        S = np.zeros((A.shape[0], 0), dtype=b.dtype)
        d = np.zeros((0,), dtype=b.dtype)

    def qform(v: np.ndarray) -> float:
        """
        Compute v^T V_n v for V_n = S (S^T A S)^{-1} S^T.
        Because S^T A S is diagonal with entries d_i, this is:
            sum_i ( (p_i^T v)^2 / d_i ).
        """
        if d.size == 0:
            return 0.0
        t = S.T @ v                    # (m,)
        return float((t * t / d).sum())

    info = {
        "iterations": float(iters),
        "residual_norm": residual_norm,
        "S": S,
        "d": d,
        "qform_fn": qform,             # calibrated quadratic-form accessor
    }
    return x, info


jax.config.update("jax_enable_x64", True)


@dataclass
class _PackedParams:
    length_scales: jnp.ndarray
    rq_length: jnp.ndarray
    rq_alpha: jnp.ndarray
    lin_scale: jnp.ndarray
    lin_sigma0: jnp.ndarray
    outer_amp: jnp.ndarray
    noise_scale: jnp.ndarray


class JaxGaussianProcessRegressor:
    """Gaussian process regressor implemented with JAX for accelerated kernels.

    The implementation mirrors the additive kernel used previously:
        Constant * ( Matérn_ARD + RationalQuadratic + Constant * DotProduct ) + White
    """

    def __init__(
        self,
        nu: float = 1.5,
        normalize_y: bool = False,
        n_restarts_optimizer: int = 10,
        alpha: float = 1e-10,
        jitter: float = 1e-6,
        max_iters: int = 150,
        learning_rate: float = 0.05,
        random_state: int = 0,
        solver: str = "bayes_cg",
        bayes_cg_maxiter: Optional[int] = None,
        bayes_cg_tol: float = 1e-6,
        bayes_cg_prior_var: float = 1.0,
    ) -> None:
        self.nu = float(nu)
        self.normalize_y = bool(normalize_y)
        self.n_restarts_optimizer = int(max(n_restarts_optimizer, 0))
        self.base_alpha = float(alpha)
        self.jitter = float(jitter)
        self.max_iters = int(max_iters)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        solver = solver.lower()
        if solver not in {"cholesky", "bayes_cg"}:
            raise ValueError("solver must be 'cholesky' or 'bayes_cg'")
        self.solver = solver
        self.bayes_cg_maxiter = bayes_cg_maxiter
        self.bayes_cg_tol = bayes_cg_tol
        self.bayes_cg_prior_var = bayes_cg_prior_var

        self.alpha_vector: Optional[np.ndarray] = None
        self._is_fit: bool = False

        # placeholders filled during fit
        self.X_train_: Optional[jnp.ndarray] = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        self.params_: Optional[_PackedParams] = None
        self.theta_: Optional[jnp.ndarray] = None
        self.L_: Optional[jnp.ndarray] = None
        self.alpha_: Optional[jnp.ndarray] = None
        self.n_features_: Optional[int] = None
        self.K_train_: Optional[np.ndarray] = None
        self.bayes_cg_diagnostics_: Optional[Dict[str, float]] = None

        self._init_matern_fn()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_matern_fn(self) -> None:
        nu = float(self.nu)

        if np.isclose(nu, 0.5):
            def matern_from_sqdist(sqdist: jnp.ndarray) -> jnp.ndarray:
                r = jnp.sqrt(jnp.maximum(sqdist, 0.0) + 1e-12)
                return jnp.exp(-r)
        elif np.isclose(nu, 1.5):
            def matern_from_sqdist(sqdist: jnp.ndarray) -> jnp.ndarray:
                r = jnp.sqrt(jnp.maximum(sqdist, 0.0) + 1e-12)
                c = jnp.sqrt(3.0) * r
                return (1.0 + c) * jnp.exp(-c)
        elif np.isclose(nu, 2.5):
            def matern_from_sqdist(sqdist: jnp.ndarray) -> jnp.ndarray:
                r = jnp.sqrt(jnp.maximum(sqdist, 0.0) + 1e-12)
                c = jnp.sqrt(5.0) * r
                return (1.0 + c + (c**2) / 3.0) * jnp.exp(-c)
        else:
            nu_const = jnp.array(nu)

            def matern_from_sqdist(sqdist: jnp.ndarray) -> jnp.ndarray:
                r = jnp.sqrt(jnp.maximum(sqdist, 0.0) + 1e-12)
                scaled = jnp.sqrt(2.0 * nu_const) * r
                return jnp.where(
                    r < 1e-10,
                    jnp.ones_like(r),
                    (2.0 ** (1.0 - nu_const) / jsp_special.gamma(nu_const))
                    * (scaled ** nu_const)
                    * jsp_special.kv(nu_const, scaled)
                )

        self._matern_from_sqdist = jax.jit(matern_from_sqdist)

    @staticmethod
    def _pairwise_sqdist(X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        sum1 = jnp.sum(X1**2, axis=1, keepdims=True)
        sum2 = jnp.sum(X2**2, axis=1, keepdims=True)
        sqdist = sum1 + sum2.T - 2.0 * (X1 @ X2.T)
        return jnp.maximum(sqdist, 0.0)

    def _scaled_sqdist(self, X1: jnp.ndarray, X2: jnp.ndarray, length_scales: jnp.ndarray) -> jnp.ndarray:
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        return self._pairwise_sqdist(X1_scaled, X2_scaled)

    def _unpack_params(self, theta: jnp.ndarray) -> _PackedParams:
        p = self.n_features_
        idx = 0
        length_scales = jnp.exp(theta[idx:idx + p])
        idx += p
        rq_length = jnp.exp(theta[idx])
        idx += 1
        rq_alpha = jnp.exp(theta[idx]) + 1e-6
        idx += 1
        lin_scale = jnp.exp(theta[idx])
        idx += 1
        lin_sigma0 = jnp.exp(theta[idx])
        idx += 1
        outer_amp = jnp.exp(theta[idx])
        idx += 1
        noise_scale = jnp.exp(theta[idx])

        # Clamp to keep kernel well-behaved
        length_scales = jnp.clip(length_scales, 1e-3, 1e3)
        rq_length = jnp.clip(rq_length, 1e-3, 1e3)
        rq_alpha = jnp.clip(rq_alpha, 1e-6, 1e6)
        lin_scale = jnp.clip(lin_scale, 1e-6, 1e3)
        lin_sigma0 = jnp.clip(lin_sigma0, 1e-6, 1e3)
        outer_amp = jnp.clip(outer_amp, 1e-6, 1e6)
        noise_scale = jnp.clip(noise_scale, 1e-8, 1e3)

        return _PackedParams(
            length_scales=length_scales,
            rq_length=rq_length,
            rq_alpha=rq_alpha,
            lin_scale=lin_scale,
            lin_sigma0=lin_sigma0,
            outer_amp=outer_amp,
            noise_scale=noise_scale,
        )

    def _init_theta(self, n_features: int, key: Optional[jrandom.PRNGKey]) -> jnp.ndarray:
        base = [0.0] * n_features  # log(1.0) for length scales
        others = [0.0, 0.0, 0.0, 0.0, 0.0]  # rq_length, rq_alpha, lin_scale, lin_sigma0, outer_amp
        noise = [np.log(1e-3)]
        theta = jnp.array(base + others + noise, dtype=jnp.float64)
        if key is not None:
            perturb = 0.1 * jrandom.normal(key, shape=theta.shape, dtype=jnp.float64)
            theta = theta + perturb
        return theta

    # ------------------------------------------------------------------
    # Kernels & objectives
    # ------------------------------------------------------------------
    def _kernel(self, X1: jnp.ndarray, X2: jnp.ndarray, params: _PackedParams) -> jnp.ndarray:
        sqdist_ard = self._scaled_sqdist(X1, X2, params.length_scales)
        k_matern = self._matern_from_sqdist(sqdist_ard)

        # Rational Quadratic (isotropic)
        X1_rq = X1 / params.rq_length
        X2_rq = X2 / params.rq_length
        sqdist_rq = self._pairwise_sqdist(X1_rq, X2_rq)
        rq_term = (1.0 + 0.5 * sqdist_rq / params.rq_alpha) ** (-params.rq_alpha)

        # Linear term
        dot = X1 @ X2.T
        linear = params.lin_sigma0**2 + dot
        linear = params.lin_scale**2 * linear

        inner = k_matern + rq_term + linear
        return params.outer_amp**2 * inner

    def _kernel_diag(self, X: jnp.ndarray, params: _PackedParams) -> jnp.ndarray:
        # Diagonal of kernel without white noise
        diag_matern = jnp.ones(X.shape[0])  # Matérn diag is 1
        diag_rq = jnp.ones(X.shape[0])      # RQ diag is 1
        diag_linear = params.lin_scale**2 * (params.lin_sigma0**2 + jnp.sum(X * X, axis=1))
        return params.outer_amp**2 * (diag_matern + diag_rq + diag_linear)

    def _nll(self, theta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, noise_vector: Optional[jnp.ndarray]) -> jnp.ndarray:
        params = self._unpack_params(theta)
        K = self._kernel(X, X, params)
        n = X.shape[0]

        diag_noise = params.noise_scale**2 + self.base_alpha
        if noise_vector is not None:
            diag = noise_vector + diag_noise
        else:
            diag = jnp.full((n,), diag_noise)
        K = K + jnp.diag(diag + self.jitter)

        L = jnp.linalg.cholesky(K)
        alpha = jsp_linalg.solve_triangular(L, y, lower=True)
        alpha = jsp_linalg.solve_triangular(L.T, alpha, lower=False)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return 0.5 * y @ alpha + 0.5 * log_det + 0.5 * n * jnp.log(2.0 * jnp.pi)

    # ------------------------------------------------------------------
    # Fitting & prediction
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "JaxGaussianProcessRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        if self.normalize_y:
            self.y_mean_ = float(y.mean())
            std = float(y.std(ddof=0))
            self.y_scale_ = std if std > 0 else 1.0
            y_centered = (y - self.y_mean_) / self.y_scale_
        else:
            self.y_mean_ = 0.0
            self.y_scale_ = 1.0
            y_centered = y

        X_j = jnp.asarray(X)
        y_j = jnp.asarray(y_centered)
        noise_vec = None
        if self.alpha_vector is not None:
            alpha_arr = np.asarray(self.alpha_vector, dtype=np.float64)
            if alpha_arr.shape[0] != n_samples:
                warnings.warn(
                    "Ignoring heteroscedastic alpha: length does not match number of samples.",
                    RuntimeWarning,
                )
            else:
                noise_vec = jnp.asarray(alpha_arr)

        key = jrandom.PRNGKey(self.random_state)

        best_loss = np.inf
        best_theta = None

        loss_fn = lambda theta: self._nll(theta, X_j, y_j, noise_vec)
        loss_and_grad = jax.jit(value_and_grad(loss_fn))

        for restart in range(self.n_restarts_optimizer + 1):
            subkey = None if restart == 0 else jrandom.fold_in(key, restart)
            theta0 = self._init_theta(n_features, subkey)
            theta0_np = np.asarray(theta0)

            def obj(theta_flat: np.ndarray) -> Tuple[float, np.ndarray]:
                theta_j = jnp.asarray(theta_flat)
                loss_val, grad_val = loss_and_grad(theta_j)
                return float(loss_val), np.asarray(grad_val, dtype=np.float64)

            try:
                from scipy.optimize import minimize

                def fun(theta_flat: np.ndarray) -> float:
                    loss_val, _ = obj(theta_flat)
                    return loss_val

                def jac(theta_flat: np.ndarray) -> np.ndarray:
                    _, grad_val = obj(theta_flat)
                    return grad_val

                res = minimize(fun, theta0_np, method="L-BFGS-B", jac=jac)
                theta_opt = res.x
                loss_val = res.fun
            except Exception:
                # Fall back to simple gradient descent if SciPy is unavailable
                theta_opt = theta0
                m = jnp.zeros_like(theta_opt)
                v = jnp.zeros_like(theta_opt)
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                for step in range(self.max_iters):
                    loss_val, grad_val = obj(np.asarray(theta_opt))
                    grad_j = jnp.asarray(grad_val)
                    m = beta1 * m + (1.0 - beta1) * grad_j
                    v = beta2 * v + (1.0 - beta2) * (grad_j ** 2)
                    m_hat = m / (1.0 - beta1 ** (step + 1))
                    v_hat = v / (1.0 - beta2 ** (step + 1))
                    theta_opt = theta_opt - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
                    loss_val = float(loss_fn(theta_opt))

            if loss_val < best_loss:
                best_loss = loss_val
                best_theta = jnp.asarray(theta_opt)

        if best_theta is None:
            best_theta = self._init_theta(n_features, None)

        params = self._unpack_params(best_theta)
        K = self._kernel(X_j, X_j, params)
        diag_noise = params.noise_scale**2 + self.base_alpha
        if noise_vec is not None:
            diag = noise_vec + diag_noise
        else:
            diag = jnp.full((n_samples,), diag_noise)
        K = K + jnp.diag(diag + self.jitter)

        self.X_train_ = X_j
        self.params_ = params
        self.theta_ = best_theta
        self.K_train_ = np.asarray(K)
        self.bayes_cg_diagnostics_ = None

        if self.solver == "cholesky":
            L = jnp.linalg.cholesky(K)
            alpha = jsp_linalg.solve_triangular(L, y_j, lower=True)
            alpha = jsp_linalg.solve_triangular(L.T, alpha, lower=False)
            self.L_ = L
            self.alpha_ = alpha
        else:
            alpha_np, info = _bayes_cg(
            self.K_train_,
            np.asarray(y_j),
            maxiter=self.bayes_cg_maxiter,
            tol=self.bayes_cg_tol,
        )
        self.alpha_ = jnp.asarray(alpha_np)
        self.L_ = None
        self.bayes_cg_diagnostics_ = info

        self._is_fit = True

        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not self._is_fit:
            raise RuntimeError("Model must be fit before predicting.")

        X = np.asarray(X, dtype=np.float64)
        X_j = jnp.asarray(X)

        params = self.params_
        K_trans = self._kernel(self.X_train_, X_j, params)
        mean = K_trans.T @ self.alpha_

        if return_std:
            if self.solver == "cholesky":
                v = jsp_linalg.solve_triangular(self.L_, K_trans, lower=True)
                diag = self._kernel_diag(X_j, params) + (params.noise_scale**2 + self.base_alpha)
                var = diag - jnp.sum(v**2, axis=0)
                var = jnp.maximum(var, 1e-12)
                std = jnp.sqrt(var) * self.y_scale_
            else:
                diag = self._kernel_diag(X_j, params) + (params.noise_scale**2 + self.base_alpha)
                K_trans_np = np.asarray(K_trans)
            
                var_vals = []
                for col in range(K_trans_np.shape[1]):
                    rhs = K_trans_np[:, col]
            
                    # BayesCG solve for A x ≈ rhs, and get calibrated V_n ingredients
                    sol, info = _bayes_cg(
                        self.K_train_,
                        rhs,
                        maxiter=self.bayes_cg_maxiter,
                        tol=self.bayes_cg_tol,
                    )
            
                    # Calibrated correction: rhs^T V_n rhs = qform(rhs)
                    q_corr = info["qform_fn"](rhs)
            
                    # GP var with solver uncertainty: k(x*,x*) - rhs^T A^{-1} rhs
                    # Approximate A^{-1} term with CG mean and add BayesCG calibration:
                    #   rhs^T A^{-1} rhs  ≈  rhs^T sol  -  rhs^T V_n rhs
                    # so: diag - rhs^T A^{-1} rhs  ≈  diag - rhs^T sol + q_corr
                    var_i = float(diag[col] - rhs @ sol + q_corr)
                    var_vals.append(max(var_i, 1e-12))
            
                std = np.sqrt(np.asarray(var_vals)) * self.y_scale_
        mean = mean * self.y_scale_ + self.y_mean_

        mean_np = np.asarray(mean)
        if return_std:
            return mean_np, np.asarray(std)
        return mean_np

    def negative_log_predictive_density(self, X: np.ndarray, y: np.ndarray) -> float:
        mu, std = self.predict(X, return_std=True)
        var = np.maximum(std**2, 1e-12)
        ll = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((y - mu) ** 2) / var
        return float(-np.mean(ll))


class GaussianProcessModel(BaseRegressionModel):
    """
    Gaussian Process Regression with an additive kernel implemented in JAX:
        Constant * ( Matern(ARD) + RationalQuadratic + Linear ) + White

    Tunable (via `hparams` / `hparam_grid`):
        - nu (float): Matern smoothness, typical {0.5, 1.5, 2.5}
        - n_restarts_optimizer (int)
        - normalize_y (bool)
        - Optional: heteroscedastic noise via self.set_heteroscedastic_alpha(alpha_vec)
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        feature_selection: dict,
        target_name: str,
        test_split_size: float = 0.2,
        save_path: Optional[str] = None,
        top_n: int = -1,
        hparams: Optional[dict] = None,
        hparam_grid: Optional[dict] = None,
        standardize: str = "zscore",
        split_shaps: Optional[bool] = None,
        logging=None,
    ) -> None:
        super().__init__(
            data_df,
            feature_selection,
            target_name,
            test_split_size,
            save_path,
            top_n,
            standardize=standardize,
            split_shaps=split_shaps,
            logging=logging,
        )

        default_hparams = {
            "nu": 1.5,
            "normalize_y": False,
            "n_restarts_optimizer": 10,
            "solver": "bayes_cg",
            "bayes_cg_maxiter": None,
            "bayes_cg_tol": 1e-6,
            "bayes_cg_prior_var": 1.0,
        }
        self.gp_hparams = default_hparams.copy()
        if hparams:
            self.gp_hparams.update(hparams)

        self.param_grid = hparam_grid if hparam_grid is not None else {}
        self._hetero_alpha: Optional[np.ndarray] = None
        self.solve_mode = self.gp_hparams.get("solver", "bayes_cg")

        p = len(self.feature_selection["features"])
        self.kernel_info = {"input_dim": p, "nu": self.gp_hparams["nu"]}

        self.model = self._build_model(self.gp_hparams)
        self.model_name = "Gaussian Process (JAX additive kernel)"
        if top_n == -1:
            self.top_n = len(self.feature_selection["features"])

        self.feature_scores: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_model(self, params: dict, solver: Optional[str] = None) -> JaxGaussianProcessRegressor:
        solver_choice = solver if solver is not None else params.get("solver", "bayes_cg")
        model = JaxGaussianProcessRegressor(
            nu=params.get("nu", 1.5),
            normalize_y=params.get("normalize_y", False),
            n_restarts_optimizer=params.get("n_restarts_optimizer", 10),
            alpha=1e-10,
            random_state=getattr(self, "random_state", 0),
            solver=solver_choice,
            bayes_cg_maxiter=params.get("bayes_cg_maxiter"),
            bayes_cg_tol=params.get("bayes_cg_tol", 1e-6),
            bayes_cg_prior_var=params.get("bayes_cg_prior_var", 1.0),
        )
        if self._hetero_alpha is not None:
            model.alpha_vector = self._hetero_alpha
        return model

    @staticmethod
    def _nlpd_scorer(est: JaxGaussianProcessRegressor, X: pd.DataFrame, y: pd.Series) -> float:
        """Negative Log Predictive Density (to minimize)."""
        mu, std = est.predict(X, return_std=True)
        var = np.maximum(std**2, 1e-12)
        ll = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((y - mu) ** 2) / var
        return -float(np.mean(ll))

    def set_heteroscedastic_alpha(self, alpha_vec: np.ndarray) -> None:
        self._hetero_alpha = np.asarray(alpha_vec, dtype=float)
        if isinstance(self.model, JaxGaussianProcessRegressor):
            self.model.alpha_vector = self._hetero_alpha

    # ------------------------------------------------------------------
    # Preprocessing (unchanged)
    # ------------------------------------------------------------------
    def model_specific_preprocess(self, data_df: pd.DataFrame):
        self.logging.info("Starting model-specific preprocessing...")

        data_df = data_df.dropna(subset=self.feature_selection['features'] + [self.feature_selection['target']])

        X = data_df[self.feature_selection['features']].copy()
        bad_cols = [c for c in X.columns if X[c].apply(lambda v: isinstance(v, str)).any()]
        if bad_cols:
            self.logging.info(f"Dropping non-numeric columns: {bad_cols}")
            X = X.drop(columns=bad_cols)

        y = data_df[self.feature_selection['target']]

        X = X.apply(pd.to_numeric, errors='coerce')
        missing_rate = X.isna().mean()

        X = X.fillna(X.mean())
        X = X.dropna(axis=0, how='any')
        y = y.loc[X.index]

        m = y.mean()
        std = y.std(ddof=0) if y.std(ddof=0) != 0 else 1.0
        z = (y - m) / std

        try:
            pearson_r, pearson_p = [], []
            spearman_r, spearman_p = [], []
            for col in X.columns:
                xi = X[col].values
                if np.all(xi == xi[0]):
                    pr, pp, sr, sp = 0.0, 1.0, 0.0, 1.0
                else:
                    pr, pp = pearsonr(xi, y.values)
                    sr, sp = spearmanr(xi, y.values)
                pearson_r.append(pr)
                pearson_p.append(pp)
                spearman_r.append(sr)
                spearman_p.append(sp)

            nunique = X.nunique(dropna=False)
            discrete_mask = nunique.le(10).values
            mi = mutual_info_regression(
                X.values, y.values,
                discrete_features=discrete_mask,
                random_state=42
            )

            variance = X.var(ddof=0)

            scores_df = pd.DataFrame({
                "feature": X.columns,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_r,
                "spearman_p": spearman_p,
                "mutual_info": mi,
                "variance": variance.values,
                "missing_rate": missing_rate.reindex(X.columns).fillna(0).values,
                "nunique": nunique.reindex(X.columns).fillna(0).astype(int).values,
                "is_discrete_for_mi": discrete_mask.astype(bool)
            }).sort_values(["mutual_info", "pearson_r"], ascending=[False, False]).reset_index(drop=True)

            self.feature_scores = scores_df
            os.makedirs(self.save_path, exist_ok=True)
            scores_df.to_csv(f"{self.save_path}/{self.target_name}_feature_univariate_scores.csv", index=False)

            self.logging.info("Computed univariate feature scores (MI, Pearson, Spearman).")
        except Exception as e:
            self.logging.warning(f"Feature scoring failed: {e}")

        self.logging.info("Finished model-specific preprocessing.")
        return X, y, z, m, std

    # ------------------------------------------------------------------
    # Importance & SHAP (unchanged except title)
    # ------------------------------------------------------------------
    def feature_importance(self, X, top_n: int = None, save_results: bool = True,
                           iter_idx: int = None, ablation_idx: int = None):
        if iter_idx is None and hasattr(self, 'logging') and self.logging:
            self.logging.info("Starting SHAP feature attribution for GP...")

        shap.initjs()
        bg = X.sample(min(50, len(X)), random_state=42)

        try:
            explainer = shap.KernelExplainer(self.model.predict, bg)
            shap_values = explainer.shap_values(X, nsamples=min(200, 50 * X.shape[1]))
        except Exception as e:
            if hasattr(self, 'logging') and self.logging:
                self.logging.warning(f"SHAP KernelExplainer failed ({e}); using zeros).")
            shap_values = np.zeros_like(X.values)

        mean_abs = np.abs(shap_values).mean(axis=0)
        denom = mean_abs.sum() if mean_abs.sum() != 0 else 1.0
        attribution = mean_abs / denom
        feature_names = list(X.columns)
        k = self.top_n if top_n is None else top_n
        idx = np.argsort(attribution)[-k:][::-1]
        self.importances = {feature_names[i]: float(attribution[i]) for i in idx}

        if save_results:
            os.makedirs(self.save_path, exist_ok=True)
            np.save(f'{self.save_path}/{self.target_name}_feature_importance.npy', attribution)

        shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
        plt.title('SHAP Summary Plot (GP additive kernel)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                sp = os.path.join(self.save_path, "singleSHAPs")
                os.makedirs(sp, exist_ok=True)
                plt.savefig(f'{sp}/{self.target_name}_shap_aggregated_beeswarm_{iter_idx}.png', dpi=150, bbox_inches='tight')
            elif ablation_idx is not None:
                sp = os.path.join(self.save_path, "ablationSHAPs")
                os.makedirs(sp, exist_ok=True)
                plt.savefig(f'{sp}/{self.target_name}_shap_aggregated_beeswarm_ablation_{ablation_idx}.png', dpi=150, bbox_inches='tight')
            else:
                plt.savefig(f'{self.save_path}/{self.target_name}_shap_aggregated_beeswarm.png', dpi=150, bbox_inches='tight')
            plt.close()

        if iter_idx is None and hasattr(self, 'logging') and self.logging:
            self.logging.info("Finished SHAP feature attribution for GP.")
        return shap_values

    # ------------------------------------------------------------------
    # Hyperparameter Tuning
    # ------------------------------------------------------------------
    def _cv_splitter(self, folds: int, groups: Optional[np.ndarray]):
        if folds == -1:
            return (LeaveOneOut() if groups is None else LeaveOneGroupOut())
        if groups is None:
            rs = getattr(self, 'random_state', 42)
            return KFold(n_splits=folds, shuffle=True, random_state=rs)
        return GroupKFold(n_splits=folds)

    def _evaluate_params(self, X, y, folds: int, groups: Optional[np.ndarray], params: dict) -> float:
        splitter = self._cv_splitter(folds, groups)
        X_values = X.values if hasattr(X, "values") else np.asarray(X)
        y_values = y.values if hasattr(y, "values") else np.asarray(y)
        if y_values.ndim > 1:
            y_values = np.asarray(y_values).ravel()

        alpha_vec = self._hetero_alpha
        if alpha_vec is not None and len(alpha_vec) != X_values.shape[0]:
            if hasattr(self, "logging") and self.logging:
                self.logging.warning(
                    "Heteroscedastic alpha length mismatch during tuning; ignoring provided vector."
                )
            alpha_vec = None

        groups_array = None
        if groups is not None:
            groups_array = groups.values if hasattr(groups, "values") else np.asarray(groups)

        split_args = (X_values,)
        split_kwargs = {}
        if groups_array is not None:
            split_kwargs["groups"] = groups_array

        nlpd_scores = []
        for tr_idx, val_idx in splitter.split(*split_args, **split_kwargs):
            X_train, X_val = X_values[tr_idx], X_values[val_idx]
            y_train, y_val = y_values[tr_idx], y_values[val_idx]
            model = self._build_model(params, solver="cholesky")
            if alpha_vec is not None:
                model.alpha_vector = alpha_vec[tr_idx]
            model.fit(X_train, y_train)
            score = model.negative_log_predictive_density(X_val, y_val)
            nlpd_scores.append(score)
        return float(np.mean(nlpd_scores))

    def tune_hparams(
        self,
        X,
        y,
        param_grid: Optional[dict] = None,
        folds: int = 5,
        groups: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Dict:
        if weights is not None and hasattr(self, 'logging') and self.logging:
            self.logging.warning("Sample weights are ignored by the JAX GP implementation.")

        egrid = param_grid if param_grid is not None else (self.param_grid or {})
        X_values = X.values if hasattr(X, "values") else np.asarray(X)
        y_values = y.values if hasattr(y, "values") else np.asarray(y)
        if y_values.ndim > 1:
            y_values = np.asarray(y_values).ravel()
        groups_array = None
        if groups is not None:
            groups_array = groups.values if hasattr(groups, "values") else np.asarray(groups)

        nu_vals = egrid.get('nu', [self.gp_hparams.get('nu', 1.5)])
        norm_vals = egrid.get('normalize_y', [self.gp_hparams.get('normalize_y', False)])
        restart_vals = egrid.get('n_restarts_optimizer', [self.gp_hparams.get('n_restarts_optimizer', 10)])

        best_score = np.inf
        best_params = None

        for nu, norm_y, n_restart in product(nu_vals, norm_vals, restart_vals):
            params = {
                "nu": nu,
                "normalize_y": norm_y,
                "n_restarts_optimizer": n_restart,
            }
            score = self._evaluate_params(X_values, y_values, folds, groups_array, params)
            if hasattr(self, 'logging') and self.logging:
                self.logging.info(f"Tuning params {params} | NLPD={score:.5f}")
            if score < best_score:
                best_score = score
                best_params = params

        if best_params is None:
            best_params = self.gp_hparams.copy()
        else:
            self.gp_hparams.update(best_params)

        self.solve_mode = self.gp_hparams.get("solver", self.solve_mode)

        self.model = self._build_model(self.gp_hparams)
        self.model.fit(X_values, y_values)
        if hasattr(self, 'logging') and self.logging:
            self.logging.info(f"Best parameters found: {self.gp_hparams}")
            self.logging.info(f"Best CV score (NLPD): {best_score:.6f}")

        return self.gp_hparams
