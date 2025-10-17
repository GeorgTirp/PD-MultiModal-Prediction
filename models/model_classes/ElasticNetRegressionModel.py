# Standard Libraries
import os
from typing import Dict, Union, List, Tuple

# Data handling and numerical computation
import numpy as np
import pandas as pd

# Machine Learning and Modeling
from sklearn.linear_model import ElasticNet  # <-- L1 + L2

# Visualization
import matplotlib.pyplot as plt
import shap
from utils.my_logging import Logging
from sklearn.model_selection import (
    KFold, LeaveOneOut,
    GroupKFold, LeaveOneGroupOut
)
from sklearn.model_selection import GridSearchCV


# Custom Base Model
from model_classes.BaseRegressionModel import BaseRegressionModel


class ElasticNetRegressionModel(BaseRegressionModel):
    """
    Elastic Net Regression (combined L1/L2 regularization).

    Args:
        data_df (pd.DataFrame): Input dataset with features and target.
        feature_selection (dict): {'features': [...], 'target': '...'}.
        target_name (str): Name of the target column.
        test_split_size (float, optional): Test split fraction. Defaults to 0.2.
        save_path (str, optional): Directory for saving outputs. Defaults to None.
        top_n (int, optional): Number of top features to consider, -1 for all. Defaults to 10.
        alpha (float, optional): Overall regularization strength. Defaults to 1.0.
        l1_ratio (float, optional): Mix between L1 and L2 (0→Ridge, 1→Lasso). Defaults to 0.5.
        fit_intercept (bool, optional): Fit intercept. Defaults to True.
        max_iter (int, optional): Max solver iterations. Defaults to 10000.
        logging: Logger instance.
    """
    def __init__(
            self,
            data_df: pd.DataFrame,
            feature_selection: dict,
            target_name: str,
            test_split_size: float = 0.2,
            save_path: str = None,
            top_n: int = -1,
            hparams: dict = None,          
            hparam_grid: dict = None,
            standardize_features: str = "zscore",
            split_shaps: bool = None,
            logging=None):

        super().__init__(
            data_df,
            feature_selection,
            target_name,
            test_split_size,
            save_path,
            top_n,
            standardize=standardize_features,
            split_shaps=split_shaps,
            logging=logging)

        default_hparams = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'fit_intercept': True,
            'max_iter': 10000,
        }
        self.enet_hparams = default_hparams.copy()
        if hparams:
            self.enet_hparams.update(hparams)
        self.param_grid = hparam_grid if hparam_grid is not None else {}
        # Elastic Net model (L1 + L2)
        self.model = ElasticNet(
            alpha=self.enet_hparams['alpha'],
            l1_ratio=self.enet_hparams['l1_ratio'],
            fit_intercept=self.enet_hparams['fit_intercept'],
            max_iter=self.enet_hparams['max_iter'],
            # random_state only used if selection='random', harmless otherwise
            random_state=self.random_state,
        )

        self.model_name = "Elastic Net Regression"
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])

    def model_specific_preprocess(self, data_df: pd.DataFrame):
        """
        Preprocess data for Elastic Net.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series, float, float]:
            X, y, z (standardized target), mean(y), std(y)
        """
        self.logging.info("Starting model-specific preprocessing...")

        # Drop rows with missing in selected features/target
        data_df = data_df.dropna(subset=self.feature_selection['features'] + [self.feature_selection['target']])

        # Features: remove columns that contain strings anywhere
        X = data_df[self.feature_selection['features']]
        bad_cols = [c for c in X.columns if X[c].apply(lambda v: isinstance(v, str)).any()]
        X = X.drop(columns=bad_cols)

        # Target
        y = data_df[self.feature_selection['target']]

        # Coerce numerics, simple impute remaining NaNs with column means
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.mean())
        X = X.dropna(axis=0, how='any')  # drop any rows still containing NaNs after coercion

        # Align y with cleaned X index
        y = y.loc[X.index]

        # Standardize target (if your BaseRegressionModel uses it downstream)
        m = y.mean()
        std = y.std(ddof=0) if y.std(ddof=0) != 0 else 1.0
        z = (y - m) / std

        self.logging.info("Finished model-specific preprocessing.")
        return X, y, z, m, std

    def feature_importance(self, X, top_n: int = None, save_results: bool = True, iter_idx: int = None, ablation_idx: int = None) -> Dict:
        """
        Computes feature importance using absolute normalized coefficients and SHAP values.
        """
        if iter_idx is None:
            self.logging.info("Starting feature importance evaluation for Elastic Net...")

        # Coefficients (Elastic Net encourages sparsity; zeros mean pruned features)
        coefs = np.asarray(self.model.coef_)
        abs_sum = np.sum(np.abs(coefs)) if np.sum(np.abs(coefs)) != 0 else 1.0
        attribution = np.abs(coefs) / abs_sum

        feature_names = list(X.columns)
        k = self.top_n if top_n is None else top_n
        indices = np.argsort(attribution)[-k:][::-1]
        top_features = {feature_names[i]: float(attribution[i]) for i in indices}
        self.importances = top_features

        if save_results:
            os.makedirs(self.save_path, exist_ok=True)
            np.save(f'{self.save_path}/{self.target_name}_feature_importance.npy', attribution)

        # SHAP for linear models
        shap.initjs()
        background_data = X.sample(min(25, len(X)), random_state=42)
        explainer = shap.LinearExplainer(self.model, background_data)
        shap_values = explainer.shap_values(X)

        # Beeswarm summary
        shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
        plt.title('SHAP Summary Plot (Aggregated)', fontsize=16)

        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                save_path = os.path.join(self.save_path, "singleSHAPs")
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.target_name}_shap_aggregated_beeswarm_{iter_idx}.png', dpi=150, bbox_inches='tight')
            elif ablation_idx is not None:
                save_path = os.path.join(self.save_path, "ablationSHAPs")
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.target_name}_shap_aggregated_beeswarm_ablation_{ablation_idx}.png', dpi=150, bbox_inches='tight')
            else:
                plt.savefig(f'{self.save_path}/{self.target_name}_shap_aggregated_beeswarm.png', dpi=150, bbox_inches='tight')
            plt.close()

        if iter_idx is None:
            self.logging.info("Finished feature importance evaluation for Elastic Net.")
        return shap_values

    def tune_hparams(self,
                     X,
                     y,
                     param_grid: dict = None,
                     folds: int = 5,
                     groups: np.ndarray = None,
                     weights: np.ndarray = None
                    ) -> Dict:
        """
        Tune Elastic Net hyperparameters via GridSearchCV.

        - Uses self.hparam_grid if param_grid is None.
        - folds == -1 → Leave-One-Out (or Leave-One-Group-Out if groups provided).
        - If groups provided, uses group-aware CV.
        - Passes sample_weight to fit() when provided.
        """
        # choose grid: passed-in overrides class-level
        grid = param_grid if param_grid is not None else (getattr(self, "hparam_grid", None) or {})
        rs = getattr(self, 'random_state', 42)
        # CV splitter
        if folds == -1:
            cv = LeaveOneOut() if groups is None else LeaveOneGroupOut()
        else:
            if groups is None:
                cv = KFold(n_splits=folds, shuffle=True, random_state=rs)
            else:
                # GroupKFold has no shuffle/random_state
                cv = GroupKFold(n_splits=folds)

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0,
        )

        # assemble fit kwargs
        fit_kwargs = {}
        if groups is not None:
            fit_kwargs['groups'] = groups
        if weights is not None:
            fit_kwargs['sample_weight'] = weights

        # run the search
        grid_search.fit(X, y, **fit_kwargs)

        best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        # keep a record and sync model params
        if not hasattr(self, 'enet_hparams'):
            self.enet_hparams = {}
        self.enet_hparams.update(best_params)
        self.model.set_params(**best_params)

        if hasattr(self, 'logging') and self.logging:
            self.logging.info(f"Best parameters found: {best_params}")
            self.logging.info(f"Best CV score (neg MSE): {grid_search.best_score_:.6f}")

        return best_params

