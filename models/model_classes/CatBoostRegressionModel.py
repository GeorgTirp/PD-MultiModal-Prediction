# Standard Libraries
import os
from typing import Dict, Optional, Tuple

# Data Handling and Numeric Computation
import numpy as np
import pandas as pd

# Machine Learning and Modeling
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    KFold, LeaveOneOut,
    GroupKFold, LeaveOneGroupOut
)
from sklearn.metrics import mean_squared_error

# Visualization and Explainability
import matplotlib.pyplot as plt
import shap

# Custom Base Model
from model_classes.BaseRegressionModel import BaseRegressionModel


class CatBoostRegressionModel(BaseRegressionModel):
    """CatBoost regression model supporting grid search tuning and SHAP explainability."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        feature_selection: dict,
        target_name: str,
        cat_hparams: Optional[dict] = None,
        test_split_size: float = 0.2,
        save_path: str = None,
        top_n: int = -1,
        param_grid: Optional[dict] = None,
        logging=None,
        Pat_IDs=None,
        split_shaps: bool = False,
        sample_weights: Optional[np.ndarray] = None,
        random_state: int = 420,
    ) -> None:
        super().__init__(
            data_df,
            feature_selection,
            target_name,
            test_split_size,
            save_path,
            top_n,
            logging=logging,
            Pat_IDs=Pat_IDs,
            split_shaps=split_shaps,
            random_state=random_state,
        )

        default_hparams = {
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 6,
            "loss_function": "RMSE",
            "random_seed": random_state,
            "verbose": 0,
            "allow_writing_files": False,
        }
        if cat_hparams is None:
            cat_hparams = {}
        self.cat_hparams = {**default_hparams, **cat_hparams}

        self.model = CatBoostRegressor(**self.cat_hparams)
        self.model_name = "CatBoost Regression"
        self.param_grid = param_grid
        self.weights = sample_weights
        self.random_state = random_state
        if top_n == -1:
            self.top_n = len(self.feature_selection["features"])

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def model_specific_preprocess(self, data_df: pd.DataFrame) -> Tuple:
        """
        Preprocess for CatBoost:
        - Keep categorical columns (incl. 'SEX') as category dtype (no manual encoding).
        - Impute numeric columns only.
        - Store cat feature indices in self.cat_features for CatBoost.fit(..., cat_features=...).
        """
        self.logging.info("Starting CatBoost preprocessing...")

        # Only require target to be present; allow feature NaNs (CatBoost can handle them)
        target_col = self.feature_selection["target"]
        feature_cols = self.feature_selection["features"]

        # drop rows where target is missing
        data_df = data_df.dropna(subset=[target_col])

        X = data_df[feature_cols].copy()
        y = data_df[target_col]

        # Identify categorical columns: object, category, or boolean (CatBoost handles bool as categorical fine)
        cat_cols = (
            list(X.select_dtypes(include=["object", "category", "bool"]).columns)
        )

        # Ensure 'SEX' (if present) is treated as categorical
        if "SEX" in X.columns and "SEX" not in cat_cols:
            cat_cols.append("SEX")

        # Cast categorical columns to pandas category (stable + efficient)
        for col in cat_cols:
            X[col] = X[col].astype("category")

            # Optional: unify missing label for categorical (CatBoost can handle NaN, but a distinct 'Unknown' can be clearer)
            # Comment out the next line if you prefer to keep NaNs.
            X[col] = X[col].cat.add_categories(["<Unknown>"]).fillna("<Unknown>")

        # Numeric columns: impute mean (CatBoost handles NaN in numeric too; keeping this for determinism)
        num_cols = list(X.select_dtypes(include=["number"]).columns)
        if len(num_cols) > 0:
            X[num_cols] = X[num_cols].astype(float)
            X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

        # Save categorical feature indices for CatBoost
        self.cat_features = [X.columns.get_loc(c) for c in cat_cols]

        # Convert y to numeric
        y = pd.to_numeric(y, errors="coerce")
        # If target had non-numeric, drop those rows
        non_nan_mask = ~y.isna()
        if not non_nan_mask.all():
            dropped = (~non_nan_mask).sum()
            self.logging.info(f"Dropping {dropped} rows due to non-numeric target.")
            y = y[non_nan_mask]
            X = X.loc[y.index]

        # Standardization info for target (as your original code)
        m = y.mean()
        std = y.std(ddof=0)
        z = (y - m) / (std if std != 0 else 1.0)

        self.logging.info(
            f"Finished CatBoost preprocessing. "
            f"{len(cat_cols)} categorical features detected: {cat_cols}"
        )
        return X, y, z, m, std

    def fit(self, X, y, **kwargs):
        if "sample_weight" not in kwargs and self.weights is not None:
            kwargs["sample_weight"] = self.weights
        kwargs.setdefault("cat_features", getattr(self, "cat_features", None))
        return self.model.fit(X, y, **kwargs)


    # ------------------------------------------------------------------
    # Feature Importance (SHAP)
    # ------------------------------------------------------------------
    def feature_importance(
        self,
        X,
        top_n: Optional[int] = None,
        batch_size: Optional[int] = None,
        save_results: bool = True,
        iter_idx: Optional[int] = None,
        ablation_idx: Optional[int] = None,
        validation: bool = False,
    ):
        shap.initjs()
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X, check_additivity=True)

        plt_title = f"{self.target_name} {self.model_name} SHAP Summary"
        shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False, max_display=self.top_n)
        plt.title(plt_title, fontsize=16)

        if save_results:
            plt.subplots_adjust(top=0.90)
            save_root = self.save_path
            if iter_idx is not None:
                save_root = os.path.join(save_root, "singleSHAPs")
                os.makedirs(save_root, exist_ok=True)
                suffix = "test" if validation else "train"
                filename = f"{self.target_name}_catboost_shap_{suffix}_{iter_idx}.png"
            elif ablation_idx is not None:
                save_root = os.path.join(save_root, "ablationSHAPs")
                os.makedirs(save_root, exist_ok=True)
                suffix = "test" if validation else "train"
                filename = f"{self.target_name}_catboost_shap_{suffix}_ablation_{ablation_idx}.png"
            else:
                suffix = "test" if validation else "train"
                filename = f"{self.target_name}_catboost_shap_{suffix}.png"
            plt.savefig(os.path.join(save_root, filename), dpi=150, bbox_inches="tight")
            plt.close()

        return shap_values

    # ------------------------------------------------------------------
    # Hyperparameter Tuning
    # ------------------------------------------------------------------
    def tune_hparams(self, X, y, param_grid, folds=5, groups=None, weights=None) -> Dict:
        if param_grid is None:
            raise ValueError("param_grid must be provided for tuning.")
        if weights is None:
            weights = self.weights

        splitter = (LeaveOneOut() if groups is None else LeaveOneGroupOut()) if folds == -1 \
            else (KFold(n_splits=folds, shuffle=True, random_state=self.random_state) if groups is None
                  else GroupKFold(n_splits=folds))

        estimator = CatBoostRegressor(**self.cat_hparams)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=splitter,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        fit_kwargs = {"cat_features": getattr(self, "cat_features", None)}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
        if groups is not None and folds != -1:
            # GridSearchCV handles groups via split() generator, but pass for completeness
            fit_kwargs["groups"] = groups

        grid_search.fit(X, y, **fit_kwargs)

        best_params = grid_search.best_params_
        self.cat_hparams.update(best_params)
        self.model = grid_search.best_estimator_
        if hasattr(self, "logging") and self.logging:
            self.logging.info(f"Best CatBoost params: {best_params}")
            self.logging.info(f"Best CatBoost CV score: {grid_search.best_score_:.6f}")
        return best_params


    # ------------------------------------------------------------------
    # Override predict to ensure numpy output
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.logging.info("Starting CatBoost prediction...")
        preds = self.model.predict(X)
        self.logging.info("Finished CatBoost prediction.")
        return np.asarray(preds)
