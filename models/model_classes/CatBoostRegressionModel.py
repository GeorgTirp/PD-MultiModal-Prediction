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
        self.logging.info("Starting CatBoost preprocessing...")

        data_df = data_df.dropna(subset=self.feature_selection["features"] + [self.feature_selection["target"]])
        X = data_df[self.feature_selection["features"]].copy()
        y = data_df[self.feature_selection["target"]]

        # Encode categorical/string columns via pandas category codes
        from pandas.api.types import is_string_dtype

        string_cols = [col for col in X.columns if is_string_dtype(X[col])]
        for col in string_cols:
            self.logging.info(f"Encoding column '{col}' as categorical codes for CatBoost.")
            X[col] = X[col].astype("category").cat.codes

        numeric_cols = X.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        X = X.apply(pd.to_numeric, errors="raise")
        y = y.apply(pd.to_numeric, errors="raise")

        m = y.mean()
        std = y.std(ddof=0)
        z = (y - m) / (std if std != 0 else 1.0)

        self.logging.info("Finished CatBoost preprocessing.")
        return X, y, z, m, std

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
    def tune_hparams(
        self,
        X,
        y,
        param_grid: dict,
        folds: int = 5,
        groups: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Dict:
        if param_grid is None:
            raise ValueError("param_grid must be provided for tuning.")

        if weights is None:
            weights = self.weights

        if folds == -1:
            splitter = LeaveOneOut() if groups is None else LeaveOneGroupOut()
        else:
            if groups is None:
                splitter = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
            else:
                splitter = GroupKFold(n_splits=folds)

        estimator = CatBoostRegressor(**self.cat_hparams)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=splitter,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights

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
