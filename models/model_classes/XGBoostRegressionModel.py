# Standard Libraries
import os
from typing import Dict

# Data Handling and Numeric Computation
import numpy as np
import pandas as pd

# Machine Learning and Modeling
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from typing import Union, List, Tuple
# Visualization and Explainability
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import (
    KFold, LeaveOneOut,
    GroupKFold, LeaveOneGroupOut
)
# Custom Base Model
from model_classes.BaseRegressionModel import BaseRegressionModel
from model_classes.NGBoostRegressionModel import NGBoostRegressionModel


class XGBoostRegressionModel(BaseRegressionModel):
    """ XGBoost Regression Model """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            target_name: str,
            xgb_hparams: dict, 
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = -1,
            param_grid: dict = None,
            logging = None,
            Pat_IDs= None,
            split_shaps=None,
            sample_weights=None
            ):
        
        super().__init__(
            data_df, 
            feature_selection, 
            target_name, 
            test_split_size, 
            save_path, 
            identifier, 
            top_n, 
            logging=logging, 
            Pat_IDs=Pat_IDs,
            split_shaps=split_shaps)

        self.xgb_hparams = xgb_hparams
        self.model = XGBRegressor(**self.xgb_hparams)
        self.model_name = "XGBoost Regression"
        self.param_grid = param_grid
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])
        self.weights = sample_weights

    def model_specific_preprocess(self, data_df: pd.DataFrame) -> Tuple:
        """
        Preprocess data specific to model requirements, including feature extraction,
        optional ceiling adjustments, and automatic categorical encoding of string columns.

        Args:
            data_df (pd.DataFrame): Original input data.
            ceiling (list or str, optional): Ceiling transformations to apply. Defaults to ["BDI", "MoCA"].

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series, float, float]:
                - Preprocessed features (X)
                - Original target (y)
                - Standardized target (z)
                - Target mean (m)
                - Target std (std)
        """
        self.logging.info("Starting model-specific preprocessing...")

        # Drop rows with missing values for features and target
        data_df = data_df.dropna(subset=self.feature_selection['features'] + [self.feature_selection['target']])
        X = data_df[self.feature_selection['features']].copy()

        # Apply ceiling transformations
        # Allow ceiling to be a single string or a list
        # Extract target
        y = data_df[self.feature_selection['target']]

        # Identify and encode string columns as categorical
        from pandas.api.types import is_string_dtype
        string_cols = [col for col in X.columns if is_string_dtype(X[col])]
        for col in string_cols:
            # Convert to pandas 'category' dtype, then use integer codes
            self.logging.info(f"Encoding column '{col}' as categorical codes.")
            X[col] = X[col].astype('category').cat.codes

        # Fill numeric missing values
        numeric_cols = X.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        # Ensure all values are numeric
        X = X.apply(pd.to_numeric, errors='raise')
        y = y.apply(pd.to_numeric, errors='raise')
        # Standardize target variable
        m = y.mean()
        std = y.std()
        z = (y - m) / std

        self.logging.info("Finished model-specific preprocessing.")

        return X, y, z, m, std

    def feature_importance(
        self, 
        X, 
        top_n: int = None, 
        batch_size: int = 10, 
        save_results: bool = True, 
        iter_idx=None, 
        ablation_idx=None,
        validation=False
        ) -> Dict:
        """ Compute feature importance for the predicted mean using SHAP KernelExplainer. """
        shap.initjs()

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X, check_additivity=True)
        shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False, max_display=self.top_n)
        plt.title(f'{self.identifier} NGBoost Mean SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                save_path = self.save_path + "/singleSHAPs"
                os.makedirs(save_path, exist_ok=True)
                if validation:
                    plt.savefig(f'{save_path}/{self.identifier}_mean_shap_aggregated_beeswarm_{iter_idx}_test.png')
                else:
                    plt.savefig(f'{save_path}/{self.identifier}_mean_shap_aggregated_beeswarm_train{iter_idx}.png')
            elif ablation_idx is not None:
                save_path = self.save_path + "/ablationSHAPs"
                if validation:
                    plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm{ablation_idx}_test.png')
                else:    
                    plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm{ablation_idx}_train.png')
            else:
                if validation:
                    plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_aggregated_beeswarm_test.png')
                else:
                    plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_aggregated_beeswarm_train.png')
            plt.close()
        return shap_values
    
    def tune_hparams(self,
                 X,
                 y,
                 param_grid: dict,
                 folds: int = 5,
                 groups: np.ndarray = None,
                 weights: np.ndarray = None
                ) -> Dict:
        """
        Tune hyperparameters using GridSearchCV.
        If `groups` is provided, uses GroupKFold; otherwise standard KFold.
        If `weights` is provided, passes them in as sample_weight to .fit().
    
        Args:
            X:           Training features.
            y:           Training targets.
            param_grid:  Dict of parameters to search.
            folds:       Number of CV folds (or -1 → leave‑one‑out).
            groups:      Optional array of group labels for group‑aware CV.
            weights:     Optional array of sample‐weights aligned with X, y.
        """
        # allow -1 → leave‑one‑out
        if folds == -1:
            folds = len(X)
    
        # pick our CV splitter
        if groups is None:
            cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            cv = GroupKFold(n_splits=folds)
    
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
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
        self.xgb_hparams.update(best_params)
        self.model.set_params(**best_params)
    
        if hasattr(self, 'logging') and self.logging:
            self.logging.info(f"Best parameters found: {best_params}")
            self.logging.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
        return best_params