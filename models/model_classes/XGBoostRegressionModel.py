# Standard Libraries
import os
from typing import Dict

# Data Handling and Numeric Computation
import numpy as np
import pandas as pd

# Machine Learning and Modeling
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Visualization and Explainability
import matplotlib.pyplot as plt
import shap

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
            logging = None):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n, logging=logging)
        self.xgb_hparams = xgb_hparams
        self.model = XGBRegressor(**self.xgb_hparams)
        self.model_name = "XGBoost Regression"
        self.param_grid = param_grid
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])

    def feature_importance(self, top_n: int = None, batch_size: int = 10, save_results: bool = True, iter_idx=None, ablation_idx=None) -> Dict:
        """ Compute feature importance for the predicted mean using SHAP KernelExplainer. """
        shap.initjs()

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X, check_additivity=True)
        shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
        plt.title(f'{self.identifier} NGBoost Mean SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                save_path = self.save_path + "/singleSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_mean_shap_aggregated_beeswarm_{iter_idx}.png')
            elif ablation_idx is not None:
                save_path = self.save_path + "/ablationSHAPs"
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm{ablation_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_aggregated_beeswarm.png')
            plt.close()
        return shap_values
    
    def tune_hparams(self, X, y, param_grid: dict, folds=5) -> Dict:
        """Tune hyperparameters using GridSearchCV with 5-fold cross-validation.

        Args:
            param_grid (dict): Dictionary of parameter grid to search over.

        Returns:
            dict: Best hyperparameters found.
        """
        if folds == -1:
            folds = len(X)
        #self.logging.info(f"Starting hyperparameter tuning using GridSearchCV with {folds}-fold CV...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.xgb_hparams.update(best_params)
        self.model.set_params(**best_params)
        return best_params