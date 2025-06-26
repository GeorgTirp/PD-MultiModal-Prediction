# Standard Libraries
import os
from typing import Dict

# Data Handling and Numerical Computation
import numpy as np
import pandas as pd

# Machine Learning and Modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Visualization
import matplotlib.pyplot as plt
import shap

# Custom Base Model
from model_classes.BaseRegressionModel import BaseRegressionModel


class RandomForestModel(BaseRegressionModel):
    """ Random Forest Regression Model """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            target_name: str,
            rf_hparams: dict, 
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10,
            param_grid: dict = None):
        """
        Initializes the Random Forest model with specified hyperparameters.

        Args:
            data_df (pd.DataFrame): Input dataset containing features and target.
            feature_selection (dict): Dictionary with selected features, typically {'features': [...] }.
            target_name (str): Name of the target column in the dataset.
            rf_hparams (dict): Dictionary of hyperparameters for RandomForestRegressor.
            test_split_size (float, optional): Fraction of the data to use as test set. Defaults to 0.2.
            save_path (str, optional): Path to directory for saving outputs. Defaults to None.
            identifier (str, optional): Optional run identifier used for saving results. Defaults to None.
            top_n (int, optional): Number of top features to consider. Use -1 for all. Defaults to 10.
            param_grid (dict, optional): Grid of parameters to use for tuning (if needed). Defaults to None.
        """
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
        self.rf_hparams = rf_hparams
        self.param_grid = param_grid
        self.model = RandomForestRegressor(**self.rf_hparams)
        self.model_name = "Random Forest"
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])

    def feature_importance(self, top_n: int = None, save_results=True, iter_idx=None, ablation_idx=None) -> Dict:
        """
        Computes feature importance using built-in feature importances and SHAP values.
    
        Args:
            top_n (int, optional): Number of top features to display. If None, uses `self.top_n`.
            save_results (bool, optional): Whether to save the importance scores and SHAP plots. Defaults to True.
            iter_idx (int, optional): Optional iteration index for naming the saved SHAP plot. Used during repeated runs.
            ablation_idx (int, optional): Optional ablation index for naming the saved SHAP plot. Used in ablation studies.
    
        Returns:
            Dict: SHAP values for each feature across the dataset.
        """
        if iter_idx is None:
            self.logging.info("Starting feature importance evaluation for Random Forest...")
        # Use the feature_importances_ attribute of RandomForest
        attribution = self.model.feature_importances_
        feature_names = self.feature_selection['features']
        indices = np.argsort(attribution)[-self.top_n:][::-1]
        top_features = {feature_names[i]: attribution[i] for i in indices}
        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_{self.target_name}_feature_importance.npy', top_features)
        
        self.importances = top_features

        # Compute SHAP values using a tree explainer
        shap.initjs()
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X)
        # Plot aggregated SHAP values (beeswarm and bar plots)
        shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
        plt.title(f'{self.identifier} {self.target_name}  SHAP Summary Plot (aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                save_path = self.save_path + "/singleSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_rf_shap_aggregated_beeswarm_{iter_idx}.png')
            elif ablation_idx is not None:
                save_path = self.save_path + "/ablationSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm_ablation_{ablation_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_rf_shap_aggregated_beeswarm.png')
            plt.close()
            
        if iter_idx is None:
            self.logging.info("Finished feature importance evaluation for Random Forest.")
        return shap_values

    def tune_hparams(self, X, y, param_grid: dict, folds=5) -> Dict:
        """
        Tunes hyperparameters using GridSearchCV with k-fold cross-validation.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target variable.
            param_grid (dict): Dictionary of hyperparameter ranges to search.
            folds (int, optional): Number of cross-validation folds. Use -1 for leave-one-out. Defaults to 5.

        Returns:
            Dict: Dictionary of best hyperparameters found.
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
        self.model.set_params(**best_params)
        self.rf_hparams.update(best_params)
        #self.logging.info(f"Best parameters found: {best_params}")
        return best_params