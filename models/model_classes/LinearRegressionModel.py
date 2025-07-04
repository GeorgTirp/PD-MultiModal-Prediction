# Standard Libraries
import os
from typing import Dict

# Data handling and numerical computation
import numpy as np
import pandas as pd

# Machine Learning and Modeling
from sklearn.linear_model import LinearRegression

# Visualization
import matplotlib.pyplot as plt
import shap

# Custom Base Model
from model_classes.BaseRegressionModel import BaseRegressionModel


class LinearRegressionModel(BaseRegressionModel):
    """
    Initializes the Linear Regression model and sets up data and metadata.

    Args:
        data_df (pd.DataFrame): Input dataset containing features and target.
        feature_selection (dict): Dictionary with selected features, typically {'features': [...] }.
        target_name (str): Name of the target column in the dataset.
        test_split_size (float, optional): Fraction of the data to use as test set. Defaults to 0.2.
        save_path (str, optional): Path to directory for saving outputs. Defaults to None.
        identifier (str, optional): Optional run identifier used for saving results. Defaults to None.
        top_n (int, optional): Number of top features to consider. Use -1 for all. Defaults to 10.
    """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            target_name: str,
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10,
            logging=None):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n, logging)
        self.model = LinearRegression()
        self.model_name = "Linear Regression"
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])

    def feature_importance(self, top_n: int = None, save_results=True, iter_idx=None, ablation_idx=None) -> Dict:
        """
        Computes feature importance using model coefficients and SHAP values.

        Args:
            top_n (int, optional): Number of top features to display. If None, uses `self.top_n`.
            save_results (bool, optional): Whether to save the importance scores and SHAP plots. Defaults to True.
            iter_idx (int, optional): Optional iteration index for naming the saved SHAP plot. Used during repeated runs.
            ablation_idx (int, optional): Optional ablation index for naming the saved SHAP plot. Used in ablation studies.

        Returns:
            Dict: SHAP values for each feature across the dataset.
        """
    
        if iter_idx is None:
            self.logging.info("Starting feature importance evaluation for Linear Regression...")
        # Use absolute value of coefficients (normalized)
        attribution = np.abs(self.model.coef_) / np.sum(np.abs(self.model.coef_))
        feature_names = self.feature_selection['features']
        indices = np.argsort(attribution)[-self.top_n:][::-1]
        top_features = {feature_names[i]: attribution[i] for i in indices}
        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_{self.target_name}_feature_importance.npy', top_features)
        
        self.importances = top_features

        # Compute SHAP values using a linear explainer
        shap.initjs()
        background_data = self.X.sample(25, random_state=42)
        explainer = shap.LinearExplainer(self.model, background_data)
        shap_values = explainer.shap_values(self.X)
        # Plot aggregated SHAP values (beeswarm and bar plots)
        shap.summary_plot(shap_values, self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
        plt.title(f'{self.identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                save_path = self.save_path + "/singleSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm_{iter_idx}.png')
            elif ablation_idx is not None:
                save_path = self.save_path + "/ablationSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm_ablation_{ablation_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm.png')
            plt.close()
            
        if iter_idx is None:
            self.logging.info("Finished feature importance evaluation for Linear Regression.")
        return shap_values