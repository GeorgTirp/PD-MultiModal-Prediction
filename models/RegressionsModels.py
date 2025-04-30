import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import shap
import logging
from tqdm import tqdm
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
import seaborn as sns
from contextlib import contextmanager#
from IPython.utils import io
import torch
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor
from evidential_boost import NormalInverseGamma
from joblib import Parallel, delayed
import scipy.stats as st
from properscoring import crps_ensemble

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseRegressionModel:
    """ Base class for regression models """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            target_name: str,
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10):
        
        logging.info("Initializing BaseRegressionModel class...")
        self.feature_selection = feature_selection
        self.top_n = top_n
        self.X, self.y = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(self.X, self.y, test_size=test_split_size, random_state=42)
        self.save_path = save_path
        self.identifier = identifier
        self.metrics = None
        self.model_name = None
        self.target_name = target_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info("Finished initializing BaseRegressionModel class.")

    def model_specific_preprocess(self, data_df: pd.DataFrame, ceiling: list =["BDI", "MoCA"]) -> Tuple:
        """ Preprocess the data for the model """
        logging.info("Starting model-specific preprocessing...")
        # Drop rows with missing values for features and target
        data_df = data_df.dropna(subset=self.feature_selection['features'] + [self.feature_selection['target']])
        X = data_df[self.feature_selection['features']]
        if ceiling == "BDI":
            if "BDI_sum_pre" in X.columns:
                X["Distance_ceiling"] = 63 - X["BDI_sum_pre"]  
                self.feature_selection["features"].append("Distance_ceiling")  
            elif "MoCA_sum_pre" in X.columns:
                X["Distance_ceiling"] = 30 - X["BDI_sum"]
            else:
                raise ValueError("Neither BDI_sum_pre nor MoCA_sum_pre found in the DataFrame.")
        y = data_df[self.feature_selection['target']]
        X = X.fillna(X.mean())
        X = X.apply(pd.to_numeric, errors='coerce')
        
        logging.info("Finished model-specific preprocessing.")
        return X, y

    def fit(self) -> None:
        """ Train the model """
        logging.info(f"Starting {self.model_name} model training...")
        X_train, X_test, y_train, y_test = self.train_split
        self.model.fit(X_train, y_train)
        logging.info(f"Finished {self.model_name} model training.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """ Predict using the trained model """
        logging.info("Starting prediction...")
        pred = self.model.predict(X)
        logging.info("Finished prediction.")
        return pred

    def evaluate(self, folds=10, get_shap=True, tune=False, tune_folds=10, nested=False, uncertainty=False) -> Dict:
        if nested==True:
            return self.nested_eval(folds, get_shap, tune, tune_folds, uncertainty=uncertainty)
        else:
            return self.sequential_eval(folds, get_shap, tune, tune_folds)
        
    def sequential_eval(self, folds=10, get_shap=True, tune=False, tune_folds=10) -> Dict:
        """ Evaluate the model using cross-validation """
        logging.info("Starting model evaluation...")
        if tune:
            if self.param_grid is None:
                raise ValueError("When calling tune=True, a param_grid has to be passed when initializing the model.")
            self.tune_hparams(self.X, self.y,  self.param_grid, tune_folds)

        if folds == -1:
            kf = LeaveOneOut()
        else:
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        preds = []
        y_vals = []
        all_shap_values = []
        for train_index, val_index in tqdm(kf.split(self.X), total=kf.get_n_splits(self.X), desc="Cross-validation", leave=True):
            X_train_kf, X_val_kf = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_kf, y_val_kf = self.y.iloc[train_index], self.y.iloc[val_index]
            if not tune:
                self.model.fit(X_train_kf, y_train_kf)
            pred = self.model.predict(X_val_kf)
            preds.append(pred)
            y_vals.append(y_val_kf)
            if get_shap:             
                 # Compute SHAP values on the whole dataset per fold
                with io.capture_output():
                    shap_values = self.feature_importance(top_n=-1, save_results=True, iter_idx=val_index)
                all_shap_values.append(shap_values) 

        preds = np.concatenate(preds)
        y_vals = np.concatenate(y_vals)
        r2, p = pearsonr(y_vals, preds)
        mse = mean_squared_error(y_vals, preds)

        metrics = {
            'mse': mse,
            'r2': r2,
            'p_value': p,
            'y_pred': preds,
            'y_test': y_vals
        }
        self.metrics = metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{self.save_path}/{self.identifier}_metrics.csv', index=False)

        if get_shap:
            all_shap_values_array = np.stack(all_shap_values, axis=0)
            # Average over the folds to get an aggregated array of shape (n_samples, n_features)
            mean_shap_values = np.mean(all_shap_values_array, axis=0)
            np.save(f'{self.save_path}/{self.identifier}_mean_shap_values.npy', mean_shap_values)
            shap.summary_plot(mean_shap_values , features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
            plt.title(f'{self.identifier} Summary Plot (Aggregated)', fontsize=16)
            plt.subplots_adjust(top=0.90)
            plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm.png')
            plt.close()

        logging.info("Finished model evaluation.")
        return metrics

    def nested_eval(self, folds=10, get_shap=True, tune=False, tune_folds=10, uncertainty=False) -> Dict:
        """ Evaluate the model using cross-validation """
        logging.info("Starting model evaluation...")
        if folds == -1:
            kf = LeaveOneOut()
        else:
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        if tune and self.param_grid is None:
                raise ValueError("When calling tune=True, a param_grid has to be passed when initializing the model.")
        preds = []
        epistemics = []
        aleatorics = []
        pred_dists = []
        y_vals = []
        all_shap_values = []
        all_shap_mean = []
        all_shap_variance = []

        for train_index, val_index in tqdm(kf.split(self.X), total=kf.get_n_splits(self.X), desc="Cross-validation", leave=False):
            X_train_kf, X_val_kf = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_kf, y_val_kf = self.y.iloc[train_index], self.y.iloc[val_index]
            if tune:
                self.tune_hparams(X_train_kf, y_train_kf, self.param_grid, tune_folds)
                #tune = False
            else:
                self.model.fit(X_train_kf, y_train_kf)

            pred = self.model.predict(X_val_kf)
            preds.append(pred)
            y_vals.append(y_val_kf)

            # Get uncertainties
            if uncertainty == True:
                _ ,epistemic, aleatoric = self.compute_uncertainties(mode="nig", X=X_val_kf)
                epistemics.append(epistemic)
                aleatorics.append(aleatoric)

            # Get paramters of predictive Distribution
            if isinstance(self, NGBoostRegressionModel):
                pred_dist = self.model.pred_dist(X_val_kf)
                pred_dists.append(pred_dist)
            
            # Get eplanations
            if get_shap:             
                 # Compute SHAP values on the whole dataset per fold
                with io.capture_output():
                    if isinstance(self, NGBoostRegressionModel):
                        shap_values_mean = self.feature_importance_mean(top_n=-1, save_results=True,  iter_idx=val_index)
                        shap_values_variance = self.feature_importance_variance(top_n=-1, save_results=True, iter_idx=val_index)
                        all_shap_mean.append(shap_values_mean)
                        all_shap_variance.append(shap_values_variance)
                    else:
                        shap_values = self.feature_importance(top_n=-1, save_results=True, iter_idx=val_index)
                        all_shap_values.append(shap_values) 

        #pred_dists = np.hstack(pred_dists)
        preds = np.concatenate(preds)
        y_vals = np.concatenate(y_vals)
        r2, p = pearsonr(y_vals, preds)
        
        if uncertainty == True:
            epistemic_uncertainty = np.concatenate(epistemics)
            aleatoric_uncertainty = np.concatenate(aleatorics)
        mse = mean_squared_error(y_vals, preds)

        if get_shap:
            if isinstance(self, NGBoostRegressionModel):
                all_shap_mean_array = np.stack(all_shap_mean, axis=0)
                all_shap_variance_array = np.stack(all_shap_variance, axis=0)
                # Average over the folds to get an aggregated array of shape (n_samples, n_features)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                variance_shap_values = np.mean(all_shap_variance_array, axis=0)
                np.save(f'{self.save_path}/{self.identifier}_mean_shap_values.npy', mean_shap_values)
                np.save(f'{self.save_path}/{self.identifier}_variance_shap_values.npy', variance_shap_values)
                # Plot for mean SHAP values
                shap.summary_plot(mean_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier} Summary Plot (Aggregated - Mean)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm_mean.png')
                plt.close()
                
                shap.summary_plot(variance_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier} Summary Plot (Aggregated - Variance)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm_variance.png')
                plt.close()
            else:
                all_shap_values_array = np.stack(all_shap_values, axis=0)
                # Average over the folds to get an aggregated array of shape (n_samples, n_features)
                mean_shap_values = np.mean(all_shap_values_array, axis=0)
                np.save(f'{self.save_path}/{self.identifier}_mean_shap_values.npy', mean_shap_values)
                shap.summary_plot(mean_shap_values , features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier}  Summary Plot (Aggregated)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm.png')
                plt.close()
            
            # Compute the mean of the absolute SHAP values for each feature
            feature_importances = np.mean(np.abs(mean_shap_values), axis=0)
            feature_importance_dict = dict(zip(self.X.columns, feature_importances))
            # Save feature importances to a file
            

        metrics = {
        'mse': mse,
        'r2': r2,
        'p_value': p,
        'y_pred': preds,
        'y_test': y_vals,
        'pred_dist': pred_dists,
        'epistemic': epistemic_uncertainty if uncertainty else None,
        'aleatoric': aleatoric_uncertainty if uncertainty else None,
        'feature_importance': feature_importances if get_shap else None
        }
        self.metrics = metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{self.save_path}/{self.identifier}_metrics.csv', index=False)
        logging.info("Finished model evaluation.")
        return metrics
    
    def feature_ablation(self) -> Dict:
        """ Compute the feature ablation"""
        r2s = []
        p_values = []
        number_of_features = len(self.feature_selection['features'])
        for i in range(number_of_features):
            metrics = self.nested_eval(folds=10, get_shap=True, tune=False)
            r2s.append(metrics['r2'])
            p_values.append(metrics['p_value'])
            importance = metrics['feature_importance']
            importance_indices = np.argsort(importance)
            least_important_feature = self.feature_selection['features'][importance_indices[0]]
            self.X = self.X.drop(columns=[least_important_feature])
            self.y = self.y.drop(columns=[least_important_feature])
            self.feature_selection['features'].remove(least_important_feature)
        
        # Define the custom color palette from your image
        custom_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#525252"]
                # Set Seaborn style, context, and custom palette
        sns.set_theme(style="whitegrid", context="paper")
        sns.set_palette(custom_palette)
        path = self.save_path + 'feature_ablation.png'
                # Read in the CSV
        
        
        # Create a figure
        plt.figure(figsize=(6, 4))
        x = np.arange(number_of_features)
                # Create a figure
        plt.figure(figsize=(6, 4))
                # Plot each model's R² scores in a loop, using sample_sizes on the x-axis
        #for model_name, r2_scores in results.items():
        plot_df = pd.DataFrame({'x': x, 'r2s': r2s})
        sns.lineplot(data=plot_df, x='x', y='r2s', label="R Score", marker='o')

                # Optionally use a log scale for the x-axis if you want to emphasize the “logarithmic” nature
        # Label the axes and set the title
        plt.xlabel("Number of removed features")
        plt.ylabel("R Score")
        plt.title("R Scores Over Feature Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return r2s, p_values
            

    
    def feature_importance(self, top_n: int = 10, batch_size=None, save_results=True) -> Dict:
        """ To be implemented in the subclass """
        raise NotImplementedError("Subclasses must implement feature_importance method")

    def compute_uncertainties(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ Compute uncertainties using the model """
        raise NotImplementedError("Uncertainty computation not implemented for this model.")
    
    
        

    def plot(self, title, modality='') -> None:
        """ Plot predicted vs. actual values """
        logging.info("Starting plot generation.")
         # Use a context suitable for publication-quality figures
        sns.set_context("paper")
        # Optionally choose a style you like
        sns.set_style("whitegrid")

        # Create a wider (landscape) figure
        plt.figure(figsize=(10, 6))

        # Create a DataFrame for Seaborn
        plot_df = pd.DataFrame({
            'Actual': self.metrics['y_test'],
            'Predicted': self.metrics['y_pred']
        })

        # Scatter plot only (no regression line)
        sns.scatterplot(
            x='Actual', 
            y='Predicted', 
            data=plot_df, 
            alpha=0.7
        )
        
        # Plot a reference line with slope = 1
        min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
        max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='grey', alpha=0.4, linestyle='--')

        # Fit a regression line
        sns.regplot(
            x='Actual', 
            y='Predicted', 
            data=plot_df, 
            scatter=False, 
            color='red', 
            line_kws={'label': 'Regression Line'}
        )

        # Plot confidence intervals
        ci = 95  # Confidence interval percentage
        sns.regplot(
            x='Actual', 
            y='Predicted', 
            data=plot_df, 
            scatter=False, 
            color='red', 
            ci=ci, 
            line_kws={'label': f'{ci}% Confidence Interval'}
        )
        # Add text (R and p-value) in the top-left corner inside the plot
        # using axis coordinates (0–1 range) so it doesn't get cut off
        plt.text(
            0.05, 0.95, 
            f'R: {self.metrics["r2"]:.2f}\nP-value: {self.metrics["p_value"]:.6f}', 
            fontsize=12, 
            transform=plt.gca().transAxes,  # use axis coordinates
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5)
        )

        # Label axes and set title
        plt.xlabel(f'Actual {modality} {self.target_name}', fontsize=12)
        plt.ylabel(f'Predicted {modality} {self.target_name}', fontsize=12)
        plt.title(title + "  N=" + str(len(self.y)), fontsize=14)

        # Show grid and ensure everything fits nicely
        plt.grid(True)
        plt.tight_layout()

        # Save and close
        plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_actual_vs_predicted.png')
        plt.close()

        # Log info (optional)
        logging.info("Plot saved to %s/%s_%s_actual_vs_predicted.png", 
                 self.save_path, self.identifier, self.target_name)

class LinearRegressionModel(BaseRegressionModel):
    """ Linear Regression Model """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            target_name: str,
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
        self.model = LinearRegression()

        self.model_name = "Linear Regression"
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])

    def feature_importance(self, top_n: int = None, save_results=True, iter_idx=None) -> Dict:
        """ Compute feature importance using coefficients for Linear Regression """
    
        if iter_idx is None:
            logging.info("Starting feature importance evaluation for Linear Regression...")
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
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm.png')
            plt.close()
            
        if iter_idx is None:
            logging.info("Finished feature importance evaluation for Linear Regression.")
        return shap_values

class RandomForestModel(BaseRegressionModel):
    """ Random Forest Model """
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
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
        self.rf_hparams = rf_hparams
        self.param_grid = param_grid
        self.model = RandomForestRegressor(**self.rf_hparams)
        self.model_name = "Random Forest"
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])

    def feature_importance(self, top_n: int = None, save_results=True, iter_idx=None) -> Dict:
        """ Compute feature importance using the built-in attribute for Random Forest """
        if iter_idx is None:
            logging.info("Starting feature importance evaluation for Random Forest...")
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
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_rf_shap_aggregated_beeswarm.png')
            plt.close()
            
        if iter_idx is None:
            logging.info("Finished feature importance evaluation for Random Forest.")
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
        #logging.info(f"Starting hyperparameter tuning using GridSearchCV with {folds}-fold CV...")
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
        #logging.info(f"Best parameters found: {best_params}")
        return best_params
    

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
            param_grid: dict = None):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
        self.xgb_hparams = xgb_hparams
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(device)
            print("Device name:", props.name)
            print("Number of SMs:", props.multi_processor_count)
            print("Total GPU memory (bytes):", props.total_memory)
            self.model = XGBRegressor(**self.xgb_hparams)
        else:
            print("No CUDA device found.")
            self.model = XGBRegressor(**self.xgb_hparams)
        self.model_name = "XGBoost Regression"
        self.param_grid = param_grid
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])

    def feature_importance(self, top_n: int = None, save_results=True, iter_idx = None) -> Dict:
        """ Compute feature importance using the built-in attribute for XGBoost """
        
        if iter_idx is None:
            logging.info("Starting feature importance evaluation for XGBoost Regression...")
        # Use the feature_importances_ attribute of XGBoost
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
                plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_xgb_shap_beeswarm_{iter_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_xgb_shap_aggregated_beeswarm.png')
            plt.close()
            
        if iter_idx is None:  
            logging.info("Finished feature importance evaluation for XGBoost Regression.")
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
        #logging.info(f"Starting hyperparameter tuning using GridSearchCV with {folds}-fold CV...")
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
        #logging.info(f"Best parameters found: {best_params}")
        return best_params
    

class NGBoostRegressionModel(BaseRegressionModel):
    """ NGBoost Regression Model for heteroscedastic regression.
        This model returns the predictive mean by default, and its predictive distribution can be
        accessed via the `pred_dist` method.
    """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            target_name: str,
            ngb_hparams: dict = None, 
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = -1,
            param_grid: dict = None):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
        # Set default hyperparameters if not provided
        if ngb_hparams is None:
            ngb_hparams = {
                'Dist': NormalInverseGamma,
                'n_estimators': 500,
                'learning_rate': 0.01,
                'verbose': False
            }
        self.ngb_hparams = ngb_hparams
        self.model = NGBRegressor(**self.ngb_hparams)
        self.model_name = "NGBoost Regression"
        self.param_grid = param_grid
        if top_n == -1:
            self.top_n = len(self.feature_selection['features'])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """ Predict using the trained NGBoost model.
            By default, return the mean predictions.
        """
        logging.info("Starting NGBoost prediction...")
        # NGBoost's predict method returns the mean predictions
        pred = self.model.predict(X)
        logging.info("Finished NGBoost prediction.")
        return pred

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns both the mean predictions and the variance (from the predictive distribution). """
        logging.info("Starting NGBoost prediction with uncertainty estimation...")
        pred_dist = self.model.pred_dist(X)
        pass

    def tune_hparams(self, X, y, param_grid: dict, folds=5) -> Dict:
        """Tune hyperparameters using GridSearchCV with 5-fold cross-validation."""
            # Ensure base estimator supports tree-based parameters
        if folds == -1:
            folds = len(X)
        # Perform grid search on the current NGBoost model
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        
        # Separate base learner parameters and other NGBoost hyperparameters
        base_params = {k.replace("Base__", ""): v for k, v in best_params.items() if k.startswith("Base__")}
        non_base_params = {k: v for k, v in best_params.items() if not k.startswith("Base__")}
        
        # Rebuild the base learner using the tuned parameters.
        # Here we assume the base learner is a DecisionTreeRegressor.
        if base_params:
            new_base_learner = DecisionTreeRegressor(**base_params)
        else:
            # Use a default base learner if no parameters were provided
            new_base_learner = DecisionTreeRegressor(max_depth=3)
        
        # Update the current NGBoost hyperparameters with the tuned non-base parameters
        new_ngb_hparams = self.ngb_hparams.copy()
        new_ngb_hparams.update(non_base_params)
        new_ngb_hparams['Base'] = new_base_learner
        
        # Reinitialize the NGBoost model with updated hyperparameters.
        self.model = NGBRegressor(**new_ngb_hparams)
        # Force an immediate fit on the tuning data to initialize internal parameters.
        self.model.fit(X, y)
        self.model_name += " (Tuned)"
        return best_params

    def feature_importance_mean(self, top_n: int = None, batch_size: int = 10, save_results: bool = True, iter_idx=None) -> Dict:
        """ Compute feature importance for the predicted mean using SHAP KernelExplainer. """
        shap.initjs()
        # Use a small random sample as background
        background = shap.sample(self.X, 20)

        explainer = shap.TreeExplainer(self.model, model_output=0)
        shap_values = explainer.shap_values(self.X, check_additivity=True)
        shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
        plt.title(f'{self.identifier} NGBoost Mean SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                save_path = self.save_path + "/singleSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_ngboost_mean_shap_aggregated_beeswarm_{iter_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_ngboost_mean_shap_aggregated_beeswarm.png')
            plt.close()
        return shap_values

    def feature_importance_variance(self, top_n: int = None, batch_size: int = 10, save_results: bool = True, iter_idx=None) -> Dict:
        """ Compute feature importance for the predicted variance using SHAP KernelExplainer. """
        shap.initjs()
        background = shap.sample(self.X, 20)

        explainer = shap.TreeExplainer(self.model, model_output=1)
        shap_values = explainer.shap_values(self.X)
        shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
        plt.title(f'{self.identifier} NGBoost Variance SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                save_path = self.save_path + "/singleSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_ngboost_variance_shap_aggregated_beeswarm_{iter_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_ngboost_variance_shap_aggregated_beeswarm.png')
            plt.close()
        return shap_values
    
    def compute_uncertainties(self, mode=["nig", "ensemble"], X: pd.DataFrame = None,  members: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute aleatoric and epistemic uncertainty using an ensemble of NGBoost models.

        Parameters:
            X (pd.DataFrame): Input data for prediction.
            members (int): Number of ensemble members.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - mean_prediction: Ensemble mean predictions
                - aleatoric_uncertainty: Mean of predicted variances across ensemble
                - epistemic_uncertainty: Variance of predicted means across ensemble
        """
        if mode not in ["nig", "ensemble"]:
            raise ValueError("Invalid mode. Choose either 'nig' or 'ensemble'.")
        elif mode == "nig":
            mean_prediction, aleatoric_uncertainty, epistemic_uncertainty = self.model.pred_uncertainty(X).items()
        else:
            mean_predictions = []
            variance_predictions = []
            members = 2
            for i in range(members):
                # Shuffle data with different random seed
                shuffled_df = self.X.copy()
                shuffled_df['target'] = self.y
                shuffled_df = shuffled_df.sample(frac=1.0, random_state=i).reset_index(drop=True)
                X_shuffled = shuffled_df[self.feature_selection['features']]
                y_shuffled = shuffled_df['target']

                # Fit a new NGBoost model with different seed
                ngb_model = NGBRegressor(**self.ngb_hparams, random_state=i)
                ngb_model.fit(X_shuffled, y_shuffled)

                dist = ngb_model.pred_dist(X)
                mean_pred = dist.loc  # mean
                var_pred = dist.scale**2  # variance

                mean_predictions.append(mean_pred)
                variance_predictions.append(var_pred)

            mean_predictions = np.array(mean_predictions)  # shape (members, n_samples)
            variance_predictions = np.array(variance_predictions)

            # Compute uncertainties
            mean_prediction = np.mean(mean_predictions, axis=0)
            aleatoric_uncertainty = np.mean(variance_predictions, axis=0)
            epistemic_uncertainty = np.var(mean_predictions, axis=0)

        return mean_prediction, aleatoric_uncertainty, epistemic_uncertainty

    

    def calibration_analysis(self):
        """
        Generate PIT histogram, QQ-plot, quantile‐calibration diagram, and CRPS.
        Assumes that after LOOCV you have stored in self.metrics:
          - 'pred_dist': a tuple/ list of 4 numpy arrays (mu, lam, alpha, beta), each of length n_samples
          - 'y_test'   : the true y's for those held-out samples, length n_samples
        """
        # prepare output folder
        save_path = os.path.join(self.save_path, "calibration")
        os.makedirs(save_path, exist_ok=True)

        # pull out parameters & truths
        mu_arr, lam_arr, alpha_arr, beta_arr = self.metrics['pred_dist']
        y_test = np.asarray(self.metrics['y_test'])
        n = len(y_test)

        # reconstruct a frozen Student-t for each case
        #   ν = 2α,   Ω = 2β(1+λ),   scale = sqrt(Ω / (λ ν))
        nu    = 2 * alpha_arr
        Omega = 2 * beta_arr * (1 + lam_arr)
        scale = np.sqrt(Omega / (lam_arr * nu))

        dists = [ st.t(df=nu[i], loc=mu_arr[i], scale=scale[i])
                  for i in range(n) ]

        # 1) PIT values
        pit = np.array([d.cdf(y) for d,y in zip(dists, y_test)])

        # --- PIT histogram ---
        plt.figure()
        plt.hist(pit, bins=20, range=(0,1), edgecolor='k', alpha=0.7)
        plt.axhline(n/20, color='r', linestyle='--', label='ideal uniform')
        plt.title('PIT Histogram')
        plt.xlabel('PIT')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f'{save_path}/pit_hist.png')
        plt.close()

        # --- PIT QQ-plot ---
        sorted_pit = np.sort(pit)
        uniform_q  = np.linspace(0,1,n)
        plt.figure()
        plt.plot(uniform_q, sorted_pit, marker='.', linestyle='none')
        plt.plot([0,1],[0,1], 'r--')
        plt.title('PIT QQ-Plot')
        plt.xlabel('Uniform Quantile')
        plt.ylabel('Empirical PIT Quantile')
        plt.savefig(f'{save_path}/pit_qq.png')
        plt.close()

        # 2) Quantile Calibration Diagram
        qs  = np.linspace(0.05, 0.95, 19)
        obs = []
        for q in qs:
            # predicted q-quantile for each case
            yq = np.array([d.ppf(q) for d in dists])
            # fraction of truths ≤ that
            obs.append(np.mean(y_test <= yq))

        plt.figure()
        plt.plot(qs, obs, marker='o', linestyle='-')
        plt.plot([0,1],[0,1],'r--')
        plt.title('Quantile Calibration Plot')
        plt.xlabel('Nominal Quantile')
        plt.ylabel('Observed Fraction ≤ Predicted')
        plt.savefig(f'{save_path}/quantile_calib.png')
        plt.close()

        # 3) CRPS (draw 500 samples per predictive distribution)
        samples = np.stack([d.sample(500) for d in dists], axis=1)
        # samples.shape == (500, n)
        crps_vals = crps_ensemble(y_test, samples.T)
        print(f'Average CRPS: {crps_vals.mean():.4f}')

        return {
            'pit':          pit,
            'quantile_cal': (qs, np.array(obs)),
            'crps':         crps_vals,
        }



class TabPFNRegression(BaseRegressionModel):
    """ TabPFN Regression Model """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            target_name: str,
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
        self.model = TabPFNRegressor()
        self.model_name = "TabPFN Regression"

    def model_specific_preprocess(self, data_df: pd.DataFrame) -> Tuple:
        """ Preprocess the data for the TabPFN model """
        logging.info("Starting TabPFN model-specific preprocessing...")
        # Drop rows with missing values for features and target
        data_df = data_df.dropna(subset=self.feature_selection['features'] + [self.feature_selection['target']])
        X = data_df[self.feature_selection['features']]
        y = data_df[self.feature_selection['target']]
        # Ensure features are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        logging.info("Finished TabPFN model-specific preprocessing.")
        return X, y

    def feature_importance(self, top_n: int = 10, batch_size: int = 10, save_results=True) -> Tuple:
        """ Compute feature importance for TabPFN using LOCO and SHAP evaluations """
        logging.info("Starting feature importance evaluation for TabPFN Regression.")
        X_train, X_test, y_train, y_test = self.train_split

        def loco_importances(X_train, y_test):
            logging.info("Starting LOCO importance evaluation...")
            importances = {}
            for i, feature in enumerate(X_train.columns):
                logging.info(f"Evaluating LOCO importance for feature {i + 1}/{len(X_train.columns)}: {feature}")
                # Remove the feature from the training and test sets
                X_train_loco = X_train.drop(columns=[feature])
                X_test_loco = X_test.drop(columns=[feature])
                # Fit the model on the modified data and predict
                self.model.fit(X_train_loco, y_train)
                loco_pred = self.model.predict(X_test_loco)
                # Compute the MSE and record the relative change
                loco_mse = mean_squared_error(y_test, loco_pred)
                importances[feature] = abs(loco_mse - self.metrics['mse']) / self.metrics['mse']
                if (i + 1) % 10 == 0 or (i + 1) == len(X_train.columns):
                    logging.info(f"Progress: {i + 1}/{len(X_train.columns)} features evaluated.")
            logging.info("Finished LOCO importance evaluation.")
            return importances

        def shap_importances(batch_size):
            logging.info("Starting SHAP importance evaluation for TabPFN...")
            shap.initjs()
            # Use a small background sample for the KernelExplainer
            background = shap.sample(self.X, 20)
            explainer = shap.KernelExplainer(
                lambda x: self.model.predict(pd.DataFrame(x, columns=self.X.columns)),
                self.X, background
            )
            num_samples = len(self.X)
            all_shap_values = []
            # Process the data in batches
            for i in range(0, num_samples, batch_size):
                batch = self.X.iloc[i:i+batch_size]
                shap_values_batch = explainer.shap_values(batch, nsamples=300)
                all_shap_values.append(shap_values_batch)
            shap_values = np.concatenate(all_shap_values, axis=0)
            logging.info(f"SHAP values shape: {shap_values.shape}")
            # Plot aggregated SHAP values (beeswarm and bar plots)
            shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=top_n)
            plt.title(f'{self.identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
            if save_results:
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{self.save_path}/{self.identifier}_tabpfn_shap_aggregated_beeswarm.png')
                plt.close()
                shap.summary_plot(shap_values, self.X, plot_type="bar", show=False)
                plt.savefig(f'{self.save_path}/{self.identifier}_tabpfn_shap_aggregated_bar.png')
                plt.close()
            logging.info("Finished SHAP importance evaluation for TabPFN.")
            return shap_values

        logging.info("Evaluating SHAP feature importances for TabPFN...")
        shap_attributions = shap_importances(batch_size)
        logging.info("SHAP importance evaluation completed for TabPFN.")
    
        logging.info("Evaluating LOCO feature importances for TabPFN...")
        loco_attributions = loco_importances(X_train, y_test)
        logging.info("LOCO importance evaluation completed for TabPFN.")
    
        if save_results:
            logging.info(f"Saving SHAP attributions to {self.save_path}/{self.identifier}_tabpfn_mean_shap_values.npy")
            np.save(f'{self.save_path}/{self.identifier}_tabpfn_mean_shap_values.npy', shap_attributions)
    
        return loco_attributions, shap_attributions
