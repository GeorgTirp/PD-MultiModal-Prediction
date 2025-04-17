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

    def model_specific_preprocess(self, data_df: pd.DataFrame) -> Tuple:
        """ Preprocess the data for the model """
        logging.info("Starting model-specific preprocessing...")
        # Drop rows with missing values for features and target
        data_df = data_df.dropna(subset=self.feature_selection['features'] + [self.feature_selection['target']])
        X = data_df[self.feature_selection['features']]
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

    def evaluate(self, folds=10, get_shap=True, tune=False, tune_folds=10, nested=False) -> Dict:
        if nested==True:
            return self.nested_eval(folds, get_shap, tune, tune_folds)
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

    def nested_eval(self, folds=10, get_shap=True, tune=False, tune_folds=10) -> Dict:
        """ Evaluate the model using cross-validation """
        logging.info("Starting model evaluation...")
        if folds == -1:
            kf = LeaveOneOut()
        else:
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        if tune and self.param_grid is None:
                raise ValueError("When calling tune=True, a param_grid has to be passed when initializing the model.")
        preds = []
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
            if isinstance(self, NGBoostRegressionModel):
                pred_dist = self.model.pred_dist(X_val_kf)._params
            preds.append(pred)
            pred_dists.append(pred_dist)
            y_vals.append(y_val_kf)
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

        pred_dists = np.hstack(pred_dists)
        preds = np.concatenate(preds)
        y_vals = np.concatenate(y_vals)
        r2, p = pearsonr(y_vals, preds)
        mse = mean_squared_error(y_vals, preds)

        metrics = {
            'mse': mse,
            'r2': r2,
            'p_value': p,
            'y_pred': preds,
            'y_test': y_vals,
            'pred_dist': pred_dists
        }
        self.metrics = metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{self.save_path}/{self.identifier}_metrics.csv', index=False)

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

        logging.info("Finished model evaluation.")
        return metrics
    
    def feature_ablation(self) -> Dict:
        pass
    
    def feature_importance(self, top_n: int = 10, batch_size=None, save_results=True) -> Dict:
        """ To be implemented in the subclass """
        raise NotImplementedError("Subclasses must implement feature_importance method")

    def compute_unertainties(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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

        explainer = shap.TreeExplainer(self.model, background, model_output=0)
        shap_values = explainer.shap_values(self.X)
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

        explainer = shap.TreeExplainer(self.model, background, model_output=1)
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
    
    def compute_unertainties(self, X: pd.DataFrame, y: pd.DataFrame, members: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """ Compute uncertainties using the NGBoost model """
        


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
