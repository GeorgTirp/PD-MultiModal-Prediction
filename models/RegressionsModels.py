import os

os.environ["OMP_NUM_THREADS"] = "1"            # Limits OpenMP (used by NumPy, Numba, etc.)
os.environ["OPENBLAS_NUM_THREADS"] = "1"       # OpenBLAS (used by NumPy)
os.environ["MKL_NUM_THREADS"] = "1"            # MKL (used by scikit-learn on Intel Macs)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"     # Apple's Accelerate framework
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
import torch
torch.set_num_threads(1)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, LeaveOneOut, ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import shap
import logging
from tqdm import tqdm
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
import seaborn as sns
from IPython.utils import io
import torch
from ngboost import NGBRegressor
from ngboost.distns import Normal
from faster_evidential_boost import NIGLogScore
from sklearn.tree import DecisionTreeRegressor
from faster_evidential_boost import NormalInverseGamma
import scipy.stats as st
from properscoring import crps_ensemble
from scipy.stats import norm
import pickle
#from pycens.censored_regression import Tobit as PycensTobit
from statsmodels.base.model import GenericLikelihoodModel
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.basic_variant import BasicVariantGenerator

search_alg = BasicVariantGenerator()

from hyperopt import hp
#matplotlib.use('TkAgg') 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
class BaseRegressionModel:
    """Base class for supervised regression models using scikit-learn and SHAP.

    This class handles shared logic for preparing data, training models, evaluating
    performance, saving results, and computing feature importance. Subclasses are expected
    to define the model instance and may override feature attribution logic.

    Attributes:
        data_df (pd.DataFrame): Full input dataset.
        feature_selection (dict): Dictionary containing selected features, e.g. {'features': [...]}.
        target_name (str): Name of the target variable column in `data_df`.
        test_split_size (float): Fraction of data to use as the test set.
        save_path (str): Optional directory where results, models, and plots are saved.
        identifier (str): Optional experiment identifier for naming saved files.
        top_n (int): Number of top features to retain when computing feature importances.
        model (sklearn.BaseEstimator): Regression model (defined by subclass).
        model_name (str): Human-readable name of the model (defined by subclass).
        X (pd.DataFrame): Training features after split.
        y (pd.Series): Training target after split.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        importances (dict): Dictionary of feature importance scores (populated after evaluation).
    """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            target_name: str,
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10):
        """
        Initialize the regression model framework with dataset, feature selection, and settings.

        Args:
            data_df (pd.DataFrame): Input dataset containing features and target.
            feature_selection (dict): Dictionary with keys 'features' and 'target' specifying columns.
            target_name (str): Name of the target variable.
            test_split_size (float, optional): Fraction of data to hold out for testing. Defaults to 0.2.
            save_path (str, optional): Path to save model outputs and results. Defaults to None.
            identifier (str, optional): Unique identifier for this model run. Defaults to None.
            top_n (int, optional): Number of top features to display in SHAP plots. Defaults to 10.
        """
        
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
        """
        Preprocess data specific to model requirements, including feature extraction and optional ceiling adjustments.

        Args:
            data_df (pd.DataFrame): Original input data.
            ceiling (list, optional): List of ceiling transformations to apply (e.g., ['BDI', 'MoCA']). Defaults to ["BDI", "MoCA"].

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Tuple containing preprocessed features (X) and target variable (y).
        """

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
        """
        Train the model on the training dataset split.
        """
        logging.info(f"Starting {self.model_name} model training...")
        X_train, X_test, y_train, y_test = self.train_split
        self.model.fit(X_train, y_train)
        logging.info(f"Finished {self.model_name} model training.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from a trained model.

        Args:
            X (pd.DataFrame): Feature data.

        Returns:
            np.ndarray: Predicted target values.
        """
        logging.info("Starting prediction...")
        pred = self.model.predict(X)
        logging.info("Finished prediction.")
        return pred

    def evaluate(self, folds=10, get_shap=True, tune=False, tune_folds=10, nested=False, uncertainty=False) -> Dict:
        """
        Evaluate model performance using cross-validation, with options for tuning and SHAP explanation.

        Args:
            folds (int, optional): Number of cross-validation folds. Defaults to 10.
            get_shap (bool, optional): Whether to compute SHAP feature importances. Defaults to True.
            tune (bool, optional): Whether to perform hyperparameter tuning. Defaults to False.
            tune_folds (int, optional): Number of folds for tuning process. Defaults to 10.
            nested (bool, optional): Use nested cross-validation if True. Defaults to False.
            uncertainty (bool, optional): Whether to compute uncertainty estimates (if supported). Defaults to False.

        Returns:
            dict: Dictionary of evaluation metrics such as MSE, R², p-values, and SHAP values.
        """
        if nested==True:
            return self.nested_eval(folds, get_shap, tune, tune_folds, uncertainty=uncertainty)
        else:
            return self.sequential_eval(folds, get_shap, tune, tune_folds)
        
    def sequential_eval(
        self, 
        folds: int = 10, 
        get_shap: bool = True, 
        tune: bool = False, 
        tune_folds: int = 10,
        uncertainty: bool = False,
        ablation_idx: int = None
        ) -> Dict:
        """
        Perform sequential cross-validation evaluation with optional tuning, SHAP, uncertainty estimation, and ablation.
        Tuning is performed once before cross-validation begins.

        Args:
            folds (int): Number of outer CV folds (-1 for Leave-One-Out).
            get_shap (bool): Whether to compute SHAP values.
            tune (bool): Whether to perform hyperparameter tuning before CV.
            tune_folds (int): Number of inner CV folds for tuning.
            uncertainty (bool): Whether to estimate predictive uncertainty.
            ablation_idx (int or None): Index for feature ablation tracking.

        Returns:
            dict: Evaluation metrics including MSE, R², SHAP, and optional uncertainty.
        """
        logging.info("Starting model evaluation...")

        if tune:
            if self.param_grid is None:
                raise ValueError("When calling tune=True, a param_grid has to be passed when initializing the model.")
            self.tune_hparams(self.X, self.y, self.param_grid, tune_folds)

        if folds == -1:
            kf = LeaveOneOut()
        else:
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        preds = []
        epistemics = []
        aleatorics = []
        pred_dists = []
        y_vals = []
        all_shap_values = []
        all_shap_mean = []
        all_shap_variance = []

        for train_index, val_index in tqdm(kf.split(self.X), total=kf.get_n_splits(self.X), desc="Cross-validation", leave=True):
            X_train_kf, X_val_kf = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_kf, y_val_kf = self.y.iloc[train_index], self.y.iloc[val_index]

            self.model.fit(X_train_kf, y_train_kf)
            pred = self.model.predict(X_val_kf)
            preds.append(pred)
            y_vals.append(y_val_kf)

            if uncertainty:
                _, epistemic, aleatoric = self.compute_uncertainties(mode="nig", X=X_val_kf)
                epistemics.append(epistemic)
                aleatorics.append(aleatoric)

            if isinstance(self, NGBoostRegressionModel):
                pred_dist = self.model.pred_dist(X_val_kf).params
                pred_dist = np.column_stack([pred_dist[key] for key in pred_dist.keys()])
                pred_dists.append(pred_dist)

            if get_shap:
                with io.capture_output():
                    if ablation_idx is not None:
                        val_index = None
                    if isinstance(self, NGBoostRegressionModel):
                        shap_values_mean = self.feature_importance_mean(top_n=-1, save_results=True, iter_idx=val_index)
                        shap_values_variance, _, _ = self.feature_importance_variance(top_n=-1, save_results=True, iter_idx=val_index)
                        all_shap_mean.append(shap_values_mean)
                        all_shap_variance.append(shap_values_variance)
                    else:
                        shap_values = self.feature_importance(top_n=-1, save_results=True, iter_idx=val_index)
                        all_shap_values.append(shap_values)

        preds = np.concatenate(preds)
        y_vals = np.concatenate(y_vals)
        r2, p = pearsonr(y_vals, preds)
        mse = mean_squared_error(y_vals, preds)

        if uncertainty:
            epistemic_uncertainty = np.concatenate(epistemics)
            aleatoric_uncertainty = np.concatenate(aleatorics)

        if get_shap:
            if ablation_idx is not None:
                save_path = f'{self.save_path}/ablation/'
                os.makedirs(save_path, exist_ok=True)
                save_path = f'{save_path}{self.identifier}_{self.target_name}_{ablation_idx}'
            else:
                save_path = f'{self.save_path}/{self.identifier}_{self.target_name}'

            if isinstance(self, NGBoostRegressionModel):
                all_shap_mean_array = np.stack(all_shap_mean, axis=0)
                all_shap_variance_array = np.stack(all_shap_variance, axis=0)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                variance_shap_values = np.mean(all_shap_variance_array, axis=0)

                np.save(f'{save_path}_mean_shap_values.npy', mean_shap_values)
                np.save(f'{save_path}_predicitve_uncertainty_shap_values.npy', variance_shap_values)
                np.save(f'{save_path}_all_shap_values(variance).npy', all_shap_variance_array)

                shap.summary_plot(mean_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier} Summary Plot (Aggregated - Mean)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_mean_shap_aggregated.png')
                plt.close()

                shap.summary_plot(variance_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier} Summary Plot (Aggregated - Variance)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_preditive_uncertainty_shap_aggregated.png')
                plt.close()
            else:
                all_shap_mean_array = np.stack(all_shap_values, axis=0)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                np.save(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_values.npy', mean_shap_values)
                shap.summary_plot(mean_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier} Summary Plot (Aggregated)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_shap_aggregated_beeswarm.png')
                with open(f'{save_path}_shap_explanations.pkl', 'wb') as fp:
                    pickle.dump(mean_shap_values, fp)
                plt.close()
            np.save(f'{save_path}_all_shap_values(mu).npy', all_shap_mean_array)

            feature_importances = np.mean(np.abs(mean_shap_values), axis=0)
            feature_importance_dict = dict(zip(self.X.columns, feature_importances))

        metrics = {
            'mse': mse,
            'r2': r2,
            'p_value': p,
            'y_pred': preds,
            'y_test': y_vals,
            'pred_dist': np.vstack(pred_dists) if isinstance(self, NGBoostRegressionModel) else None,
            'epistemic': epistemic_uncertainty if uncertainty else None,
            'aleatoric': aleatoric_uncertainty if uncertainty else None,
            'feature_importance': feature_importances if get_shap else None
        }

        self.metrics = metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{self.save_path}/{self.identifier}_metrics.csv', index=False)

        model_save_path = f'{self.save_path}/{self.identifier}_{ablation_idx}_trained_model.pkl' if ablation_idx is not None else f'{self.save_path}/{self.identifier}_trained_model.pkl'
        with open(model_save_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        logging.info(f"Trained model saved to {model_save_path}.")
        logging.info("Finished model evaluation.")

        return metrics


    def nested_eval(self, folds=10, get_shap=True, tune=False, tune_folds=10, uncertainty=False, ablation_idx=None) -> Dict:
        """
        Perform nested cross-validation evaluation with optional tuning, SHAP, uncertainty estimation, and ablation.

        Args:
            folds (int, optional): Number of outer CV folds (-1 for Leave-One-Out). Defaults to 10.
            get_shap (bool, optional): Whether to compute SHAP values. Defaults to True.
            tune (bool, optional): Whether to perform hyperparameter tuning. Defaults to False.
            tune_folds (int, optional): Number of inner CV folds for tuning. Defaults to 10.
            uncertainty (bool, optional): Whether to estimate predictive uncertainty. Defaults to False.
            ablation_idx (int or None, optional): Index for feature ablation tracking. Defaults to None.

        Returns:
            dict: Dictionary of performance metrics, uncertainties, SHAP values, and predictions.
        """
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
        iter_idx = 0
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
                #if
                pred_dist = self.model.pred_dist(X_val_kf).params
                pred_dist = np.column_stack([pred_dist[key] for key in pred_dist.keys()])
                pred_dists.append(pred_dist)
            
            # Get eplanations
            if get_shap:             
                 # Compute SHAP values on the whole dataset per fold
                with io.capture_output():
                    if ablation_idx is not None:
                        val_index = None
                    if isinstance(self, NGBoostRegressionModel):
                        shap_values_mean = self.feature_importance_mean(
                            top_n=-1, save_results=True,  
                            iter_idx=iter_idx)
                        if self.prob_func == NormalInverseGamma:
                            shap_values_variance, _, _ = self.feature_importance_variance(
                                mode="nig",
                                top_n=-1, 
                                save_results=True, 
                                iter_idx=iter_idx)
                        elif self.prob_func == Normal:
                            shap_values_variance = self.feature_importance_variance(
                                mode="normal",
                                top_n=-1, 
                                save_results=True, 
                                iter_idx=iter_idx)
                        all_shap_mean.append(shap_values_mean)
                        all_shap_variance.append(shap_values_variance)
                    else:
                        shap_values = self.feature_importance(
                            top_n=-1, 
                            save_results=True, 
                            iter_idx=iter_idx)
                        all_shap_values.append(shap_values) 
            iter_idx += 1
        if isinstance(self, NGBoostRegressionModel):
            pred_dists = np.vstack(pred_dists)
        preds = np.concatenate(preds)
        y_vals = np.concatenate(y_vals)
        r2, p = pearsonr(y_vals, preds)
        
        if uncertainty == True:
            epistemic_uncertainty = np.concatenate(epistemics)
            aleatoric_uncertainty = np.concatenate(aleatorics)
        mse = mean_squared_error(y_vals, preds)

        if ablation_idx is not None:
            save_path = f'{self.save_path}/ablation/ablation_step[{ablation_idx}]/'
            os.makedirs(save_path, exist_ok=True)
            save_path = f'{save_path}{self.identifier}_{self.target_name}'
        else:
            save_path = f'{self.save_path}/{self.identifier}_{self.target_name}'

        
        if get_shap:
            
            
            if isinstance(self, NGBoostRegressionModel):
                all_shap_mean_array = np.stack(all_shap_mean, axis=0)
                all_shap_variance_array = np.stack(all_shap_variance, axis=0)
                # Average over the folds to get an aggregated array of shape (n_samples, n_features)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                variance_shap_values = np.mean(all_shap_variance_array, axis=0)

                # Save SHAP values to a file
                np.save(f'{save_path}_mean_shap_values.npy', mean_shap_values)
                np.save(f'{save_path}_predicitve_uncertainty_shap_values.npy', variance_shap_values)
                np.save(f'{save_path}_all_shap_values(variance).npy', all_shap_variance_array)
        
                # Plot for mean SHAP values
                shap.summary_plot(mean_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier} Summary Plot (Aggregated - Mean)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_mean_shap_aggregated.png')
                plt.close()
                
                shap.summary_plot(variance_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier} Summary Plot (Aggregated - Variance)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                if self.prob_func == NormalInverseGamma:
                    plt.savefig(f'{save_path}_preditive_uncertainty_shap_aggregated.png')
                elif self.prob_func == Normal:
                    plt.savefig(f'{save_path}_std_shap_aggregated.png')
                plt.close()
            else:
                all_shap_mean_array = np.stack(all_shap_values, axis=0)
                # Average over the folds to get an aggregated array of shape (n_samples, n_features)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                np.save(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_values.npy', mean_shap_values)
                shap.summary_plot(mean_shap_values , features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier}  Summary Plot (Aggregated)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_shap_aggregated_beeswarm.png')
                with open(f'{save_path}_shap_explanations.pkl', 'wb') as fp:
                    pickle.dump(mean_shap_values, fp)
                plt.close()
            np.save(f'{save_path}_all_shap_values(mu).npy', all_shap_mean_array)
            
            
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
        metrics_df.to_csv(f'{save_path}_metrics.csv', index=False)
        
        # Save the trained model to a file
        model_save_path = f'{save_path}_trained_model.pkl'
        with open(model_save_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        
        logging.info(f"Trained model saved to {model_save_path}.")
        logging.info("Finished model evaluation.")
        return metrics
    
    def feature_ablation(self, folds: int = -1, get_shap=True, tune=False, tune_folds: int = 10) -> Dict:
        """
        Perform iterative feature ablation analysis using nested cross-validation.

        Returns:
            Tuple[list, list, list]: Lists of R² scores, p-values, and removed feature names after each ablation step.
        """
        r2s = []
        p_values = []
        removals = []
        number_of_features = len(self.feature_selection['features'])
        for i in range(number_of_features):
            save_path = f'{self.save_path}/ablation/'
            metrics = self.nested_eval(folds=folds, get_shap=get_shap, tune=tune, tune_folds=tune_folds, ablation_idx=i)
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f'{save_path}/ablation_step[{i}]/{self.identifier}_{self.target_name}_metrics.csv', index=False)
            r2s.append(metrics['r2'])
            p_values.append(metrics['p_value'])
            importance = metrics['feature_importance']
            importance_indices = np.argsort(importance)
            least_important_feature = self.feature_selection['features'][importance_indices[0]]
            self.X = self.X.drop(columns=[least_important_feature])
            self.y = self.y.drop(columns=[least_important_feature])
            self.feature_selection['features'].remove(least_important_feature)
            removals.append(least_important_feature)
            if isinstance(self, NGBoostRegressionModel):
                self.calibration_analysis(ablation_idx=i)
        
        # Define the custom color palette from your image
        custom_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#525252"]
                # Set Seaborn style, context, and custom palette
        sns.set_theme(style="whitegrid", context="paper")
        sns.set_palette(custom_palette)
        path = f'{save_path}{self.identifier}_{self.target_name}_feature_ablation.png'
                # Read in the CSV
        
        # Save the removals list as a CSV file
        removals_df = pd.DataFrame({'Removed_Features': removals})
        removals_df.to_csv(f'{save_path}{self.identifier}_ablation_history.csv', index=False)
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
        return r2s, p_values, removals
            

    
    def feature_importance(self, top_n: int = 10, batch_size=None, save_results=True) -> Dict:
        """
        Compute feature importance scores (e.g., SHAP values). Must be implemented by subclasses.

        Args:
            top_n (int, optional): Number of top features to return. Defaults to 10.
            batch_size (int or None, optional): Batch size for computation. Defaults to None.
            save_results (bool, optional): Whether to save the importance results. Defaults to True.

        Returns:
            dict: Feature importance scores.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement feature_importance method")

    def compute_uncertainties(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute predictive uncertainties (epistemic and aleatoric) for the input features.

        Args:
            X (pd.DataFrame): Input features for uncertainty estimation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Epistemic and aleatoric uncertainty arrays.

        Raises:
            NotImplementedError: If uncertainty computation is not implemented for the model.
        """
        raise NotImplementedError("Uncertainty computation not implemented for this model.")
    
    
        

    def plot(self, title, modality='') -> None:
        """
        Generate a scatter plot of predicted vs. actual target values with regression line and confidence intervals.

        Args:
            title (str): Title of the plot.
            modality (str, optional): Modality string to label axes (e.g., "MRI"). Defaults to empty string.
        """
        logging.info("Starting plot generation.")
        
        colors = {
            "deterioration": "04E762",
            "improvement": "FF5714",
            "line": "grey",
            "scatter": "grey",
            "ideal_line": "black",
        }

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

        ax = plt.gca()
        # get current y‐limits
        min_y, max_y = ax.get_ylim()

        # shade below y=0 (improvement)
        ax.axhspan(min_y, 0,
                   color="#" + colors["improvement"],
                   alpha=0.08, zorder=0)
        # shade above y=0 (deterioration)
        ax.axhspan(0, max_y,
                   color="#" + colors["deterioration"],
                   alpha=0.08, zorder=0)
        # Label axes and set title
        plt.xlabel(f'Actual {modality} {self.target_name}', fontsize=12)
        plt.ylabel(f'Predicted {modality} {self.target_name}', fontsize=12)
        plt.title(title + "  N=" + str(len(self.y)), fontsize=14)

        # Show grid and ensure everything fits nicely
        plt.grid(False)
        sns.set_context("paper")
        # Optionally choose a style you like
        sns.despine()
        plt.tight_layout()
        # Save and close
        plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_actual_vs_predicted.png')
        plt.close()

        # Log info (optional)
        logging.info("Plot saved to %s/%s_%s_actual_vs_predicted.png", 
                 self.save_path, self.identifier, self.target_name)



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
            top_n: int = 10):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
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
            elif ablation_idx is not None:
                save_path = self.save_path + "/ablationSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm_ablation_{ablation_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm.png')
            plt.close()
            
        if iter_idx is None:
            logging.info("Finished feature importance evaluation for Linear Regression.")
        return shap_values


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
            elif ablation_idx is not None:
                save_path = self.save_path + "/ablationSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm_ablation_{ablation_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_rf_shap_aggregated_beeswarm.png')
            plt.close()
            
        if iter_idx is None:
            logging.info("Finished feature importance evaluation for Random Forest.")
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
        self.model_name = "NGBoost"
        self.prob_func = ngb_hparams["Dist"]
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
        #ss = ShuffleSplit(n_splits=50, test_size= 0.01, random_state=7)
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
        score_params = {k.replace("Score__", ""): v for k, v in best_params.items() if k.startswith("Score__")}
        NIGLogScore.set_params(**score_params)

        # Remove score params from best_params before passing to NGBoost
        clean_params = {k: v for k, v in best_params.items() if not k.startswith("Score__") and not k.startswith("Base__")}

        # Extract and create base learner if needed
        base_params = {k.replace("Base__", ""): v for k, v in best_params.items() if k.startswith("Base__")}
        if base_params:
            base_learner = DecisionTreeRegressor(**base_params)
        else:
            base_learner = DecisionTreeRegressor(max_depth=3)

        # Insert base learner into NGBoost params
        clean_params['Base'] = base_learner

        # Create NGBoost model with clean params
        self.model = NGBRegressor(Dist=NormalInverseGamma, Score=NIGLogScore, verbose=False, **clean_params)
        self.model.fit(X, y)
        # Force an immediate fit on the tuning data to initialize internal parameters.
        self.model.fit(X, y)
        self.model_name += " (Tuned)"
        print(f"Best parameters found: {best_params}")
        return best_params

    def tune_hparams_ray(self, X, y, param_grid: dict, folds=5, algo: str ="BayesOpt") -> dict:
        """Tune hyperparameters using Ray Tune with cross-validation."""

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/folds, random_state=42)

        def train_ngboost(config):
            # Extract base learner parameters
            if algo == "BayesOpt":
                config['n_estimators'] = int(round(config['n_estimators']))
                config['Base__max_depth'] = int(round(config['Base__max_depth']))

            base_params = {k.replace("Base__", ""): v for k, v in config.items() if k.startswith("Base__")}
            base_learner = DecisionTreeRegressor(**base_params) if base_params else DecisionTreeRegressor(max_depth=3)

            # Extract score parameters
            score_params = {k.replace("Score__", ""): v for k, v in config.items() if k.startswith("Score__")}
            NIGLogScore.set_params(**score_params)

            # Extract NGBoost parameters
            ngb_params = {k: v for k, v in config.items() if not (k.startswith("Base__") or k.startswith("Score__"))}
            ngb_params['Base'] = base_learner

            # Initialize and train NGBoost model
            model = NGBRegressor(Dist=NormalInverseGamma, Score=NIGLogScore, verbose=False, **ngb_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            tune.report({"mse": mse})

        # Define the search space
        search_space = {}
        for param, values in param_grid.items():
            if isinstance(values, list):
                search_space[param] = tune.choice(values)
            else:
                search_space[param] = values  # Fixed value

        # Configure the scheduler for early stopping
        scheduler = ASHAScheduler(
            max_t=500,
            grace_period=100,
            reduction_factor=2
        )
        def convert_to_hyperopt_space(param_grid):
                space = {}
                for k, v in param_grid.items():
                    if isinstance(v, list):
                        space[k] = hp.choice(k, v)
                    else:
                        space[k] = v
                return space
        
        def convert_to_bayesopt_space(param_grid):
            space = {}
            for k, v in param_grid.items():
                if isinstance(v, list):
                    if all(isinstance(i, int) for i in v):
                        # Treat integer parameters as continuous and use a uniform distribution
                        space[k] = tune.uniform(min(v), max(v))
                    elif all(isinstance(i, float) for i in v):
                        # Use uniform distribution for float parameters
                        space[k] = tune.uniform(min(v), max(v))
                    else:
                        # Handle categorical parameters
                        space[k] = tune.choice(v)
                else:
                    # Handle fixed values
                    space[k] = v
            return space
    

        if algo == "HyperOpt":
            search_space = convert_to_hyperopt_space(param_grid)
            algo = HyperOptSearch(search_space, metric="mse", mode="min")

        elif algo == "BayesOpt":
            search_space = convert_to_bayesopt_space(param_grid)
            algo = BayesOptSearch(search_space, metric="mse", mode="min")
        
        elif algo == "Optuna":
            algo = OptunaSearch(search_space, metric="mse", mode="min")

        else:
            algo = BasicVariantGenerator(search_space, metric="mse", mode="min")
        
        # Execute hyperparameter tuning
        tuner = tune.Tuner(
            train_ngboost,
            #param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="mse",
                mode="min",
                scheduler=scheduler,
                search_alg= algo,
                num_samples=200  # Adjust as needed
            )
        )
        results = tuner.fit()

        # Retrieve the best hyperparameters
        best_result = results.get_best_result()
        best_params = best_result.config

        # Separate and apply score parameters
        score_params = {k.replace("Score__", ""): v for k, v in best_params.items() if k.startswith("Score__")}
        NIGLogScore.set_params(**score_params)

        # Separate and apply base learner parameters
        base_params = {k.replace("Base__", ""): v for k, v in best_params.items() if k.startswith("Base__")}
        base_learner = DecisionTreeRegressor(**base_params) if base_params else DecisionTreeRegressor(max_depth=3)

        # Prepare NGBoost parameters
        ngb_params = {k: v for k, v in best_params.items() if not (k.startswith("Base__") or k.startswith("Score__"))}
        ngb_params['Base'] = base_learner

        # Initialize and train the final NGBoost model
        self.model = NGBRegressor(Dist=NormalInverseGamma, Score=NIGLogScore, verbose=False, **ngb_params)
        self.model.fit(X, y)
        self.model_name += " (Tuned)"
        print(f"Best parameters found: {best_params}")
        return best_params

    def feature_importance_mean(self, top_n: int = None, batch_size: int = 10, save_results: bool = True, iter_idx=None, ablation_idx=None) -> Dict:
        """ Compute feature importance for the predicted mean using SHAP KernelExplainer. """
        shap.initjs()

        explainer = shap.TreeExplainer(self.model, model_output=0)
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

    def feature_importance_variance(
            self, 
            mode = "nig",
            top_n: int = None, 
            batch_size: int = 10, 
            save_results: bool = True, 
            iter_idx=None, 
            ablation_idx=None) -> Dict:
        """ Compute feature importance for the predicted variance using SHAP KernelExplainer. """
        
        if mode == "nig":
            pred_dist = self.model.pred_dist(self.X).params
            pred_dist = np.column_stack([pred_dist[key] for key in pred_dist.keys()]).T  # shape = (n_samples, 4)
            lam_vals   =  pred_dist[1]                # λ: precision
            alpha_vals =  pred_dist[2]            # α: shape
            beta_vals  =  pred_dist[3]             # β: rate

            # 2) Compute predictive, epistemic, and aleatoric variances
            var_pred = beta_vals / (lam_vals * (alpha_vals - 1))  # predictive
            var_epi  = beta_vals / (lam_vals * (alpha_vals - 1)**2)  # epistemic
            var_alea = beta_vals / (alpha_vals - 1)                  # aleatoric

            # 3) Derivatives for Taylor approx (∂var/∂param)
            # Predictive
            dpred_dbeta  = 1 / (lam_vals * (alpha_vals - 1))
            dpred_dalpha = -beta_vals / (lam_vals * (alpha_vals - 1)**2)
            dpred_dlam   = -beta_vals / (lam_vals**2 * (alpha_vals - 1))

            # Epistemic
            depi_dbeta  = 1 / (lam_vals * (alpha_vals - 1)**2)
            depi_dalpha = -2 * beta_vals / (lam_vals * (alpha_vals - 1)**3)
            depi_dlam   = -beta_vals / (lam_vals**2 * (alpha_vals - 1)**2)

            # Aleatoric
            dalea_dbeta  = 1 / (alpha_vals - 1)
            dalea_dalpha = -beta_vals / (alpha_vals - 1)**2
            # dalpha_dlam = 0 (lam not involved)

            # 4) Get SHAP values per param
            explainer = shap.TreeExplainer(self.model, model_output=1)
            sh_lam = explainer.shap_values(self.X)
            explainer = shap.TreeExplainer(self.model, model_output=2)
            sh_alpha = explainer.shap_values(self.X)
            explainer = shap.TreeExplainer(self.model, model_output=3)
            sh_beta = explainer.shap_values(self.X)


            # 5) Apply chain rule (broadcast derivatives over feature axis)
            shap_pred = (dpred_dbeta[:, None]  * sh_beta
                       + dpred_dalpha[:, None] * sh_alpha
                       + dpred_dlam[:, None]   * sh_lam)

            shap_epi = (depi_dbeta[:, None]  * sh_beta
                      + depi_dalpha[:, None] * sh_alpha
                      + depi_dlam[:, None]   * sh_lam)

            shap_alea = (dalea_dbeta[:, None]  * sh_beta
               + dalea_dalpha[:, None] * sh_alpha)

            shap.summary_plot(shap_pred, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
            plt.title(f'{self.identifier} NGBoost Variance SHAP Summary Plot (Aggregated)', fontsize=16)
            if save_results:
                plt.subplots_adjust(top=0.90)
                if iter_idx is not None:
                    save_path = self.save_path + "/singleSHAPs"
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(f'{save_path}/{self.identifier}_ngboost_predicitve_uncertainty_shap_aggregated{iter_idx}.png')
                elif ablation_idx is not None:
                    save_path = self.save_path + "/ablationSHAPs"
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_predicitve_uncertainty_shap_aggregated{ablation_idx}.png')
                else:
                    plt.savefig(f'{self.save_path}/{self.identifier}_predicitve_uncertainty_shap_aggregated.png')
                plt.close()
            return shap_pred, shap_epi, shap_alea
        
        elif mode == "normal":
            shap.initjs()
            explainer = shap.TreeExplainer(self.model, model_output=1)
            shap_values = explainer.shap_values(self.X, check_additivity=True)
            shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
            plt.title(f'{self.identifier} NGBoost Variance SHAP Summary Plot (Aggregated)', fontsize=16)
            if save_results:
                plt.subplots_adjust(top=0.90)
                if iter_idx is not None:
                    save_path = self.save_path + "/singleSHAPs"
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(f'{save_path}/{self.identifier}_std_shap_aggregated_beeswarm_{iter_idx}.png')
                elif ablation_idx is not None:
                    save_path = self.save_path + "/ablationSHAPs"
                    plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_std_shap_aggregated_beeswarm{ablation_idx}.png')
                else:
                    plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_std_shap_aggregated_beeswarm.png')
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

    

    def calibration_analysis(self, ablation_idx = None):
        """
        Generate PIT histogram, QQ-plot, quantile‐calibration diagram, and CRPS.
        Assumes that after LOOCV you have stored in self.metrics:
          - 'pred_dist': a tuple/ list of 4 numpy arrays (mu, lam, alpha, beta), each of length n_samples
          - 'y_test'   : the true y's for those held-out samples, length n_samples
        """
        # prepare output folder
        if ablation_idx is not None:
            save_path = f'{self.save_path}/ablation/ablation_step[{ablation_idx}]/calibration'
        else:
            save_path = f'{self.save_path}/calibration'
        
        os.makedirs(save_path, exist_ok=True)

        # pull out true values + predicted parameters
        pred_dist = self.metrics['pred_dist'].T
        y_test    = np.asarray(self.metrics['y_test'])
        n         = len(y_test)

        # decide which family
        if self.model.Dist == NormalInverseGamma:
            # NIG→Student-t
            mu_arr, lam_arr, alpha_arr, beta_arr = pred_dist[0], pred_dist[1], pred_dist[2], pred_dist[3]
            nu    = 2 * alpha_arr
            Omega = 2 * beta_arr * (1 + lam_arr)
            scale = np.sqrt(Omega / (lam_arr * nu))
            dists = [
                st.t(df=nu[i], loc=mu_arr[i], scale=scale[i])
                for i in range(n)
            ]

        elif self.model.Dist == Normal:
            # Gaussian
            mu_arr, sigma_arr = pred_dist[0], pred_dist[1]
            dists = [
                st.norm(loc=mu_arr[i], scale=sigma_arr[i])
                for i in range(n)
            ]

        else:
            raise ValueError(f"Expected 2 or 4 distribution parameters, got {len(pred_dist)}")

        # ---- 1) PIT ----
        pit = np.array([dist.cdf(y) for dist, y in zip(dists, y_test)])

        plt.figure()
        plt.hist(pit, bins=20, range=(0,1), edgecolor='k', alpha=0.7)
        plt.axhline(n/20, color='r', linestyle='--', label='ideal')
        plt.title('PIT Histogram')
        plt.xlabel('PIT')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f'{save_path}/pit_hist.png')
        plt.close()

        # ---- 2) PIT QQ plot ----
        sorted_pit = np.sort(pit)
        uniform_q  = np.linspace(0,1,n)
        plt.figure()
        plt.plot(uniform_q, sorted_pit, marker='.', linestyle='none')
        plt.plot([0,1],[0,1], 'r--')
        plt.title('PIT QQ‐Plot')
        plt.xlabel('Uniform Quantile')
        plt.ylabel('Empirical PIT Quantile')
        plt.savefig(f'{save_path}/pit_qq.png')
        plt.close()

        # ---- 3) Quantile‐Calibration ----
        qs  = np.linspace(0.05, 0.95, 19)
        obs = []
        for q in qs:
            yq = np.array([dist.ppf(q) for dist in dists])
            obs.append(np.mean(y_test <= yq))

        plt.figure()
        plt.plot(qs, obs, marker='o', linestyle='-')
        plt.plot([0,1],[0,1],'r--')
        plt.title('Quantile Calibration')
        plt.xlabel('Nominal Quantile')
        plt.ylabel('Observed Fraction ≤ Predicted')
        plt.savefig(f'{save_path}/quantile_calib.png')
        plt.close()

        # ---- 4) CRPS ----
        # draw 500 samples per predictive distribution
        samples = np.stack([dist.rvs(size=500) for dist in dists], axis=1)
        # samples.shape == (500, n)
        crps_vals = crps_ensemble(y_test, samples.T)
        avg_crps  = crps_vals.mean()

        # baseline degenerate‐median model
        median_pred   = np.median(y_test)
        baseline_crps = np.mean(np.abs(y_test - median_pred))

        print(f"Average CRPS: {avg_crps:.4f}")
        print(f"Baseline CRPS (degenerate at median): {baseline_crps:.4f}")

        # ---- 5) ECE ----
        ece = np.mean(np.abs(np.array(obs) - qs))
        print(f"Expected Calibration Error (ECE): {ece:.4f}")

        calibration_results = {
            'ece': ece,
            'avg_crps': avg_crps,
            'baseline_crps': baseline_crps,
        }
        # Save calibration_results as CSV
        # Flatten the dict for CSV saving
        cal_df = pd.DataFrame([calibration_results],)
        cal_df.to_csv(f'{save_path}/calibration_metrics.csv', index=False)

        # Save PIT and quantile calibration arrays as separate CSVs for clarity
        pd.DataFrame({'pit': pit}).to_csv(f'{save_path}/pit_values.csv', index=False)
        pd.DataFrame({'qs': qs, 'obs': obs}).to_csv(f'{save_path}/quantile_calibration.csv', index=False)
        pd.DataFrame({'crps': crps_vals}).to_csv(f'{save_path}/crps_values.csv', index=False)
        
        return calibration_results
       
        

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
