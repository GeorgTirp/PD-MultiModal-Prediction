import os

os.environ["OMP_NUM_THREADS"] = "1"            # Limits OpenMP (used by NumPy, Numba, etc.)
os.environ["OPENBLAS_NUM_THREADS"] = "1"       # OpenBLAS (used by NumPy)
os.environ["MKL_NUM_THREADS"] = "1"            # MKL (used by scikit-learn on Intel Macs)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"     # Apple's Accelerate framework
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
import torch
torch.set_num_threads(1)
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import shap
from tqdm import tqdm
import seaborn as sns
from IPython.utils import io
import torch
from ngboost.distns import Normal
from model_classes.faster_evidential_boost import NormalInverseGamma
import pickle
from model_classes.scalers import RobustTanhScaler, RobustSigmoidScaler, ZScoreScaler
from sklearn.model_selection import (
    KFold, LeaveOneOut,
    GroupKFold, LeaveOneGroupOut
)

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
            top_n: int = 10,
            logging = None,
            standardize: str = "",
            Pat_IDs = None,
            split_shaps=False,
            random_key=420) -> None:
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
            logging (optional): Logger instance for logging messages. Defaults to None.
        """
        self.logging = logging if logging is not None else print
        self.logging.info("Initializing BaseRegressionModel class...")
        self.feature_selection = feature_selection
        self.top_n = top_n
        self.random_key = random_key
        self.X, self.y, self.z, self.m, self.std = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(self.X, self.y, test_size=test_split_size, random_state=self.random_key)
        self.save_path = save_path
        self.identifier = identifier
        self.metrics = None
        self.model_name = None
        self.target_name = target_name
        self.Pat_IDs = Pat_IDs
        self.split_shaps = split_shaps
        
        if standardize == "zscore":
            self.scaler = ZScoreScaler()
        elif standardize == "tanh":
            self.scaler = RobustTanhScaler()
        elif standardize == "sigmoid":
            self.scaler = RobustSigmoidScaler()
        else: self.scaler = None

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.logging.info("Finished initializing BaseRegressionModel class.")

    def model_specific_preprocess(self, data_df: pd.DataFrame, ceiling: list =["BDI", "MoCA"]) -> Tuple:
        """
        Preprocess data specific to model requirements, including feature extraction and optional ceiling adjustments.
        """
        raise NotImplementedError("Subclasses must implement model_specific_preprocess method")

    def fit(self) -> None:
        """
        Train the model on the training dataset split.
        """
        self.logging.info(f"Starting {self.model_name} model training...")
        X_train, X_test, y_train, y_test = self.train_split
        self.model.fit(X_train, y_train)
        self.logging.info(f"Finished {self.model_name} model training.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from a trained model.

        Args:
            X (pd.DataFrame): Feature data.

        Returns:
            np.ndarray: Predicted target values.
        """
        self.logging.info("Starting prediction...")
        pred = self.model.predict(X)
        self.logging.info("Finished prediction.")
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
        ablation_idx: int = None,
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
        self.logging.info("Starting model evaluation...")

        if tune:
            if self.param_grid is None:
                raise ValueError("When calling tune=True, a param_grid has to be passed when initializing the model.")
            self.tune_hparams(self.X, self.y, self.param_grid, tune_folds)

        if folds == -1:
            kf = LeaveOneOut()
        else:
            kf = KFold(n_splits=folds, shuffle=True, random_state=self.random_key)

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
        r, p = pearsonr(y_vals, preds)
        f_squared = (r**2) / (1 - r**2)
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
            'r': r,
            'f_squared':f_squared,
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
        self.logging.info("Finished model evaluation.")

        return metrics


    def nested_eval(
            self, 
            folds=10, 
            get_shap=True, 
            tune=False, 
            tune_folds=10, 
            uncertainty=False, 
            ablation_idx=None) -> Dict:
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
        self.logging.info("Starting model evaluation...")

        #1) Choose outer CV depending on whether we have pat_ids
        if self.Pat_IDs is None:
            # fallback to standard CV
            if folds == -1:
                outer_cv = LeaveOneOut()
            else:
                outer_cv = KFold(n_splits=folds, shuffle=True, random_state=42)
            split_args = (self.X,)
            split_kwargs = {}      # no groups
        else:
            # group‐aware CV
            if folds == -1:
                outer_cv = LeaveOneGroupOut()
            else:
                outer_cv = GroupKFold(n_splits=folds)
            split_args = (self.X, self.y)
            split_kwargs = {'groups': self.Pat_IDs}

        if tune and self.param_grid is None:
                raise ValueError("When calling tune=True, a param_grid has to be passed when initializing the model.")
        preds = []
        preds_train = []
        epistemics = []
        aleatorics = []
        pred_dists = []
        y_vals = []
        y_trains = []
        all_shap_values = []
        all_shap_mean = []
        all_shap_variance = []
        all_shap_train = []
        all_shap_test = []
        iter_idx = 0
        # Farzin was here
        #cv_r_values = []
        
        # 2) Outer CV loop
        for train_idx, val_idx in tqdm(
            outer_cv.split(*split_args, **split_kwargs),
            total=outer_cv.get_n_splits(*split_args, **split_kwargs),
            desc="Cross-validation",
            leave=False
        ):
            X_train_kf, X_val_kf = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train_kf, y_val_kf = self.y.iloc[train_idx], self.y.iloc[val_idx]
            if hasattr(self, 'weights') and self.weights is not None:
                w_train, w_test = self.weights[train_idx], self.weights[val_idx]
            else:
                w_train, w_test = None, None

            if self.scaler is not None:
                self.scaler.fit(y_train_kf)
                y_train_kf = self.scaler.transform(y_train_kf)
                y_val_kf = self.scaler.transform(y_val_kf)

            groups_tr = None
            if self.Pat_IDs is not None:
                groups_tr = self.Pat_IDs.iloc[train_idx].to_numpy()

            if tune:
                self.tune_hparams(
                    X_train_kf, 
                    y_train_kf, 
                    self.param_grid, 
                    tune_folds, 
                    groups_tr, 
                    w_train)
            else:
                if hasattr(self, 'weights') and self.weights is not None:
                    self.model.fit(X_train_kf, y_train_kf, sample_weight=w_train)
                else:
                    self.model.fit(X_train_kf, y_train_kf)

            pred = self.model.predict(X_val_kf)
            pred_train = self.model.predict(X_train_kf)

            if self.scaler is not None:
                pred = self.scaler.inverse_transform(pred)
                y_val_kf = self.scaler.inverse_transform(y_val_kf)
            
            #r, _ = pearsonr(y_val_kf, pred)
            #cv_r_values.append(r)
            preds.append(pred)
            preds_train.append(pred_train)
            y_vals.append(y_val_kf)
            y_trains.append(y_train_kf)

            if hasattr(self, 'weights'):
                mse = mean_squared_error(y_val_kf, pred, sample_weight=w_test)
                train_mse = mean_squared_error(y_train_kf, pred_train, sample_weight=w_train)
            else:
                mse = mean_squared_error(y_val_kf, pred)
                train_mse = mean_squared_error(y_train_kf, pred_train)
            if len(preds) != 1:
                tqdm.write(f'Current Pearson-r: {pearsonr(np.concatenate(y_vals), np.concatenate(preds))[0]}, Train MSE = {train_mse}, Test MSE {mse}')

            # Get uncertainties
            if uncertainty == True:
                _ ,epistemic, aleatoric = self.compute_uncertainties(mode="nig", X=X_val_kf)
                epistemics.append(epistemic)
                aleatorics.append(aleatoric)

            # Get paramters of predictive Distribution
            if self.model_name == "NGBoost":
                pred_dist = self.model.pred_dist(X_val_kf).params
                pred_dist = np.column_stack([pred_dist[key] for key in pred_dist.keys()])
                pred_dists.append(pred_dist)
            
            # Get eplanations
            if get_shap:             
                 # Compute SHAP values on the whole dataset per fold
                with io.capture_output():
                    if ablation_idx is not None:
                        val_index = None
                    if self.model_name == "NGBoost":
                        shap_values_mean = self.feature_importance_mean(
                            self.X,
                            top_n=-1, 
                            save_results=True,  
                            iter_idx=iter_idx)
                        if self.prob_func == NormalInverseGamma:
                            shap_values_variance, _, _ = self.feature_importance_variance(
                                self.X,
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
                        if self.split_shaps:
                            
                            shap_values_test = self.feature_importance(
                                X_val_kf,
                                top_n=-1, 
                                save_results=True, 
                                iter_idx=iter_idx,
                                validation=True
                                )
                            
                            all_shap_test.append(shap_values_test)
                    
                        shap_values = self.feature_importance(
                            self.X,
                            top_n=-1, 
                            save_results=True, 
                            iter_idx=iter_idx,
                            )
                        all_shap_values.append(shap_values) 
            iter_idx += 1

        # Farzin was here :D
        #print(f"Mean CV R: {np.mean(cv_r_values):.3f}")
        #print(f"Std CV R: {np.std(cv_r_values):.3f}")
        
        if self.model_name == "NGBoost":
            pred_dists = np.vstack(pred_dists)
        preds = np.concatenate(preds)
        preds_train = np.concatenate(preds_train)
        y_vals = np.concatenate(y_vals)
        y_trains = np.concatenate(y_trains)
        r, p = pearsonr(y_vals, preds)
        f_squared = (r**2) / (1 - r**2)
        mse = mean_squared_error(y_vals, preds)
        train_mse = mean_squared_error(y_trains, preds_train)

        if uncertainty == True:
            epistemic_uncertainty = np.concatenate(epistemics)
            aleatoric_uncertainty = np.concatenate(aleatorics)

        if ablation_idx is not None:
            save_path = f'{self.save_path}/ablation/ablation_step[{ablation_idx}]/'
            os.makedirs(save_path, exist_ok=True)
            save_path = f'{save_path}{self.identifier}_{self.target_name}'
        else:
            save_path = f'{self.save_path}/{self.identifier}_{self.target_name}'

        
        if get_shap:
            if self.model_name == "NGBoost":
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
                if self.split_shaps:
                    #all_shap_mean_train_array = np.stack(all_shap_train, axis=0)
                    ## Average over the folds to get an aggregated array of shape (n_samples, n_features)
                    #train_shap_values = np.mean(all_shap_mean_train_array, axis=0)
                    #np.save(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_values_train.npy', train_shap_values)
                    #shap.summary_plot(train_shap_values , features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                    #plt.title(f'{self.identifier}  Summary Plot (Aggregated)', fontsize=16)
                    #plt.subplots_adjust(top=0.90)
                    #plt.savefig(f'{save_path}_shap_aggregated_beeswarm_train.png')
                    #plt.close()

                    test_shap_values = np.concatenate(all_shap_test, axis=0)
                    np.save(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_values_test.npy', test_shap_values)
                    shap.summary_plot(test_shap_values , features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                    plt.title(f'{self.identifier}  Summary Plot (Aggregated)', fontsize=16)
                    plt.subplots_adjust(top=0.90)
                    plt.savefig(f'{save_path}_shap_aggregated_beeswarm_test.png')
                    plt.close()
                
                all_shap_mean_array = np.stack(all_shap_values, axis=0)
                # Average over the folds to get an aggregated array of shape (n_samples, n_features)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                np.save(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_values.npy', mean_shap_values)
                shap.summary_plot(mean_shap_values , features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.identifier}  Summary Plot (Aggregated)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_shap_aggregated_beeswarm.png')
                #with open(f'{save_path}_mean_shap_explanations.pkl', 'wb') as fp:
                #    pickle.dump(mean_shap_values, fp)
                plt.close()
            np.save(f'{save_path}_all_shap_values(mu).npy', all_shap_mean_array)
            
            
            # Compute the mean of the absolute SHAP values for each feature
            feature_importances = np.mean(np.abs(mean_shap_values), axis=0)
            feature_importance_dict = dict(zip(self.X.columns, feature_importances))
            # Save feature importances to a file
            

        metrics = {
        'mse': mse,
        'train_mse': train_mse,
        'r': r,
        'f_squared':f_squared,
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
        
        self.logging.info(f"Trained model saved to {model_save_path}.")
        self.logging.info("Finished model evaluation.")
        return metrics
    
    def feature_ablation(self, folds: int = -1, get_shap=True, tune=False, tune_folds: int = 10, features_per_step: int = 1, threshold_to_one_fps: int = 10) -> Dict:
        """
        Perform iterative feature ablation analysis using nested cross-validation.

        Returns:
            Tuple[list, list, list]: Lists of R² scores, p-values, and removed feature names after each ablation step.
        """
        rs = []
        p_values = []
        removals = []
        number_of_features = len(self.feature_selection['features'])
        i = 1
        while number_of_features > 0:
            self.logging.info(f"---- Starting ablation step {i} with {number_of_features} features remaining. ----")
            # Determine the number of features to remove in this step
            if number_of_features > threshold_to_one_fps:
                # Remove multiple features (up to `features_per_step` if more than the threshold)
                features_to_remove = min(features_per_step, number_of_features)
            else:
                # Remove just one feature at a time once the threshold is passed
                features_to_remove = 1

            save_path = f'{self.save_path}/ablation/'
            metrics = self.nested_eval(folds=folds, get_shap=get_shap, tune=tune, tune_folds=tune_folds, ablation_idx=i)
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f'{save_path}/ablation_step[{i}]/{self.identifier}_{self.target_name}_metrics.csv', index=False)

            # let's save the preditcions for inspection
            self.plot(
                f"Actual vs. Prediction {self.model_name}- {self.identifier} No. features: {number_of_features}", 
                modality='',
                plot_df= pd.DataFrame({'Actual': metrics_df['y_test'].values[0],'Predicted': metrics_df['y_pred'].values[0]}),
                save_path_file =f'{save_path}/ablation_step[{i}]/{self.identifier}_{self.target_name}_actual_vs_predicted.png' )

            rs.append(metrics['r'])
            p_values.append(metrics['p_value'])
            importance = metrics['feature_importance']
            importance_indices = np.argsort(importance)

            # Remove the least important features
            least_important_features = [self.feature_selection['features'][idx] for idx in importance_indices[:features_to_remove]]
            
            # Update X, y, and feature list
            self.X = self.X.drop(columns=least_important_features)
            self.y = self.y.drop(columns=least_important_features)
            for feature in least_important_features:
                self.feature_selection['features'].remove(feature)
            
            removals.extend(least_important_features)

            # Update the number of remaining features
            number_of_features -= features_to_remove
            
            # If using a specific model, perform additional calibration if needed
            if self.model_name == "NGBoost":
                self.calibration_analysis(ablation_idx=i)

            # Increment the iteration counter (this is crucial when removing multiple features per step)
            i += 1

            # If features remaining are fewer than the threshold, switch to one-by-one ablation
            if number_of_features <= threshold_to_one_fps:
                features_per_step = 1  # From here on, remove only one feature at a time

            self.logging.info(f"Feature ablation finished. Final r: {rs[-1] if rs else 'N/A'}, min r: {np.min(rs) if rs else 'N/A'}, max r: {np.max(rs) if rs else 'N/A'}")
        
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
        x = range(i - 1)
                # Plot each model's R² scores in a loop, using sample_sizes on the x-axis
        #for model_name, r_scores in results.items():
        plot_df = pd.DataFrame({'x': x, 'rs': rs})
        sns.lineplot(data=plot_df, x='x', y='rs', label="R Score", marker='o')
        # Label the axes and set the title
        plt.xlabel("Number of removed features")
        plt.ylabel("Pearson-R Score")
        plt.title("Pearson-R Scores Over Feature Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return rs, p_values, removals
            

    
    def feature_importance(self, top_n: int = 10, batch_size=None, iter_idx=None, ablation_idx=None, save_results=True) -> Dict:
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
    
    
        

    def plot(self, title, modality='', plot_df=None, save_path_file=None) -> None:
        """
        Generate a scatter plot of predicted vs. actual target values with regression line and confidence intervals.

        Args:
            title (str): Title of the plot.
            modality (str, optional): Modality string to label axes (e.g., "MRI"). Defaults to empty string.
        """
        self.logging.info("Starting plot generation.")
        
        colors = {
            "deterioration": "04E762",
            "improvement": "FF5714",
            "line": "grey",
            "scatter": "grey",
            "ideal_line": "black",
        }

        # Create a wider (landscape) figure
        plt.figure(figsize=(10, 6))

        if plot_df is None:
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
            f'R: {self.metrics["r"]:.2f}\nP-value: {self.metrics["p_value"]:.6f}\nf²: {self.metrics["f_squared"]:.3f}', 
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

        if save_path_file:
            plt.savefig(save_path_file)

        else:
            # Save and close
            plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_actual_vs_predicted.png')

        plt.close()

        # Log info (optional)
        self.logging.info("Plot saved to %s/%s_%s_actual_vs_predicted.png", 
                 self.save_path, self.identifier, self.target_name)