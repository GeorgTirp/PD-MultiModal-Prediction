import os

os.environ["OMP_NUM_THREADS"] = "1"            # Limits OpenMP (used by NumPy, Numba, etc.)
os.environ["OPENBLAS_NUM_THREADS"] = "1"       # OpenBLAS (used by NumPy)
os.environ["MKL_NUM_THREADS"] = "1"            # MKL (used by scikit-learn on Intel Macs)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"     # Apple's Accelerate framework
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
#import torch
#torch.set_num_threads(1)
import json
from collections import Counter
import copy
import inspect
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy import stats
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
    GroupKFold, LeaveOneGroupOut,
    ParameterGrid
)
from sklearn.base import clone as skl_clone
from typing import Optional, List

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
            #identifier: str = None,
            top_n: int = 10,
            logging = None,
            standardize: str = "",
            standardize_features: str = "",
            Pat_IDs = None,
            split_shaps=False,
            random_state=420) -> None:
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
        self.random_state = random_state
        self.X, self.y, self.z, self.m, self.std = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(self.X, self.y, test_size=test_split_size, random_state=self.random_state)
        self.save_path = save_path
        #self.identifier = identifier
        self.metrics = None
        self.model_name = None
        self.target_name = target_name
        self.Pat_IDs = Pat_IDs
        self.split_shaps = split_shaps
        self.test_shap_mean = None
        self.test_shap_variance = None
        self.shap_mean  = None 
        self.shap_variance = None
        self.X_test = None
        self._raw_df = data_df.copy()

        if standardize == "zscore":
            self.scaler = ZScoreScaler()
        elif standardize == "tanh":
            self.scaler = RobustTanhScaler()
        elif standardize == "sigmoid":
            self.scaler = RobustSigmoidScaler()
        else: 
            self.scaler = None

        if standardize_features == "zscore":
            self.feature_scaler = ZScoreScaler()
        elif standardize_features == "tanh":
            self.feature_scaler = RobustTanhScaler()
        elif standardize_features == "sigmoid":
            self.feature_scaler = RobustSigmoidScaler()
        else:
            self.feature_scaler = None

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

    def evaluate(
        self,
        folds: int = 10,
        get_shap: bool = True,
        tune: bool = False,
        tune_folds: int = 10,
        nested: bool = False,
        uncertainty: bool = False,
        safe_best_hparams: bool = False,
    ) -> Dict:
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
        if safe_best_hparams and not nested:
            raise ValueError("safe_best_hparams=True requires nested=True.")

        if nested:
            return self.nested_eval(
                folds,
                get_shap,
                tune,
                tune_folds,
                uncertainty=uncertainty,
                safe_best_hparams=safe_best_hparams,
            )
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
            kf = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)

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
        test_indices = np.concatenate(test_indices)
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
                save_path = f'{save_path}_{self.target_name}_{ablation_idx}'
            else:
                save_path = f'{self.save_path}/{self.target_name}'

            if isinstance(self, NGBoostRegressionModel):
                all_shap_mean_array = np.stack(all_shap_mean, axis=0)
                all_shap_variance_array = np.stack(all_shap_variance, axis=0)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                variance_shap_values = np.mean(all_shap_variance_array, axis=0)

                np.save(f'{save_path}_mean_shap_values.npy', mean_shap_values)
                np.save(f'{save_path}_predicitve_uncertainty_shap_values.npy', variance_shap_values)
                np.save(f'{save_path}_all_shap_values(variance).npy', all_shap_variance_array)

                shap.summary_plot(mean_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.target_name} Summary Plot (Aggregated - Mean)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_mean_shap_aggregated.png')
                plt.close()

                shap.summary_plot(variance_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.target_name} Summary Plot (Aggregated - Variance)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_preditive_uncertainty_shap_aggregated.png')
                plt.close()
            else:
                all_shap_mean_array = np.stack(all_shap_values, axis=0)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                np.save(f'{self.save_path}/{self.target_name}_mean_shap_values.npy', mean_shap_values)
                shap.summary_plot(mean_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.target_name} Summary Plot (Aggregated)', fontsize=16)
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
        metrics_df.to_csv(f'{self.save_path}/{self.target_name}_metrics.csv', index=False)

        model_save_path = f'{self.save_path}/{self.target_name}_{ablation_idx}_trained_model.pkl' if ablation_idx is not None else f'{self.save_path}/{self.target_name}_trained_model.pkl'
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
            ablation_idx=None,
            member_idx=None,
            safe_best_hparams: bool = False) -> Dict:
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
        X, y, z, m, std = self.model_specific_preprocess(self._raw_df.copy())
        self.X, self.y, self.z, self.m, self.std = X, y, z, m, std
        #1) Choose outer CV depending on whether we have pat_ids
        if self.Pat_IDs is None:
            # fallback to standard CV
            if folds == -1:
                outer_cv = LeaveOneOut()
            else:
                outer_cv = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
            split_args = (self.X,)
            split_kwargs = {}      # no groups
        else:
            # group‐aware CV
            if folds == -1:
                outer_cv = LeaveOneGroupOut()
            else:
                outer_cv = GroupKFold(n_splits=folds,)
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
        test_indices = []
        all_shap_values = []
        all_shap_mean = []
        all_shap_variance = []
        all_shap_train = []
        all_shap_test = []
        all_test_shap_mean = []
        all_test_shap_variance = []
        X_vals = []
        iter_idx = 0
        # Farzin was here
        #cv_r_values = []

        hyperparam_votes = []
        if safe_best_hparams:
            def _to_serializable(value):
                if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(value)
                if isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                    return float(value)
                if isinstance(value, np.ndarray):
                    return [_to_serializable(v) for v in value.tolist()]
                if isinstance(value, (list, tuple)):
                    return [_to_serializable(v) for v in value]
                if isinstance(value, dict):
                    return {k: _to_serializable(v) for k, v in value.items()}
                if hasattr(value, "item"):
                    try:
                        return value.item()
                    except Exception:
                        return str(value)
                if isinstance(value, (str, bool)) or value is None:
                    return value
                return str(value)

            def _normalize_params(params: Dict) -> Dict:
                return {k: _to_serializable(v) for k, v in params.items()}
        
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

            if self.feature_scaler is not None:
                self.feature_scaler.fit(X_train_kf)
                X_train_kf = self.feature_scaler.transform(X_train_kf)
                X_val_kf = self.feature_scaler.transform(X_val_kf)

            groups_tr = None
            if self.Pat_IDs is not None:
                groups_tr = self.Pat_IDs.iloc[train_idx].to_numpy()

            best_params = None

            if tune:
                best_params = self.tune_hparams(
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
                if safe_best_hparams and hasattr(self.model, "get_params"):
                    try:
                        best_params = self.model.get_params()
                    except Exception:
                        best_params = None

            if safe_best_hparams and best_params:
                normalized_params = _normalize_params(best_params)
                hyperparam_votes.append(normalized_params)

            pred = self.model.predict(X_val_kf)
            pred_train = self.model.predict(X_train_kf)

            if self.scaler is not None:
                pred = self.scaler.inverse_transform(pred)
                y_val_kf = self.scaler.inverse_transform(y_val_kf)

            if self.feature_scaler is not None:
                X_val_kf = self.feature_scaler.inverse_transform(X_val_kf)
                X_train_kf = self.feature_scaler.inverse_transform(X_train_kf)
            
            test_std = pred.std()
            r, _ = pearsonr(y_val_kf, pred)
            #cv_r_values.append(r)
            preds.append(pred)
            preds_train.append(pred_train)
            y_vals.append(y_val_kf)
            y_trains.append(y_train_kf)
            X_vals.append(X_val_kf)
            test_indices.append(val_idx)
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
                        if self.split_shaps:
                            test_shap_mean = self.feature_importance_mean(
                                X_val_kf,
                                top_n=-1, 
                                save_results=True,  
                                iter_idx=iter_idx)
                            #test_shap_mean /= test_std
                            if self.prob_func == NormalInverseGamma:
                                test_shap_variance, _, _ = self.feature_importance_variance(
                                    X_val_kf,
                                    mode="nig",
                                    top_n=-1, 
                                    save_results=True, 
                                    iter_idx=iter_idx)
                                #test_shap_variance /= test_std
                            elif self.prob_func == Normal:
                                test_shap_variance = self.feature_importance_variance(
                                    X_val_kf,
                                    mode="normal",
                                    top_n=-1, 
                                    save_results=True, 
                                    iter_idx=iter_idx)
                                #test_shap_variance /= test_std
                            all_test_shap_mean.append(test_shap_mean)
                            all_test_shap_variance.append(test_shap_variance)
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
                                )
                            #shap_values_test /= test_std
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

        if safe_best_hparams:
            if hyperparam_votes:
                vote_strings = [json.dumps(v, sort_keys=True) for v in hyperparam_votes]
                vote_counter = Counter(vote_strings)
                majority_key, _ = vote_counter.most_common(1)[0]
                majority_params = json.loads(majority_key)
                filename = f"{self.target_name}_best_hparams.json"
                if ablation_idx is not None and member_idx is not None:
                    filename = f"{self.target_name}_best_hparams_ablation_{ablation_idx}_member_{member_idx}.json"
                elif ablation_idx is not None:
                    filename = f"{self.target_name}_best_hparams_ablation_{ablation_idx}.json"
                elif member_idx is not None:
                    filename = f"{self.target_name}_best_hparams_member_{member_idx}.json"

                os.makedirs(self.save_path, exist_ok=True)
                best_params_path = os.path.join(self.save_path, filename)
                with open(best_params_path, "w", encoding="utf-8") as fh:
                    json.dump(majority_params, fh, indent=2)

                self.best_hparams_ = majority_params
                self.best_hparams_path_ = best_params_path
                self.logging.info(
                    f"Saved majority-vote hyperparameters (votes={len(hyperparam_votes)}) to {best_params_path}."
                )
            else:
                self.logging.info(
                    "safe_best_hparams=True but no hyperparameter votes were collected during nested evaluation."
                )
        
        if self.model_name == "NGBoost":
            pred_dists = np.vstack(pred_dists)
        preds = np.concatenate(preds)
        preds_train = np.concatenate(preds_train)
        y_vals = np.concatenate(y_vals)
        y_trains = np.concatenate(y_trains)
        X_vals = np.concatenate(X_vals)
        r, p = pearsonr(y_vals, preds)
        f_squared = (r**2) / (1 - r**2)
        rho, pval_spearman = spearmanr(y_vals, preds)
        test_indices = [self.X.iloc[idx].index.to_numpy() for idx in test_indices]
        test_indices = np.concatenate(test_indices)
        mse = mean_squared_error(y_vals, preds)
        train_mse = mean_squared_error(y_trains, preds_train)

        os.makedirs(self.save_path, exist_ok=True)
        test_pred_path = os.path.join(self.save_path, f"{self.target_name}_cv_test_predictions.csv")
        pd.DataFrame({
            "y_true": y_vals,
            "y_pred": preds
        }).to_csv(test_pred_path, index=False)

        # (Optional) also save the train-fold predictions
        train_pred_path = os.path.join(self.save_path, f"{self.target_name}_cv_train_predictions.csv")
        pd.DataFrame({
            "y_true": y_trains,
            "y_pred": preds_train
        }).to_csv(train_pred_path, index=False)

        if uncertainty == True:
            epistemic_uncertainty = np.concatenate(epistemics)
            aleatoric_uncertainty = np.concatenate(aleatorics)

        if ablation_idx is not None:
            save_path = f'{self.save_path}/ablation/ablation_step[{ablation_idx}]/'
            if member_idx is not None:
                save_path = f'{save_path}/member[{member_idx}]/'
            os.makedirs(save_path, exist_ok=True)
            save_path = f'{save_path}_{self.target_name}'
        else:
            if member_idx is not None:
                save_path = f'{save_path}/member[{member_idx}]/'
            save_path = f'{self.save_path}/{self.target_name}'

        
            os.makedirs(save_path, exist_ok=True)
        
        if get_shap:
            feature_names = list(self.feature_selection.get('features', self.X.columns.tolist()))

            def _save_shap_csv(array: np.ndarray, file_path: str) -> None:
                """Persist SHAP arrays with feature headers as CSV."""
                arr = np.asarray(array)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                elif arr.ndim > 2:
                    arr = arr.reshape(-1, arr.shape[-1])
                # Fall back to current X columns if feature list length mismatched
                if arr.shape[1] != len(feature_names):
                    cols = list(self.X.columns[:arr.shape[1]])
                else:
                    cols = feature_names
                shap_df = pd.DataFrame(arr, columns=cols)
                shap_df.to_csv(file_path, index=False)

            if self.model_name == "NGBoost":
                if self.split_shaps:
                    all_test_shap_mean_array = np.concatenate(all_test_shap_mean, axis=0)
                    all_test_shap_variance_array = np.concatenate(all_test_shap_variance, axis=0)
                    # Average over the folds to get an aggregated array of shape (n_samples, n_features)
                    test_shap_mean = all_test_shap_mean_array
                    test_shap_variance = all_test_shap_variance_array
                    # Save SHAP values to a file
                    _save_shap_csv(test_shap_mean, f'{save_path}_mean_shap_values_test.csv')
                    _save_shap_csv(test_shap_variance, f'{save_path}_predicitve_uncertainty_shap_values_test.csv')
                    _save_shap_csv(all_test_shap_variance_array, f'{save_path}_all_shap_values(variance)_test.csv')

                    # Plot for mean SHAP values
                    shap.summary_plot(test_shap_mean, features=X_vals, feature_names=self.X.columns, show=False, max_display=self.top_n)
                    plt.title(f'{self.target_name} Summary Plot (Aggregated - Mean)', fontsize=16)
                    plt.subplots_adjust(top=0.90)
                    plt.savefig(f'{save_path}_mean_shap_aggregated_test.png')
                    plt.close()

                    shap.summary_plot(test_shap_variance, features=X_vals, feature_names=self.X.columns, show=False, max_display=self.top_n)
                    plt.title(f'{self.target_name} Summary Plot (Aggregated - Variance)', fontsize=16)
                    plt.subplots_adjust(top=0.90)
                    if self.prob_func == NormalInverseGamma:
                        plt.savefig(f'{save_path}_preditive_uncertainty_shap_aggregated_test.png')
                    elif self.prob_func == Normal:
                        plt.savefig(f'{save_path}_std_shap_aggregated_test.png')
                    plt.close()
                    
                all_shap_mean_array = np.stack(all_shap_mean, axis=0)
                all_shap_variance_array = np.stack(all_shap_variance, axis=0)
                # Average over the folds to get an aggregated array of shape (n_samples, n_features)
                mean_shap_values = np.mean(all_shap_mean_array, axis=0)
                variance_shap_values = np.mean(all_shap_variance_array, axis=0)

                # Save SHAP values to a file
                _save_shap_csv(mean_shap_values, f'{save_path}_mean_shap_values.csv')
                _save_shap_csv(variance_shap_values, f'{save_path}_predicitve_uncertainty_shap_values.csv')
                _save_shap_csv(all_shap_variance_array, f'{save_path}_all_shap_values(variance).csv')
        
                # Plot for mean SHAP values
                shap.summary_plot(mean_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.target_name} Summary Plot (Aggregated - Mean)', fontsize=16)
                plt.subplots_adjust(top=0.90)
                plt.savefig(f'{save_path}_mean_shap_aggregated.png')
                plt.close()
                
                shap.summary_plot(variance_shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
                plt.title(f'{self.target_name} Summary Plot (Aggregated - Variance)', fontsize=16)
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

                    test_shap_mean = np.concatenate(all_shap_test, axis=0)
                    _save_shap_csv(test_shap_mean, f'{self.save_path}/{self.target_name}_mean_shap_values_test.csv')
                    shap.summary_plot(test_shap_mean , features=X_vals, feature_names=self.X.columns, show=False, max_display=self.top_n)
                    plt.title(f'{self.target_name}  Summary Plot (Aggregated)', fontsize=16)
                    plt.subplots_adjust(top=0.90)
                    plt.savefig(f'{save_path}_shap_aggregated_beeswarm_test.png')
                    plt.close()
            
            all_shap_mean_array = np.stack(all_shap_values, axis=0)
            # Average over the folds to get an aggregated array of shape (n_samples, n_features)
            mean_shap_values = np.mean(all_shap_mean_array, axis=0)
            _save_shap_csv(mean_shap_values, f'{self.save_path}/{self.target_name}_mean_shap_values.csv')
            shap.summary_plot(mean_shap_values , features=self.X, feature_names=self.X.columns, show=False, max_display=self.top_n)
            plt.title(f'{self.target_name}  Summary Plot (Aggregated)', fontsize=16)
            plt.subplots_adjust(top=0.90)
            plt.savefig(f'{save_path}_shap_aggregated_beeswarm.png')
            #with open(f'{save_path}_mean_shap_explanations.pkl', 'wb') as fp:
            #    pickle.dump(mean_shap_values, fp)
            plt.close()

        _save_shap_csv(all_shap_mean_array, f'{save_path}_all_shap_values(mu).csv')
        test_idx = np.asarray(test_indices)            # original row positions in self.X
        aligned_test_shap = pd.DataFrame(
            test_shap_mean,
            index=test_idx,
            columns=self.X.columns
        )
        
        # 2) Reindex to the FULL original order (train rows become NaN)
        aligned_test_shap = aligned_test_shap.reindex(self.X.index)
        
        # 3) Save an aligned CSV (one row per original sample, same order as self.X)
        aligned_path = os.path.join(self.save_path, f"{self.target_name}_mean_shap_values_test_ALIGNED.csv")
        aligned_test_shap.to_csv(aligned_path)
        try:
            self.X_test = pd.DataFrame(X_vals, columns=self.X.columns)
        except Exception:
            self.X_test = X_vals
        self.shap_mean = mean_shap_values
        if self.split_shaps:
            self.test_shap_mean = test_shap_mean
        if self.model_name == "NGBoost":
            self.shap_variance = variance_shap_values
            if self.split_shaps:
                self.test_shap_variance = test_shap_variance
        
        # Compute the mean of the absolute SHAP values for each feature
        feature_importances = np.mean(np.abs(mean_shap_values), axis=0)
        feature_importance_dict = dict(zip(self.X.columns, feature_importances))
        if self.split_shaps:
            feature_importances_test = np.mean(np.abs(test_shap_mean), axis=0)
            feature_importance_test_dict = dict(zip(self.X.columns, feature_importances))
        # Save feature importances to a file
            

        metrics = {
        'mse': mse,
        'train_mse': train_mse,
        'r': r,
        'rho': rho,
        'f_squared':f_squared,
        'p_value': p,
        'y_pred': preds,
        'y_test': y_vals,
        'pred_dist': pred_dists,
        'test_index': test_indices,
        'epistemic': epistemic_uncertainty if uncertainty else None,
        'aleatoric': aleatoric_uncertainty if uncertainty else None,
        'feature_importance': feature_importances if get_shap else None,
        'feature_importance_test': feature_importances_test if get_shap & self.split_shaps else None
        }
        self.metrics = metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{save_path}_metrics.csv', index=False)
        
        # Save the trained model to a file
        #model_save_path = f'{save_path}_trained_model.pkl'
        #with open(model_save_path, 'wb') as model_file:
        #    pickle.dump(self.model, model_file)
        
        #self.logging.info(f"Trained model saved to {model_save_path}.")
        self.logging.info("Finished model evaluation.")
        return metrics
    
    def inference(
        self,
        inference_csv_path: str,
        param_grid: Optional[dict] = None,
        members: int = 1,
        folds: int = 5,
        target_col: Optional[str] = None,
        save_dir: Optional[str] = None,
        random_seed: Optional[int] = None,
        drop_nan: bool = True,
    ) -> Dict:
        """
        Perform ensemble inference on a new dataset with optional hyperparameter tuning.

        Args:
            inference_csv_path (str): Path to CSV containing features (and optional target).
            param_grid (dict, optional): Hyperparameter combinations to evaluate.
            members (int, optional): Number of ensemble members. Defaults to 1.
            folds (int, optional): Number of CV folds per member for voting. Defaults to 5.
            target_col (str, optional): Target column name. Defaults to self.target_name.
            save_dir (str, optional): Directory to store inference artefacts.
            random_seed (int, optional): Seed for ensemble member generation.

        Returns:
            Dict: Dictionary containing predictions, metrics, chosen hyperparameters, and paths.
        """
        if members < 1:
            raise ValueError("members must be >= 1.")

        if target_col is None:
            target_col = self.target_name

        if not os.path.exists(inference_csv_path):
            raise FileNotFoundError(f"Inference CSV not found: {inference_csv_path}")

        inference_df = pd.read_csv(inference_csv_path)

        # Determine common feature set
        original_features = list(self.feature_selection.get('features', []))
        common_features = [f for f in original_features if f in inference_df.columns]
        if not common_features:
            raise ValueError("No overlapping features between training features and inference CSV.")

        if len(common_features) < len(original_features):
            dropped = set(original_features) - set(common_features)
            self.logging.warning(f"Dropping missing features for inference: {sorted(dropped)}")

        X_train_full = self.X[common_features].copy()
        y_train_full = self.y.copy()
        X_infer = inference_df[common_features].copy()
        y_infer = inference_df[target_col].copy() if target_col in inference_df.columns else None

        if drop_nan:
            train_mask = ~X_train_full.isna().any(axis=1)
            dropped_train = len(train_mask) - train_mask.sum()
            if dropped_train > 0:
                self.logging.info(f"[Inference] Dropping {dropped_train} training rows with NaNs in {len(common_features)}-feature set.")
            X_train_full = X_train_full.loc[train_mask].reset_index(drop=True)
            y_train_full = y_train_full.loc[train_mask].reset_index(drop=True)

            infer_mask = ~X_infer.isna().any(axis=1)
            if y_infer is not None:
                infer_mask &= ~y_infer.isna()
            dropped_infer = len(infer_mask) - infer_mask.sum()
            if dropped_infer > 0:
                self.logging.info(f"[Inference] Dropping {dropped_infer} inference rows with NaNs among selected features/target.")
            X_infer = X_infer.loc[infer_mask].reset_index(drop=True)
            if y_infer is not None:
                y_infer = y_infer.loc[infer_mask].reset_index(drop=True)

        # Prepare save paths
        inference_root = save_dir or os.path.join(self.save_path, "inference")
        os.makedirs(inference_root, exist_ok=True)

        # Base parameters and parameter grid
        base_params = {}
        if hasattr(self.model, "get_params"):
            try:
                base_params = self.model.get_params(deep=True)
            except Exception:
                base_params = {}

        param_candidates = list(ParameterGrid(param_grid)) if param_grid else [{}]

        # Seed handling for ensemble members
        seed_source = random_seed if random_seed is not None else getattr(self, "random_state", 0)
        rng = np.random.default_rng(seed_source)
        member_seeds = rng.integers(low=0, high=np.iinfo(np.uint32).max, size=members, dtype=np.uint32)

        if folds == -1:
            if self.Pat_IDs is not None:
                cv_splitter = LeaveOneGroupOut()
                groups = self.Pat_IDs.to_numpy()
            else:
                cv_splitter = LeaveOneOut()
                groups = None
        else:
            if self.Pat_IDs is not None:
                cv_splitter = GroupKFold(n_splits=folds)
                groups = self.Pat_IDs.to_numpy()
            else:
                cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=seed_source)
                groups = None

        if param_grid and folds == 1:
            raise ValueError("When param_grid is provided, folds must be >= 2 to enable majority voting.")


        member_models = []
        member_params = []
        member_predictions = []
        member_shap_values_train = []
        member_shap_values_test = []
        member_metrics = []

        original_model = self.model
        original_features_state = list(self.feature_selection.get('features', []))

        for idx in range(members):
            self.logging.info(f"[Inference] Starting ensemble member {idx}")

            # --- Seed pro Member setzen, damit tune_hparams den auch nutzt ---
            seed_m = int(member_seeds[idx])
            # Viele deiner Modelklassen lesen self.random_state in tune_hparams ein
            if hasattr(self, "random_state"):
                self.random_state = seed_m
                self.logging.info(f"[Inference] Member {idx}: Set self.random_state = {seed_m}")
            # CatBoost-Spezialfall: cat_hparams.random_seed
            if hasattr(self, "cat_hparams") and isinstance(self.cat_hparams, dict):
                if "random_seed" in self.cat_hparams:
                    self.cat_hparams["random_seed"] = seed_m

            # --- (Optional) CV-Splitter-Info für Gruppen ---
            if folds == -1:
                groups = self.Pat_IDs.to_numpy() if self.Pat_IDs is not None else None
            else:
                groups = self.Pat_IDs.to_numpy() if self.Pat_IDs is not None else None

            # --- Hyperparametertuning über die MODELLEIGENE Methode ---
            # Erwartete Signatur wie in deinem CatBoost-Beispiel:
            #   tune_hparams(X, y, param_grid, folds=..., groups=..., weights=...)
            best_params = self.tune_hparams(
                X=X_train_full,
                y=y_train_full,
                param_grid=param_grid if param_grid is not None else {},
                folds=folds,
                groups=groups,
                weights=getattr(self, "weights", None),
            )
            self.logging.info(f"[Inference] Member {idx}: best params {best_params}")

            # --- Skalierer wie bisher: auf FULL TRAIN fitten, train/infer transformieren ---
            feature_scaler_final = None
            if self.feature_scaler is not None:
                feature_scaler_final = copy.deepcopy(self.feature_scaler)
                feature_scaler_final.fit(X_train_full)
                X_train_final = feature_scaler_final.transform(X_train_full)
                X_infer_final  = feature_scaler_final.transform(X_infer)
            else:
                X_train_final = X_train_full.to_numpy()
                X_infer_final  = X_infer.to_numpy()

            target_scaler_final = None
            if self.scaler is not None:
                target_scaler_final = copy.deepcopy(self.scaler)
                target_scaler_final.fit(y_train_full)
                y_train_final = np.asarray(target_scaler_final.transform(y_train_full)).ravel()
            else:
                y_train_final = y_train_full.to_numpy()

            # --- WICHTIG: self.model ist jetzt schon der best_estimator_ vom Tuning.
            # Wir klonen ihn pro Member, damit sich Mitglieder nicht überschreiben. ---
            member_model = skl_clone(self.model)

            fit_kwargs = {}
            if hasattr(self, "weights") and self.weights is not None:
                fit_kwargs["sample_weight"] = self.weights
            member_model.fit(X_train_final, y_train_final, **fit_kwargs)

            # --- Inferenz + ggf. Inverse-Target-Transform ---
            preds_proc = member_model.predict(X_infer_final)
            if target_scaler_final is not None:
                preds = np.asarray(target_scaler_final.inverse_transform(np.asarray(preds_proc))).ravel()
            else:
                preds = np.asarray(preds_proc).ravel()
            member_predictions.append(preds)

            # --- Member-Artefakte speichern (wie bisher) ---
            member_artifacts = {
                "model": member_model,
                "feature_scaler": feature_scaler_final,
                "target_scaler": target_scaler_final,
                "features": common_features,
            }
            model_path = os.path.join(inference_root, f"member_{idx}_model.pkl")
            with open(model_path, "wb") as fh:
                pickle.dump(member_artifacts, fh)
            self.logging.info(f"[Inference] Saved member {idx} artefacts to {model_path}")

            # --- SHAP wie bisher, temporär self.model = member_model ---
            try:
                self.model = member_model
                self.feature_selection["features"] = common_features
                fi_signature = inspect.signature(self.feature_importance)
                base_kwargs = {
                    "X": X_train_full[common_features],
                    "top_n": -1,
                    "save_results": False,
                    "iter_idx": None,
                }
                if "validation" in fi_signature.parameters:
                    base_kwargs["validation"] = False

                shap_train = self.feature_importance(**base_kwargs)
                if isinstance(shap_train, list):
                    shap_train = np.asarray(shap_train)
                shap_train = np.asarray(shap_train)
                if shap_train.ndim == 1:
                    shap_train = shap_train.reshape(1, -1)
                member_shap_values_train.append(shap_train)

                if self.split_shaps and not X_infer.empty:
                    infer_kwargs = base_kwargs.copy()
                    infer_kwargs["X"] = X_infer[common_features]
                    if "validation" in fi_signature.parameters:
                        infer_kwargs["validation"] = True
                    shap_test = self.feature_importance(**infer_kwargs)
                    if isinstance(shap_test, list):
                        shap_test = np.asarray(shap_test)
                    shap_test = np.asarray(shap_test)
                    if shap_test.ndim == 1:
                        shap_test = shap_test.reshape(1, -1)
                    member_shap_values_test.append(shap_test)
            except Exception as shap_err:
                self.logging.warning(f"[Inference] Failed to compute SHAP for member {idx}: {shap_err}")

            # --- Member-Metriken (falls y_infer vorhanden) wie bisher ---
            metrics_member = {}
            if y_infer is not None:
                preds_series = pd.Series(preds, index=X_infer.index)
                y_series = y_infer
                r, _ = pearsonr(y_series, preds_series)
                rho, _ = spearmanr(y_series, preds_series)
                mse = mean_squared_error(y_series, preds_series)
                metrics_member = {"r": r, "rho": rho, "mse": mse, "rmse": np.sqrt(mse)}
            member_metrics.append(metrics_member)

            member_models.append({
                "model": member_model,
                "feature_scaler": feature_scaler_final,
                "target_scaler": target_scaler_final,
            })


        # Restore original state
        self.model = original_model
        self.feature_selection['features'] = original_features_state

        member_predictions_array = np.vstack(member_predictions)
        ensemble_predictions = member_predictions_array.mean(axis=0)

        results_df = pd.DataFrame({"Prediction": ensemble_predictions})
        if y_infer is not None:
            results_df["Actual"] = y_infer.values
        results_df_path = os.path.join(inference_root, "ensemble_predictions.csv")
        results_df.to_csv(results_df_path, index=False)
        self.logging.info(f"[Inference] Saved ensemble predictions to {results_df_path}")

        metrics = {}
        plot_df = None
        if y_infer is not None:
            r, p_val = pearsonr(y_infer, ensemble_predictions)
            rho, _ = spearmanr(y_infer, ensemble_predictions)
            mse = mean_squared_error(y_infer, ensemble_predictions)
            rmse = np.sqrt(mse)
            metrics = {
                "r": r,
                "rho": rho,
                "p_value": p_val,
                "mse": mse,
                "rmse": rmse,
                "y_test": y_infer.to_numpy(),
                "y_pred": ensemble_predictions,
            }
            plot_df = pd.DataFrame({"Actual": y_infer, "Predicted": ensemble_predictions})
            plot_path = os.path.join(inference_root, f"{self.target_name}_ensemble_actual_vs_predicted.png")
            self.plot(
                f"Actual vs. Prediction (Ensemble of {members}) - {self.model_name}",
                plot_df=plot_df,
                save_path_file=plot_path,
                N=len(plot_df),
                metrics_override=metrics
            )

        # Ensemble SHAP aggregation
        shap_plot_path = None
        shap_plot_test_path = None
        shap_csv_path = None
        shap_csv_test_path = None
        if member_shap_values_train:
            try:
                shap_stack = np.stack(member_shap_values_train, axis=0)
                shap_mean = np.mean(shap_stack, axis=0)
                feature_matrix = X_train_full[common_features]
                shap_plot_path = os.path.join(inference_root, f"{self.target_name}_ensemble_shap_beeswarm.png")
                plt.figure(figsize=(8, 6))
                shap.summary_plot(
                    shap_mean,
                    features=feature_matrix,
                    feature_names=common_features,
                    show=False,
                    max_display=self.top_n if self.top_n > 0 else None,
                )
                plt.title(f"{self.target_name} Ensemble SHAP Summary", fontsize=14)
                plt.tight_layout()
                plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                shap_csv_path = os.path.join(inference_root, f"{self.target_name}_ensemble_shap_values.csv")
                pd.DataFrame(shap_mean, columns=common_features).to_csv(shap_csv_path, index=False)
            except Exception as shap_agg_err:
                self.logging.warning(f"[Inference] Failed to create ensemble SHAP plot: {shap_agg_err}")

        if self.split_shaps and member_shap_values_test:
            try:
                shap_stack_test = np.stack(member_shap_values_test, axis=0)
                shap_mean_test = np.mean(shap_stack_test, axis=0)
                feature_matrix_test = X_infer[common_features]
                shap_plot_test_path = os.path.join(inference_root, f"{self.target_name}_ensemble_shap_beeswarm_test.png")
                plt.figure(figsize=(8, 6))
                shap.summary_plot(
                    shap_mean_test,
                    features=feature_matrix_test,
                    feature_names=common_features,
                    show=False,
                    max_display=self.top_n if self.top_n > 0 else None,
                )
                plt.title(f"{self.target_name} Ensemble SHAP Summary (Inference Set)", fontsize=14)
                plt.tight_layout()
                plt.savefig(shap_plot_test_path, dpi=300, bbox_inches="tight")
                plt.close()
                shap_csv_test_path = os.path.join(inference_root, f"{self.target_name}_ensemble_shap_values_test.csv")
                pd.DataFrame(shap_mean_test, columns=common_features).to_csv(shap_csv_test_path, index=False)
            except Exception as shap_test_err:
                self.logging.warning(f"[Inference] Failed to create inference SHAP plot: {shap_test_err}")

        summary_path = os.path.join(inference_root, "ensemble_summary.json")
        def _json_sanitize(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.uint32, np.int16, np.int8)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return [_json_sanitize(v) for v in obj.tolist()]
            if isinstance(obj, list):
                return [_json_sanitize(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _json_sanitize(v) for k, v in obj.items()}
            return obj

        summary_payload = _json_sanitize({
            "members": members,
            "folds": folds,
            "param_grid": param_grid,
            "member_params": member_params,
            "metrics": metrics,
            "member_metrics": member_metrics,
            "prediction_csv": results_df_path,
            "shap_plot_train": shap_plot_path,
            "shap_plot_test": shap_plot_test_path,
            "shap_csv_train": shap_csv_path,
            "shap_csv_test": shap_csv_test_path,
        })
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2)

        return {
            "predictions": ensemble_predictions,
            "metrics": metrics,
            "member_params": member_params,
            "member_predictions": member_predictions_array,
            "prediction_csv": results_df_path,
            "summary_json": summary_path,
            "shap_plot_train": shap_plot_path,
            "shap_plot_test": shap_plot_test_path,
            "shap_csv_train": shap_csv_path,
            "shap_csv_test": shap_csv_test_path,
        }
    def feature_ablation(
        self, 
        folds: int = -1, 
        get_shap=True, 
        tune=False, 
        tune_folds: int = 10, 
        features_per_step: int = 1, 
        members: int = 1,
        threshold_to_one_fps: int = 10,
        test_set=True):

        """
        Perform iterative feature ablation analysis using nested cross-validation.

        Returns:
            Tuple[list, list, list]: Lists of R² scores, p-values, and removed feature names after each ablation step.
        """

        rs, rhos, p_values, removals = [], [], [], []
        train_rmse_list, test_rmse_list = [], []
        r_std_list, rho_std_list, train_rmse_std_list, test_rmse_std_list = [], [], [], []
        

        number_of_features = len(self.feature_selection['features'])
        i = 1
        global_random_state = int(getattr(self, "random_state", 0))
        rng_global = np.random.default_rng(global_random_state)

        # Use a large space to avoid collisions (not 0..1000)
        member_seeds = rng_global.integers(
            low=0, high=np.iinfo(np.uint32).max, size=members, dtype=np.uint32
        )

        while number_of_features > 0:
            ensemble_y_tests, ensemble_y_preds, ensemble_test_idx = [], [], []
            members_shap_values_test, members_X_test = [], []
            members_shap_values_train, members_X_train = [], []
            self.logging.info(f"---- Starting ablation step {i} with {number_of_features} features remaining. ----")

            if number_of_features > threshold_to_one_fps:
                features_to_remove = min(features_per_step, number_of_features)
            else:
                features_to_remove = 1

            feature_votes = {}
            ensemble_rs, ensemble_rhos, ensemble_p_values = [], [], []
            ensemble_train_mse, ensemble_test_mse = [], []

            members_shap_mean, members_shap_mean_test = [], []
            members_shap_variance, members_shap_variance_test = [], []

            for m in range(members):
                self.random_state = int(member_seeds[m])   # same seed for this member at every step
                self.logging.info(f"[ablation step {i}] member {m} seed={self.random_state}")

                save_path = f'{self.save_path}/ablation/ablation_step[{i}]/member[{m}]'
                os.makedirs(save_path, exist_ok=True)

                metrics = self.nested_eval(
                    folds=folds, 
                    get_shap=get_shap, 
                    tune=tune, 
                    tune_folds=tune_folds, 
                    ablation_idx=i, 
                    member_idx=m)

                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(f'{save_path}/{self.target_name}_metrics.csv', index=False)

                ensemble_y_tests.append(metrics['y_test'])
                ensemble_y_preds.append(metrics['y_pred'])
                ensemble_test_idx.append(metrics.get('test_index', None))

                if hasattr(self, "test_shap_values") and self.test_shap_values is not None:
                    members_shap_values_test.append(self.test_shap_values)
                    if hasattr(self, "test_X_for_shap"):
                        members_X_test.append(self.test_X_for_shap)
                if hasattr(self, "shap_values") and self.shap_values is not None:
                    members_shap_values_train.append(self.shap_values)
                    if hasattr(self, "X_for_shap"):
                        members_X_train.append(self.X_for_shap)
                if self.shap_mean is not None:
                    members_shap_mean.append(self.shap_mean)

                if hasattr(self, 'test_shap_mean') and self.test_shap_mean is not None:
                    members_shap_mean_test.append(self.test_shap_mean)

                if hasattr(self, 'shap_variance') and self.shap_variance is not None:
                    members_shap_variance.append(self.shap_variance)

                if hasattr(self, 'test_shap_variance') and self.test_shap_variance is not None:
                    members_shap_variance_test.append(self.test_shap_variance)

                self.plot(
                    f"Actual vs. Prediction {self.model_name}- {self.target_name} No. features: {number_of_features}", 
                    modality='',
                    plot_df=pd.DataFrame({'Actual': metrics_df['y_test'].values[0], 'Predicted': metrics_df['y_pred'].values[0]}),
                    save_path_file=f'{save_path}/{self.target_name}_actual_vs_predicted.png')

                ensemble_rs.append(metrics['r'])
                ensemble_rhos.append(metrics['rho'])
                ensemble_p_values.append(metrics['p_value'])
                ensemble_train_mse.append(metrics['train_mse'])
                ensemble_test_mse.append(metrics['mse'])

                importance = metrics['feature_importance_test'] if test_set else metrics['feature_importance']
                importance_indices = np.argsort(importance)
                least_important = [self.feature_selection['features'][idx] for idx in importance_indices[:features_per_step]]
                for feature in least_important:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1

            # SHAP plots saved as before (no change)
            # [Omitted here for brevity, unchanged from original]

            # ========= NEW: step-level (across members) averages =========
            step_dir = f'{self.save_path}/ablation/ablation_step[{i}]'
            os.makedirs(step_dir, exist_ok=True)
            feature_names_current = list(self.feature_selection['features'])

            def save_avg_beeswarm(shap_list, X_list, tag: str):
                if not shap_list: 
                    return
                # If all members used identical rows, take a true mean across members per-row; else pool rows.
                same_shape = len({(sv.shape[0], sv.shape[1]) for sv in shap_list}) == 1
                if same_shape:
                    # Try true per-row mean (best)
                    shap_avg = np.mean(np.stack(shap_list, axis=0), axis=0)  # (n_rows, n_features)
                    data_avg = None
                    if X_list and all(x is not None for x in X_list) and \
                       len({(x.shape[0], x.shape[1]) for x in X_list}) == 1:
                        data_avg = np.mean(np.stack([x.values for x in X_list], axis=0), axis=0)  # (n_rows, n_features)
                    expl = shap.Explanation(
                        values=shap_avg,
                        base_values=np.zeros(shap_avg.shape[0]),
                        data=data_avg,
                        feature_names=feature_names_current
                    )
                else:
                    # Pool all rows from all members (good proxy of average distribution)
                    shap_pool = np.vstack(shap_list)                                  # (sum_rows, n_features)
                    data_pool = None
                    if X_list and any(x is not None for x in X_list):
                        data_pool = np.vstack([x.values for x in X_list if x is not None])
                    expl = shap.Explanation(
                        values=shap_pool,
                        base_values=np.zeros(shap_pool.shape[0]),
                        data=data_pool,
                        feature_names=feature_names_current
                    )

                plt.figure(figsize=(8, 6))
                shap.plots.beeswarm(expl, show=False)  # do not pass color/style; let SHAP choose
                plt.tight_layout()
                plt.savefig(f"{step_dir}/{self.target_name}_SHAP_beeswarm_{tag}_AVG.png", dpi=300, bbox_inches="tight")
                plt.close()

            # Save averaged test/train beeswarms if available
            save_avg_beeswarm(members_shap_values_test,  members_X_test,  tag="test")
            save_avg_beeswarm(members_shap_values_train, members_X_train, tag="train")
            # --- 1) Average scalar metrics across members ---
            avg_r        = float(np.mean(ensemble_rs))
            avg_rho      = float(np.mean(ensemble_rhos))
            avg_p        = float(np.mean(ensemble_p_values))
            avg_train_mse= float(np.mean(ensemble_train_mse))
            avg_test_mse = float(np.mean(ensemble_test_mse))
            avg_train_rmse = float(np.sqrt(avg_train_mse))
            avg_test_rmse  = float(np.sqrt(avg_test_mse))
            
            pd.DataFrame([{
                "r": avg_r,
                "rho": avg_rho,
                "p_value": avg_p,
                "train_mse": avg_train_mse,
                "mse": avg_test_mse,
                "train_rmse": avg_train_rmse,
                "rmse": avg_test_rmse,
                "members": members,
                "features_remaining": number_of_features
            }]).to_csv(f"{step_dir}/{self.target_name}_metrics_avg_over_members.csv", index=False)
            
            # --- 2) “Average” Actual vs Predicted plot across members ---
            # We’ll try a *true* mean prediction per sample if we can align test indices.
            # Your nested_eval already returns y_test and y_pred in metrics.
            # If you can also add 'test_index' to metrics, we can align perfectly.
            try:
                # Build a dict of predictions by row index
                preds_by_index = {}
                y_by_index = {}
                have_indices = True
                for m_metrics in [metrics_df.iloc[0] for _ in range(1)]:  # placeholder to keep scope tools quiet
                    pass
                # Re-read actual member-level metrics we collected
                members_payload = []  # (y_test, y_pred, test_index or None)
                for m in range(members):
                    # reconstruct from files we just wrote (fast) OR
                    # keep them in lists while looping members (recommended).
                    # If you still have them in memory from the loop, use that instead:
                    #   y_t = ensemble_y_tests[m]; y_p = ensemble_y_preds[m]; idx = ensemble_test_idx[m] (optional)
                    pass
            except Exception:
                have_indices = False
            
            # Simpler: if you *can* keep these during the loop, do this:
            # (Add these lists before the loop)
            #   ensemble_y_tests, ensemble_y_preds, ensemble_test_idx = [], [], []
            # (Inside the loop after metrics_df):
            #   ensemble_y_tests.append(metrics['y_test'])
            #   ensemble_y_preds.append(metrics['y_pred'])
            #   ensemble_test_idx.append(metrics.get('test_index', None))
            
            avg_plot_save = f"{step_dir}/{self.target_name}_actual_vs_predicted_AVG.png"
            if members > 0:
                if all(x is not None for x in locals().get('ensemble_test_idx', [])) \
                   and len(set(tuple(np.asarray(idx).ravel()) for idx in ensemble_test_idx)) == 1:
                    # True per-row mean (same test rows across members)
                    y_test_ref = np.asarray(ensemble_y_tests[0]).ravel()
                    y_pred_stack = np.vstack([np.asarray(p).ravel() for p in ensemble_y_preds])
                    y_pred_mean = np.mean(y_pred_stack, axis=0)
                    plot_df_avg = pd.DataFrame({"Actual": y_test_ref, "Predicted": y_pred_mean})
                    avg_metrics = {
                        "r": avg_r,
                        "rho": avg_rho,
                        "p_value": avg_p,
                        "mse": avg_test_mse,
                        "rmse": avg_test_rmse,
                        "y_test": y_test_ref,
                        "y_pred": y_pred_mean,
                    }
                else:
                    # Fallback: pool all member predictions for a stable scatter cloud
                    plot_df_avg = pd.DataFrame({
                        "Actual": np.concatenate([np.asarray(y).ravel() for y in ensemble_y_tests]),
                        "Predicted": np.concatenate([np.asarray(p).ravel() for p in ensemble_y_preds]),
                    })
                    avg_metrics = {
                        "r": avg_r,
                        "rho": avg_rho,
                        "p_value": avg_p,
                        "mse": avg_test_mse,
                        "rmse": avg_test_rmse,
                        "y_test": plot_df_avg["Actual"].to_numpy(),
                        "y_pred": plot_df_avg["Predicted"].to_numpy(),
                    }

                self.plot(
                    f"Actual vs. Prediction (AVERAGED over {members} members) - {self.model_name} - {self.target_name} - No. features: {number_of_features}",
                    modality='',
                    plot_df=plot_df_avg,
                    save_path_file=avg_plot_save,
                    N=len(plot_df_avg),
                    metrics_override=avg_metrics
                )
            
            # --- 3) Averaged SHAP beeswarm across members ---
            # We support both train and test SHAP; prefer test if present.
            # During the member loop, also append raw SHAP arrays + feature names:
            #   members_shap_values_test = []; members_X_test = []
            #   (inside member loop):
            #   if hasattr(self, "test_shap_values") and self.test_shap_values is not None:
            #       members_shap_values_test.append(self.test_shap_values)  # (n_samples, n_features)
            #       members_X_test.append(self.test_X_for_shap)             # Data used to color beeswarm (optional)
            
            def _save_avg_beeswarm(shap_list, X_list, tag="test"):
                if not shap_list:
                    return
                save_bee = f"{step_dir}/{self.target_name}_SHAP_beeswarm_{tag}_AVG.png"
            
                # Try true mean per row if rows align; else pool values.
                same_rows = False
                if X_list and all(x is not None for x in X_list):
                    # crude alignment check on shape/indices if you keep an index
                    try:
                        same_rows = all(getattr(X_list[0], "index", None) is not None and
                                        getattr(X_list[k], "index", None) is not None and
                                        (X_list[k].index.equals(X_list[0].index)) for k in range(len(X_list)))
                    except Exception:
                        same_rows = False
            
                import shap
                import matplotlib.pyplot as plt
            
                if same_rows:
                    # True mean across members: average SHAP per-sample, per-feature
                    shap_stack = np.stack(shap_list, axis=0)  # (members, n_samples, n_features)
                    shap_avg = np.mean(shap_stack, axis=0)    # (n_samples, n_features)
                    X_ref = X_list[0]
                    plt.figure(figsize=(8, 6))
                    shap.plots.beeswarm(shap.Explanation(values=shap_avg, feature_names=list(X_ref.columns)),
                                        show=False)
                    plt.tight_layout()
                    plt.savefig(save_bee, dpi=300, bbox_inches="tight")
                    plt.close()
                else:
                    # Pool all SHAP rows from all members (good approximation to the average distribution)
                    shap_pool = np.vstack(shap_list)  # (sum_samples, n_features)
                    # Feature names from first X if available, otherwise from self.feature_selection
                    feature_names = list(X_list[0].columns) if (X_list and X_list[0] is not None) \
                                    else list(self.feature_selection['features'])
                    plt.figure(figsize=(8, 6))
                    shap.plots.beeswarm(shap.Explanation(values=shap_pool, feature_names=feature_names),
                                        show=False)
                    plt.tight_layout()
                    plt.savefig(save_bee, dpi=300, bbox_inches="tight")
                    plt.close()
            
            # Call the helper for test/train if you collected them during the loop
            try:
                _save_avg_beeswarm(members_shap_values_test, members_X_test, tag="test")
            except NameError:
                pass
            try:
                _save_avg_beeswarm(members_shap_values_train, members_X_train, tag="train")
            except NameError:
                pass
            # ========= END NEW =========
            voted_features = sorted(feature_votes.items(), key=lambda x: -x[1])[:features_to_remove]
            least_important_features = [feat for feat, count in voted_features]

            self.feature_selection['features'] = [f for f in self.feature_selection['features'] if f not in least_important_features]
            removals.extend(least_important_features)
            number_of_features -= features_to_remove

            # Store mean + std metrics
            rs.append(np.mean(ensemble_rs))
            rhos.append(np.mean(ensemble_rhos))
            p_values.append(np.mean(ensemble_p_values))
            train_rmse_list.append(np.mean(np.sqrt(ensemble_train_mse)))
            test_rmse_list.append(np.mean(np.sqrt(ensemble_test_mse)))

            r_std_list.append(np.std(ensemble_rs, ddof=1) if len(ensemble_rs) > 1 else 0.0)
            rho_std_list.append(np.std(ensemble_rhos, ddof=1) if len(ensemble_rhos) > 1 else 0.0)
            train_rmse_std_list.append(np.std(train_rmse_list, ddof=1) if len(train_rmse_list) > 1 else 0.0)
            test_rmse_std_list.append(np.std(test_rmse_list, ddof=1) if len(test_rmse_list) > 1 else 0.0)

            def ci95(x):
                if len(x) <= 1: return 0.0
                return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
            r_err  = ci95(ensemble_rs)
            rho_err = ci95(ensemble_rhos)
            train_rmse_err = ci95(ensemble_train_mse)
            test_rmse_err = ci95(ensemble_test_mse)

            # Final ensemble plot
            if self.model_name == "NGBoost":
                self.calibration_analysis(ablation_idx=i)

            i += 1
            if number_of_features <= threshold_to_one_fps:
                features_per_step = 1

            self.logging.info(f"Feature ablation finished. Final r: {rs[-1]}, min r: {np.min(rs)}, max r: {np.max(rs)}")

        # Final plotting with error bands
        custom_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#525252"]
        sns.set_theme(style="whitegrid", context="paper")
        sns.set_palette(custom_palette)

        x = range(i - 1)
        plot_df = pd.DataFrame({
            'x': x, 'rs': rs, 'rhos': rhos, 'train_rmse': train_rmse_list, 'test_rmse': test_rmse_list,
            'r_std': r_std_list, 'rho_std': rho_std_list,
            'train_rmse_std': train_rmse_std_list, 'test_rmse_std': test_rmse_std_list
        })

        save_path_ablation = f'{self.save_path}/ablation/'

        x = range(i - 1)
        plot_df = pd.DataFrame({
            'x': x, 'rs': rs, 'rhos': rhos, 'train_rmse': train_rmse_list, 'test_rmse': test_rmse_list,
            'r_std': r_std_list, 'rho_std': rho_std_list,
            'train_rmse_std': train_rmse_std_list, 'test_rmse_std': test_rmse_std_list
        })

        save_path_ablation = f'{self.save_path}/ablation/'

        # R Score plot with std as whiskers
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, rs, yerr=r_err, label="R Score", marker='o', capsize=4)
        plt.xlabel("Number of removed features")
        plt.ylabel("Pearson-R Score")
        plt.title("Pearson-R Scores Over Feature Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path_ablation}_{self.target_name}_feature_ablation_R.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Rho plot with std as whiskers
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, rhos, yerr=rho_err, label="Spearman-Rho", marker='o', capsize=4)
        plt.xlabel("Number of removed features")
        plt.ylabel("Spearman-Rho Score")
        plt.title("Spearman-Rho Scores Over Feature Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path_ablation}_{self.target_name}_feature_ablation_Rho.png', dpi=300, bbox_inches='tight')
        plt.close()

        # RMSE plot with std as whiskers
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, test_rmse_list, yerr=test_rmse_err, label="Test RMSE", marker='o', capsize=4)
        plt.errorbar(x, train_rmse_list, yerr=train_rmse_err, label="Train RMSE", marker='o', capsize=4)
        plt.xlabel("Number of removed features")
        plt.ylabel("RMSE")
        plt.title("Train and Test RMSE Over Feature Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path_ablation}_{self.target_name}_feature_ablation_errors.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save feature removal history
        pd.DataFrame({'Removed_Features': removals}).to_csv(f'{save_path_ablation}{self.target_name}_ablation_history.csv', index=False)

        return rs, p_values, removals
    
    def feature_ablation_ensemble(
            self, 
            folds: int = -1, 
            get_shap: bool = True, 
            tune: bool = False, 
            tune_folds: int = 10, 
            features_per_step: int = 1, 
            members: int = 1,
            threshold_to_one_fps: int = 10,
            test_set: bool = True):
        """
        Feature ablation where, at each step, we form an *ensemble* by averaging member predictions,
        then recompute all scores (RMSE, r, rho, p_value) w.r.t. that ensemble prediction.
    
        Also:
          • Keeps distinct member metrics so we can draw CIs in the ablation plots.
          • Saves an averaged (ensemble) SHAP beeswarm per step *and* writes the ensemble SHAP values to CSV
            with current column names (similar to nested_eval). Beeswarm includes color gradient by passing
            the matching feature matrix used for SHAP (averaged or pooled across members).
          • Removes features by majority vote over per-member importances.
    
        Outputs saved under: {self.save_path}/ablation/ablation_step[i]/ ...
        """

    
        # -------------------- small helpers --------------------
        def _pearson_r(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            if y_true.size < 2 or np.allclose(np.std(y_true), 0) or np.allclose(np.std(y_pred), 0):
                return 0.0, 1.0
            r, p = stats.pearsonr(y_true, y_pred)
            return float(r), float(p)
    
        def _spearman_rho(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            if y_true.size < 2:
                return 0.0
            rho, _ = stats.spearmanr(y_true, y_pred)
            return float(rho)
    
        def _mse(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            return float(np.mean((y_true - y_pred) ** 2))
    
        def _rmse(y_true, y_pred):
            return float(np.sqrt(_mse(y_true, y_pred)))
    
        # -------------------- containers for ablation curves (ENSEMBLE-level metrics) --------------------
        rs, rhos, p_values, removals = [], [], [], []
        train_rmse_list, test_rmse_list = [], []
        # stds from per-member metrics (for whiskers)
        r_std_list, rho_std_list, train_rmse_std_list, test_rmse_std_list = [], [], [], []
    
        number_of_features = len(self.feature_selection['features'])
        i = 1
    
        # Deterministic but distinct seeds per member across steps
        global_random_state = int(getattr(self, "random_state", 0))
        rng_global = np.random.default_rng(global_random_state)
        member_seeds = rng_global.integers(
            low=0, high=np.iinfo(np.uint32).max, size=members, dtype=np.uint32
        )
    
        while number_of_features > 0:
            step_dir = f'{self.save_path}/ablation/ablation_step[{i}]'
            os.makedirs(step_dir, exist_ok=True)
            self.logging.info(f"---- Starting ablation step {i} with {number_of_features} features remaining. ----")
    
            features_to_remove = min(features_per_step, number_of_features) if number_of_features > threshold_to_one_fps else 1
    
            # Per-member collections
            feature_votes = {}
            member_rs, member_rhos, member_p_values = [], [], []
            member_train_mse, member_test_mse = [], []
    
            member_y_tests, member_y_preds, member_test_index = [], [], []
    
            # SHAP collectors (arrays) and matching feature matrices for coloring
            shap_test_list, shap_train_list = [], []
            x_test_list,   x_train_list   = [], []
    
            for m in range(members):
                self.random_state = int(member_seeds[m])   # same seed for this member at every step
                self.logging.info(f"[ablation step {i}] member {m} seed={self.random_state}")
    
                member_dir = f'{step_dir}/member[{m}]'
                os.makedirs(member_dir, exist_ok=True)
    
                metrics = self.nested_eval(
                    folds=folds, 
                    get_shap=get_shap, 
                    tune=tune, 
                    tune_folds=tune_folds, 
                    ablation_idx=i, 
                    member_idx=m)
    
                # Persist member metrics (unchanged)
                pd.DataFrame([metrics]).to_csv(f'{member_dir}/{self.target_name}_metrics.csv', index=False)
    
                # Predictions & indices for building ensemble
                member_y_tests.append(np.asarray(metrics['y_test']).ravel())
                member_y_preds.append(np.asarray(metrics['y_pred']).ravel())
                member_test_index.append(metrics.get('test_index', None))
    
                # Member-level scatter plot
                self.plot(
                    f"Actual vs. Prediction {self.model_name}- {self.target_name} No. features: {number_of_features}", 
                    modality='',
                    plot_df=pd.DataFrame({'Actual': metrics['y_test'], 'Predicted': metrics['y_pred']}),
                    save_path_file=f'{member_dir}/{self.target_name}_actual_vs_predicted.png')
    
                # Distinct member scalar metrics (for CIs)
                member_rs.append(metrics['r'])
                member_rhos.append(metrics['rho'])
                member_p_values.append(metrics['p_value'])
                member_train_mse.append(metrics['train_mse'])
                member_test_mse.append(metrics['mse'])
    
                # Collect SHAP matrices + feature matrices for coloring.
                # ---- TEST ----
                if 'test_shap_values' in metrics and metrics['test_shap_values'] is not None:
                    shap_test_list.append(np.asarray(metrics['test_shap_values']))
                    # NEW: always try both sources for the features matrix
                    Xcand = metrics.get('X_test_for_shap', None)
                    if Xcand is None and hasattr(self, 'X_test') and self.X_test is not None:
                        Xcand = self.X_test
                    if Xcand is not None:
                        try:
                            x_test_list.append(Xcand[self.feature_selection['features']].copy())
                        except Exception:
                            pass
                elif hasattr(self, 'test_shap_mean') and getattr(self, 'test_shap_mean') is not None:
                    shap_test_list.append(np.asarray(self.test_shap_mean))
                    if hasattr(self, 'X_test') and self.X_test is not None:
                        try:
                            x_test_list.append(self.X_test[self.feature_selection['features']].copy())
                        except Exception:
                            pass
                        
                # ---- TRAIN ----
                if 'shap_values' in metrics and metrics['shap_values'] is not None:
                    shap_train_list.append(np.asarray(metrics['shap_values']))
                    if 'X_for_shap' in metrics and metrics['X_for_shap'] is not None:
                        try:
                            x_train_list.append(metrics['X_for_shap'][self.feature_selection['features']].copy())
                        except Exception:
                            pass
                elif hasattr(self, 'shap_mean') and getattr(self, 'shap_mean') is not None:
                    shap_train_list.append(np.asarray(self.shap_mean))
                    try:
                        x_train_list.append(self.X[self.feature_selection['features']].copy())
                    except Exception:
                        pass
                    
                # Majority vote using per-member importances
                importance = metrics['feature_importance_test'] if test_set else metrics['feature_importance']
                importance_indices = np.argsort(importance)
                least_important = [self.feature_selection['features'][idx] for idx in importance_indices[:features_per_step]]
                for feat in least_important:
                    feature_votes[feat] = feature_votes.get(feat, 0) + 1
    
            # -------------------- Build ensemble prediction (aligned) --------------------
            can_simple_stack = (
                members > 0 and
                all(idx is not None for idx in member_test_index) and
                len({tuple(np.asarray(idx).ravel()) for idx in member_test_index}) == 1 and
                len({len(y) for y in member_y_tests}) == 1 and
                len({len(p) for p in member_y_preds}) == 1
            )
            if can_simple_stack:
                y_test_ref = member_y_tests[0]
                y_pred_stack = np.vstack(member_y_preds)  # (members, n_rows)
                y_pred_ens = np.mean(y_pred_stack, axis=0)
            else:
                # robust alignment by index; averages across available predictions per row-id
                pred_map, y_map = {}, {}
                for y_t, y_p, idx in zip(member_y_tests, member_y_preds, member_test_index):
                    if idx is None:
                        n = min(len(y_t), len(y_p))
                        for k in range(n):
                            pred_map.setdefault(k, []).append(y_p[k])
                            y_map[k] = y_t[k]
                    else:
                        idx = np.asarray(idx).ravel()
                        for k, rid in enumerate(idx):
                            pred_map.setdefault(rid, []).append(y_p[k])
                            y_map[rid] = y_t[k]
                aligned_ids = list(y_map.keys())
                aligned_ids.sort(key=lambda z: (isinstance(z, (str, bytes)), z))
                y_test_ref = np.array([y_map[rid] for rid in aligned_ids], dtype=float)
                y_pred_ens = np.array([np.mean(pred_map[rid]) for rid in aligned_ids], dtype=float)
    
            # Ensemble-level metrics
            ens_r, ens_p = _pearson_r(y_test_ref, y_pred_ens)
            ens_rho = _spearman_rho(y_test_ref, y_pred_ens)
            ens_mse_test = _mse(y_test_ref, y_pred_ens)
            ens_rmse_test = float(np.sqrt(ens_mse_test))
            ens_mse_train = float(np.mean(member_train_mse)) if member_train_mse else np.nan
            ens_rmse_train = float(np.sqrt(ens_mse_train)) if not np.isnan(ens_mse_train) else np.nan

            # Save ensemble metrics + ensemble Actual vs Predicted
            pd.DataFrame([{
                "r_ensemble": ens_r,
                "rho_ensemble": ens_rho,
                "p_value_ensemble": ens_p,
                "train_mse_proxy_mean": ens_mse_train,
                "mse_ensemble": ens_mse_test,
                "train_rmse_proxy": ens_rmse_train,
                "rmse_ensemble": ens_rmse_test,
                "members": members,
                "features_remaining": number_of_features,
                "y_test": json.dumps(y_test_ref.tolist()),
                "y_pred": json.dumps(y_pred_ens.tolist())
            }]).to_csv(f"{step_dir}/{self.target_name}_metrics_ENSEMBLE.csv", index=False)

            denom = 1 - (ens_r ** 2)
            f_squared_ens = np.inf if np.isclose(denom, 0) else (ens_r ** 2) / denom
            ensemble_metrics = {
                "r": ens_r,
                "rho": ens_rho,
                "p_value": ens_p,
                "f_squared": f_squared_ens,
                "mse": ens_mse_test,
                "rmse": ens_rmse_test,
                "y_test": y_test_ref,
                "y_pred": y_pred_ens,
            }
            self.metrics = ensemble_metrics

            self.plot(
                f"Actual vs. Prediction (ENSEMBLE of {members}) - {self.model_name} - {self.target_name} - No. features: {number_of_features}",
                modality='',
                plot_df=pd.DataFrame({"Actual": y_test_ref, "Predicted": y_pred_ens}),
                save_path_file=f"{step_dir}/{self.target_name}_actual_vs_predicted_ENSEMBLE.png",
                N=len(y_test_ref),
                metrics_override=ensemble_metrics
            )
    
            # ===================== ENSEMBLE SHAP: collect & save (CSV + beeswarm w/ color) =====================
            feature_names_current = list(self.feature_selection['features'])
            n_feat_current = len(feature_names_current)
    
            def _collect_member_shap_arrays(shap_src_list):
                out = []
                for sv in shap_src_list:
                    if sv is None:
                        continue
                    arr = np.asarray(sv)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    if arr.shape[1] < n_feat_current:
                        continue
                    if arr.shape[1] > n_feat_current:
                        arr = arr[:, :n_feat_current]
                    out.append(arr)
                return out
    
            def _collect_member_X_arrays(X_src_list):
                outs = []
                for X in (X_src_list or []):
                    if X is None:
                        continue
                    try:
                        if isinstance(X, pd.DataFrame):
                            Xc = X[feature_names_current]
                            arr = Xc.to_numpy(dtype=float)
                        elif isinstance(X, pd.Series):
                            arr = X.loc[:, feature_names_current].to_numpy(dtype=float)
                        else:
                            arr = np.asarray(X, dtype=float)
                            if arr.ndim == 1:
                                arr = arr.reshape(-1, 1)
                            if arr.shape[1] < n_feat_current:
                                continue
                            if arr.shape[1] > n_feat_current:
                                arr = arr[:, :n_feat_current]
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        outs.append(arr)
                    except Exception:
                        pass
                return outs
    
            def _average_or_pool(arr_list: list) -> np.ndarray:
                if not arr_list:
                    return None
                shapes = {(a.shape[0], a.shape[1]) for a in arr_list}
                if len(shapes) == 1:
                    stack = np.stack(arr_list, axis=0)      # (members, n_rows, n_feat)
                    return np.mean(stack, axis=0)           # (n_rows, n_feat)
                return np.vstack(arr_list)                  # (sum_rows, n_feat)
    
            def _save_csv(arr2d: np.ndarray, tag: str):
                csv_path = os.path.join(step_dir, f"{self.target_name}_mean_shap_values_{tag}_ENSEMBLE.csv")
                pd.DataFrame(arr2d, columns=feature_names_current).to_csv(csv_path, index=False)
                return csv_path
    
            def _save_beeswarm(arr2d: np.ndarray, X_arr: np.ndarray, tag: str):
                import shap
                max_disp = getattr(self, "top_n", None)
                plt.figure(figsize=(8, 6))
                if X_arr is not None and X_arr.shape == arr2d.shape:
                    shap.summary_plot(arr2d, features=X_arr, feature_names=feature_names_current,
                                      show=False, max_display=max_disp)
                else:
                    shap.summary_plot(arr2d, features=None, feature_names=feature_names_current,
                                      show=False, max_display=max_disp)
                plt.title(f"{self.target_name} SHAP Beeswarm (Ensemble, {tag})", fontsize=14)
                plt.tight_layout()
                out_png = os.path.join(step_dir, f"{self.target_name}_shap_aggregated_beeswarm_{tag}_ENSEMBLE.png")
                plt.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.close()
                self.logging.info(f"[ablation {i}] Saved ensemble SHAP beeswarm ({tag}) to: {out_png}")
    
            def _save_placeholder(reason_txt: str, tag: str):
                with open(os.path.join(step_dir, f"{self.target_name}_SHAP_{tag}_ENSEMBLE_reason.txt"),
                          "w", encoding="utf-8") as fh:
                    fh.write(reason_txt)
                fig = plt.figure(figsize=(4, 2))
                plt.axis('off')
                plt.text(0.02, 0.5, f"No averaged SHAP ({tag})\n{reason_txt}", va='center')
                plt.tight_layout()
                fig.savefig(os.path.join(step_dir, f"{self.target_name}_shap_aggregated_beeswarm_{tag}_ENSEMBLE.png"),
                            dpi=200, bbox_inches="tight")
                plt.close(fig)
                self.logging.warning(f"[ablation {i}] {reason_txt}")
    
            # Build normalized sets
            test_mats  = _collect_member_shap_arrays(shap_test_list)
            train_mats = _collect_member_shap_arrays(shap_train_list)
            Xtest_mats = _collect_member_X_arrays(x_test_list)
            Xtrain_mats= _collect_member_X_arrays(x_train_list)
    
            # TEST save
            # === TEST save (index-aware averaging) ===
            if test_mats and member_test_index and any(idx is not None for idx in member_test_index):
                # Build: row_id -> list of SHAP vectors (across members and folds)
                per_id = {}
            
                # We need per-member, per-fold SHAP matrices and their exact row indices
                # 'shap_test_list' aligns 1:1 with members (we appended once per member)
                # but each entry can be pooled over folds; we therefore rely on metrics['test_index']
                # which we already collected in member_test_index.
                for sv, idx in zip(shap_test_list, member_test_index):
                    if sv is None or idx is None:
                        continue
                    sv = np.asarray(sv)
                    idx = np.asarray(idx).ravel()
                    # If the member concatenated folds, idx length should match sv.shape[0]
                    n = min(len(idx), sv.shape[0])
                    for rid, rowvec in zip(idx[:n], sv[:n, :n_feat_current]):
                        per_id.setdefault(int(rid), []).append(rowvec.astype(float))
            
                if not per_id:
                    _save_placeholder("No valid test SHAP matrices with indices collected.", tag="test")
                else:
                    # Average across members for each row-id; then reindex to full X order
                    target_order = list(self.X.index)
                    rows = []
                    keep_mask = []
                    for rid in target_order:
                        vecs = per_id.get(int(rid), None)
                        if vecs is None:
                            rows.append(np.full((n_feat_current,), np.nan))
                            keep_mask.append(False)  # unseen in test → NaNs
                        else:
                            rows.append(np.mean(np.stack(vecs, axis=0), axis=0))
                            keep_mask.append(True)
            
                    ens_test_aligned = np.vstack(rows)  # (n_full_rows, n_feat_current)
                    # Save two files:
                    # 1) Strictly aligned to full X (includes NaNs where a row never appeared in any test fold)
                    aligned_full_path = os.path.join(step_dir, f"{self.target_name}_mean_shap_values_test_ALIGNED.csv")
                    pd.DataFrame(ens_test_aligned, index=self.X.index, columns=feature_names_current)\
                      .to_csv(aligned_full_path, index=True)  # KEEP the index
            
                    # 2) Compact version containing only rows that actually appeared in test
                    compact = ens_test_aligned[np.array(keep_mask, dtype=bool)]
                    compact_path = os.path.join(step_dir, f"{self.target_name}_mean_shap_values_test_ENSEMBLE.csv")
                    pd.DataFrame(compact, columns=feature_names_current).to_csv(compact_path, index=False)
            
                    # Beeswarm (only rows seen in test, to match compact)
                    try:
                        X_test_ens = None
                        if Xtest_mats:
                            # index-aware average for X too (optional)
                            per_id_X = {}
                            for Xcand, idx in zip(Xtest_mats, member_test_index):
                                if Xcand is None or idx is None:
                                    continue
                                Xcand = np.asarray(Xcand, dtype=float)
                                idx = np.asarray(idx).ravel()
                                n = min(len(idx), Xcand.shape[0])
                                for rid, rowx in zip(idx[:n], Xcand[:n, :n_feat_current]):
                                    per_id_X.setdefault(int(rid), []).append(rowx)
                            if per_id_X:
                                X_rows = [np.mean(np.stack(per_id_X[rid], axis=0), axis=0)
                                          for rid, keep in zip(target_order, keep_mask) if keep and rid in per_id_X]
                                if X_rows:
                                    X_test_ens = np.vstack(X_rows)
            
                        _save_beeswarm(compact, X_test_ens, tag="test")
                    except Exception as e:
                        _save_placeholder(f"Beeswarm failed for test ({type(e).__name__}: {e})", tag="test")
            else:
                _save_placeholder("No valid test SHAP matrices collected.", tag="test")
            
    
            # TRAIN save
            if train_mats:
                ens_train = _average_or_pool(train_mats)
                X_train_ens = _average_or_pool(Xtrain_mats) if Xtrain_mats else None
                csv_path = _save_csv(ens_train, tag="train")
                if ens_train is not None and isinstance(ens_train, np.ndarray):
                    aligned_train_path = os.path.join(step_dir, f"{self.target_name}_mean_shap_values_train_ALIGNED.csv")
                    pd.DataFrame(ens_train, columns=feature_names_current).to_csv(aligned_train_path, index=False)
                try:
                    _save_beeswarm(ens_train, X_train_ens, tag="train")
                except Exception as e:
                    _save_placeholder(f"Beeswarm failed for train ({type(e).__name__}: {e})", tag="train")
                self.logging.info(f"[ablation {i}] Saved ensemble TRAIN SHAP CSV: {csv_path}")
            else:
                _save_placeholder("No valid train SHAP matrices collected (width < current feature count or empty).", tag="train")
            # =================== END ENSEMBLE SHAP save block ===================
    
            # -------------------- Remove features by majority vote --------------------
            voted_features = sorted(feature_votes.items(), key=lambda x: -x[1])[:features_to_remove]
            least_important_features = [feat for feat, _ in voted_features]
    
            self.X = self.X.drop(columns=least_important_features)
            self.feature_selection['features'] = [f for f in self.feature_selection['features'] if f not in least_important_features]
            removals.extend(least_important_features)
            number_of_features -= features_to_remove
    
            # -------------------- Append ENSEMBLE curves + CI from member metrics --------------------
            rs.append(ens_r)
            rhos.append(ens_rho)
            p_values.append(ens_p)
            train_rmse_list.append(ens_rmse_train)
            test_rmse_list.append(ens_rmse_test)
    
            r_std_list.append(np.std(member_rs, ddof=1) if len(member_rs) > 1 else 0.0)
            rho_std_list.append(np.std(member_rhos, ddof=1) if len(member_rhos) > 1 else 0.0)
            member_train_rmse = [np.sqrt(x) for x in member_train_mse] if member_train_mse else []
            member_test_rmse  = [np.sqrt(x) for x in member_test_mse]  if member_test_mse  else []
            train_rmse_std_list.append(np.std(member_train_rmse, ddof=1) if len(member_train_rmse) > 1 else 0.0)
            test_rmse_std_list.append(np.std(member_test_rmse,  ddof=1) if len(member_test_rmse)  > 1 else 0.0)
    
            if self.model_name == "NGBoost":
                self.calibration_analysis(ablation_idx=i)
    
            i += 1
            if number_of_features <= threshold_to_one_fps:
                features_per_step = 1
    
            self.logging.info(f"[ablation step {i-1}] Ensemble r={rs[-1]:.4f} | min/max member r=({np.min(member_rs):.4f},{np.max(member_rs):.4f})")
    
        # -------------------- Final plots (whiskers = member stds) --------------------
        custom_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#525252"]
        sns.set_theme(style="whitegrid", context="paper")
        sns.set_palette(custom_palette)
    
        x = list(range(i - 1))
        save_path_ablation = f'{self.save_path}/ablation/'
    
        r_err   = r_std_list
        rho_err = rho_std_list
        train_rmse_err = train_rmse_std_list
        test_rmse_err  = test_rmse_std_list
    
        # R (ensemble)
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, rs, yerr=r_err, label="Pearson-R (Ensemble)", marker='o', capsize=4)
        plt.xlabel("Number of removed features")
        plt.ylabel("Pearson-R")
        plt.title("Ensemble Pearson-R Over Feature Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path_ablation}_{self.target_name}_feature_ablation_ENSEMBLE_R.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        # Rho (ensemble)
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, rhos, yerr=rho_err, label="Spearman-Rho (Ensemble)", marker='o', capsize=4)
        plt.xlabel("Number of removed features")
        plt.ylabel("Spearman-Rho")
        plt.title("Ensemble Spearman-Rho Over Feature Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path_ablation}_{self.target_name}_feature_ablation_ENSEMBLE_Rho.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        # RMSE (ensemble)
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, test_rmse_list, yerr=test_rmse_err, label="Test RMSE (Ensemble)", marker='o', capsize=4)
        plt.errorbar(x, train_rmse_list, yerr=train_rmse_err, label="Train RMSE (proxy mean)", marker='o', capsize=4)
        plt.xlabel("Number of removed features")
        plt.ylabel("RMSE")
        plt.title("Ensemble Train/Test RMSE Over Feature Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path_ablation}_{self.target_name}_feature_ablation_ENSEMBLE_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        # Save feature removal history
        pd.DataFrame({'Removed_Features': removals}).to_csv(
            f'{save_path_ablation}{self.target_name}_ENSEMBLE_ablation_history.csv', index=False)
    
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
    
    
        

    def plot(self, title, modality='', plot_df=None, save_path_file=None, N=None, metrics_override=None) -> None:
        """
        Generate a scatter plot of predicted vs. actual target values with regression line and confidence intervals.

        Args:
            title (str): Title of the plot.
            modality (str, optional): Modality string to label axes (e.g., "MRI"). Defaults to empty string.
            metrics_override (dict, optional): Metrics dictionary to use for annotations/derived values.
                Falls back to `self.metrics` if not provided.
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

        metrics_src = metrics_override if metrics_override is not None else self.metrics
        if metrics_src is None:
            raise ValueError("No metrics available for plotting. Provide metrics_override or ensure self.metrics is set.")

        if plot_df is None:
            # Create a DataFrame for Seaborn
            if 'y_test' not in metrics_src or 'y_pred' not in metrics_src:
                raise ValueError("Metrics must contain 'y_test' and 'y_pred' when plot_df is not provided.")
            plot_df = pd.DataFrame({
                'Actual': metrics_src['y_test'],
                'Predicted': metrics_src['y_pred']
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
        text_lines = []
        r_val = metrics_src.get("r")
        if r_val is not None and not pd.isna(r_val):
            text_lines.append(f'Pearson R: {r_val:.2f}')

        rho_val = metrics_src.get("rho")
        if rho_val is not None and not pd.isna(rho_val):
            text_lines.append(f'Spearman ρ: {rho_val:.2f}')

        p_val = metrics_src.get("p_value")
        if p_val is not None and not pd.isna(p_val):
            text_lines.append(f'P-value: {p_val:.6f}')

        f_sq = metrics_src.get("f_squared")
        if (f_sq is None or pd.isna(f_sq)) and r_val is not None and not pd.isna(r_val):
            denom = 1 - r_val ** 2
            if np.isclose(denom, 0):
                f_sq = np.inf
            else:
                f_sq = (r_val ** 2) / denom
        if f_sq is not None and not pd.isna(f_sq):
            text_lines.append(f'f²: {f_sq:.3f}' if np.isfinite(f_sq) else 'f²: ∞')

        rmse_val = metrics_src.get("rmse")
        mse_val = metrics_src.get("mse")
        if rmse_val is None and mse_val is not None and not pd.isna(mse_val):
            rmse_val = np.sqrt(mse_val)
        if rmse_val is not None and not pd.isna(rmse_val):
            text_lines.append(f'RMSE: {rmse_val:.6f}')

        if text_lines:
            plt.text(
                0.05, 0.95, 
                "\n".join(text_lines),
                fontsize=12, 
                transform=plt.gca().transAxes,
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
        if N is not None:
            plt.title(title + "  N=" + str(N), fontsize=14)
        else:   
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
            plt.savefig(f'{self.save_path}/{self.target_name}_actual_vs_predicted.png')

        plt.close()

        # Log info (optional)
        self.logging.info("Plot saved to %s/%s_actual_vs_predicted.png", 
        self.save_path, self.target_name)
