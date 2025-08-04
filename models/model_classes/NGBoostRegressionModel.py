# Standard Libraries
import os
from typing import Dict, Tuple

# Data Handling and Numeric Computation
import numpy as np
import pandas as pd

# Machine Learning and Modeling
from ngboost import NGBRegressor
from model_classes.faster_evidential_boost import NIGLogScore
from model_classes.faster_evidential_boost import NormalInverseGamma
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    KFold, LeaveOneOut,
    GroupKFold, LeaveOneGroupOut
)
# Visualization and Explainability
import matplotlib.pyplot as plt
import shap
import pickle
# Uncertainty Quantification and Calibration
import scipy.stats as st
from properscoring import crps_ensemble

# Hyperparameter Tuning with Ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp

# Custom Base Model
from model_classes.BaseRegressionModel import BaseRegressionModel


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
            param_grid: dict = None,
            logging = None,
            standardize=False,
            split_shaps=False,
            Pat_IDs=None):
        
        super().__init__(
            data_df, 
            feature_selection, 
            target_name, 
            test_split_size, 
            save_path, 
            identifier, 
            top_n, 
            logging=logging, 
            standardize=standardize,
            split_shaps=split_shaps,
            Pat_IDs=Pat_IDs)

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

    def model_specific_preprocess(self, data_df: pd.DataFrame, ceiling: list = ["BDI", "MoCA"]) -> Tuple:
        """
        Preprocess data specific to model requirements, including feature extraction,
        optional ceiling adjustments, and one-hot encoding of categorical features.

        Args:
            data_df (pd.DataFrame): Original input data.
            ceiling (list, optional): List of ceiling transformations to apply (e.g., ['BDI', 'MoCA']).
                Defaults to ["BDI", "MoCA"].

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series, float, float]:
                - X: Preprocessed feature DataFrame (with dummies for categorical cols).
                - y: Original target Series.
                - z: Standardized target Series.
                - m: Mean of original target.
                - std: Standard deviation of original target.
        """
        self.logging.info("Starting model-specific preprocessing...")

        # Drop rows with missing values for features and target
        data_df = data_df.dropna(subset=self.feature_selection['features'] + [self.feature_selection['target']])

        # Select features
        X = data_df[self.feature_selection['features']].copy()

        # Apply ceiling adjustment if requested
        if ceiling == "BDI":
            if "BDI_sum_pre" in X.columns:
                X["Distance_ceiling"] = 63 - X["BDI_sum_pre"]
                self.feature_selection["features"].append("Distance_ceiling")
            elif "MoCA_sum_pre" in X.columns:
                X["Distance_ceiling"] = 30 - X["MoCA_sum_pre"]
                self.feature_selection["features"].append("Distance_ceiling")
            else:
                raise ValueError("Neither BDI_sum_pre nor MoCA_sum_pre found in the DataFrame.")

        # Extract target
        y = data_df[self.feature_selection['target']]

        # One-hot encode any columns containing strings or categorical dtype
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            self.logging.info(f"One-hot encoding categorical columns: {cat_cols}")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            # Update features list to new dummy columns
            self.feature_selection['features'] = X.columns.tolist()

        # Fill numeric NaNs with column means
        X = X.fillna(X.mean())
        # Convert all to numeric (will coerce non-numeric if any slipped through)
        X = X.apply(pd.to_numeric, errors='coerce')

        # Standardize the target variable
        m = y.mean()
        std = y.std()
        z = (y - m) / std
        self.logging.info("Finished model-specific preprocessing.")
        return X, y, z, m, std


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """ Predict using the trained NGBoost model.
            By default, return the mean predictions.
        """
        self.logging.info("Starting NGBoost prediction...")
        # NGBoost's predict method returns the mean predictions
        pred = self.model.predict(X)
        self.logging.info("Finished NGBoost prediction.")
        return pred

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns both the mean predictions and the variance (from the predictive distribution). """
        self.logging.info("Starting NGBoost prediction with uncertainty estimation...")
        pred_dist = self.model.pred_dist(X)
        pass

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
    
        Args:
            X:           Training features.
            y:           Training targets.
            param_grid:  Dict of parameters to search (use prefixes 'Score__', 'Base__', etc).
            folds:       Number of CV folds (or -1 → leave‑one‑out).
            groups:      Optional array of group labels for group‑aware CV.
        """
        # allow -1 → leave‐one‐out#
        del weights  # TODO: implement weights
        if folds == -1:
            folds = len(X)
    
        # choose splitter
        if groups is None:
            cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            cv = GroupKFold(n_splits=folds)
    
        # Pearson‐r scorer
        def pearson_corr(y_true, y_pred):
            return pearsonr(y_true, y_pred)[0]
        #pearson_scorer = make_scorer(pearson_corr, greater_is_better=True)
    
        # set up grid search
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0
        )
    
        # fit, passing groups only if needed
        if groups is None:
            grid_search.fit(X, y)
        else:
            grid_search.fit(X, y, groups=groups)
    
        best_params = grid_search.best_params_
        best_score  = grid_search.best_score_
    
        # extract Score__ params
        score_params = {
            k.replace("Score__", ""): v
            for k, v in best_params.items()
            if k.startswith("Score__")
        }
        NIGLogScore.set_params(**score_params)
    
        # extract Base__ params
        base_params = {
            k.replace("Base__", ""): v
            for k, v in best_params.items()
            if k.startswith("Base__")
        }
        if base_params:
            base_learner = DecisionTreeRegressor(**base_params)
        else:
            base_learner = DecisionTreeRegressor(max_depth=3)
    
        # clean out Score__ and Base__ keys
        clean_params = {
            k: v for k, v in best_params.items()
            if not (k.startswith("Score__") or k.startswith("Base__"))
        }
        clean_params['Base'] = base_learner
    
        # rebuild and refit NGBoost
        self.model = NGBRegressor(
            Dist=NormalInverseGamma,
            Score=NIGLogScore,
            verbose=False,
            **clean_params
        )
        # double‐fit to initialize
        self.model.fit(X, y)
        self.model.fit(X, y)
    
        if hasattr(self, 'logging') and self.logging:
            self.logging.info(f"NGBoost best_params: {best_params}")
            self.logging.info(f"NGBoost best_score: {best_score:.4f}")
    
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

    def feature_importance_mean(
            self, 
            X,
            top_n: int = None, 
            batch_size: int = 10, 
            save_results: bool = True, 
            iter_idx=None, 
            ablation_idx=None) -> Dict:
        """ Compute feature importance for the predicted mean using SHAP KernelExplainer. """

        shap.initjs()
        explainer = shap.TreeExplainer(self.model, model_output=0)
        shap_values = explainer.shap_values(X, check_additivity=True)

        shap.summary_plot(shap_values, features=X, feature_names=self.X.columns, show=False, max_display=self.top_n)
        plt.title(f'{self.identifier} NGBoost Mean SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            if iter_idx is not None:
                save_path = self.save_path + "/singleSHAPs"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{self.identifier}_mean_shap_aggregated_beeswarm_{iter_idx}.png')
                with open(f'{save_path}/{self.identifier}_mean_shap_explanations{iter_idx}.pkl', 'wb') as fp:
                    pickle.dump(explainer, fp)
            elif ablation_idx is not None:
                save_path = self.save_path + "/ablationSHAPs"
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_shap_aggregated_beeswarm{ablation_idx}.png')
            else:
                plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_mean_shap_aggregated_beeswarm.png')
            plt.close()

        if self.scaler is not None:
            if hasattr(self.scaler, 'std_'):
                scale = self.scaler.std_
            elif hasattr(self.scaler, 'mad_'):
                scale = self.scaler.mad_
            else:
                raise AttributeError("Scaler has neither 'std_' nor 'mad_' attribute.")
            shap_values *= scale 

        return shap_values

    def feature_importance_variance(
            self, 
            X,
            mode = "nig",
            top_n: int = None, 
            batch_size: int = 10, 
            save_results: bool = True, 
            iter_idx=None, 
            ablation_idx=None) -> Dict:
        """ Compute feature importance for the predicted variance using SHAP KernelExplainer. """
        
        if mode == "nig":
            pred_dist = self.model.pred_dist(X).params
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
            explainer_lam = shap.TreeExplainer(self.model, model_output=1)
            sh_lam = explainer_lam.shap_values(X)
            explainer_alpha = shap.TreeExplainer(self.model, model_output=2)
            sh_alpha = explainer_alpha.shap_values(X)
            explainer_beta = shap.TreeExplainer(self.model, model_output=3)
            sh_beta = explainer_beta.shap_values(X)


            # 5) Apply chain rule (broadcast derivatives over feature axis)
            shap_pred = (dpred_dbeta[:, None]  * sh_beta
                       + dpred_dalpha[:, None] * sh_alpha
                       + dpred_dlam[:, None]   * sh_lam)

            shap_epi = (depi_dbeta[:, None]  * sh_beta
                      + depi_dalpha[:, None] * sh_alpha
                      + depi_dlam[:, None]   * sh_lam)

            shap_alea = (dalea_dbeta[:, None]  * sh_beta
               + dalea_dalpha[:, None] * sh_alpha)

            shap.summary_plot(shap_pred, features=X, feature_names=self.X.columns, show=False, max_display=self.top_n)
            plt.title(f'{self.identifier} NGBoost Variance SHAP Summary Plot (Aggregated)', fontsize=16)
            if save_results:
                plt.subplots_adjust(top=0.90)
                if iter_idx is not None:
                    save_path = self.save_path + "/singleSHAPs"
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(f'{save_path}/{self.identifier}_ngboost_predicitve_uncertainty_shap_aggregated{iter_idx}.png')
                    #with open(f'{save_path}/{self.identifier}_ngboost_predicitve_uncertainty_shap_explanations{iter_idx}.pkl', 'wb') as fp:
                    #    pickle.dump((explainer_lam, explainer_alpha, explainer_beta), fp)
                elif ablation_idx is not None:
                    save_path = self.save_path + "/ablationSHAPs"
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(f'{save_path}/{self.identifier}_{self.target_name}_predicitve_uncertainty_shap_aggregated{ablation_idx}.png')
                else:
                    plt.savefig(f'{self.save_path}/{self.identifier}_predicitve_uncertainty_shap_aggregated.png')
                plt.close()

           
            if self.scaler is not None:
                if hasattr(self.scaler, 'std_'):
                    scale = self.scaler.std_
                elif hasattr(self.scaler, 'mad_'):
                    scale = self.scaler.mad_
                else:
                    raise AttributeError("Scaler has neither 'std_' nor 'mad_' attribute.")
                shap_pred *= scale
                shap_epi *= scale
                shap_alea *= scale

            return shap_pred, shap_epi, shap_alea
        
        elif mode == "normal":
            shap.initjs()
            explainer = shap.TreeExplainer(self.model, model_output=1)
            shap_values = explainer.shap_values(X, check_additivity=True)
            shap.summary_plot(shap_values, features=X, feature_names=self.X.columns, show=False, max_display=self.top_n)
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

            if self.scaler is not None:
                shap_values = self.scaler.inverse_transform(shap_values)
            
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

    

    #def calibration_analysis(self, ablation_idx = None):
    #    """
    #    Generate PIT histogram, QQ-plot, quantile‐calibration diagram, and CRPS.
    #    Assumes that after LOOCV you have stored in self.metrics:
    #      - 'pred_dist': a tuple/ list of 4 numpy arrays (mu, lam, alpha, beta), each of length n_samples
    #      - 'y_test'   : the true y's for those held-out samples, length n_samples
    #    """
    #    # prepare output folder
    #    if ablation_idx is not None:
    #        save_path = f'{self.save_path}/ablation/ablation_step[{ablation_idx}]/calibration'
    #    else:
    #        save_path = f'{self.save_path}/calibration'
    #    
    #    os.makedirs(save_path, exist_ok=True)
#
    #    # pull out true values + predicted parameters
    #    pred_dist = self.metrics['pred_dist'].T
    #    y_test    = np.asarray(self.metrics['y_test'])
    #    n         = len(y_test)
#
    #    # decide which family
    #    if self.model.Dist == NormalInverseGamma:
    #        # NIG→Student-t
    #        mu_arr, lam_arr, alpha_arr, beta_arr = pred_dist[0], pred_dist[1], pred_dist[2], pred_dist[3]
    #        nu    = 2 * alpha_arr
    #        Omega = 2 * beta_arr * (1 + lam_arr)
    #        scale = np.sqrt(Omega / (lam_arr * nu))
    #        dists = [
    #            st.t(df=nu[i], loc=mu_arr[i], scale=scale[i])
    #            for i in range(n)
    #        ]
#
    #    elif self.model.Dist == Normal:
    #        # Gaussian
    #        mu_arr, sigma_arr = pred_dist[0], pred_dist[1]
    #        dists = [
    #            st.norm(loc=mu_arr[i], scale=sigma_arr[i])
    #            for i in range(n)
    #        ]
#
    #    else:
    #        raise ValueError(f"Expected 2 or 4 distribution parameters, got {len(pred_dist)}")
#
    #    # ---- 1) PIT ----
    #    pit = np.array([dist.cdf(y) for dist, y in zip(dists, y_test)])
#
    #    plt.figure()
    #    plt.hist(pit, bins=20, range=(0,1), edgecolor='k', alpha=0.7)
    #    plt.axhline(n/20, color='r', linestyle='--', label='ideal')
    #    plt.title('PIT Histogram')
    #    plt.xlabel('PIT')
    #    plt.ylabel('Count')
    #    plt.legend()
    #    plt.savefig(f'{save_path}/pit_hist.png')
    #    plt.close()
#
    #    # ---- 2) PIT QQ plot ----
    #    sorted_pit = np.sort(pit)
    #    uniform_q  = np.linspace(0,1,n)
    #    plt.figure()
    #    plt.plot(uniform_q, sorted_pit, marker='.', linestyle='none')
    #    plt.plot([0,1],[0,1], 'r--')
    #    plt.title('PIT QQ‐Plot')
    #    plt.xlabel('Uniform Quantile')
    #    plt.ylabel('Empirical PIT Quantile')
    #    plt.savefig(f'{save_path}/pit_qq.png')
    #    plt.close()
#
    #    # ---- 3) Quantile‐Calibration ----
    #    qs  = np.linspace(0.05, 0.95, 19)
    #    obs = []
    #    for q in qs:
    #        yq = np.array([dist.ppf(q) for dist in dists])
    #        obs.append(np.mean(y_test <= yq))
#
    #    plt.figure()
    #    plt.plot(qs, obs, marker='o', linestyle='-')
    #    plt.plot([0,1],[0,1],'r--')
    #    plt.title('Quantile Calibration')
    #    plt.xlabel('Nominal Quantile')
    #    plt.ylabel('Observed Fraction ≤ Predicted')
    #    plt.savefig(f'{save_path}/quantile_calib.png')
    #    plt.close()
#
    #    # ---- 4) CRPS ----
    #    # draw 500 samples per predictive distribution
    #    samples = np.stack([dist.rvs(size=500) for dist in dists], axis=1)
    #    # samples.shape == (500, n)
    #    crps_vals = crps_ensemble(y_test, samples.T)
    #    avg_crps  = crps_vals.mean()
#
    #    # baseline degenerate‐median model
    #    median_pred   = np.median(y_test)
    #    baseline_crps = np.mean(np.abs(y_test - median_pred))
#
    #    print(f"Average CRPS: {avg_crps:.4f}")
    #    print(f"Baseline CRPS (degenerate at median): {baseline_crps:.4f}")
#
    #    # ---- 5) ECE ----
    #    ece = np.mean(np.abs(np.array(obs) - qs))
    #    print(f"Expected Calibration Error (ECE): {ece:.4f}")
#
    #    calibration_results = {
    #        'ece': ece,
    #        'avg_crps': avg_crps,
    #        'baseline_crps': baseline_crps,
    #    }
    #    # Save calibration_results as CSV
    #    # Flatten the dict for CSV saving
    #    cal_df = pd.DataFrame([calibration_results],)
    #    cal_df.to_csv(f'{save_path}/calibration_metrics.csv', index=False)
#
    #    # Save PIT and quantile calibration arrays as separate CSVs for clarity
    #    pd.DataFrame({'pit': pit}).to_csv(f'{save_path}/pit_values.csv', index=False)
    #    pd.DataFrame({'qs': qs, 'obs': obs}).to_csv(f'{save_path}/quantile_calibration.csv', index=False)
    #    pd.DataFrame({'crps': crps_vals}).to_csv(f'{save_path}/crps_values.csv', index=False)
    #    
    #    return calibration_results
    def calibration_analysis(self, ablation_idx=None):
        """
        Generate PIT histogram, QQ‐plot, quantile‐calibration diagram, and CRPS.
        Assumes that after LOOCV you have stored in self.metrics:
          - 'pred_dist': a (n_samples × 4) array of (mu, lam, alpha, beta) or (mu, sigma)
          - 'y_test'   : the true y's (raw, unscaled), length n_samples
        And that self.scaler implements fit/transform/inverse_transform on y.
        """
        # 1) Prepare paths
        if ablation_idx is not None:
            save_path = f'{self.save_path}/ablation/ablation_step[{ablation_idx}]/calibration'
        else:
            save_path = f'{self.save_path}/calibration'
        os.makedirs(save_path, exist_ok=True)

        # 2) Pull out preds + truths
        pred_dist = np.asarray(self.metrics['pred_dist']).T
        y_test_raw = np.asarray(self.metrics['y_test'])
        n = len(y_test_raw)

        # 3) Scale the true y's into model‐space
        #    (so we can feed them to the CDF of our scaled‐space distributions)
        y_test = self.scaler.transform(y_test_raw.reshape(-1,1)).ravel()

        # 4) Build scipy Distr objects in *scaled* space
        dists = []
        if self.model.Dist == NormalInverseGamma:
            mu_arr, lam_arr, alpha_arr, beta_arr = pred_dist
            nu    = 2 * alpha_arr
            Omega = 2 * beta_arr * (1 + lam_arr)
            scale = np.sqrt(Omega / (lam_arr * nu))
            for i in range(n):
                dists.append(st.t(df=nu[i], loc=mu_arr[i], scale=scale[i]))
        elif self.model.Dist == Normal:
            mu_arr, sigma_arr = pred_dist
            for i in range(n):
                dists.append(st.norm(loc=mu_arr[i], scale=sigma_arr[i]))
        else:
            raise ValueError(f"Expected 2 or 4 distribution parameters, got {len(pred_dist)}")

        # 5) PIT histogram (scaled)
        pit = np.array([dist.cdf(y_test[i]) for i, dist in enumerate(dists)])
        plt.figure()
        plt.hist(pit, bins=20, range=(0,1), edgecolor='k', alpha=0.7)
        plt.axhline(n/20, color='r', linestyle='--', label='ideal')
        plt.title('PIT Histogram')
        plt.xlabel('PIT')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f'{save_path}/pit_hist.png')
        plt.close()

        # 6) PIT QQ‐plot
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

        # 7) Quantile‐Calibration
        qs  = np.linspace(0.05, 0.95, 19)
        obs = []
        for q in qs:
            # 7a) compute predicted q‐quantile in scaled space
            yq_scaled = np.asarray([dist.ppf(q) for dist in dists])
            # 7b) bring it back to raw space
            yq_raw = self.scaler.inverse_transform(yq_scaled.reshape(-1,1)).ravel()
            # 7c) fraction of true‐raw ≤ predicted‐raw
            obs.append(np.mean(y_test_raw <= yq_raw))

        plt.figure()
        plt.plot(qs, obs, marker='o', linestyle='-')
        plt.plot([0,1],[0,1],'r--')
        plt.title('Quantile Calibration')
        plt.xlabel('Nominal Quantile')
        plt.ylabel('Observed Fraction ≤ Predicted')
        plt.savefig(f'{save_path}/quantile_calib.png')
        plt.close()

        # 8) CRPS in raw space
        # draw 500 samples *in scaled space*, then inverse‐transform them to raw
        samples_scaled = np.stack([dist.rvs(size=500) for dist in dists], axis=1)  # (500, n)
        flat = samples_scaled.flatten().reshape(-1,1)
        samples_raw = self.scaler.inverse_transform(flat).reshape(samples_scaled.shape)
        # now samples_raw[:, i] are 500 draws in raw‐space for i-th test point
        crps_vals = crps_ensemble(y_test_raw, samples_raw.T)  # crps_ensemble expects shape (n, draws)
        avg_crps  = crps_vals.mean()
        # baseline: degenerate at median
        median_pred   = np.median(y_test_raw)
        baseline_crps = np.mean(np.abs(y_test_raw - median_pred))

        # 9) ECE
        ece = np.mean(np.abs(np.array(obs) - qs))

        # 10) print/save
        print(f"Average CRPS: {avg_crps:.4f}")
        print(f"Baseline CRPS: {baseline_crps:.4f}")
        print(f"Expected Calibration Error (ECE): {ece:.4f}")

        calibration_results = {
            'ece': ece,
            'avg_crps': avg_crps,
            'baseline_crps': baseline_crps,
        }
        pd.DataFrame([calibration_results]).to_csv(
            f'{save_path}/calibration_metrics.csv', index=False
        )
        pd.DataFrame({'pit':     pit}).to_csv(f'{save_path}/pit_values.csv', index=False)
        pd.DataFrame({'qs':      qs, 'obs': obs}).to_csv(
            f'{save_path}/quantile_calibration.csv', index=False
        )
        pd.DataFrame({'crps': crps_vals}).to_csv(f'{save_path}/crps_values.csv', index=False)

        return calibration_results