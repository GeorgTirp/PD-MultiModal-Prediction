import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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

    def evaluate(self, folds=10) -> Dict:
        """ Evaluate the model using cross-validation """
        logging.info("Starting model evaluation...")
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        cv_p_values = []
        cv_r2_scores = []
        preds = []
        y_vals = []
        for train_index, val_index in kf.split(self.X):
            X_train_kf, X_val_kf = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_kf, y_val_kf = self.y.iloc[train_index], self.y.iloc[val_index]
            self.model.fit(X_train_kf, y_train_kf)
            pred = self.model.predict(X_val_kf)
            mse = mean_squared_error(y_val_kf, pred)
            r2, p = pearsonr(y_val_kf, pred)
            cv_p_values.append(p)
            cv_r2_scores.append(r2)
            preds.append(pred)
            y_vals.append(y_val_kf)

        preds = np.concatenate(preds)
        y_vals = np.concatenate(y_vals)
        avg_cv_mse = np.mean(cv_p_values)
        avg_cv_r2 = np.mean(cv_r2_scores)

        metrics = {
            'mse': avg_cv_mse,
            'r2': avg_cv_r2,
            'p_value': p,
            'y_pred': preds,
            'y_test': y_vals
        }
        self.metrics = metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{self.save_path}/{self.identifier}_metrics.csv', index=False)
        logging.info("Finished model evaluation.")
        return metrics

    def feature_importance(self, top_n: int = 10, batch_size=None, save_results=True) -> Dict:
        """ To be implemented in the subclass """
        raise NotImplementedError("Subclasses must implement feature_importance method")

    def plot(self, title, modality='') -> None:
        """ Plot predicted vs. actual values """
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.metrics['y_test'], self.metrics['y_pred'], alpha=0.5)
        plt.plot([self.metrics['y_test'].min(), self.metrics['y_test'].max()], 
                 [self.metrics['y_test'].min(), self.metrics['y_test'].max()], 
                 color='red', linestyle='--', linewidth=2)
        plt.text(self.metrics['y_test'].min(), 
                 self.metrics['y_pred'].max(), 
                 f'R: {self.metrics["r2"]:.2f}\nP-value: {self.metrics["p_value"]:.2e}', 
                 fontsize=12, 
                 verticalalignment='top', 
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.xlabel('Actual ' + modality + self.target_name)
        plt.ylabel('Predicted ' + modality + self.target_name)
        plt.title(title)
        plt.grid(True)
        plt.savefig(f'{self.save_path}/{self.identifier}_actual_vs_predicted.png')
        plt.close()

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

    def feature_importance(self, top_n: int = None, save_results=True) -> Dict:
        """ Compute feature importance using coefficients for Linear Regression """
        top_n = top_n or self.top_n
        logging.info("Starting feature importance evaluation for Linear Regression...")
        # Use absolute value of coefficients (normalized)
        attribution = np.abs(self.model.coef_) / np.sum(np.abs(self.model.coef_))
        feature_names = self.feature_selection['features']
        indices = np.argsort(attribution)[-top_n:][::-1]
        top_features = {feature_names[i]: attribution[i] for i in indices}
        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_feature_importance.npy', top_features)
        
        self.importances = top_features

        # Compute SHAP values using a linear explainer
        shap.initjs()
        background_data = self.X.sample(25, random_state=42)
        explainer = shap.LinearExplainer(self.model, background_data)
        shap_values = explainer.shap_values(self.X)
        # Plot aggregated SHAP values (beeswarm and bar plots)
        shap.summary_plot(shap_values, self.X, feature_names=self.X.columns, show=False, max_display=top_n)
        plt.title(f'{self.identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_aggregated_beeswarm.png')
            plt.close()
            shap.summary_plot(shap_values, self.X, plot_type="bar", show=False)
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_aggregated_bar.png')
            plt.close()
        
        logging.info("Finished feature importance evaluation for Linear Regression.")
        return top_features

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
            top_n: int = 10):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
        self.rf_hparams = rf_hparams
        self.model = RandomForestRegressor(**self.rf_hparams)
        self.model_name = "Random Forest"

    def feature_importance(self, top_n: int = None, save_results=True) -> Dict:
        """ Compute feature importance using the built-in attribute for Random Forest """
        top_n = top_n or self.top_n
        logging.info("Starting feature importance evaluation for Random Forest...")
        # Use the feature_importances_ attribute of RandomForest
        attribution = self.model.feature_importances_
        feature_names = self.feature_selection['features']
        indices = np.argsort(attribution)[-top_n:][::-1]
        top_features = {feature_names[i]: attribution[i] for i in indices}
        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_feature_importance.npy', top_features)
        
        self.importances = top_features

        # Compute SHAP values using a tree explainer
        shap.initjs()
        background_data = self.X.sample(25, random_state=42)
        explainer = shap.TreeExplainer(self.model)
        shap_values = []
        for row in tqdm(self.X.itertuples(index=False), total=len(self.X), desc="Computing SHAP values"):
            row_df = pd.DataFrame([row], columns=self.X.columns)
            row_shap = explainer.shap_values(row_df)
            shap_values.append(row_shap)
        # Convert list of arrays to a single array
        shap_values = np.array(shap_values).squeeze()
        # Plot aggregated SHAP values (beeswarm and bar plots)
        shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=top_n)
        plt.title(f'{self.identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_aggregated_beeswarm.png')
            plt.close()
            shap.summary_plot(shap_values, self.X, plot_type="bar", show=False)
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_aggregated_bar.png')
            plt.close()
        
        logging.info("Finished feature importance evaluation for Random Forest.")
        return top_features

    def tune_haparams(self, param_grid: dict) -> Dict:
        """Tune hyperparameters using GridSearchCV with 5-fold cross-validation.

        Args:
            param_grid (dict): Dictionary of parameter grid to search over.

        Returns:
            dict: Best hyperparameters found.
        """
        logging.info("Starting hyperparameter tuning using GridSearchCV with 5-fold CV...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(self.X, self.y)
        best_params = grid_search.best_params_
        self.rf_hparams.update(best_params)
        logging.info(f"Best parameters found: {best_params}")
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
            top_n: int = 10):
        
        super().__init__(data_df, feature_selection, target_name, test_split_size, save_path, identifier, top_n)
        self.xgb_hparams = xgb_hparams
        self.model = XGBRegressor(**self.xgb_hparams)
        self.model_name = "XGBoost Regression"

    def feature_importance(self, top_n: int = None, save_results=True) -> Dict:
        """ Compute feature importance using the built-in attribute for XGBoost """
        top_n = top_n or self.top_n
        logging.info("Starting feature importance evaluation for XGBoost Regression...")
        # Use the feature_importances_ attribute of XGBoost
        attribution = self.model.feature_importances_
        feature_names = self.feature_selection['features']
        indices = np.argsort(attribution)[-top_n:][::-1]
        top_features = {feature_names[i]: attribution[i] for i in indices}
        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_feature_importance.npy', top_features)
        
        self.importances = top_features

        # Compute SHAP values using a tree explainer
        shap.initjs()
        background_data = self.X.sample(25, random_state=42)
        explainer = shap.TreeExplainer(self.model)
        shap_values = []
        for row in tqdm(self.X.itertuples(index=False), total=len(self.X), desc="Computing SHAP values for XGBoost"):
            row_df = pd.DataFrame([row], columns=self.X.columns)
            row_shap = explainer.shap_values(row_df)
            shap_values.append(row_shap)
        shap_values = np.array(shap_values).squeeze()
        # Plot aggregated SHAP values (beeswarm and bar plots)
        shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=top_n)
        plt.title(f'{self.identifier} XGBoost SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            plt.savefig(f'{self.save_path}/{self.identifier}_xgb_shap_aggregated_beeswarm.png')
            plt.close()
            shap.summary_plot(shap_values, self.X, plot_type="bar", show=False)
            plt.savefig(f'{self.save_path}/{self.identifier}_xgb_shap_aggregated_bar.png')
            plt.close()
        
        logging.info("Finished feature importance evaluation for XGBoost Regression.")
        return top_features
    
    def tune_haparams(self, param_grid: dict) -> Dict:
        """Tune hyperparameters using GridSearchCV with 5-fold cross-validation.

        Args:
            param_grid (dict): Dictionary of parameter grid to search over.

        Returns:
            dict: Best hyperparameters found.
        """
        logging.info("Starting hyperparameter tuning using GridSearchCV with 5-fold CV...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=10,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(self.X, self.y)
        best_params = grid_search.best_params_
        self.xgb_hparams.update(best_params)
        logging.info(f"Best parameters found: {best_params}")
        return best_params