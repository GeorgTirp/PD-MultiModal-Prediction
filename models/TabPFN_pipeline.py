
from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict
import shap
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import os
import torch
# Setting test split size


class TabPFNRegression():
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            Feature_Selection: dict, 
            test_split_size:float = 0.2,
            save_path: str = None,
            identifier: str = None):
        
        self.save_path = save_path
        self.identifier = identifier
        self.Feature_Selection = Feature_Selection
        
        self.reg_model = None
        self.X, self.y = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(self.X, self.y, test_size=test_split_size, random_state=42)
        self.metrics = None
        
        save_path = f'{save_path}/{identifier}' 
        if not os.path.exists(save_path):
                os.makedirs(save_path)

    def model_specific_preprocess(self, data_df: pd.DataFrame, Feature_Selection: dict = None) -> Tuple:
        """ Preprocess the data for the TabPFN model"""
        # Ensure all features are numeric
        if Feature_Selection is None:
            Feature_Selection = self.Feature_Selection

        data_df = data_df.dropna(subset=Feature_Selection['features'] + [Feature_Selection['target']])
        X = data_df[self.Feature_Selection['features']]
        y = data_df[self.Feature_Selection['target']]
        X = X.apply(pd.to_numeric, errors='coerce')
        # Remove dollar sign and convert to float
        y = pd.to_numeric(y, errors='coerce')
        #data = data[data['price'] < 1000]
        return X, y
    
    def fit(self) -> None:
        """ Train and predict using Linear Regression and Random Forest"""
        reg = TabPFNRegressor()
        reg.fit(self.X, self.y)
        self.reg_model = reg
        return self.reg_model
       
    def predict(self, X_in: pd.DataFrame, save_results=False) -> Dict:
        """Predict using the trained model"""
        if self.reg_model is None:
            raise ValueError("Model not fitted yet")
        #X_in = X_in[Feature_Selection['features']]
        predictions = self.reg_model.predict(X_in)

        # Optionally save predictions
        if save_results:
            results_df = pd.DataFrame({'y_pred': predictions})
            results_df.to_csv(f'{self.save_path}/{self.identifier}_results.csv', index=False)

        return predictions

    def evaluate(self) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        reg_pred = self.reg_model.predict(X_test)
        reg_mse = mean_squared_error(y_test, reg_pred)
        reg_r2 = r2_score(y_test, reg_pred)
        
        
        # Cross-validation for Random Forest
        #reg_cv_scores = cross_val_score(self.reg_model, X_train, y_train, cv=10, scoring='accuracy')
        #reg_cv_mean_score = reg_cv_scores.mean()

        
        print(f"TabPFN MSE: {reg_mse}, R2: {reg_r2}")
        print(f"TabPFN R^2: {reg_r2}")
        
    
        tabpfn_metrics = {
            'mse': reg_mse,
            'r2': reg_r2,
        }
        self.metrics = tabpfn_metrics

        return tabpfn_metrics
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def feature_importance(self, save_results=True) -> Dict:
        """Return the feature importance for the Random Forest and linear model"""
        logging.info("Starting feature importance evaluation.")

        X_train, X_test, y_train, y_test = self.train_split

        def loco_importances(self, X_train, y_test):
            logging.info("Starting Loco importance evaluation...")
            importances = {}
            for i, feature in enumerate(X_train.columns):
                logging.info(f"Evaluating Loco importance for feature {i + 1}/{len(X_train.columns)}: {feature}")

                # Remove the feature from the data
                X_train_loco = X_train.drop(columns=[feature])
                X_test_loco = X_test.drop(columns=[feature])

                # Train the model and get predictions
                self.reg_model.fit(X_train_loco, y_train)
                loco_pred = self.reg_model.predict(X_test_loco)

                # Compute the MSE and store the importance
                loco_mse = mean_squared_error(y_test, loco_pred)
                importances[feature] = abs(loco_mse - self.metrics['mse']) / self.metrics['mse']

                # Log progress for every 10th feature evaluated
                if (i + 1) % 10 == 0 or (i + 1) == len(X_train.columns):
                    logging.info(f"Progress: {i + 1}/{len(X_train.columns)} features evaluated.")

            logging.info("Finished Loco importance evaluation.")
            return importances

        def shap_importances(self, X_train, y_test):
            logging.info("Starting SHAP importance evaluation...")

            # Initialize SHAP
            shap.initjs()

            # Sample background data
            background_data = X_train.sample(25, random_state=42)
            logging.info("Background data for SHAP initialized.")

            # Initialize SHAP Explainer
            explainer = shap.KernelExplainer(self.reg_model.predict, background_data)
            logging.info("SHAP explainer initialized.")

            # Calculate SHAP values with tqdm progress bar
            shap_values = []
            for i in tqdm(range(len(X_train)), desc="Computing SHAP values", unit="sample"):
                shap_values.append(explainer.shap_values(X_train.iloc[i:i+1]))
                # Clear CUDA cache to avoid out of memory error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Plot SHAP summary
            shap.summary_plot(shap_values, X_train)
            logging.info("SHAP summary plot generated.")

            return shap_values

        # Run Loco and SHAP importance evaluations
        logging.info("Evaluating SHAP feature importances...")
        shap_attributions = shap_importances(self, X_train, y_test)
        logging.info("SHAP importance evaluation completed.")

        logging.info("Evaluating LOCO feature importances...")
        loco_attributions = loco_importances(self, X_train, y_test)
        logging.info("LOCO importance evaluation completed.")

        # Save results if specified
        #if save_results:
        #    logging.info(f"Saving results to {self.save_path}/{self.identifier}_mean_shap_values.npy")
        #    np.save(f'{self.save_path}/{self.identifier}_mean_shap_values.npy', shap_attributions)

        return loco_attributions#, shap_attributions

    def plot():
        """ Plot """
        pass

if __name__ == "__main__":
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction"
    data_df = pd.read_csv(folder_path + "/data/bdi_df.csv")
    test_split_size= 0.2
    Feature_Selection = {}
    Feature_Selection['target'] = 'BDI_diff'
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = folder_path + "/results/TabPFN"
    identifier = "bdi"
    model = TabPFNRegression(data_df, Feature_Selection, test_split_size, safe_path, identifier)
    model.fit()
    X, y = model.model_specific_preprocess(data_df, Feature_Selection)
    preds = model.predict(X, save_results=True)
    metrics = model.evaluate()
    importances = model.feature_importance()
    