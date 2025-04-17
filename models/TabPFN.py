from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict
from shapiq import TabPFNExplainer
import shap
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import torch
from scipy.stats import pearsonr
import os
import warnings
import seaborn as sns

warnings.resetwarnings()
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TabPFNRegression():
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            Feature_Selection: dict, 
            target_name: str,
            test_split_size:float = 0.2,
            save_path: str = None,
            identifier: str = None):
        
        self.save_path = save_path
        self.identifier = identifier
        self.Feature_Selection = Feature_Selection
        self.target_name = target_name
        self.X, self.y = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(self.X, self.y, test_size=test_split_size, random_state=42)
        self.metrics = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TabPFNRegressor(device=self.device, n_jobs=4)
        logging.info("Initialized TabPFNRegression with identifier: %s", self.identifier)

    def model_specific_preprocess(self, data_df: pd.DataFrame, y: pd.DataFrame = None, Feature_Selection: dict = None) -> Tuple:
        """ Preprocess the data for the TabPFN model"""
        logging.info("Starting model-specific preprocessing.")
        if Feature_Selection is None:
            Feature_Selection = self.Feature_Selection
        data_df = data_df.dropna(subset=Feature_Selection['features'] + [Feature_Selection['target']])
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        X = X.apply(pd.to_numeric, errors='coerce')
        logging.info("Completed model-specific preprocessing.")
        return X, y
    
    def fit(self) -> None:
        """ Train and predict using Linear Regression and Random Forest"""
        logging.info("Starting model training.")
        X_train, X_test, y_train, y_test = self.train_split
        reg = self.model
        reg.fit(X_train, y_train)
        self.model = reg
        logging.info("Model training completed.")
        return self.model
       
    def predict(self, X_in: pd.DataFrame) -> Dict:
        """Predict using the trained model"""
        logging.info("Starting prediction.")
        if self.model is None:
            raise ValueError("Model not fitted yet")
        predictions = self.model.predict(X_in)
        logging.info("Prediction completed.")
        return predictions

    def evaluate(self, folds=10) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
        
        if folds == -1:
            kf = LeaveOneOut()
            logging.info("Starting model evaluation with Leave-One-Out cross-validation.")
        else:
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            logging.info("Starting model evaluation with %d folds.", folds)

        preds = []
        y_vals = []
        for train_index, val_index in tqdm(kf.split(self.X), total=kf.get_n_splits(self.X), desc="Cross-validation"):
            X_train_kf, X_val_kf = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_kf, y_val_kf = self.y.iloc[train_index], self.y.iloc[val_index]
            self.model.fit(X_train_kf, y_train_kf)
            pred = self.model.predict(X_val_kf)
            preds.append(pred)
            y_vals.append(y_val_kf)
            

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
        logging.info("Model evaluation completed")
        return metrics

    def feature_importance(self, top_n: int = 10, batch_size=10, save_results=True) -> Dict:
        """Return the feature importance for the Random Forest and linear model"""
        logging.info("Starting feature importance evaluation.")

        X_train, X_test, y_train, y_test = self.train_split

        def loco_importances(X_train, y_test):
            logging.info("Starting Loco importance evaluation...")
            importances = {}
            for i, feature in enumerate(X_train.columns):
                logging.info(f"Evaluating Loco importance for feature {i + 1}/{len(X_train.columns)}: {feature}")

                X_train_loco = X_train.drop(columns=[feature])
                X_test_loco = X_test.drop(columns=[feature])

                self.model.fit(X_train_loco, y_train)
                loco_pred = self.model.predict(X_test_loco)

                loco_mse = mean_squared_error(y_test, loco_pred)
                importances[feature] = abs(loco_mse - self.metrics['mse']) / self.metrics['mse']

                if (i + 1) % 10 == 0 or (i + 1) == len(X_train.columns):
                    logging.info(f"Progress: {i + 1}/{len(X_train.columns)} features evaluated.")

            logging.info("Finished Loco importance evaluation.")
            return importances

        def shap_importances(top_n=10, save_results=True):
            logging.info("Starting feature importance evaluation using shapiq for TabPFN...")
        
            # Initialize the TabPFNExplainer
            # Note: Ensure that self.model is your TabPFN regressor,
            # self.X is your feature DataFrame, and self.y is your target vector.
            explainer = TabPFNExplainer(
                model=self.model,
                data=self.X.values,  # pass data as a NumPy array for compatibility
                labels=self.y,       # target values; for regression, these are continuous
                index="SV",          # using standard Shapley Values
                max_order=1
            )
        
            num_samples = len(self.X)
            all_shap_values = []
        
            # Iterate over each sample individually
            for i in tqdm(range(num_samples), desc="SHAP feature importance evaluation"):
                # Extract a single sample as a 1D NumPy array
                sample = self.X.values[i]
                # Explain the sample. The demo you saw uses this pattern.
                shap_values_sample = explainer.explain(sample)
                # Append the shap_values to the list
                all_shap_values.append(shap_values_sample)
        
            # Convert list of dict_values (assumed to be arrays) into a NumPy array.
            # This array should have shape (n_samples, n_features).
            shap_values = np.vstack(all_shap_values)
            logging.info("Feature importance computed using shapiq.")
        
            # Compute mean absolute importance across samples
            mean_importance = np.mean(np.abs(shap_values), axis=0)
        
            # Sort by importance and select top_n features
            sorted_indices = np.argsort(mean_importance)[::-1][:top_n]
            sorted_features = np.array(self.X.columns)[sorted_indices]
            sorted_importance = mean_importance[sorted_indices]
        
            # Bar plot of feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(sorted_features[::-1], sorted_importance[::-1], color='skyblue')
            plt.xlabel("Mean Absolute Importance")
            plt.title(f"{self.identifier} Feature Importance (Top {top_n})")
            plt.grid(axis="x", linestyle="--", alpha=0.7)
        
            if save_results:
                plt.savefig(f'{self.save_path}/{self.identifier}_shapiq_feature_importance.png')
                plt.show()
            else:
                plt.show()
        
            logging.info("Feature importance plot generated.")
            return shap_values


        logging.info("Evaluating SHAP feature importances...")
        shap_attributions = shap_importances(batch_size)
        logging.info("SHAP importance evaluation completed.")

        logging.info("Evaluating LOCO feature importances...")
        loco_attributions = loco_importances(X_train, y_test)
        logging.info("LOCO importance evaluation completed.")
    
        if save_results:
            logging.info(f"Saving results to {self.save_path}/{self.identifier}_mean_shap_values.npy")
            np.save(f'{self.save_path}/{self.identifier}_mean_shap_values.npy', shap_attributions)
    
        return loco_attributions, shap_attributions

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
        plt.title(title  + "  N=" + str(len(self.y)), fontsize=14)

        # Show grid and ensure everything fits nicely
        plt.grid(True)
        plt.tight_layout()

        # Save and close
        plt.savefig(f'{self.save_path}/{self.identifier}_{self.target_name}_actual_vs_predicted.png')
        plt.close()

        # Log info (optional)
        logging.info("Plot saved to %s/%s_%s_actual_vs_predicted.png", 
                 self.save_path, self.identifier, self.target_name)

        #
