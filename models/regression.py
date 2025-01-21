import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from add_custom_features import AddCustomFeatures
#from preprocessing.preprocessing_test import preprocess_data
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np


# Setting Hyperparameters and features
RandomForest_Hparams = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

# Setting features and target
Feature_Selection = {
    'features': [
        'latitude', 
        'longitude'],

    'target': 'price'
}

# Setting test split size
test_split_size= 0.2

class RegressionModels:
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            rf_hparams: dict = RandomForest_Hparams, 
            Feature_Selection: dict= Feature_Selection, 
            test_split_size:float = test_split_size,
            save_path: str = None,
            identifier: str = None):
        
        self.rf_hparams = rf_hparams
        self.linear_model = None
        self.rf_model = None
        X,y = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(X, y, test_size=test_split_size, random_state=42)
        self.save_path = save_path
        self.identifier_rf, self.identifier_linear = identifier

    def model_specific_preprocess(self, data_df: pd.DataFrame) -> Tuple:
        """ Preprocess the data for the TabPFN model"""
        # Ensure all features are numeric
        data_df = data_df.dropna(subset=Feature_Selection['features'] + [Feature_Selection['target']])
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        X = X.apply(pd.to_numeric, errors='coerce')
        # Z-score normalization
        X = (X - X.mean()) / X.std()
        # Remove dollar sign and convert to float
        if y.dtype == object:
            y = y.replace('[\$,]', '', regex=True).astype(float)
        
    
        return X, y
    
    def fit(self) -> None:
        """ Train and predict using Linear Regression and Random Forest"""
    
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        rf_model = RandomForestRegressor(**self.rf_hparams)
        rf_model.fit(X_train, y_train)

        self.linear_model = linear_model
        self.rf_model = rf_model
       
    def predict(self, X: pd.DataFrame , y: pd.DataFrame, save_results=False) -> Tuple:
        """ Predict using the trained models (for class extern usage)"""
        rf_pred = self.rf_model.predict(X)
        linear_pred = self.linear_model.predict(X)
        results_df = pd.DataFrame({'y_test': y, 'y_pred':linear_pred})
        results_df = pd.DataFrame({'y_test': y, 'y_pred': rf_pred})
        if save_results == True:
            results_df.to_csv(f'{self.save_path}/{self.identifier_rf}_results.csv', index=False)
        if save_results == True:
            results_df.to_csv(f'{self.save_path}/{self.identifier_linear}_results.csv', index=False)
        return rf_pred, linear_pred

    def evaluate(self) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        rf_pred = self.rf_model.predict(X_test)
        linear_pred = self.linear_model.predict(X_test)

        rf_mse = mean_squared_error(y_test, rf_pred)
        linear_mse = mean_squared_error(y_test, linear_pred)

        rf_r2 = r2_score(y_test, rf_pred)
        linear_r2 = r2_score(y_test, linear_pred)

        # Cross-validation for Random Forest
        rf_cv_scores = cross_val_score(self.rf_model, X_train, y_train, cv=10, scoring='r2')
        rf_cv_mean_score = rf_cv_scores.mean()

        # Cross-validation for Linear Regression
        linear_cv_scores = cross_val_score(self.linear_model, X_train, y_train, cv=10, scoring='r2')
        linear_cv_mean_score = linear_cv_scores.mean()

        print(f"Random Forest MSE: {rf_mse}, R2: {rf_r2}")
        print(f"Linear Regression MSE: {linear_mse}, R2: {linear_r2}")
        print(f"Random Forest 10-fold CV Mean R2 Score: {rf_cv_mean_score}")
        print(f"Linear Regression 10-fold CV Mean R2 Score: {linear_cv_mean_score}")

        linear_metrics = {
            'mse': linear_mse,
            'r2': linear_r2,
            'cv_mean_score': linear_cv_mean_score
        }

        rf_metrics = {
            'mse': rf_mse,
            'r2': rf_r2,
            'cv_mean_score': rf_cv_mean_score
        }
        
        return linear_metrics, rf_metrics
    
    def feature_importance(self, top_n: int = 10, save_results = True) -> Tuple[Dict, Dict]:
        """ Return the feature importance for the Random Forest and linear model"""
        # Get attributions
        rf_attribution = self.rf_model.feature_importances_
        linear_attribution = abs(self.linear_model.coef_) / sum(abs(self.linear_model.coef_))
        feature_names = Feature_Selection['features']
        rf_indices = rf_attribution.argsort()[-top_n:][::-1]
        linear_indices = linear_attribution.argsort()[-top_n:][::-1]

        # Get top n features
        rf_top_features = {feature_names[i]: rf_attribution[i] for i in rf_indices}
        linear_top_features = {feature_names[i]: linear_attribution[i] for i in linear_indices}

        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_feature_importance.npy', rf_top_features)
        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_feature_importance.npy', linear_top_features)
        return rf_top_features, linear_top_features

    def plot():
        """ Plot """
        pass

if __name__ == "__main__":
    
    folder_path = "/home/georg/Documents/Master/Data_Literacy"
    data_df = pd.read_csv(folder_path + "/city_listings.csv")
    add_custom_features = ['distance_to_city_center', 'average_review_length']
    Feature_Adder = AddCustomFeatures(data_df, add_custom_features)
    model = RegressionModels(data_df)
    model.fit()
    evals =model.evaluate()
    feature_importances = model.feature_importance(10)
     