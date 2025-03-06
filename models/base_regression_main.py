import os
import logging
import pandas as pd
from models.base_regressions import LinearRegressionModel, RandomForestModel

def bdi_main():
    logging.info("Starting main execution...")
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction" # Gets directory of current script
    data_df = pd.read_csv(folder_path + "/data/BDI/bdi_df_normalized.csv")
    data_df = data_df.drop(columns=['Pat_ID'])
    test_split_size= 0.2
    Feature_Selection = {}
    Feature_Selection['target'] = 'BDI_efficacy'
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path_linear= folder_path + "/results/LinearRegression"
    if not os.path.exists(safe_path_linear):
        os.makedirs(safe_path_linear)
    safe_path_rf = folder_path + "/results/RandomForest"
    if not os.path.exists(safe_path_rf):
        os.makedirs(safe_path_rf)
    identifier_linear = "BDI"
    identifier_rf = "BDI"
    RandomForest_Hparams = {
        'n_estimators': 100,
        'max_depth': 6,
        'random_state': 42
    }
    n_top_features = 15

    # Linear Regression Model
    linear_model = LinearRegressionModel(
        data_df, 
        Feature_Selection, 
        test_split_size, 
        safe_path_linear, 
        identifier_linear, 
        n_top_features)
    linear_model.fit()
    linear_preds = linear_model.predict(linear_model.X, linear_model.y, save_results=True)
    linear_metrics = linear_model.evaluate(n_splits=10)
    linear_importances = linear_model.feature_importance(19)
    linear_model.plot("Actual vs. Prediction (Linear Regression)")

    # Random Forest Model
    rf_model = RandomForestModel(
        data_df, 
        Feature_Selection, 
        RandomForest_Hparams, 
        test_split_size, 
        safe_path_rf, 
        identifier_rf, 
        n_top_features)
    rf_model.fit()
    rf_preds = rf_model.predict(rf_model.X, rf_model.y, save_results=True)
    rf_metrics = rf_model.evaluate(n_splits=10)
    rf_importances = rf_model.feature_importance(19)
    rf_model.plot("Actual vs. Prediction (Random Forest)")

   
    logging.info("Finished main execution.")

def moca_main():
    logging.info("Starting main execution...")
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction" # Gets directory of current script
    data_df = pd.read_csv(folder_path + "/data/MOCA/bdi_df_normalized.csv")
    data_df = data_df.drop(columns=['Pat_ID'])
    test_split_size= 0.2
    Feature_Selection = {}
    Feature_Selection['target'] = 'MoCA_efficacy'
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path_linear= folder_path + "/results/LinearRegression"
    if not os.path.exists(safe_path_linear):
        os.makedirs(safe_path_linear)
    safe_path_rf = folder_path + "/results/RandomForest"
    if not os.path.exists(safe_path_rf):
        os.makedirs(safe_path_rf)
    identifier_linear = "MoCA"
    identifier_rf = "MoCA"
    RandomForest_Hparams = {
        'n_estimators': 100,
        'max_depth': 6,
        'random_state': 42
    }
    n_top_features = 15

    # Linear Regression Model
    linear_model = LinearRegressionModel(
        data_df, 
        Feature_Selection, 
        test_split_size, 
        safe_path_linear, 
        identifier_linear, 
        n_top_features)
    linear_model.fit()
    linear_preds = linear_model.predict(linear_model.X, linear_model.y, save_results=True)
    linear_metrics = linear_model.evaluate(n_splits=10)
    linear_importances = linear_model.feature_importance(19)
    linear_model.plot("Actual vs. Prediction (Linear Regression)")

    # Random Forest Model
    rf_model = RandomForestModel(
        data_df, 
        Feature_Selection, 
        RandomForest_Hparams, 
        test_split_size, 
        safe_path_rf, 
        identifier_rf, 
        n_top_features)
    rf_model.fit()
    rf_preds = rf_model.predict(rf_model.X, rf_model.y, save_results=True)
    rf_metrics = rf_model.evaluate(n_splits=10)
    rf_importances = rf_model.feature_importance(19)
    rf_model.plot("Actual vs. Prediction (Random Forest)")


if __name__ == "__main__":
    moca_main()
    bdi_main()