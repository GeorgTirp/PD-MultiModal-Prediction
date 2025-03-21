import os
import logging
import pandas as pd
from RegressionsModels import LinearRegressionModel, RandomForestModel
#from sklearn.datasets import load_diabetes


def main(folder_path, data_path, target, identifier, folds=10):
    logging.info("Starting main execution...")
    target_col = identifier + "_" + target
    possible_targets = ["efficacy", "ratio", "diff"] 
    ignored_targets = [t for t in possible_targets if t != target]
    ignored_target_cols = [identifier + "_" + t for t in ignored_targets]
    data_df = pd.read_csv(folder_path + data_path)
    data_df = data_df.drop(columns=['Pat_ID']+ ignored_target_cols)
    test_split_size = 0.2
    Feature_Selection = {}
    ### test
    #X, y = load_diabetes(return_X_y=True, as_frame=True)
    #data_df = pd.concat([X, y.rename("target")], axis=1)
    #Feature_Selection['target'] = "target"
    #Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    ### test ende
    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path_linear = folder_path + "/results/LinearRegression/"
    if not os.path.exists(safe_path_linear):
        os.makedirs(safe_path_linear)
    safe_path_rf = folder_path + "/results/RandomForest/"
    if not os.path.exists(safe_path_rf):
        os.makedirs(safe_path_rf)
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
        target,
        test_split_size, 
        safe_path_linear, 
        identifier, 
        n_top_features)
    linear_metrics = linear_model.evaluate(folds=folds, get_shap=True)
    linear_model.plot(f"Actual vs. Prediction (Linear Regression) - {identifier}", identifier)

    rf_hparams  = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 5, 6, 7],
    }

    # Random Forest Model
    rf_model = RandomForestModel(
        data_df, 
        Feature_Selection, 
        target,
        RandomForest_Hparams, 
        test_split_size, 
        safe_path_rf, 
        identifier, 
        n_top_features,
        param_grid=rf_hparams)
    rf_model.fit()
    rf_metrics = rf_model.evaluate(folds=folds, get_shap=True, tune=True, nested=True, tune_folds=10)
    rf_model.plot(f"Actual vs. Prediction (Random Forest) - {identifier}", identifier)

    logging.info("Finished main execution.")


if __name__ == "__main__":
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/BDI/bdi_df_normalized.csv", "diff", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df_normalized.csv", "diff", "MoCA", -1)
    main(folder_path, "data/BDI/bdi_df_normalized.csv", "ratio", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df_normalized.csv", "ratio", "MoCA", -1)
    main(folder_path, "data/BDI/bdi_df_normalized.csv", "efficacy", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df_normalized.csv", "efficacy", "MoCA", -1)