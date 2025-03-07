import os
import logging
import pandas as pd
from RegressionsModels import LinearRegressionModel, RandomForestModel

def main(folder_path, data_path, target, identifier):
    logging.info("Starting main execution...")
    target_col = identifier + "_" + target
    possible_targets = ["efficacy", "ratio", "diff"] 
    ignored_targets = [t for t in possible_targets if t != target]
    ignored_target_cols = [identifier + "_" + t for t in ignored_targets]
    data_df = pd.read_csv(folder_path + data_path)
    data_df = data_df.drop(columns=['Pat_ID']+ ignored_target_cols)
    test_split_size = 0.2
    Feature_Selection = {}
    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path_linear = folder_path + "/results/LinearRegression"
    if not os.path.exists(safe_path_linear):
        os.makedirs(safe_path_linear)
    safe_path_rf = folder_path + "/results/RandomForest"
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
    linear_model.fit()
    linear_preds = linear_model.predict(linear_model.X)
    linear_metrics = linear_model.evaluate(folds=10)
    linear_importances = linear_model.feature_importance(19)
    linear_model.plot(f"Actual vs. Prediction (Linear Regression) - {identifier}", identifier)

    # Random Forest Model
    rf_model = RandomForestModel(
        data_df, 
        Feature_Selection, 
        target,
        RandomForest_Hparams, 
        test_split_size, 
        safe_path_rf, 
        identifier, 
        n_top_features)
    rf_model.fit()
    rf_preds = rf_model.predict(rf_model.X)
    rf_metrics = rf_model.evaluate(folds=10)
    rf_importances = rf_model.feature_importance(19)
    rf_model.plot(f"Actual vs. Prediction (Random Forest) - {identifier}")

    logging.info("Finished main execution.")


if __name__ == "__main__":
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/BDI/bdi_df.csv", "efficacy", "BDI")
    main(folder_path, "data/MoCA/moca_df.csv", "diff", "MoCA")