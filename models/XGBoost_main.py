import os
from RegressionsModels import XGBoostRegressionModel
import pandas as pd

def main(folder_path, data_path, target, identifier):
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
    safe_path = os.path.join(folder_path, "results/XGBoost")
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    # Random Forest Model
     #XGBoost hyperparameters grid    
    param_grid_xgb = {
        'n_estimators': [150, 200, 300, 100, 50],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 5, 6, 7, 8, 4],
        'subsample': [0.9, 0.8, 1.0],
        'colsample_bytree': [0.9, 0.8, 1.0]
    }
    XGB_Hparams = {
        'n_estimators': param_grid_xgb['n_estimators'][0],
        'learning_rate': param_grid_xgb['learning_rate'][0],
        'max_depth': param_grid_xgb['max_depth'][0],
        'subsample': param_grid_xgb['subsample'][0],
        'colsample_bytree': param_grid_xgb['colsample_bytree'][0]
    }

    model = XGBoostRegressionModel(
        data_df, 
        Feature_Selection, 
        target,
        XGB_Hparams, 
        test_split_size, 
        safe_path, 
        identifier)
    model.fit()
    preds = model.predict(model.X)
    model.tune_haparams(param_grid_xgb)
    metrics = model.evaluate(folds=10)
    importances = model.feature_importance(19)
    model.plot(f"Actual vs. Prediction (XGBoost) - {identifier}")

if __name__ == "__main__":
    possible_targets = ["BDI_efficacy", "MoCA_efficacy"]
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/BDI/bdi_df.csv", "efficacy", "BDI")
    main(folder_path, "data/MoCA/moca_df.csv", "efficacy", "MoCA")
