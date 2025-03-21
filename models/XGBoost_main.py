import os
from RegressionsModels import XGBoostRegressionModel
import pandas as pd
#from sklearn.datasets import load_diabetes

def main(folder_path, data_path, target, identifier, folds=10):
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
    #safe_path = os.path.join(folder_path, "test/results/XGBoost")
    ### test ende
    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = os.path.join(folder_path, "results/XGBoost")
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    # Random Forest Model
     #XGBoost hyperparameters grid    
    param_grid_xgb = {
        'n_estimators': [10, 25, 50, 100],               # Keep trees small due to sample size
        'max_depth': [2, 3, 4],                     # Shallow trees to prevent overfitting
        'learning_rate': [0.01, 0.05, 0.1],         # Smaller learning rate for better generalization
        'subsample': [0.6, 0.8, 1.0],               # Some randomness can help
        'colsample_bytree': [0.6, 0.8, 1.0],        # Feature sampling to reduce variance
        'reg_alpha': [0, 0.1, 1],                   # L1 regularization (sparse models)
        'reg_lambda': [0.1, 1, 10],                 # L2 regularization (shrink weights)
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
        identifier,
        -1,
        param_grid_xgb)
    metrics = model.evaluate(folds=folds, tune=True, nested=True, tune_folds=10, get_shap=False)
    model.plot(f"Actual vs. Prediction (XGBoost) - {identifier}")

if __name__ == "__main__":
    possible_targets = ["BDI_efficacy", "MoCA_efficacy"]
    #folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    main(folder_path, "data/BDI/bdi_df.csv", "diff", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df.csv", "diff", "MoCA", -1)
    main(folder_path, "data/BDI/bdi_df.csv", "ratio", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df.csv", "ratio", "MoCA", -1)
    main(folder_path, "data/BDI/bdi_df.csv", "efficacy", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df.csv", "efficacy", "MoCA", -1)
