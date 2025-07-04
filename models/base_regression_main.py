import os
import logging
import pandas as pd
from model_classes.LinearRegressionModel import LinearRegressionModel
#from sklearn.datasets import load_diabetes
from utils.my_logging import Logging    


def main(folder_path, data_path, target, identifier, out, folds=10):
    target_col = identifier + "_" + target
    possible_targets = ["ratio", "diff"] 
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
    safe_path_linear = folder_path + out + "LinearRegression/"
    if not os.path.exists(safe_path_linear):
        os.makedirs(safe_path_linear)
    safe_path_rf = folder_path + out + "RandomForest/"

    os.makedirs(safe_path_linear, exist_ok=True)
    os.makedirs(os.path.join(safe_path_linear, 'log'), exist_ok=True)
    logging = Logging(f'{safe_path_linear}/log').get_logger()

    # --- Print all selected parameters ---
    logging.info('-------------------------------------------')
    logging.info(f"Folder Path: {folder_path}")
    logging.info(f"Data Path: {data_path}")
    logging.info(f"Target: {target}")
    logging.info(f"Identifier: {identifier}")
    logging.info(f"Output Path: {out}")
    logging.info(f"Folds: {folds}")
    logging.info('-------------------------------------------\n')

    if not os.path.exists(safe_path_rf):
        os.makedirs(safe_path_rf)
    RandomForest_Hparams = {
        'n_estimators': 100,
        'max_depth': 6,
        'random_state': 42
    }
    n_top_features = 15

    if target == "sum_post":
        if identifier == "BDI":
            lower_bound, upper_bound = 0, 63
        elif identifier == "MoCA":
            lower_bound, upper_bound = 0, 30
    
    ## Linear Regression Model
    linear_model = LinearRegressionModel(
        data_df, 
        Feature_Selection, 
        target,
        test_split_size, 
        safe_path_linear, 
        identifier, 
        n_top_features,
        logging=logging
        )
    linear_metrics = linear_model.evaluate(folds=folds, nested=True, get_shap=False)
    linear_model.plot(f"Actual vs. Prediction (Linear Regression) - {identifier}", identifier)
    _,_, removals= linear_model.feature_ablation()
#
    rf_hparams  = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [4, 5, 6, 7],
    }

    # Random Forest Model
    #rf_model = RandomForestModel(
    #    data_df, 
    #    Feature_Selection, 
    #    target,
    #    RandomForest_Hparams, 
    #    test_split_size, 
    #    safe_path_rf, 
    #    identifier, 
    #    n_top_features,
    #    param_grid=rf_hparams)
    #rf_model.fit()
    #rf_metrics = rf_model.evaluate(folds=folds, get_shap=True, tune=True, nested=True, tune_folds=-1)
    #rf_model.plot(f"Actual vs. Prediction (Random Forest) - {identifier}", identifier)
    #_,_, removals= rf_model.feature_ablation()
   ## _,_, removals= linear_model.feature_ablation()

    logging.info("Finished main execution.")


if __name__ == "__main__":
    #folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/MoCA/level2/moca_df.csv", "ratio", "MoCA", "results/MoCA_Linear/level2/NGBoost", -1)
    
    