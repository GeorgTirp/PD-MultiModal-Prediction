import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
from model_classes.XGBoostRegressionModel import XGBoostRegressionModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Logging
from utils.my_logging import Logging
from utils.messages import Messages
from pprint import pformat


#from sklearn.datasets import load_diabetes
# --- Dynamic Tobit bound functions ---


def main(folder_path, data_path, target, identifier, out, folds=10):
    test_split_size = 0.2
    Feature_Selection = {}
    target_col = identifier + "_" + target
    possible_targets = ["ratio", "diff"] 
    ignored_targets = [t for t in possible_targets if t != target]
    ignored_target_cols = [identifier + "_" + t for t in ignored_targets]
    data_df = pd.read_csv(folder_path + data_path)
    trend = data_df["TimeSinceSurgery"] * 1
    data_df[identifier + "_diff"] = data_df[target_col] + trend
    data_df[identifier +"_ratio"] = (data_df[identifier +"_diff"] / (data_df[identifier +"_sum_pre"]* 2 + data_df[identifier +"_diff"])) * 10
    columns_to_drop = ['Pat_ID'] + [col for col in ignored_target_cols if col in data_df.columns]
    data_df = data_df.drop(columns=columns_to_drop)
    data_df.drop("TimeSinceSurgery", axis=1, inplace=True)
    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = os.path.join(folder_path, out)

    ### test
    #X, y = load_diabetes(return_X_y=True, as_frame=True)
    #X = X.sample(n=150, random_state=42)
    #y = y.loc[X.index]
    #std = y.std()
    #y = (y - y.mean()) / std  # Standardize the target variable
    #data_df = pd.concat([X, y.rename("target")], axis=1)
    #Feature_Selection['target'] = "target"
    #Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    #safe_path = os.path.join(folder_path, "test/test_diabetes/NGBoost")
    ### test ende
    # --- SETUP LOGGING ---
    os.makedirs(safe_path, exist_ok=True)
    os.makedirs(os.path.join(safe_path, 'log'), exist_ok=True)
    logging = Logging(f'{safe_path}/log').get_logger()

    # --- Print all selected parameters ---
    logging.info('-------------------------------------------')
    logging.info(f"Folder Path: {folder_path}")
    logging.info(f"Data Path: {data_path}")
    logging.info(f"Target: {target}")
    logging.info(f"Identifier: {identifier}")
    logging.info(f"Output Path: {out}")
    logging.info(f"Folds: {folds}")
    logging.info('-------------------------------------------\n')

    
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    # Random Forest Model
     #XGBoost hyperparameters grid    
    # Define the parameter grid for NGBoost
    #define bounds
    #if target == "sum_post":
        #if identifier == "BDI":
        #    NIGLogScore.set_bounds(0, 63)
        #elif identifier == "MoCA":
        #    NIGLogScore.set_bounds(0, 30)
    
    

    # XGBoost hyperparameter grid
    # XGBoost hyperparameter grid
    param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.9],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.1, 0.5, 0.7],
    'reg_lambda': [2, 5, 8],
    'random_state': [42],
    'verbosity': [0]
}

    # Default XGBoost hyperparameters
    XGB_Hparams = {
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'random_state': 42,
        'verbosity': 0
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
        param_grid_xgb,
        logging=logging)
    metrics = model.evaluate(
        folds=folds, 
        tune=True, 
        nested=True, 
        tune_folds=20, 
        get_shap=True,
        uncertainty=False)
    
    # Log the metrics
    logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}")
    _,_, removals= model.feature_ablation(folds=folds, tune=True, tune_folds=20)
    #model.calibration_analysis()
    
    
    #summands = [0, 0, 1, 0]
    #param_names  = ["mu", "lambda", "alpha", "beta"]
    #for i in range(len(param_names)):
    #    if i == 0:
    #        param = metrics["pred_dist"][i] + summands[i]
    #    param = np.exp(metrics["pred_dist"][i]) + summands[i]
    #    print(metrics["pred_dist"][i].shape)
    #    plt.figure()
    #    plt.hist(param, bins=20, alpha=0.7, color='blue', edgecolor='black')
    #    plt.title(f"Histogram of {param_names[i]} - Sample {i+1}")
    #    plt.xlabel(f"{param_names[i]} values")
    #    plt.ylabel("Frequency")
    #    plt.grid(True)
    #    plt.savefig(os.path.join(safe_path, f"histogram_{param_names[i]}_sample.png"))
    #    plt.close()
        
        

if __name__ == "__main__":
    #folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"
    folder_path = "/home/ubuntu/PD-MultiModal-Prediction/"
    #main(folder_path, "data/BDI/level2/bdi_df.csv", "diff", "BDI", "results/BDI_test/level2/XGBoost", 20)
    main(folder_path, "data/MoCA/level2/moca_df.csv", "ratio", "MoCA", "results/MoCA_ratio/level2/XGBoost", 20)