import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
from RegressionsModels import NGBoostRegressionModel
import pandas as pd
from faster_evidential_boost import NormalInverseGamma, NIGLogScore
from ngboost.distns.normal import Normal, NormalCRPScore, NormalLogScore
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np
import logging
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from utils.my_logging import Logging



#from sklearn.datasets import load_diabetes
# --- Dynamic Tobit bound functions ---


def main(folder_path, data_path, target, identifier, out, folds=10):
    # --- SETUP LOGGING ---
    
    test_split_size = 0.2
    Feature_Selection = {}
    #target_col = identifier + "_" + target
    #possible_targets = ["ratio", "diff"] 
    #ignored_targets = [t for t in possible_targets if t != target]
    #ignored_target_cols = [identifier + "_" + t for t in ignored_targets]
    #data_df = pd.read_csv(folder_path + data_path)
    #columns_to_drop = ['Pat_ID'] + [col for col in ignored_target_cols if col in data_df.columns]
    #data_df = data_df.drop(columns=columns_to_drop)
    #Feature_Selection['target'] = target_col
    #Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    #save_path = os.path.join(folder_path, out)

    ### test
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.sample(n=150, random_state=42)
    y = y.loc[X.index]
    
    data_df = pd.concat([X, y.rename("target")], axis=1)
    Feature_Selection['target'] = "target"
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    save_path = os.path.join(folder_path, "test/test_diabetes/NGBoost")
    ### test ende

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'log'), exist_ok=True)
    logging = Logging(f'{save_path}/log').get_logger()

    # --- Print all selected parameters ---
    logging.info('-------------------------------------------')
    logging.info(f"Folder Path: {folder_path}")
    logging.info(f"Data Path: {data_path}")
    logging.info(f"Target: {target}")
    logging.info(f"Identifier: {identifier}")
    logging.info(f"Output Path: {out}")
    logging.info(f"Folds: {folds}")
    logging.info('-------------------------------------------\n')
    # Random Forest Model
     #XGBoost hyperparameters grid    
    # Define the parameter grid for NGBoost
    #define bounds
    #if target == "sum_post":
        #if identifier == "BDI":
        #    NIGLogScore.set_bounds(0, 63)
        #elif identifier == "MoCA":
        #    NIGLogScore.set_bounds(0, 30)
    
    

    param_grid_ngb = {
    'n_estimators': [450, 500, 550, 600, 650],
    'learning_rate': [0.5, 0.01, 0.1],
    'Base__max_depth': [3, 4, 5],
    'Score__evid_strength': [0.05, 0.1, 0.2],
    'Score__kl_strength': [0.01, 0.05, 0.1],
    }      
    
    
    # BEST ONES: 600, 0.1 and for regs 0.1 and 0.001
    NGB_Hparams = {
        'Dist': NormalInverseGamma,
        'Score' : NIGLogScore,
        'n_estimators': 100,
        'learning_rate': 0.01,
        'natural_gradient': True,
        #'Score_kwargs': {'evid_strength': 0.1, 'kl_strength': 0.01},
        'verbose': False,
        'Base': DecisionTreeRegressor(max_depth=3)  # specify the depth here
    }

    model = NGBoostRegressionModel(
        data_df=data_df, 
        feature_selection=Feature_Selection, 
        target_name=target,
        ngb_hparams=NGB_Hparams,
        test_split_size=test_split_size,
        save_path=save_path,
        identifier=identifier,
        top_n=-1,
        param_grid=param_grid_ngb,
        standardize=True,
        logging=logging)
    
    metrics = model.evaluate(
        folds=folds, 
        tune=False, 
        nested=True, 
        tune_folds=20, 
        get_shap=True,
        uncertainty=False)
    
    # Set up logging
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log the metrics
    #logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    #logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}")
    _,_, removals= model.feature_ablation(folds=folds, tune=False, tune_folds=-1)
    model.calibration_analysis()
    
    
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
    folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/BDI/level2/bdi_df.csv", "diff", "BDI", "results/level2_test/NGBoost", 20)
    
