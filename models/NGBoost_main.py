import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
from model_classes.NGBoostRegressionModel import NGBoostRegressionModel
import pandas as pd
from model_classes.faster_evidential_boost import NormalInverseGamma, NIGLogScore
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


def main(folder_path, data_path, target, identifier, out, folds=10, tune_folds=5, detrend=True, tune=False, uncertainty=False):
   
    test_split_size = 0.2
    Feature_Selection = {}
    target_col = identifier + "_" + target
    possible_targets = ["ratio", "diff"] 
    ignored_targets = [t for t in possible_targets if t != target]
    ignored_target_cols = [identifier + "_" + t for t in ignored_targets]
    data_df = pd.read_csv(folder_path + data_path)

    if detrend:
        trend = data_df["TimeSinceSurgery"] * 1
        data_df[identifier + "_diff"] = data_df[identifier + "_diff"] + trend
        data_df[identifier +"_ratio"] = (data_df[identifier +"_diff"] / (data_df[identifier +"_sum_pre"]* 2 + data_df[identifier +"_diff"]))
        columns_to_drop = ['Pat_ID', "TimeSinceSurgery"] + [col for col in ignored_target_cols if col in data_df.columns] 
    else:
        columns_to_drop = ['Pat_ID'] + [col for col in ignored_target_cols if col in data_df.columns] 

    # Restrict FUs between 1-3 years
    data_df = data_df[(data_df["TimeSinceSurgery"] >= 1) & (data_df["TimeSinceSurgery"] <= 3)]

    # Left locations in our dataframe is negated! otherwise sweetspot is [-12.08, -13.94,-6.74]
    data_df['L_distance'] = signed_euclidean_distance(data_df[['X_L', 'Y_L', 'Z_L']].values, [12.08, -13.94,-6.74])
    data_df['R_distance'] = signed_euclidean_distance(data_df[['X_R', 'Y_R', 'Z_R']].values, [11.90, -13.28, -6.74])
    columns_to_drop += ['X_L', 'Y_L', 'Z_L', 'X_R', 'Y_R', 'Z_R']

    data_df = data_df.drop(columns=columns_to_drop)

    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]

    # Let's perform VUF analysis
    vif_results = calculate_vif(data_df[Feature_Selection['features']], exclude_cols=[])

    # Let's test the OLS models
    import statsmodels.api as sm

    X_const = sm.add_constant(data_df[Feature_Selection['features']])
    model = sm.OLS(data_df[Feature_Selection['target']], X_const).fit()
    print(model.summary())
    
    ### test
    #X, y = load_diabetes(return_X_y=True, as_frame=True)
    #X = X.sample(n=150, random_state=42)
    #y = y.loc[X.index]
    #test_split_size = 0.2
    #Feature_Selection = {}
    #data_df = pd.concat([X, y.rename("target")], axis=1)
    #Feature_Selection['target'] = "target"
    #Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    #save_path = os.path.join(folder_path, "test/test_diabetes/NGBoost")
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
    #'Dist': [NormalInverseGamma],
    #'Score' : [NIGLogScore],
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.01, 0.1, 0.005, 0.02],
    'Base__max_depth': [ 4, 5, 6],
    'Score__evid_strength': [0.1, 0.05, 0.15],
    'Score__kl_strength': [0.01, 0.05],
    }
    
    
    # BEST ONES: 600, 0.1 and for regs 0.1 and 0.001
    NGB_Hparams = {
        'Dist': NormalInverseGamma,
        'Score' : NIGLogScore,
        'n_estimators': 350,
        'learning_rate': 0.01,
        'natural_gradient': True,
        #'Score_kwargs': {'evid_strength': 0.1, 'kl_strength': 0.01},
        'verbose': False,
        'Base': DecisionTreeRegressor(max_depth=4)  # specify the depth here
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
        standardize="zscore",
        logging=logging)
    
    metrics = model.evaluate(
        folds=folds, 
        tune=True, 
        nested=True, 
        tune_folds=25, 
        get_shap=True,
        uncertainty=False)
    
    # Set up logging
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log the metrics
    #logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    #logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}")
    _,_, removals= model.feature_ablation(folds=folds, tune=True, tune_folds=25)
    model.calibration_analysis()
        

if __name__ == "__main__":
     folder_path = "/home/ubuntu/PD-MultiModal-Prediction/"

    main(
        folder_path, 
        "data/MoCA/level2/moca_wo_mmse_df.csv", 
        "diff", 
        "MoCA", 
        "results/MoCA_NGBoost_diff/level2/NGBoost", 
        folds=10, 
        tune_folds=10, 
        detrend=False,
        tune=True)
