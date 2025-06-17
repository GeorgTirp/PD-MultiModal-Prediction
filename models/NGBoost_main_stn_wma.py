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


#from sklearn.datasets import load_diabetes
# --- Dynamic Tobit bound functions ---


def main(folder_path, data_path, target, identifier, out, folds=10):
    df = pd.read_excel(data_path)
    data_df = df[[target] + [col for col in df.columns if col.startswith('nmf_')]]
    Feature_Selection = {}
    ### test
    #X, y = load_diabetes(return_X_y=True, as_frame=True)
    #data_df = pd.concat([X, y.rename("target")], axis=1)
    #Feature_Selection['target'] = "target"
    #Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    #safe_path = os.path.join(folder_path, "test/results/XGBoost")
    ### test ende
    Feature_Selection['target'] = target
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = os.path.join(folder_path, out)
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
    
    

    param_grid_ngb = {
    'n_estimators': [300, 400, 500, 600],
    'learning_rate': [0.05, 0.1],
    'Base__max_depth': [3, 4]
    }   
    
    # BEST ONES: 600, 0.1 and for regs 0.1 and 0.001
    NGB_Hparams = {
        'Dist': NormalInverseGamma,
        'Score' : NIGLogScore,
        'n_estimators': 600,
        'learning_rate': 0.1,
        'natural_gradient': True,
        #'minibatch_frac': 0.1,
        'verbose': False,
        'Base': DecisionTreeRegressor(max_depth=3)  # specify the depth here
    }

    model = NGBoostRegressionModel(
        data_df, 
        Feature_Selection, 
        target,
        NGB_Hparams, 
        0.2, 
        safe_path, 
        identifier,
        -1,
        param_grid_ngb)
    metrics = model.evaluate(
        folds=folds, 
        tune=False, 
        nested=True, 
        tune_folds=20, 
        get_shap=True,
        uncertainty=False)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log the metrics
    logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}")
    #_,_, removals= model.feature_ablation()
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
    folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/frieder/merged_demographics_features.xlsx", "updrs_totscore", "WMA", "results/WMA", 20)
    