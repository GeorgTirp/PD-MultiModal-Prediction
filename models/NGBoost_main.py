import os
from RegressionsModels import NGBoostRegressionModel
import pandas as pd
from evidential_boost import NormalInverseGamma
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np
import logging
#from sklearn.datasets import load_diabetes

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
    #safe_path = os.path.join(folder_path, "test/results/XGBoost")
    ### test ende
    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = os.path.join(folder_path, out)
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    # Random Forest Model
     #XGBoost hyperparameters grid    
    # Define the parameter grid for NGBoost
    
    param_grid_ngb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'Base__max_depth': [3, 4, 5]
    }  
    
    # Initialize NGBoost hyperparameters using default values from the grid
    
    NGB_Hparams = {
        #'Dist': NormalInverseGamma,
        'n_estimators': 150, 
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
        test_split_size, 
        safe_path, 
        identifier,
        -1,
        param_grid_ngb)
    metrics = model.evaluate(
        folds=folds, 
        tune=False, 
        nested=True, 
        tune_folds=20, 
        get_shap=False,
        uncertainty=True)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log the metrics
    logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}")
    r2s, p_values = model.feature_ablation()
    

    #param_names  = ["mu", "lambda", "alpha", "beta"]
    #for i in range(len(param_names)):
    #    param = metrics["pred_dist"][i]
    #    plt.figure()
    #    plt.hist(param, bins=20, alpha=0.7, color='blue', edgecolor='black')
    #    plt.title(f"Histogram of {param_names[i]} - Sample {i+1}")
    #    plt.xlabel(f"{param_names[i]} values")
    #    plt.ylabel("Frequency")
    #    plt.grid(True)
    #    plt.savefig(os.path.join(safe_path, f"histogram_{param_names[i]}_sample.png"))
    #    plt.close()
        
        
    

if __name__ == "__main__":
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"
    #main(folder_path, "data/BDI/level2/bdi_df.csv", "diff", "BDI", "results/level2/NGBoost", -1)
    #main(folder_path, "data/MoCA/level2/moca_df.csv", "diff", "MoCA", "results/level2/NGBoost", -1)
    main(folder_path, "data/BDI/level2/bdi_df.csv", "ratio", "BDI", "results/level2/NGBoost", -1)
    #main(folder_path, "data/MoCA/level2/moca_df.csv", "ratio", "MoCA","results/level2/NGBoost", -1)
    
