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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from scipy.stats import zscore

def signed_euclidean_distance(points: np.ndarray, sweetspot: np.ndarray) -> np.ndarray:
    """
    Compute signed Euclidean distance from each point to a sweet spot.
    Distance is negative if the point is below the sweet spot in Z.

    Parameters:
        points (np.ndarray): shape (n_samples, 3), where each row is (X, Y, Z)
        sweetspot (np.ndarray or tuple): shape (3,), (X, Y, Z) of the sweet spot

    Returns:
        np.ndarray: shape (n_samples,), signed distances
    """
    deltas = points - sweetspot
    dists = np.linalg.norm(deltas, axis=1)
    signed_dists = np.where(points[:, 2] < sweetspot[2], -dists, dists)
    return signed_dists


def calculate_vif(df, exclude_cols=None, verbose=True):
    """
    Calculates Variance Inflation Factor (VIF) for a dataframe's features.

    Parameters:
        df (pd.DataFrame): DataFrame with only numerical features to assess.
        exclude_cols (list): Columns to exclude from VIF analysis (e.g. IDs).
        verbose (bool): If True, prints the results.

    Returns:
        pd.DataFrame: VIF scores for each variable.
    """
    if exclude_cols is not None:
        features_df = df.drop(columns=exclude_cols)
    else:
        features_df = df.copy()

    # Drop NaNs
    features_df = features_df.dropna()

    # Add constant term for intercept
    features_df = sm.add_constant(features_df)

    vif_data = pd.DataFrame()
    vif_data["feature"] = features_df.columns
    vif_data["VIF"] = [variance_inflation_factor(features_df.values, i) 
                       for i in range(features_df.shape[1])]

    # Remove constant from results
    vif_data = vif_data[vif_data["feature"] != "const"]

    if verbose:
        print(vif_data.sort_values("VIF", ascending=False))

    return vif_data.sort_values("VIF", ascending=False)

#from sklearn.datasets import load_diabetes
# --- Dynamic Tobit bound functions ---

def compute_stimulation_density(mA: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Compute stimulation density as mA / (distance^2).
    Assumes distance in mm.

    Parameters:
        mA (np.ndarray): Current in mA, shape (n,)
        distance (np.ndarray): Euclidean distance in mm, shape (n,)

    Returns:
        np.ndarray: stimulation density, shape (n,)
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    return mA / (distance**2 + epsilon)

def main(folder_path, data_path, target, identifier, out, folds=10,tune_folds=5, detrend=True, tune=False, uncertainty=False):
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
    'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.9, 0.95],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.1, 0.5, 0.7],
    'reg_lambda': [2, 5, 8],
    'random_state': [42],
    'verbosity': [0]
}

    # Default XGBoost hyperparameters
    XGB_Hparams = {
        'n_estimators': 400,
        'learning_rate': 0.01,
        'max_depth': 4,                  
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,                
        'reg_lambda': 2,
        'min_child_weight': 5,           
        'gamma': 0.1,                    
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
        tune=tune, 
        nested=True, 
        tune_folds=tune_folds, 
        get_shap=True,
        uncertainty=uncertainty)
    
    # Log the metrics
    logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}")
    _,_, removals= model.feature_ablation(folds=folds, tune=tune, tune_folds=tune_folds)
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

    folder_path = "/home/ubuntu/PD-MultiModal-Prediction/"

    main(
        folder_path, 
        "data/MoCA/level2/moca_wo_mmse_df.csv", 
        "diff", 
        "MoCA", 
        "results/MoCA_XGBoost_diff/level2/XGBoost", 
        folds=10, 
        tune_folds=10, 
        detrend=True,
        tune=True)