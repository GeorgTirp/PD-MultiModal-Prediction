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
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def outlier_removal(
    data_df: pd.DataFrame, 
    cols: List, 
    Q1: float = 0.25, 
    Q3: float = 0.75, 
    logging = None):
    """
    Throw outliers in the provided list of cols    
    """

    # 2) Compute IQR on just the numeric columns
    Q1  = data_df[cols].quantile(Q1)
    Q3  = data_df[cols].quantile(Q3)
    IQR = Q3 - Q1
    # same logic, but you could customize which subset to use
    is_outlier = (data_df[cols] < (Q1 - 3 * IQR)) | \
                    (data_df[cols] > (Q3 + 3 * IQR))
    # Count and print number of outliers per feature
    outlier_counts_iqr = is_outlier.sum()
    logging.info("IQR Outliers per feature:\n", outlier_counts_iqr)
    # Print outlier rows grouped by feature
    logging.info("\nDetailed outlier rows by feature:")
    
    for col in cols:
        outlier_rows = data_df[is_outlier[col]]
        if not outlier_rows.empty:
            logging.info(f"\n--- Outliers in feature: {col} (n={len(outlier_rows)}) ---")
            logging.info(outlier_rows[[col]])
    # Optionally print full rows for all outlier-containing samples
    rows_with_any_outlier = data_df[is_outlier.any(axis=1)]
    logging.info(f"\nTotal rows with at least one outlier: {len(rows_with_any_outlier)}")
    # Drop all rows containing any outlier
    data_df = data_df[~is_outlier.any(axis=1)].reset_index(drop=True)
    logging.info(f"Remaining rows after outlier removal: {len(data_df)}")

    return data_df

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

def select_top_shap_features(shap_values: np.ndarray, feature_names: list, threshold: float = 0.95):
    """
    Select top features contributing to `threshold` proportion of total SHAP importance.
    
    Parameters:
        shap_values (np.ndarray): SHAP values (n_samples, n_features)
        feature_names (list): Names of features in correct order
        threshold (float): Proportion of total importance to retain (default: 0.95)

    Returns:
        selected_features (list): List of selected feature names
        explained_ratio (float): Actual ratio of total importance captured
    """
    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })

    # Sort features by importance
    shap_df = shap_df.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_df['cumulative_importance'] = shap_df['mean_abs_shap'].cumsum()
    shap_df['cumulative_ratio'] = shap_df['cumulative_importance'] / shap_df['mean_abs_shap'].sum()

    # Find how many features are needed to reach threshold
    idx_cutoff = (shap_df['cumulative_ratio'] < threshold).sum() + 1
    selected_features = shap_df['feature'].iloc[:idx_cutoff].tolist()
    explained_ratio = shap_df['cumulative_ratio'].iloc[idx_cutoff - 1]

    print(f"Selected {len(selected_features)} features out of {len(feature_names)}")
    print(f"These features explain {explained_ratio:.2%} of total SHAP importance")

    return selected_features, explained_ratio

def main(
    folder_path=None, 
    data_path=None, 
    feature_cols=None, 
    target_col=None, 
    outlier_cols = [],
    out=None, 
    folds=10, 
    tune_folds=5, 
    tune=False, 
    drop_iqr_outliers=False,
    uncertainty=False, 
    filtered_data_path="",
    ):
    
    safe_path = os.path.join(folder_path, out)

    # --- SETUP LOGGING ---
    os.makedirs(safe_path, exist_ok=True)
    os.makedirs(os.path.join(safe_path, 'log'), exist_ok=True)
    log_obj = Logging(f'{safe_path}/log')  # keep the object
    logging = log_obj.get_logger()          # get the logger

    # --- Print all selected parameters ---
    logging.info('-------------------------------------------')
    logging.info(f"Folder Path: {folder_path}")
    logging.info(f"Data Path: {data_path}")
    logging.info(f"Target: {target_col}")
    logging.info(f"Output Path: {out}")
    logging.info(f"Folds: {folds}")
    logging.info(f"tune_folds: {tune_folds}")
    logging.info(f"drop_iqr_outliers: {drop_iqr_outliers}")
    logging.info(f"uncertainty: {uncertainty}")
    logging.info(f"filtered_data_path: {filtered_data_path}")
    logging.info(f"feature_cols: {feature_cols}")
    logging.info('-------------------------------------------\n')

    test_split_size = 0.2
    Feature_Selection = {}

    data_df = pd.read_csv(os.path.join(folder_path,data_path))

    # Restrict FUs between 1-3 years
    logging.info(f"Dropping {(data_df["TimeSinceSurgery"] < 0.6).sum()} patients with TimeSinceSurgery less than 6 months")
    data_df = data_df[(data_df["TimeSinceSurgery"] >= 0.6)]

    # If MDS_UPDRS_III_sum_OFF is present, compute UPDRS_ratio and drop the source columns
    if "MDS_UPDRS_III_sum_OFF" in data_df.columns and "MDS_UPDRS_III_sum_ON" in data_df.columns:
        data_df["UPDRS_reduc_pre"] = (data_df["MDS_UPDRS_III_sum_OFF"] - data_df["MDS_UPDRS_III_sum_ON"]) / data_df["MDS_UPDRS_III_sum_OFF"]
        logging.info(f"Patients with negative UPDRS_reduc_pre: {(data_df["UPDRS_reduc_pre"] < 0).sum()}")
        data_df = data_df[data_df["UPDRS_reduc_pre"] >= 0]

    #data_df['AGE_AT_DIAG'] = data_df["AGE_AT_OP"] - data_df['TimeSinceDiag']
    #columns_to_drop += ['TimeSinceDiag']
    
    if 'X_L' in data_df.columns:
        # Left locations in our dataframe is negated! otherwise sweetspot is [-12.08, -13.94,-6.74]
        data_df['L_distance'] = signed_euclidean_distance(data_df[['X_L', 'Y_L', 'Z_L']].values, [12.08, -13.94,-6.74])
        data_df['R_distance'] = signed_euclidean_distance(data_df[['X_R', 'Y_R', 'Z_R']].values, [11.90, -13.28, -6.74])

    # Define target and features
    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = feature_cols
    
    data_df = outlier_removal(data_df, outlier_cols, logging=logging)
    op_dates = data_df["OP_DATUM"]

    data_df = data_df[feature_cols + [target_col]]
    # Plot corr matrix of features + target
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_df.select_dtypes(include='number').corr(), 
                annot=True, cmap='coolwarm', center=0, fmt=".2f", square=True, linewidths=0.5)
    plt.title(f'Correlation Matrix: Features and Target N={len(data_df)}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out,"all_features_corr.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    # Let's perform VUF analysis
    vif_results = calculate_vif(data_df[Feature_Selection['features']].select_dtypes(include='number'), exclude_cols=[])

    # Shuffle target values
    #data_df[Feature_Selection['target']] = data_df[Feature_Selection['target']].sample(frac=1).values
    
    if filtered_data_path != "":
        data_dir = os.path.dirname(data_path)
        filtered_df = data_df.copy()
        filtered_df["OP_DATUM"] = op_dates
        filtered_df.to_csv(data_dir + '/' + filtered_data_path)

        #op_dates.to_csv(op_dates_path, index=False)

    param_grid_xgb = {
        'learning_rate': [0.2, 0.3, 0.05],              # aka eta
        'max_depth': [5, 6, 7, 8],
        'n_estimators': [100, 200, 300],                # set during model init, can be tuned
        'random_state': [random_state],

        'min_child_weight': [1, 3],
        'reg_lambda': [1, 1.5, 2],                   # L2 regularization

    }

    # Default XGBoost hyperparameters
    XGB_Hparams = {
        'objective': 'reg:squarederror',   # or 'reg:absoluteerror', etc., depending on your loss preferences
        'learning_rate': 0.2,              # aka eta
        'max_depth': 6,
        'min_child_weight': 3,
        'gamma': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'reg_alpha': 0,                    # L1 regularization
        'reg_lambda': 2,                   # L2 regularization
        'n_estimators': 100,                # set during model init, can be tuned
        'random_state': random_state,
        'enable_categorical': True
    }

    model = XGBoostRegressionModel(
        data_df, 
        Feature_Selection, 
        target,
        XGB_Hparams, 
        0.2, 
        safe_path, 
        identifier,
        -1,
        param_grid_xgb,
        logging=logging,
        #Pat_IDs=Pat_IDs,
        #split_shaps=False,
        #sample_weights=sample_weights
        )

    metrics = model.evaluate(
        folds=folds, 
        tune=tune, 
        nested=True, 
        tune_folds=tune_folds, 
        get_shap=True,
        uncertainty=uncertainty)

    ######
    model.plot(f"Actual vs. Prediction (NGBoost)")
    _,_, removals= model.feature_ablation(folds=folds, tune=tune, tune_folds=tune_folds)
    #model.calibration_analysis()
    
    log_obj.close()

if __name__ == "__main__":

    folder_path = "/home/ubuntu/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"

    exp_infos = [
                {
                'exp_number' : 1,
                'target_col' :"BDI_sum_post", 
                'ignore_cols':[ 
                            "BDI_Harmonized_pre", "BDI_Cognitive_pre", "BDI_Affective_pre", "BDI_Somatic_pre",
                            "BDI_Cognitive_post", "BDI_Affective_post", "BDI_Somatic_post", "BDI_Harmonized_post",
                            "BDI_Cognitive_delta", "BDI_Affective_delta", "BDI_Somatic_delta","BDI_Harmonized_delta", 
                            "BDI_sum_delta",
                            #"BDI_sum_pre",
                            #"BDI_sum_post"
                            ] ,
                },   
    ]
    for exp_info in exp_infos:

        exp_number = exp_info['exp_number']
        target_col= exp_info['target_col']
        ignore_cols = exp_info['ignore_cols']
        outlier_cols = exp_info['outlier_cols']

        main(folder_path=folder_path, 
            data_path="data/BDI/level2/bdi_stim.csv", 
            ignore_cols=ignore_cols, 
            target_col=target_col,
            outlier_cols=outlier_cols, 
            out=f"results/{exp_number}_{target_col}_stim/level2/XGBoost", 
            folds=10, 
            tune_folds=5, 
            tune=True, 
            drop_iqr_outliers=True,
            uncertainty=False, 
            filtered_data_path="filtered_bdi_demo.csv") 


