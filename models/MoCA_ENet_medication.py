import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

from model_classes.ElasticNetRegressionModel import ElasticNetRegressionModel

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
from typing import List, Tuple, Optional

def outlier_removal(
    data_df: pd.DataFrame, 
    cols: List[str], 
    Q1: float = 0.25, 
    Q3: float = 0.75, 
    iqr_mult: float = 1.5,
    logging=None
) -> pd.DataFrame:
    """
    Remove outliers from specified columns using the IQR method.

    Parameters:
        data_df (pd.DataFrame): The full dataset.
        cols (List[str]): List of columns to check for outliers.
        Q1 (float): Lower quantile boundary.
        Q3 (float): Upper quantile boundary.
        logging: Logger instance.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if not cols:
        return data_df

    normalized_cols = []
    missing_cols = []
    for col in cols:
        col_name = ''.join(col) if isinstance(col, (tuple, list)) else col
        if col_name in data_df.columns:
            normalized_cols.append(col_name)
        else:
            missing_cols.append(col_name)

    if logging and missing_cols:
        logging.warning(f"Skipping outlier removal for missing columns: {missing_cols}")

    if not normalized_cols:
        return data_df

    # Compute IQR bounds
    Q1_vals = data_df[normalized_cols].quantile(Q1)
    Q3_vals = data_df[normalized_cols].quantile(Q3)
    IQR = Q3_vals - Q1_vals

    # Identify outliers
    is_outlier = (data_df[normalized_cols] < (Q1_vals - iqr_mult * IQR)) | (data_df[normalized_cols] > (Q3_vals + iqr_mult * IQR))

    # Count outliers
    outlier_counts = is_outlier.sum()

    if logging:
        logging.info("IQR Outliers per feature:\n" + str(outlier_counts))

        logging.info("\nDetailed outlier rows by feature:")
        for col in normalized_cols:
            outlier_rows = data_df.loc[is_outlier[col], [col]]
            if not outlier_rows.empty:
                logging.info(f"\n--- Outliers in feature: {col} (n={len(outlier_rows)}) ---")
                logging.info(f"\n{outlier_rows.to_string(index=True)}")

    # Rows with any outlier
    rows_with_outlier = is_outlier.any(axis=1)
    n_outliers_total = rows_with_outlier.sum()
    data_df_clean = data_df.loc[~rows_with_outlier].reset_index(drop=True)

    if logging:
        logging.info(f"\nTotal rows with at least one outlier: {n_outliers_total}")
        logging.info(f"Remaining rows after outlier removal: {len(data_df_clean)}")

    return data_df_clean


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
    members=1, 
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
    logging.info(f"uncertainty: {uncertainty}")
    logging.info(f"filtered_data_path: {filtered_data_path}")
    logging.info(f"feature_cols: {feature_cols}")
    logging.info('-------------------------------------------\n')

    test_split_size = 0.2
    Feature_Selection = {}

    data_df = pd.read_csv(os.path.join(folder_path, data_path))
    if "TEST_DATUM_pre" in data_df.columns:
        data_df = data_df.rename(columns={"TEST_DATUM_pre": "OP_DATUM"})

    # Restrict FUs between 1-3 years for the base cohort
    mask_valid_fu = data_df["TimeSinceSurgery"] >= 0.6
    logging.info(f"Dropping {(~mask_valid_fu).sum()} patients with TimeSinceSurgery less than 6 months")
    data_df = data_df[mask_valid_fu].copy()

    if "UPDRS_off" in data_df.columns and "UPDRS_on" in data_df.columns:
        data_df["UPDRS_reduc"] = (
                data_df["UPDRS_off"] - data_df["UPDRS_on"]
            ) / data_df["UPDRS_off"]
    #columns_to_drop += ['TimeSinceDiag']

    if 'X_L' in data_df.columns:
        # Left locations in our dataframe is negated! otherwise sweetspot is [-12.08, -13.94,-6.74]
        data_df['L_distance'] = signed_euclidean_distance(data_df[['X_L', 'Y_L', 'Z_L']].values, [12.08, -13.94,-6.74])
        data_df['R_distance'] = signed_euclidean_distance(data_df[['X_R', 'Y_R', 'Z_R']].values, [11.90, -13.28, -6.74])
    #data_df['L_distance'] = np.random.permutation(data_df['L_distance'].values)
    #data_df['R_distance'] = np.random.permutation(data_df['R_distance'].values)
    #data_df["STIM_distance"] = (data_df['L_distance'] + data_df['R_distance']) / 2
    # Define target and features
    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = feature_cols

    

    # Ensure critical numeric columns are properly typed
    for col in ["UPDRS_on", "MoCA_sum_pre", "TimeSinceSurgery", "AGE_AT_BASELINE", "TimeSinceDiag"]:
        if col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    ### SPLIT BY AGE###
    #data_df= data_df[data_df["AGE_AT_OP"] >= 65]
    #data_df= data_df[data_df["AGE_AT_OP"] < 65]

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
        filtered_df = data_df.copy().dropna()
        filtered_df["OP_DATUM"] = op_dates
        filtered_df.to_csv(data_dir + '/' + filtered_data_path)

        #op_dates.to_csv(op_dates_path, index=False)
    param_grid = {
    'alpha':     np.logspace(-3, 1, 5, 10),
    'l1_ratio':  np.linspace(0.05, 0.95, 5, 10),
    'fit_intercept': [True, False],
    'max_iter':  [10000],                  # keep high for convergence
    }
    params = {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'fit_intercept': True,
        'max_iter': 10000,
    }

    model = ElasticNetRegressionModel(
        data_df=data_df,
        feature_selection=Feature_Selection,
        target_name=target_col,
        hparams=params,
        test_split_size=test_split_size,
        save_path=safe_path,
        #identifier=identifier,
        top_n=-1,
        hparam_grid=param_grid,
        logging=logging,
        split_shaps=True,
        standardize_features="zscore",
        #random_state=42
        )

    metrics = model.evaluate(
        folds=folds, 
        tune=tune, 
        nested=True, 
        tune_folds=tune_folds, 
        get_shap=True,
#        safe_best_hparams=True,
        uncertainty=uncertainty)

    ######
    model.plot(f"Actual vs. Prediction (Elastic Net)")
    _,_, removals= model.feature_ablation_ensemble(folds=folds, tune=tune, tune_folds=tune_folds, members=members)
    #inference_csv = os.path.join(folder_path, "data/MoCA/level2/ppmi_ledd.csv")
    #inference_save_dir = os.path.join(safe_path, "inference_ppmi_ledd")
    #model.inference(
    #    inference_csv_path=inference_csv,
    #    param_grid=param_grid,
    #    members=members,
    #    folds=tune_folds,
    #    target_col=target_col,
    #    save_dir=inference_save_dir,
    #    random_seed=42,
    #)

    #log_obj.close()
        
if __name__ == "__main__":

    #folder_path = "/home/ubuntu/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"

    exp_infos = [
                {
                'exp_number' : 1,
                'target_col' :"MoCA_sum_post", 
                'feature_cols':[ 
                            "TimeSinceSurgery",
                            "AGE_AT_BASELINE",
                            "TimeSinceDiag",
                            #"SEX",
                            #"UPDRS_reduc",
                            "UPDRS_on",
                            #"MoCA_sum_pre",
                            "MoCA_Executive_sum_pre",
                            #"MoCA_Executive_sum_post",
                            "MoCA_Erinnerung_sum_pre",
                            #"MoCA_Erinnerung_sum_post",
                            "MoCA_Sprache_sum_pre",
                            #"MoCA_Sprache_sum_post",
                            "MoCA_Aufmerksamkeit_sum_pre",
                            #"MoCA_Aufmerksamkeit_sum_post",
                            #"MoCA_Benennen_sum_pre",
                            #"MoCA_Benennen_sum_post",
                            "MoCA_Abstraktion_sum_pre",
                            #"MoCA_Abstraktion_sum_post",
                            #"MoCA_Orientierung_sum_pre",
                            #"MoCA_Orientierung_sum_post",
                            #"LEDD_reduc",
                            "LEDD_pre",
                            #"L_distance",
                            #"R_distance",
                            #"Left_1_mA",
                            #"Right_1_mA",
                            #"STIM_distance"
                            #"X_L" ,
                            #"Y_L" ,
                            #"Z_L" ,
                            #"X_R" ,
                            #"Y_R" ,
                            #"Z_R" ,
                            ] ,
                'outlier_cols':[ 
                            "TimeSinceSurgery",
                            "AGE_AT_BASELINE",
                            "TimeSinceDiag",
                            #"SEX",
                            #"UPDRS_reduc",
                            "MoCA_sum_pre",
                            "UPDRS_on",
                            #"MoCA_Executive_sum_pre",
                            #"MoCA_Executive_sum_post",
                            #"MoCA_Erinnerung_sum_pre",
                            #"MoCA_Erinnerung_sum_post",
                            #"MoCA_Sprache_sum_pre",
                            #"MoCA_Sprache_sum_post",
                            #"MoCA_Aufmerksamkeit_sum_pre",
                            #"MoCA_Aufmerksamkeit_sum_post",
                            #"MoCA_Benennen_sum_pre",
                            #"MoCA_Benennen_sum_post",
                            #"MoCA_Abstraktion_sum_pre",
                            #"MoCA_Abstraktion_sum_post",
                            #"MoCA_Orientierung_sum_pre",
                            #"MoCA_Orientierung_sum_post",
                            #"LEDD_reduc",
                            "LEDD_pre",
                            #"L_distance",
                            #"R_distance",
                            #"STIM_distance"
                            ] ,
                },
    ]
    for exp_info in exp_infos:

        exp_number = exp_info['exp_number']
        target_col= exp_info['target_col']
        feature_cols = exp_info['feature_cols']
        outlier_cols = exp_info['outlier_cols']
        main(folder_path=folder_path, 
            data_path="data/MoCA/level2/ppmi_medication_distribution_cohort_ledd.csv", 
            feature_cols=feature_cols, 
            target_col=target_col, 
            outlier_cols=outlier_cols,
            out=f"results/{exp_number}_{target_col}_ppmi_medication_ledd/level2/ElasticNet", 
            folds=10, 
            tune_folds=5, 
            tune=True, 
            members=10,
            uncertainty=False, 
            filtered_data_path="filtered_MoCA_ppmi_medication.csv"
            )
