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
    folder_path, 
    data_path, 
    target, 
    identifier, 
    out, 
    folds=10, 
    tune_folds=5, 
    detrend=True, 
    tune=False, 
    uncertainty=False, 
    filtered_data_path=""):
    
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
    #data_df = data_df[(data_df["TimeSinceSurgery"] >= 1) & (data_df["TimeSinceSurgery"] <= 3)]
    data_df = data_df[(data_df["TimeSinceSurgery"] >= 0.3)]

    # Left locations in our dataframe is negated! otherwise sweetspot is [-12.08, -13.94,-6.74]
    # data_df['L_distance'] = signed_euclidean_distance(data_df[['X_L', 'Y_L', 'Z_L']].values, [12.08, -13.94,-6.74])
    # data_df['R_distance'] = signed_euclidean_distance(data_df[['X_R', 'Y_R', 'Z_R']].values, [11.90, -13.28, -6.74])
    # columns_to_drop += ['X_L', 'Y_L', 'Z_L', 'X_R', 'Y_R', 'Z_R']

    data_df = data_df.drop(columns=columns_to_drop)

    # Define target and features
    Feature_Selection['target'] = target_col
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]

    # Compute IQR for features
    Q1 = data_df[Feature_Selection['features']].quantile(0.25)
    Q3 = data_df[Feature_Selection['features']].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier mask
    is_outlier = (data_df[Feature_Selection['features']] < (Q1 - 1.5 * IQR)) | \
                (data_df[Feature_Selection['features']] > (Q3 + 1.5 * IQR))

    # Count and print number of outliers per feature
    outlier_counts_iqr = is_outlier.sum()
    print("IQR Outliers per feature:\n", outlier_counts_iqr)

    # Print outlier rows grouped by feature
    print("\nDetailed outlier rows by feature:")
    for col in Feature_Selection['features']:
        outlier_rows = data_df[is_outlier[col]]
        if not outlier_rows.empty:
            print(f"\n--- Outliers in feature: {col} (n={len(outlier_rows)}) ---")
            print(outlier_rows[[col]])

    # Optionally print full rows for all outlier-containing samples
    rows_with_any_outlier = data_df[is_outlier.any(axis=1)]
    print(f"\nTotal rows with at least one outlier: {len(rows_with_any_outlier)}")

    # Drop all rows containing any outlier
    data_df = data_df[~is_outlier.any(axis=1)].reset_index(drop=True)
    print(f"Remaining rows after outlier removal: {len(data_df)}")


    # Plot corr matrix of features + target
    # Combine features and target
    # cols_to_plot = Feature_Selection['features'] + [Feature_Selection['target']] + ['FU_MOCA'] + ['AGE_AT_FU']
    # data_df['AGE_AT_FU'] = data_df['AGE_AT_OP'] + data_df['TimeSinceSurgery']
    # data_df['FU_MOCA'] = data_df['MoCA_sum_pre'] + data_df['MoCA_diff']

    # corr_matrix = data_df[cols_to_plot].corr()

    # # Plot
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", square=True, linewidths=0.5)
    # plt.title(f'Correlation Matrix: Features and Target N={len(data_df)}', fontsize=14)
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig("features_corr.png", dpi=300, bbox_inches='tight')


    # Let's perform VUF analysis
    vif_results = calculate_vif(data_df[Feature_Selection['features']], exclude_cols=[])

    # Let's test the OLS models
    X = data_df[Feature_Selection['features']]
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    X_const = sm.add_constant(X_scaled)
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
    
    if filtered_data_path != "":
        data_dir = os.path.dirname(data_path)
        data_df.to_csv(data_dir + '/' + filtered_data_path)


    param_grid_xgb = {
        #'objective': 'reg:squarederror',   # or 'reg:absoluteerror', etc., depending on your loss preferences
        'learning_rate': [0.1, 0.2, 0.3],              # aka eta
        'max_depth': [3, 4, 5, 6, 7],
        # 'min_child_weight': 1,
        # 'gamma': 0,
        # 'subsample': 1,
        # 'colsample_bytree': 1,
        # 'reg_alpha': 0,                    # L1 regularization
        # 'reg_lambda': 1,                   # L2 regularization
        'n_estimators': [30, 50, 100, 200, 300, 350, 400],                # set during model init, can be tuned
        'random_state': [2025],
    }

    
    XGB_Hparams = {
        'objective': 'reg:squarederror',   # or 'reg:absoluteerror', etc., depending on your loss preferences
        'learning_rate': 0.3,              # aka eta
        'max_depth': 6,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'reg_alpha': 0,                    # L1 regularization
        'reg_lambda': 1,                   # L2 regularization
        'n_estimators': 100,                # set during model init, can be tuned
        'random_state': 2025,
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

    #### Farzin was here too, sorry :D
    
    shap_vals = np.load(f'{model.save_path}/{identifier}_{target}_mean_shap_values.npy')

    top_feats, explained = select_top_shap_features(shap_vals, Feature_Selection['features'], threshold=0.95)
    ######
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
        "results/MoCA_XGBoost_tune_bigger_3month_XYZ_detrend/level2/XGBoost", 
        folds=20, 
        tune_folds=20, 
        detrend=True,
        tune=True,
        filtered_data_path="filtered_moca_wo_mmse_df.csv")