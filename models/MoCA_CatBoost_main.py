import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

from model_classes.CatBoostRegressionModel import CatBoostRegressionModel

import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.my_logging import Logging
from utils.messages import Messages
from pprint import pformat
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple


def outlier_removal(
    data_df: pd.DataFrame,
    cols: List[str],
    Q1: float = 0.25,
    Q3: float = 0.75,
    iqr_mult: float = 1.5,
    logging=None
) -> pd.DataFrame:
    """Remove outliers from specified columns using the IQR method."""
    Q1_vals = data_df[cols].quantile(Q1)
    Q3_vals = data_df[cols].quantile(Q3)
    IQR = Q3_vals - Q1_vals

    is_outlier = (data_df[cols] < (Q1_vals - iqr_mult * IQR)) | (data_df[cols] > (Q3_vals + iqr_mult * IQR))
    outlier_counts = is_outlier.sum()

    if logging:
        logging.info("IQR Outliers per feature:\n" + str(outlier_counts))
        logging.info("\nDetailed outlier rows by feature:")
        for col in cols:
            outlier_rows = data_df.loc[is_outlier[col], [col]]
            if not outlier_rows.empty:
                logging.info(f"\n--- Outliers in feature: {col} (n={len(outlier_rows)}) ---")
                logging.info(f"\n{outlier_rows.to_string(index=True)}")

    rows_with_outlier = is_outlier.any(axis=1)
    n_outliers_total = rows_with_outlier.sum()
    data_df_clean = data_df.loc[~rows_with_outlier].reset_index(drop=True)

    if logging:
        logging.info(f"\nTotal rows with at least one outlier: {n_outliers_total}")
        logging.info(f"Remaining rows after outlier removal: {len(data_df_clean)}")

    return data_df_clean


def signed_euclidean_distance(points: np.ndarray, sweetspot: np.ndarray) -> np.ndarray:
    """Compute signed Euclidean distance from each point to a sweet spot."""
    deltas = points - sweetspot
    dists = np.linalg.norm(deltas, axis=1)
    signed_dists = np.where(points[:, 2] < sweetspot[2], -dists, dists)
    return signed_dists


def calculate_vif(df: pd.DataFrame, exclude_cols=None, verbose=True):
    if exclude_cols is not None:
        features_df = df.drop(columns=exclude_cols)
    else:
        features_df = df.copy()

    features_df = features_df.dropna()
    features_df = sm.add_constant(features_df)

    vif_data = pd.DataFrame()
    vif_data["feature"] = features_df.columns
    vif_data["VIF"] = [variance_inflation_factor(features_df.values, i)
                        for i in range(features_df.shape[1])]
    vif_data = vif_data[vif_data["feature"] != "const"]

    if verbose:
        print(vif_data.sort_values("VIF", ascending=False))

    return vif_data.sort_values("VIF", ascending=False)


def compute_stimulation_density(mA: np.ndarray, distance: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    return mA / (distance ** 2 + epsilon)


def select_top_shap_features(shap_values: np.ndarray, feature_names: list, threshold: float = 0.95):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })

    shap_df = shap_df.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_df['cumulative_importance'] = shap_df['mean_abs_shap'].cumsum()
    shap_df['cumulative_ratio'] = shap_df['cumulative_importance'] / shap_df['mean_abs_shap'].sum()

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
    outlier_cols=None,
    out=None,
    folds=10,
    tune_folds=5,
    tune=False,
    members=1,
    uncertainty=False,
    filtered_data_path="",
):

    outlier_cols = outlier_cols or []
    safe_path = os.path.join(folder_path, out)

    os.makedirs(safe_path, exist_ok=True)
    os.makedirs(os.path.join(safe_path, 'log'), exist_ok=True)
    log_obj = Logging(f'{safe_path}/log')
    logging = log_obj.get_logger()

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

    Feature_Selection = {'target': target_col, 'features': feature_cols.copy()}

    data_df = pd.read_csv(os.path.join(folder_path, data_path))

    logging.info(f"Dropping {(data_df['TimeSinceSurgery'] < 0.6).sum()} patients with TimeSinceSurgery < 6 months")
    data_df = data_df[data_df["TimeSinceSurgery"] >= 0.6]

    if "MDS_UPDRS_III_sum_OFF" in data_df.columns and "MDS_UPDRS_III_sum_ON" in data_df.columns:
        data_df["UPDRS_reduc_pre"] = (data_df["MDS_UPDRS_III_sum_OFF"] - data_df["MDS_UPDRS_III_sum_ON"]) / data_df["MDS_UPDRS_III_sum_OFF"]
        logging.info(f"Patients with negative UPDRS_reduc_pre: {(data_df['UPDRS_reduc_pre'] < 0).sum()}")
        data_df = data_df[data_df["UPDRS_reduc_pre"] >= 0]

    if 'X_L' in data_df.columns:
        data_df['L_distance'] = signed_euclidean_distance(
            data_df[['X_L', 'Y_L', 'Z_L']].values, [12.08, -13.94, -6.74]
        )
        data_df['R_distance'] = signed_euclidean_distance(
            data_df[['X_R', 'Y_R', 'Z_R']].values, [11.90, -13.28, -6.74]
        )

    data_df = outlier_removal(data_df, outlier_cols, logging=logging)
    op_dates = data_df.get("OP_DATUM")

    data_df = data_df[feature_cols + [target_col]].copy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data_df.select_dtypes(include='number').corr(),
        annot=True, cmap='coolwarm', center=0, fmt=".2f", square=True, linewidths=0.5
    )
    plt.title(f'Correlation Matrix: Features and Target N={len(data_df)}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(safe_path, "all_features_corr.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

    _ = calculate_vif(data_df[Feature_Selection['features']].select_dtypes(include='number'), exclude_cols=[])

    if filtered_data_path:
        data_dir = os.path.dirname(data_path)
        filtered_df = data_df.copy()
        if op_dates is not None:
            filtered_df["OP_DATUM"] = op_dates
        filtered_df.to_csv(os.path.join(data_dir, filtered_data_path))

    param_grid_cat = {
        'depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05,],
        'l2_leaf_reg': [1, 3, 5],
        'iterations': [300, 500, 700, 900],
        'grow_policy': ['SymmetricTree'],
        'min_data_in_leaf': [3, 5, 10],
        'random_strength': [0, 0.5]
    }

    cat_defaults = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'random_seed': 42,
        'loss_function': 'RMSE',
        'verbose': 0,
        'allow_writing_files': False
    }

    model = CatBoostRegressionModel(
        data_df=data_df,
        feature_selection=Feature_Selection,
        target_name=target_col,
        cat_hparams=cat_defaults,
        test_split_size=0.2,
        save_path=safe_path,
        top_n=-1,
        param_grid=param_grid_cat,
        logging=logging,
        split_shaps=True,
        random_state=42,
    )

    metrics = model.evaluate(
        folds=folds,
        tune=tune,
        nested=True,
        tune_folds=tune_folds,
        get_shap=True,
        uncertainty=uncertainty
    )

    model.plot("Actual vs. Prediction (CatBoost)")
    #_, _, removals = model.feature_ablation(folds=folds, tune=tune, tune_folds=tune_folds, members=members)

    inference_csv = os.path.join(folder_path, "data/MoCA/level2/ppmi_ledd.csv")
    inference_save_dir = os.path.join(safe_path, "inference_ppmi_ledd")
    model.inference(
        inference_csv_path=inference_csv,
        param_grid=param_grid_cat,
        members=members,
        folds=tune_folds,
        target_col=target_col,
        save_dir=inference_save_dir,
        random_seed=42,
    )

if __name__ == "__main__":
    #folder_path = "/home/ubuntu/PD-MultiModal-Prediction/"
    folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"

    exp_infos = [
        {
            'exp_number': 1,
            'target_col': "MoCA_sum_post",
            'feature_cols': [
                "TimeSinceSurgery",
                "AGE_AT_OP",
                "TimeSinceDiag",
                "UPDRS_reduc_pre",
                "MoCA_Executive_sum_pre",
                "MoCA_Erinnerung_sum_pre",
                "MoCA_Sprache_sum_pre",
                "MoCA_Aufmerksamkeit_sum_pre",
                "MoCA_Abstraktion_sum_pre",
                #"LEDD_reduc",
                "LEDD_pre",
                #"L_distance",
                #"R_distance",
            ],
            'outlier_cols': [
                "TimeSinceSurgery",
                "TimeSinceDiag",
                "UPDRS_reduc_pre",
                "MoCA_sum_pre",
                #"LEDD_reduc",
                "LEDD_pre",
                #"L_distance",
                #"R_distance"
            ],
        },
    ]

    for exp_info in exp_infos:
        main(
            folder_path=folder_path,
            data_path="data/MoCA/level2/moca_ledd.csv",
            feature_cols=exp_info['feature_cols'],
            target_col=exp_info['target_col'],
            outlier_cols=exp_info['outlier_cols'],
            out=f"results/{exp_info['exp_number']}_{exp_info['target_col']}_ledd/level2/CatBoost",
            folds=10,
            tune_folds=5,
            tune=True,
            members=10,
            uncertainty=False,
            filtered_data_path="filtered_MoCA_ledd.csv",
        )
