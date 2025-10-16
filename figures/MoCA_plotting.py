import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D 
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, LeaveOneOut
import pickle
from scipy.stats import pearsonr
import faster_evidential_boost
from tqdm import tqdm
import ast
from scipy.stats import linregress
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import ttest_rel, false_discovery_control
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator
import shap

def _load_shap_data(shap_path: str):
    """
    Load SHAP values from either a NumPy binary or CSV file. Returns a tuple of
    (array, feature_names) where feature_names is None for NumPy inputs.
    """
    if shap_path.lower().endswith(".csv"):
        shap_df = pd.read_csv(shap_path)
        shap_df = shap_df.loc[:, ~shap_df.columns.str.contains("^Unnamed")]
        return shap_df.to_numpy(), list(shap_df.columns)
    shap_values = np.load(shap_path, allow_pickle=True)
    return shap_values, None


def _ensure_2d_shap(shap_values: np.ndarray) -> np.ndarray:
    """Ensure SHAP values are returned as a 2D (samples x features) array."""
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        # Combine folds/samples so downstream code can operate on samples directly.
        n_folds, n_samples, n_features = shap_values.shape
        return shap_values.reshape(n_folds * n_samples, n_features)
    if shap_values.ndim == 2:
        return shap_values
    if shap_values.ndim == 1:
        return shap_values[:, np.newaxis]
    raise ValueError(f"Unsupported SHAP array shape: {shap_values.shape}")


def _prepare_dataframe(path: str) -> pd.DataFrame:
    """Load a CSV and drop unnamed index columns if they exist."""
    df = pd.read_csv(path)
    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def _resolve_remaining_features(
        df: pd.DataFrame,
        removals: pd.DataFrame,
        shap_values: np.ndarray,
        shap_feature_names: Optional[List[str]]) -> list:
    """
    Determine the list of features that correspond to the supplied SHAP values.
    Prefer explicit feature names shipped with the SHAP file; fall back to the
    ablation removal history if necessary.
    """
    if shap_feature_names:
        # Keep only columns that are present in the dataframe.
        remaining = [name for name in shap_feature_names if name in df.columns]
        if not remaining:
            raise ValueError(
                "None of the SHAP feature names match the provided dataframe columns."
            )
        return remaining

    df = df.drop(columns=["BDI_diff", "BDI_ratio", "Pat_ID"], errors="ignore")
    df = df.drop(columns=["MoCA_diff", "MoCA_ratio", "Pat_ID"], errors="ignore")
    original_features = [col for col in df.columns if col != "BDI_diff"]

    shap_matrix = np.asarray(shap_values)
    n_shap_cols = shap_matrix.shape[-1]
    n_removed_before = len(original_features) - n_shap_cols

    if n_removed_before < 0 or n_removed_before > len(removals):
        raise ValueError(
            f"Calculated removed_count = {n_removed_before} is invalid. "
            "Check that shap_values and removal_list correspond."
        )

    removed_up_to_now = removals.iloc[:n_removed_before, 0].astype(str).tolist()
    remaining_features = [f for f in original_features if f not in removed_up_to_now]
    return remaining_features


def _select_best_metrics(metrics_paths: List[str], score_columns: Optional[List[str]] = None) -> Optional[str]:
    """
    Given a list of metric file paths, return the one with the highest score.
    Score columns are checked in order; defaults to ['r2', 'r', 'r_ensemble'].
    """
    if score_columns is None:
        score_columns = ["r2", "r", "r_ensemble"]
    best_path = None
    best_score = -np.inf
    for path in metrics_paths:
        if not os.path.exists(path):
            continue
        try:
            df = _prepare_dataframe(path)
        except Exception:
            continue
        score = None
        for col in score_columns:
            if col in df.columns:
                score = df[col].iloc[0]
                break
        if score is None or pd.isna(score):
            continue
        if score > best_score:
            best_score = score
            best_path = path
    return best_path

def plot_stim_positions(positions: pd.DataFrame, safe_path: str = "") -> None:
    """ Plot the stimulation positions in the brain"""
     # Create a figure with two subplots
    data_to_plot = positions.copy()
    fig = plt.figure(figsize=(14, 7))
    
    # Left subplot for left hemisphere
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data_to_plot["X_L_stim"], data_to_plot["Y_L_stim"], data_to_plot["Z_L_stim"], c='r', label='Stim Points')
    ax1.set_xlabel('X_L')
    ax1.set_ylabel('Y_L')
    ax1.set_zlabel('Z_L')
    
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_title(f'Left Hemisphere Stim Points')
    ax1.legend()
    # Right subplot for right hemisphere
    ax2 = fig.add_subplot(122, projection='3d')
    
    ax2.scatter(data_to_plot["X_R_stim"], data_to_plot["Y_R_stim"], data_to_plot["Z_R_stim"], c='r', label='Stim Points')
    ax2.set_xlabel('X_R')
    ax2.set_ylabel('Y_R')
    ax2.set_zlabel('Z_R')
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_title(f'Right Hemisphere Stim Points')
    ax2.legend()
    # Save the plot in the current directory
    plt.savefig(safe_path + "Stim_positions.png")
    plt.close()
    return



def raincloud_plot(data: pd.DataFrame, modality_name: str, features_list: list, safe_path: str = "", show: bool = False) -> None:
    """Create a refined raincloud plot using seaborn for better aesthetics."""
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Prepare data
    data_list = [data[col].dropna().values for col in data.columns]
    x_positions = np.arange(1, len(data_list) + 1)
    
    # Define colors based on modality
    colors = {
            "deterioration": "#04E762",
            "improvement": "#FF5714",
            "det_edge": "#007C34",
            "imp_edge": "#8A3210",
            "line": "grey",
            "scatter": "grey",
            "ideal_line": "black",
        }

    base_palette = sns.color_palette("deep")
    violin_color = base_palette[0]
    violin_rgba = mcolors.to_rgba(violin_color, alpha=0.4)
    violin_rgba_light = mcolors.to_rgba(violin_color, alpha=1.0)
    scatter_color = 'black'
    
    # color palette from DL project:
    #custom_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#525252"]
    # Set width parameters
    box_width = 0.15
    violin_width = 0.5
    violin_shift = box_width / 2  # Align violin to inner edge of boxplot

    # Violin plot (half-violin) aligned to boxplot
    for idx, (x, values) in enumerate(zip(x_positions, data_list)):
        sns.violinplot(
            y=values,
            inner=None,
            cut=0,
            linewidth=0,
            color=violin_rgba,
            ax=ax,
            width=violin_width,
            #split=False
        )
        for collection in ax.collections[-1:]:
            if not isinstance(collection, PolyCollection):
                continue  # skip non-violin objects
            collection.set_facecolor(violin_rgba_light)
            collection.set_edgecolor(violin_rgba)
            for path in collection.get_paths():
                verts = path.vertices
                mean_x = np.mean(verts[:, 0])
                
                if modality_name == "MDS-UPDRS III":
                    #
                    # Keep your existing "all-same-side" logic here.
                    # For simplicity, let's do "left half" for all.
                    #
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, mean_x)
                    # Shift everything to x - violin_shift
                    shift_amount = x - violin_shift

                else:
                    # We have 2 columns. For idx=0 => left half, idx=1 => right half
                    if idx == 0:
                        # Clip to left half
                        verts[:, 0] = np.clip(verts[:, 0], -np.inf, mean_x)
                        # Shift to left
                        shift_amount = x - violin_shift
                    else:
                        # Clip to right half
                        verts[:, 0] = np.clip(verts[:, 0], mean_x, np.inf)
                        # Shift to right
                        shift_amount = x + violin_shift

                # Apply the shift
                path.vertices[:, 0] += (shift_amount)
        
    
    # Boxplot (make it slimmer)
    box_rgba = mcolors.to_rgba(violin_color, alpha=0.33)
    box_edge_rgba = mcolors.to_rgba("black", alpha=1)
    palette = [violin_rgba] * data.shape[1]
    bp = sns.boxplot(
        data=data,
        width=box_width,
        showcaps=True,
        whiskerprops={'color': box_edge_rgba},
        medianprops={'color': box_edge_rgba},
        capprops={'color': box_edge_rgba},
        flierprops={'marker': ''},  # Do not mark outliers
        palette=palette,
        boxprops={'facecolor': box_rgba, 'edgecolor': box_edge_rgba},
        ax=ax
    )

    for patch in bp.artists:
        patch.set_facecolor(box_rgba)
        patch.set_edgecolor(box_edge_rgba)
        patch.set_alpha(box_rgba[-1])
   
    
    # Scatter plot (overlay on boxplot)
    for idx, (x, values) in enumerate(zip(x_positions, data_list)):
        if modality_name == "BDI" or modality_name == "MoCA":
        #    x_jitter =   # No jitter for BDI
            np.random.seed(42)
            y_noise = np.random.uniform(low=-0.4, high=0.4, size=len(values))
            y_jitter = values + y_noise
            
        else:
            y_jitter = values
        x_noise = np.random.uniform(low=-0.05, high=0.05, size=len(values))
        x_jitter = x + x_noise
       
        
        ax.scatter(x_jitter, y_jitter, s=10, color=scatter_color , alpha=0.9, zorder=2)  # Set zorder to 2 to plot scatter points in front of boxplot
        
        # Draw lines for BDI modality
        if modality_name == "BDI"  and idx < len(x_positions) - 1:
            next_values = data_list[idx + 1]
            for i in range(len(values)):
                if i < len(next_values):
                    x_start, y_start = x_jitter[i], y_jitter[i]
                    x_end, y_end = x_positions[idx + 1] + x_noise[i]  , next_values[i] + y_noise[i]
                    slope = y_end - y_start
                    if np.isclose(y_start, y_end, atol=0.9):
                            line_color = "grey"
                    elif slope > 0:
                        line_color = colors["improvement"]
                    elif slope < 0:
                        line_color = colors["deterioration"]
                    else:
                        line_color = "black" 
                        
                             # Default color if not equal
                    ax.add_line(Line2D([x_start, x_end], [y_start, y_end], color=line_color, alpha=0.4))
        if modality_name == "MoCA" and idx < len(x_positions) - 1:
            next_values = data_list[idx + 1]
            for i in range(len(values)):
                if i < len(next_values):
                    x_start, y_start = x_jitter[i], y_jitter[i]
                    x_end, y_end = x_positions[idx + 1] + x_noise[i]  , next_values[i] + y_noise[i]
                    slope = y_end - y_start
                    if np.isclose(y_start, y_end, atol=1e-2):
                            line_color = "grey"
                    elif slope > 0:
                        line_color = colors["deterioration"]
                    elif slope < 0:
                        line_color = colors["improvement"]
                    else:
                        line_color = "black" 
                        
                             # Default color if not equal
                    ax.add_line(Line2D([x_start, x_end], [y_start, y_end], color=line_color, alpha=0.4))
    # Labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(features_list, fontsize=11)
    ax.set_yticklabels([f'{int(tick)}' for tick in ax.get_yticks()], fontsize=10)
    ax.set_ylabel(modality_name, fontsize=11)
    ax.set_title(f"{modality_name} Raincloud Plot", fontsize=14)
    ax.set_xlim(0.5, x_positions[-1] + 0.5)

    # Add legend for BDI
    if modality_name == "BDI":
        line_colors = {"increase": "red", "decrease": "green", "no change": "grey"}
        legend_elements = [
            Line2D([0], [0], color=colors["deterioration"], lw=2, label="Deterioration"),
            Line2D([0], [0], color=colors["improvement"], lw=2, label="Improvement"),
            Line2D([0], [0], color=line_colors["no change"], lw=2, label="No change"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

    # Add legend for MoCA
    elif modality_name == "MoCA":
        line_colors = {"increase": "green", "decrease": "red", "no change": "grey"}
        legend_elements = [
            Line2D([0], [0], color=colors["deterioration"], lw=2, label="Improvement"),
            Line2D([0], [0], color=colors["improvement"], lw=2, label="Deterioration"),
            Line2D([0], [0], color=line_colors["no change"], lw=2, label="No change"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")
    plt.grid(False)
    sns.set_context("paper")
    # Optionally choose a style you like
    sns.despine()
    plt.tight_layout()    
    plt.savefig(safe_path + modality_name + "_raincloud_plot.png")
    plt.savefig(safe_path + modality_name + "_raincloud_plot.svg")
    if show:
        plt.show()
    plt.close()
    



def demographics_pre_post(
        modality_path: str, 
        model_data_path = None, 
        modality_name = None,
        save_path = None,
        show = False) -> None:
    """ Plot the demographic data before and after the treatment as raincloud plot"""

    data = _prepare_dataframe(modality_path)
    if model_data_path is not None:
        model_df = _prepare_dataframe(model_data_path)
        data = data[data['OP_DATUM'].isin(model_df['OP_DATUM'])]
    

    if modality_name == "BDI":
        data["BDI_sum_post"] = data["BDI_sum_pre"] + data["BDI_diff"]
        data = data[['BDI_sum_pre', 'BDI_sum_post']]

    if modality_name == "MoCA":
        data["MoCA_sum_post"] = data["MoCA_sum_pre"] + data["MoCA_diff"]
        data = data[['MoCA_sum_pre', 'MoCA_sum_post']]

    if modality_name == "MDS-UPDRS III":
        data = data.rename(columns={'MDS_UPDRS_III_sum_pre': 'Pre', 'MDS_UPDRS_III_sum_post': 'Post'})
        data_pre_off = data[data['MEDICATION'] == 'OFF'].rename(columns={'Pre': 'Pre_OFF'})
        data_pre_on = data[data['MEDICATION'] == 'ON'].rename(columns={'Pre': 'Pre_ON'})
        data_post_off = data[data['MEDICATION'] == 'OFF'].rename(columns={'Post': 'Post_OFF'})
        data_post_on = data[data['MEDICATION'] == 'ON'].rename(columns={'Post': 'Post_ON'})
        data = pd.concat([data_pre_off[['Pre_OFF']], data_pre_on[['Pre_ON']], data_post_on[['Post_ON']]], axis=1)
        data = data.drop(columns=['MEDICATION'], errors='ignore')

    # Drop the 'OP_DATUM' column from data if not already done 
    if 'OP_DATUM' in data.columns:
        data = data.drop(columns=['OP_DATUM'])

    # Plotting demographic data before and after treatment
    if modality_name == "MDS-UPDRS III":
        raincloud_plot(data, modality_name , ['Pre OFF', 'Pre ON', 'Post ON/Stim On'], save_path, show=show)
    else:
        raincloud_plot(data, modality_name , ['Pre', 'Post'], save_path, show=show)


def histoplot(input_path: str, feature: str,  save_path: str, show: bool = False) -> None:
    """ Plot the demographic data before and after the treatment as raincloud plot"""
    # Load the data
    data = _prepare_dataframe(input_path)
    sns.set_theme(style="white", context="paper")
    sns.set_palette("deep")
    # Create a figure with two subplots
    if feature == 'TimeSinceSurgery':
        title = 'Distribution of Time Since Surgery'
        xlabel = 'Time Since Surgery (years)'
        save_path = save_path + "time_since_surgery_histoplot"
        color = 'blue'

    elif feature == 'TimeSinceDiag':
        title = 'Distribution of Time Since Diagnosis'
        xlabel = 'Time Since Diagnosis (years)'
        save_path = save_path + "time_since_diag_histoplot"
        color = 'green'

    elif feature == 'AGE_AT_OP':
        title = 'Distribution of Age at Operation'
        xlabel = 'Age at Operation (years)'
        save_path = save_path + "age_at_op_histoplot"
        color = 'orange'

    fig, ax = plt.subplots(figsize=(10, 6))
    print(data[feature].mean())
    print(data[feature].std())
    sns.histplot(data[feature], kde=True, bins=30, color=color)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(False)
    sns.set_context("paper")
    # Optionally choose a style you like
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".svg")
    if show:
        plt.show()
    plt.close()
    


def visualize_demographics(
        data_path, 
        questionnaire, 
        save_path) -> None:
    # Example usage
    
    
    stim_path = data_path + "/stim_positions.csv"
    mds_prepost_path = data_path+ "/mupdrs3_pre_vs_post.csv"
    ledd_prepost_path = data_path + "/ledd_pre_vs_post.csv"
    if questionnaire == "MoCA":  
        quest_prepost_path = data_path + "/moca_pre_vs_post.csv"
        quest = data_path + "/moca_df.csv"
    elif questionnaire == "BDI":
        quest_prepost_path = data_path + "/bdi_pre_vs_post.csv"
        quest = data_path + "/bdi_df.csv"

    op_dates_path = data_path + "/op_dates.csv"

    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    demographics_pre_post(mds_prepost_path, op_dates_path, "MDS-UPDRS III", save_path)
    demographics_pre_post(ledd_prepost_path, op_dates_path, "LEDD", save_path)
    demographics_pre_post(quest_prepost_path, op_dates_path, questionnaire, save_path)
    histoplot(quest, "TimeSinceSurgery", save_path)
    histoplot(quest, "TimeSinceDiag", save_path)
    histoplot(quest, "AGE_AT_OP", save_path)



def _load_ensemble_predictions(step_dir: str, target_name: str):
    """Aggregate member-level metrics to compute ensemble predictions."""
    ensemble_metrics_path = os.path.join(step_dir, f"{target_name}_metrics_ENSEMBLE.csv")
    if not os.path.exists(ensemble_metrics_path):
        raise FileNotFoundError(f"Ensemble metrics not found at {ensemble_metrics_path}")

    metrics_df = _prepare_dataframe(ensemble_metrics_path)
    row = metrics_df.iloc[0]
    r_val = row.get("r_ensemble")
    p_val = row.get("p_value_ensemble")
    rho_val = row.get("rho_ensemble")

    member_pattern = os.path.join(glob.escape(step_dir), "member*", f"{target_name}_metrics.csv")
    member_paths = sorted(glob.glob(member_pattern))
    if not member_paths:
        raise FileNotFoundError(
            f"No member metrics found in {step_dir}. "
            "Provide an ensemble predictions CSV or ensure member metrics are present."
        )

    member_y_tests = []
    member_y_preds = []
    for path in member_paths:
        metrics_df = _prepare_dataframe(path)
        row = metrics_df.iloc[0]
        try:
            y_test = np.asarray(ast.literal_eval(str(row["y_test"])), dtype=float)
            y_pred = np.asarray(ast.literal_eval(str(row["y_pred"])), dtype=float)
        except (ValueError, SyntaxError):
            y_test = np.fromstring(str(row["y_test"]).strip("[]"), sep=" ")
            y_pred = np.fromstring(str(row["y_pred"]).strip("[]"), sep=" ")
        member_y_tests.append(y_test)
        member_y_preds.append(y_pred)

    min_len = min(len(y) for y in member_y_tests)
    member_y_tests = [y[:min_len] for y in member_y_tests]
    member_y_preds = [y[:min_len] for y in member_y_preds]

    y_test_ref = member_y_tests[0]
    pred_stack = np.vstack(member_y_preds)
    pred_mean = np.mean(pred_stack, axis=0)

    if r_val is None or pd.isna(r_val) or p_val is None or pd.isna(p_val) or rho_val is None or pd.isna(rho_val):
        r_val, p_val = pearsonr(y_test_ref, pred_mean)
        rho_val, _ = spearmanr(y_test_ref, pred_mean)

    return (
        pd.DataFrame({"Actual": y_test_ref, "Predicted": pred_mean}),
        float(r_val),
        float(p_val),
        float(rho_val)
    )


def _load_inference_predictions(inference_dir: str):
    """Load ensemble predictions produced during inference."""
    ensemble_metrics_path = os.path.join(inference_dir, "ensemble_predictions_metrics_ENSEMBLE.csv")
    if not os.path.exists(ensemble_metrics_path):
        raise FileNotFoundError(f"Ensemble metrics not found at {ensemble_metrics_path}")

    metrics_df = _prepare_dataframe(ensemble_metrics_path)
    row = metrics_df.iloc[0]
    r_val = row.get("r_ensemble")
    p_val = row.get("p_value_ensemble")
    rho_val = row.get("rho_ensemble")

    member_pattern = os.path.join(glob.escape(inference_dir), "member*", "ensemble_predictions.csv")
    member_paths = sorted(glob.glob(member_pattern))
    if not member_paths:
        raise FileNotFoundError(
            f"No member metrics found in {inference_dir}. "
            "Provide an ensemble predictions CSV or ensure member metrics are present."
        )

    member_y_tests = []
    member_y_preds = []
    for path in member_paths:
        metrics_df = _prepare_dataframe(path)
        row = metrics_df.iloc[0]
        try:
            y_test = np.asarray(ast.literal_eval(str(row["y_test"])), dtype=float)
            y_pred = np.asarray(ast.literal_eval(str(row["y_pred"])), dtype=float)
        except (ValueError, SyntaxError):
            y_test = np.fromstring(str(row["y_test"]).strip("[]"), sep=" ")
            y_pred = np.fromstring(str(row["y_pred"]).strip("[]"), sep=" ")
        member_y_tests.append(y_test)
        member_y_preds.append(y_pred)

    min_len = min(len(y) for y in member_y_tests)
    member_y_tests = [y[:min_len] for y in member_y_tests]
    member_y_preds = [y[:min_len] for y in member_y_preds]

    y_test_ref = member_y_tests[0]
    pred_stack = np.vstack(member_y_preds)
    pred_mean = np.mean(pred_stack, axis=0)

    if r_val is None or pd.isna(r_val) or p_val is None or pd.isna(p_val) or rho_val is None or pd.isna(rho_val):
        r_val, p_val = pearsonr(y_test_ref, pred_mean)
        rho_val, _ = spearmanr(y_test_ref, pred_mean)

    return (
        pd.DataFrame({"Actual": y_test_ref, "Predicted": pred_mean}),
        float(r_val),
        float(p_val),
        float(rho_val)
    )


def regression_figures(
        ensemble_step_full: str,
        ensemble_step_best: str,
        inference_dirs: list,
        data_path: str,
        save_path: str,
        quest=None) -> None:
    """ Plot predicted vs. actual values """
    if inference_dirs is None:
        inference_dirs = []
    elif isinstance(inference_dirs, str):
        inference_dirs = [inference_dirs]
    quest_key = (quest or "").lower()
    if quest_key.startswith("bdi"):
        x_label = "Actual BDI Ratio"
        y_label = "Predicted BDI Ratio"
        linear_pre_col = "BDI_sum_pre"
        target_preferences = [("BDI_ratio", "BDI Ratio"), ("BDI_sum_post", "BDI Sum Post")]
    elif quest_key.startswith("moca"):
        linear_pre_col = "MoCA_sum_pre"
        if "sum_post" in quest_key:
            x_label = "Actual MoCA Sum Post"
            y_label = "Predicted MoCA Sum Post"
            target_preferences = [("MoCA_sum_post", "MoCA Sum Post"), ("MoCA_ratio", "MoCA Ratio")]
        else:
            x_label = "Actual MoCA Ratio"
            y_label = "Predicted MoCA Ratio"
            target_preferences = [("MoCA_ratio", "MoCA Ratio"), ("MoCA_sum_post", "MoCA Sum Post")]
    else:
        raise ValueError("Invalid questionnaire type. Choose 'BDI' or 'MoCA'.")
   
    def plot_regression(
            plot_df, 
            r, 
            p, 
            save_prefix, 
            title, 
            x_col,
            y_col,
            xlabel,
            ylabel,
            rho=None):
        # Set the context for the plot
        
        
        # Create a wider (landscape) figure
        fig = plt.figure(figsize=(8, 6))
        # Create a DataFrame for Seaborn
        colors = {
            "deterioration": "04E762",
            "improvement": "FF5714",
            "line": "grey",
            "scatter": "grey",
            "ideal_line": "black",
        }
        sns.set_theme(style="ticks", context="paper", rc={
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
        # Some matplotlib versions honor these visibility flags
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        })
        sns.scatterplot(
            x=x_col, 
            y=y_col, 
            data=plot_df, 
            alpha=0.7
        )
    

        # Plot a reference line with slope = 1
        #min_val = min(plot_df[x].min(), plot_df[y].min())
        #max_val = max(plot_df[x].max(), plot_df[y].max())
        #plt.plot([min_val, max_val], [min_val, max_val], color=colors["ideal_line"], alpha=0.5, linestyle='--')
        
        sns.regplot(
            x=x_col, 
            y=y_col, 
            data=plot_df, 
            scatter=False, 
            color=colors["line"], 
            line_kws={'label': 'Regression Line'}
        )
        # Plot confidence intervals
        ci = 95  # Confidence interval percentage
        sns.regplot(
            x=x_col, 
            y=y_col, 
            data=plot_df, 
            scatter=False, 
            color=colors["line"], 
            ci=ci, 
            line_kws={'label': f'{ci}% Confidence Interval'}
        )
        # Add text (R and p-value) in the top-left corner inside the plot
        # using axis coordinates (0–1 range) so it doesn't get cut off
        stat_lines = [f'R: {r:.2f}', f'P-value: {p:.6f}']
        if rho is not None:
            stat_lines.insert(1, f'Spearman ρ: {rho:.2f}')
        plt.text(
            0.05, 0.95, 
            "\n".join(stat_lines),
            fontsize=12, 
            transform=plt.gca().transAxes,  # use axis coordinates
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5)
        )
        N = len(plot_df)
        # Color the background: left of y=0 as improvement, right as deterioration
        ax = plt.gca()
        # get current y‐limits
        min_y, max_y = ax.get_ylim()

        # shade below y=0 (improvement)
        if min_y < 0 < max_y:
            ax.axhspan(min_y, 0,
                       color="#" + colors["improvement"],
                       alpha=0.08, zorder=0)
            # shade above y=0 (deterioration)
            ax.axhspan(0, max_y,
                       color="#" + colors["deterioration"],
                       alpha=0.08, zorder=0)

        # re‐apply limits so axes don’t auto‐expand
        pad = 0.05  # 5% padding
        combined_min = min(plot_df[x_col].min(), plot_df[y_col].min())
        combined_max = max(plot_df[x_col].max(), plot_df[y_col].max())

        # apply symmetric padding
        delta = (combined_max - combined_min)
        combined_min -= pad * delta
        combined_max += pad * delta

        ax.set_xlim(combined_min, combined_max)
        ax.set_ylim(combined_min, combined_max)

        # 2) Explicit minor locators (don’t rely on defaults)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 2 ⇒ one minor tick between majors
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # 3) Force minor tick visibility (some styles set these False)
        ax.tick_params(which='minor', bottom=True, top=False, left=True, right=False,
                       length=4, width=0.8, color='lightgrey')

        # labels/title, grid off, despine, etc.
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{title}  N={len(plot_df)}", fontsize=14)
        plt.grid(False)
        sns.despine()

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{title}.png')
        plt.savefig(f'{save_prefix}_{title}.svg')
        plt.close()
    
    def plot_linear_regression(data_path: str, title: str, save_prefix: str, xlabel: str = "BDI Pre Score", ylabel: str = "BDI Ratio"):
        # Read the BDI data CSV
        df = _prepare_dataframe(data_path)
        # Extract the y_test and y_pred arrays from the DataFrame
        if quest_key.startswith("bdi"):
            if "BDI_sum_post" not in df.columns and {"BDI_sum_pre", "BDI_diff"}.issubset(df.columns):
                df["BDI_sum_post"] = df["BDI_sum_pre"] + df["BDI_diff"]
        elif quest_key.startswith("moca"):
            if "MoCA_sum_post" not in df.columns and {"MoCA_sum_pre", "MoCA_diff"}.issubset(df.columns):
                df["MoCA_sum_post"] = df["MoCA_sum_pre"] + df["MoCA_diff"]

        if linear_pre_col not in df.columns:
            raise ValueError(f"Column '{linear_pre_col}' not found in dataframe for linear regression plot.")

        target_col = None
        target_label = ylabel
        for candidate, label in target_preferences:
            if candidate in df.columns:
                target_col = candidate
                target_label = label
                break
        if target_col is None:
            raise ValueError(
                f"None of the expected target columns {[c for c, _ in target_preferences]} were found in the dataframe."
            )

        plot_df = pd.DataFrame({
            "Pre": df[linear_pre_col],
            "Outcome": df[target_col]
        }).dropna()
        # Fit a linear regression model
        _, _, r, p, _ = linregress(plot_df["Pre"], plot_df["Outcome"])

        
        plot_regression(
            plot_df,
            r,
            p,
            save_prefix,
            title,
            "Pre",
            "Outcome",
            xlabel=xlabel,
            ylabel=target_label
        )

    # Ensemble full model (step with all features)
    full_plot_df, full_r, full_p, full_rho = _load_ensemble_predictions(ensemble_step_full, quest or "")
    plot_regression(
        full_plot_df,
        full_r,
        full_p,
        save_path,
        "Model Predicted vs. Actual - Full Ensemble",
        "Actual",
        "Predicted",
        x_label,
        y_label,
        rho=full_rho
    )

    # Best ensemble step across ablation
    if ensemble_step_best and ensemble_step_best != ensemble_step_full:
        best_plot_df, best_r, best_p, best_rho = _load_ensemble_predictions(ensemble_step_best, quest or "")
        plot_regression(
            best_plot_df,
            best_r,
            best_p,
            save_path,
            "Model Predicted vs. Actual - Best Ensemble",
            "Actual",
            "Predicted",
            x_label,
            y_label,
            rho=best_rho
        )

    # Inference runs (if available)
    for inference_dir in inference_dirs:
        try:
            result = _load_inference_predictions(inference_dir)
        except FileNotFoundError:
            continue
        if result is None:
            continue
        plot_df_inf, r_inf, p_inf, rho_inf = result
        label = os.path.basename(inference_dir)
        plot_regression(
            plot_df_inf,
            r_inf,
            p_inf,
            save_path,
            f"Inference Predicted vs. Actual - {label}",
            "Actual",
            "Predicted",
            x_label,
            y_label,
            rho=rho_inf
        )

    linear_ylabel = target_preferences[0][1]
    if quest_key.startswith("bdi"):
        linear_xlabel = "BDI Pre Score"
    else:
        linear_xlabel = "MoCA Pre Score"
    #plot_linear_regression(
    #    data_path,
    #    "Pre vs. Outcome",
    #    save_path,
    #    xlabel=linear_xlabel,
    #    ylabel=linear_ylabel
    #)


def threshold_figure(
        feature_name_mapping: dict,
        data_path: str,
        shap_data_path: str,
        removal_list_path: str,
        save_path: str) -> None:
    """
    Reads in the BDI data, SHAP values, and the list of removed features at each ablation step.
    1) Determines which column in the SHAP array corresponds to `feature_name`.
    2) Plots a histogram of the SHAP values for that feature, coloring bars by sign (negative vs. positive).
    3) Creates a simple linear‐SVM on the raw feature values to find a decision threshold for 
       predicting (BDI_diff >= 0). Plots a histogram of the feature values split by target class 
       and draws the SVM‐determined threshold as a vertical line.
    4) Saves the combined figure to `save_path`.
    """
    colors = {
            "deterioration": "#04E762",
            "improvement": "#FF5714",
            "det_edge": "#007C34",
            "imp_edge": "#8A3210",
            "line": "grey",
            "scatter": "grey",
            "ideal_line": "black",
        }
    # 1) Load inputs
    removals = pd.read_csv(removal_list_path)  # Assumes a single‐column CSV listing removed feature names
    df = _prepare_dataframe(data_path)

    shap_values_raw, shap_feature_names = _load_shap_data(shap_data_path)
    shap_values = _ensure_2d_shap(shap_values_raw)

    feature_df = df.drop(columns=["OP_DATUM"], errors="ignore")
    remaining_features = _resolve_remaining_features(feature_df, removals, shap_values, shap_feature_names)

    if shap_feature_names:
        feature_indices = [shap_feature_names.index(name) for name in remaining_features]
        shap_values = shap_values[:, feature_indices]

    #if feature_name not in remaining_features:
    #    raise ValueError(f"Feature '{feature_name}' was already removed in the ablation history; no SHAP values available.")
    for feature_name in remaining_features:

        feature = feature_df[feature_name].to_numpy()
        # The column index within shap_values for feature_name:
        shap_col_index = remaining_features.index(feature_name)
        shap_feature = shap_values[:, shap_col_index]
        # Convert shap_feature to a binary vector: 1 if SHAP ≥ 0, else 0
        target = (shap_feature >= 0).astype(int)

        X = feature.reshape(-1, 1)
        y = target

        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        svm = SVC(kernel="linear")
        grid_search = GridSearchCV(svm, param_grid, cv=5)
        grid_search.fit(X, y)
        svm_clf = grid_search.best_estimator_
        print(f"Best C: {grid_search.best_params_['C']} with accuracy: {grid_search.best_score_:.2f}")
        w = svm_clf.coef_[0][0]
        b = svm_clf.intercept_[0]
        threshold = -b / w  

        fig, ax = plt.subplots(figsize=(10, 7))

        # Split feature values by SHAP sign
        mask_neg = (shap_feature < 0)
        mask_pos = (shap_feature >= 0)

        feature_neg_shap = feature[mask_neg]
        feature_pos_shap = feature[mask_pos]


        # Common bins over the raw feature range
        fmin, fmax = feature.min(), feature.max()
        bins = np.linspace(fmin, fmax, 30)

        # Compute histogram counts manually
        counts_neg, _ = np.histogram(feature_neg_shap, bins=bins)
        counts_pos, _ = np.histogram(feature_pos_shap, bins=bins)

        # Compute error bars as Poisson (sqrt of counts)
        err_neg = np.sqrt(counts_neg)
        err_pos = np.sqrt(counts_pos)

        # Bar positions and width
        bar_positions = (bins[:-1] + bins[1:]) / 2
        bar_width = bins[1] - bins[0]

        # Plot negative‐SHAP bars
        ax.bar(
            bar_positions,
            counts_neg,
            #yerr=err_neg,
            #capsize=5,
            width=bar_width,
            color=colors["improvement"],
            alpha=0.7,
            edgecolor=colors["imp_edge"],
            linewidth=1.5,
            label="negative SHAPs"
        )
        # Plot positive‐SHAP bars
        ax.bar(
            bar_positions,
            counts_pos,
            #yerr=err_pos,
            #capsize=5,
            width=bar_width,
            color=colors["deterioration"],
            alpha=0.7,
            edgecolor=colors["det_edge"],
            linewidth=1.5,
            label="positive SHAPs"
        )

        # Draw vertical line at the SVM threshold
        ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
                   label=f"SVM threshold = {threshold:.3f}")

        feature_name_plot= feature_name_mapping[feature_name] if feature_name in feature_name_mapping else feature_name
        title = f"Histogram of {feature_name_plot} with SHAP values and threshold (Accuracy: {grid_search.best_score_:.2f})"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"{feature_name_plot}", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()

        plt.grid(False)
        sns.set_context("paper")
            # Optionally choose a style you like
        sns.despine()
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(f'{save_path}_{feature_name}.png', dpi=300)
        plt.savefig(f'{save_path}_{feature_name}.svg', dpi=300)
        plt.close(fig)


def shap_importance_histo_figure(
        feature_name_mapping: dict,
        data_path: str,
        shap_data_path: str,
        removal_list_path: str,
        save_path: str) -> None:
    """
    Plots a histogram of SHAP values for each feature in `feature_names`.
    """
    colors = {
            "deterioration": "#04E762",
            "improvement": "#FF5714",
            "line": "grey",
            "scatter": "grey",
            "ideal_line": "black",
        }
    # 1) Load inputs
    removals = pd.read_csv(removal_list_path)  # Assumes a single‐column CSV listing removed feature names
    
    df = _prepare_dataframe(data_path)
    shap_values_raw, shap_feature_names = _load_shap_data(shap_data_path)
    shap_values = _ensure_2d_shap(shap_values_raw)
    feature_df = df.drop(columns=["OP_DATUM"], errors="ignore")
    remaining_features = _resolve_remaining_features(feature_df, removals, shap_values, shap_feature_names)

    if shap_feature_names:
        feature_indices = [shap_feature_names.index(name) for name in remaining_features]
        shap_values = shap_values[:, feature_indices]

    abs_shaps = np.mean(np.abs(shap_values), axis=0)
    sorted_shap_indices = np.argsort(abs_shaps)[::-1]  # Sort indices by absolute SHAP values in descending order
    x_labels = [remaining_features[i] for i in sorted_shap_indices]
    x_labels = [feature_name_mapping.get(label, label) for label in x_labels]  # Map feature names if available
    x_values = abs_shaps[sorted_shap_indices]

    sorted_shap_matrix = np.abs(shap_values[:, sorted_shap_indices])

    # Create list to hold p-values from paired t-tests between consecutive features
    p_values = []

    # Perform paired t-tests between each consecutive pair of SHAP columns
    for i in range(1, sorted_shap_matrix.shape[1]):
        col_prev = sorted_shap_matrix[:, i - 1]
        col_curr = sorted_shap_matrix[:, i]

        t_stat, p_val = ttest_rel(col_prev, col_curr)
        p_values.append(p_val)

    #n_tests = len(p_values)
    corrected_p_values = false_discovery_control(p_values, axis=0, method='bh')

    print(f"p-values for consecutive SHAP columns: {corrected_p_values}")
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_positions = np.arange(len(x_values))
    sem_values = np.std(sorted_shap_matrix, axis=0) / np.sqrt(sorted_shap_matrix.shape[0])
    print(f"SEM values for SHAP columns: {sem_values}")
    # 1) Plot a simple bar chart of mean |SHAP| per feature
    ax.bar(
        x=bar_positions,
        height=x_values,
        yerr=sem_values,
        capsize=5,
        color="#D37B40",
        alpha=0.7,
        edgecolor="#7A431D",
        linewidth=1.5,
    )

    # 2) Labeling
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.set_title("Mean Absolute SHAP Value by Feature", fontsize=14)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Mean SHAP Value", fontsize=12)

    # Add significance annotations
    def significance_marker(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    bar_heights = x_values
    y_offset = max(x_values) * 0.05  # space above bar

    for i, p_val in enumerate(corrected_p_values):
        mark = significance_marker(p_val)
        if mark:
            x1, x2 = bar_positions[i], bar_positions[i + 1]
            y = max(bar_heights[i], bar_heights[i + 1]) + y_offset
            h = y_offset * 0.5

            # Draw bracket
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
            ax.text((x1 + x2) / 2, y + h + 0.005, mark, ha='center', va='bottom', fontsize=12)
    # 3) Styling (match threshold_figure style)
    plt.grid(False)
    sns.despine()
    sns.set_context("paper")
    plt.tight_layout()

    # 4) Save and close
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".svg", dpi=300)
    plt.close(fig)

def shap_interaction_figure(
        feature_name_mapping: dict,
        data_path: str,
        shap_data_path: str,
        removal_list_path: str,
        feature1: str,
        feature2: str,
        save_path: str) -> None:
    """    Plots a scatter plot of SHAP values for two features, coloring points by the sign of the SHAP interaction.
    """
    removals = pd.read_csv(removal_list_path)  # Assumes a single‐column CSV listing removed feature names
    df = _prepare_dataframe(data_path)

    shap_values_raw, shap_feature_names = _load_shap_data(shap_data_path)      # shape = (n_samples, n_features_remaining)
    shap_values = _ensure_2d_shap(shap_values_raw)

    feature_df = df.drop(columns=["OP_DATUM"], errors="ignore")
    remaining_features = _resolve_remaining_features(feature_df, removals, shap_values, shap_feature_names)

    if shap_feature_names:
        feature_indices = [shap_feature_names.index(name) for name in remaining_features]
        shap_values = shap_values[:, feature_indices]

    if feature1 not in remaining_features or feature2 not in remaining_features:
        raise ValueError(f"Features '{feature1}' or '{feature2}' were already removed in the ablation history; no SHAP values available.")
    
    shap_matrix = shap_values

    feature1_name = feature_name_mapping.get(feature1, feature1)
    feature2_name = feature_name_mapping.get(feature2, feature2)
    all_feature_names = [
        feature_name_mapping.get(name, name) for name in remaining_features
    ]
    feature_matrix = feature_df[remaining_features].to_numpy()
    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_matrix,
        features=feature_matrix,
        feature_names=all_feature_names,
        show=False)
    shap.dependence_plot(
        feature1,
        shap_matrix,
        feature_matrix,
        feature_names=remaining_features,
        interaction_index=feature2,
        show=False,)
    #shap.initjs()
    #shap.plots.scatter(
    #    shap_values[:, feature1_index],
    #    shap_values[:, feature2_index],)
    plt.title(f'Dependence of {feature1_name} and {feature2_name}', fontsize=16)
    plt.grid(False)
    sns.set_context("paper")
        # Optionally choose a style you like
    sns.despine()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f'{save_path}_{feature1_name}x{feature2_name}.png', dpi=300)
    plt.savefig(f'{save_path}_{feature1_name}x{feature2_name}.svg', dpi=300)
    plt.close(fig)

    
def ablation_plot(
        questionnaire: str,
        ablation_folder_path: str,
        save_path: str,
        title: str = "Pearson-R Scores Over Feature Ablation") -> None:
    history_path = os.path.join(ablation_folder_path, f"{questionnaire}_ablation_history.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Ablation history file not found at {history_path}")
    custom_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#525252"]
    sns.set_theme(style="whitegrid", context="paper")
                # Set Seaborn style, context, and custom palette
    r2_scores = []
    x = []
    step_dirs = []
    for entry in os.listdir(ablation_folder_path):
        if entry.startswith("ablation_step[") and entry.endswith("]"):
            try:
                idx = int(entry[len("ablation_step["):-1])
            except ValueError:
                continue
            step_dirs.append((idx, entry))
    step_dirs.sort()

    metric_candidates = [
        "{prefix}_ratio_metrics.csv",
        "{prefix}_metrics.csv",
        "{prefix}_metrics_ENSEMBLE.csv",
    ]
    score_columns = ["r2", "r", "r_ensemble"]

    for idx, folder in step_dirs:
        folder_path = os.path.join(ablation_folder_path, folder)
        metrics_path = None
        for candidate in metric_candidates:
            candidate_path = os.path.join(folder_path, candidate.format(prefix=questionnaire))
            if os.path.exists(candidate_path):
                metrics_path = candidate_path
                break
        if metrics_path is None:
            continue
        df = _prepare_dataframe(metrics_path)
        score = None
        for col in score_columns:
            if col in df.columns:
                score = df[col].iloc[0]
                break
        if score is None:
            raise ValueError(
                f"None of the expected score columns {score_columns} were found in {metrics_path}"
            )
        r2_scores.append(score)
        x.append(idx)

    if not r2_scores:
        raise ValueError(f"No ablation metrics found in {ablation_folder_path}")

    sns.set_palette(custom_palette)
            # Read in the CS
    # Create a figure
    plt.figure(figsize=(8, 4))
            # Create a figure
            # Plot each model's R² scores in a loop, using sample_sizes on the x-axis
    #for model_name, r2_scores in results.items():
    plot_df = pd.DataFrame({'x': x, 'r2s': r2_scores})
    sns.lineplot(data=plot_df, x='x', y='r2s', label="R Score", marker='o')
    # Label the axes and set the title
    plt.xlabel("Number of removed features")
    plt.ylabel("Pearson-R Score")
    plt.title(title)
    plt.legend()
    

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path+ ".png", dpi=300)
    plt.savefig(save_path+ ".svg", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_name = "1_MoCA_sum_post_ledd"
    results_dir = os.path.join(repo_root, "results", run_name, "level2", "ElasticNet")
    data_dir = os.path.join(repo_root, "data", "MoCA", "level2")

    figures_root = os.path.join(repo_root, "figures", "MoCA_sum_post_ledd")
    demographics_dir = os.path.join(figures_root, "demographics")
    regression_dir = os.path.join(figures_root, "regression")
    threshold_dir = os.path.join(figures_root, "threshold")
    shap_dir = os.path.join(figures_root, "shap")
    ablation_fig_dir = os.path.join(figures_root, "ablation")

    for directory in [figures_root, demographics_dir, regression_dir, threshold_dir, shap_dir, ablation_fig_dir]:
        os.makedirs(directory, exist_ok=True)

    feature_name_mapping = {
        "TimeSinceSurgery": "Time Since Surgery",
        "AGE_AT_OP": "Age at Operation",
        "TimeSinceDiag": "Time Since Diagnosis",
        "MoCA_Executive_sum_pre": "MoCA Executive (Pre)",
        "MoCA_Erinnerung_sum_pre": "MoCA Memory (Pre)",
        "MoCA_Sprache_sum_pre": "MoCA Language (Pre)",
        "MoCA_Aufmerksamkeit_sum_pre": "MoCA Attention (Pre)",
        "MoCA_Abstraktion_sum_pre": "MoCA Abstraction (Pre)",
        "UPDRS_reduc": "UPDRS Reduction",
        "UPDRS_on": "UPDRS On",
        "LEDD_pre": "LEDD Pre",
        }

    shap_data_path = os.path.join(results_dir, "MoCA_sum_post_all_shap_values(mu).csv")
    removal_list_path = os.path.join(results_dir, "ablation", "MoCA_sum_post_ENSEMBLE_ablation_history.csv")
    _, shap_feature_names = _load_shap_data(shap_data_path)
    if shap_feature_names:
        feature_name_mapping.setdefault(
            "all",
            [feature_name_mapping.get(name, name) for name in shap_feature_names]
        )
    else:
        feature_name_mapping.setdefault("all", list(feature_name_mapping.values()))

    visualize_demographics(
        data_dir,
        "MoCA",
        save_path=demographics_dir + "/"
    )

    ablation_dir = os.path.join(results_dir, "ablation")
    # Set these paths manually to the ablation steps you want to visualise.
    # Example: os.path.join(ablation_dir, "ablation_step[5]")
    full_step_dir = os.path.join(ablation_dir, "ablation_step[1]")
    best_step_dir = os.path.join(ablation_dir, "ablation_step[8]")

    if not os.path.exists(full_step_dir):
        raise FileNotFoundError(
            "Expected full ensemble step directory not found. "
            "Please update 'full_step_dir' in MoCA_plotting.py to point to the desired ablation step."
        )
    if not os.path.exists(best_step_dir):
        raise FileNotFoundError(
            "Expected best ensemble step directory not found. "
            "Please update 'best_step_dir' in MoCA_plotting.py to point to the desired ablation step."
        )

    inference_dirs = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/results/1_MoCA_sum_post_updrs/level2/ElasticNet/inference_ppmi_ledd"

    regression_figures(
        ensemble_step_full=full_step_dir,
        ensemble_step_best=best_step_dir,
        inference_dirs=inference_dirs,
        data_path=os.path.join(data_dir, "moca_demo.csv"),
        save_path=os.path.join(regression_dir, "moca_sum_post"),
        quest="MoCA_sum_post"
    )

    threshold_figure(
        feature_name_mapping,
        data_path=os.path.join(data_dir, "moca_demo.csv"),
        shap_data_path=shap_data_path,
        removal_list_path=removal_list_path,
        save_path=os.path.join(threshold_dir, "moca_sum_post_threshold")
    )

    shap_importance_histo_figure(
        feature_name_mapping,
        data_path=os.path.join(data_dir, "moca_demo.csv"),
        shap_data_path=shap_data_path,
        removal_list_path=removal_list_path,
        save_path=os.path.join(shap_dir, "moca_sum_post_importance")
    )

    shap_interaction_figure(
        feature_name_mapping,
        data_path=os.path.join(data_dir, "moca_demo.csv"),
        shap_data_path=shap_data_path,
        removal_list_path=removal_list_path,
        feature1="TimeSinceSurgery",
        feature2="AGE_AT_OP",
        save_path=os.path.join(shap_dir, "moca_sum_post_dependence")
    )

    ablation_plot(
        questionnaire="MoCA_sum_post_ENSEMBLE",
        ablation_folder_path=os.path.join(results_dir, "ablation"),
        save_path=os.path.join(ablation_fig_dir, "moca_sum_post_ablation")
    )
