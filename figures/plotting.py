import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D 
from matplotlib.collections import PolyCollection
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



def raincloud_plot(data: pd.DataFrame, modality_name: str, features_list: list, safe_path: str = "") -> None:
    """Create a refined raincloud plot using seaborn for better aesthetics."""
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Prepare data
    data_list = [data[col].dropna().values for col in data.columns]
    x_positions = np.arange(1, len(data_list) + 1)
    
    # Define colors based on modality

    boxplots_colors = sns.color_palette("deep")
    violin_colors = sns.color_palette("deep")
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
            color=violin_colors[idx],
            ax=ax,
            width=violin_width,
            #split=False
        )
        for collection in ax.collections[-1:]:
            if not isinstance(collection, PolyCollection):
                continue  # skip non-violin objects
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
    bp = sns.boxplot(
        data=data,
        width=box_width,
        showcaps=True,
        whiskerprops={'color': 'black'},
        medianprops={'color': 'black'},
        flierprops={'marker': ''},  # Do not mark outliers
        ax=ax
    )

    for patch, color in zip(bp.artists, boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
   
    
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
                        line_color = "red"
                    elif slope < 0:
                        line_color = "green"
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
                        line_color = "green"
                    elif slope < 0:
                        line_color = "red"
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
            Line2D([0], [0], color=line_colors["increase"], lw=2, label="Deterioration"),
            Line2D([0], [0], color=line_colors["decrease"], lw=2, label="Improvement"),
            Line2D([0], [0], color=line_colors["no change"], lw=2, label="No change"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

    # Add legend for MoCA
    elif modality_name == "MoCA":
        line_colors = {"increase": "green", "decrease": "red", "no change": "grey"}
        legend_elements = [
            Line2D([0], [0], color=line_colors["increase"], lw=2, label="Improvement"),
            Line2D([0], [0], color=line_colors["decrease"], lw=2, label="Deterioration"),
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
    #plt.show()
    plt.close()
    



def demographics_pre_post(modality_path: str, model_data_path: str, modality_name:str,  save_path:str) -> None:
    """ Plot the demographic data before and after the treatment as raincloud plot"""
    model_df = pd.read_csv(model_data_path)
    # Load the data
    data = pd.read_csv(modality_path)
    # Create a figure with two subplots
    
    # Filter data based on OP_DATUM in model_df
    data = data[data['OP_DATUM'].isin(model_df['OP_DATUM'])]

    if modality_name == "BDI":
        data = data.drop(columns=['BDI_diff'])

    if modality_name == "MoCA":
        data = data.drop(columns=['MoCA_diff'])  

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
        raincloud_plot(data, modality_name , ['Pre OFF', 'Pre ON', 'Post ON/Stim On'], save_path)
    else:
        raincloud_plot(data, modality_name , ['Pre', 'Post'], save_path)
    
    
def histoplot(input_path: str, feature: str,  save_path: str) -> None:
    """ Plot the demographic data before and after the treatment as raincloud plot"""
    # Load the data
    data = pd.read_csv(input_path)
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
    #plt.show()
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



def regression_figures(
        metrics_path_best: str, 
        metrics_path_full: str , 
        data_path: str,
        save_path: str,
        quest=None) -> None:
    """ Plot predicted vs. actual values """
    if quest == "BDI":
        x_label = "Actual BDI Ratio"
        y_label = "Predicted BDI Ratio"
        
    elif quest == "MoCA":
        x_label = "Actual MoCA Ratio"
        y_label = "Predicted MoCA Ratio"
    else:
        raise ValueError("Invalid questionnaire type. Choose 'BDI' or 'MoCA'.")
   
    def plot_regression(
            plot_df, 
            r, 
            p, 
            save_path, 
            title, 
            xlabel=x_label,
            ylabel = y_label,
            type="model"):
        # Set the context for the plot
        
        
        # Create a wider (landscape) figure
        fig = plt.figure(figsize=(10, 6))
        # Create a DataFrame for Seaborn
        colors = {
            "deterioration": "04E762",
            "improvement": "FF5714",
            "line": "grey",
            "scatter": "grey",
            "ideal_line": "black",
        }
        # Fit a regression line
        if type == "model":
            x,y = 'Actual', 'Predicted'
        elif type == "linear":
            x,y = 'Pre', 'Ratio'
        else:
            raise ValueError("Invalid type. Choose 'model' or 'linear'.")
        # Read your metrics CSV:
        sns.scatterplot(
            x=x, 
            y=y, 
            data=plot_df, 
            alpha=0.7
        )
    

        # Plot a reference line with slope = 1
        #min_val = min(plot_df[x].min(), plot_df[y].min())
        #max_val = max(plot_df[x].max(), plot_df[y].max())
        #plt.plot([min_val, max_val], [min_val, max_val], color=colors["ideal_line"], alpha=0.5, linestyle='--')
        
        sns.regplot(
            x=x, 
            y=y, 
            data=plot_df, 
            scatter=False, 
            color=colors["line"], 
            line_kws={'label': 'Regression Line'}
        )
        # Plot confidence intervals
        ci = 95  # Confidence interval percentage
        sns.regplot(
            x=x, 
            y=y, 
            data=plot_df, 
            scatter=False, 
            color=colors["line"], 
            ci=ci, 
            line_kws={'label': f'{ci}% Confidence Interval'}
        )
        # Add text (R and p-value) in the top-left corner inside the plot
        # using axis coordinates (0–1 range) so it doesn't get cut off
        plt.text(
            0.05, 0.95, 
            f'R: {r:.2f}\nP-value: {p:.6f}', 
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
        ax.axhspan(min_y, 0,
                   color="#" + colors["improvement"],
                   alpha=0.08, zorder=0)
        # shade above y=0 (deterioration)
        ax.axhspan(0, max_y,
                   color="#" + colors["deterioration"],
                   alpha=0.08, zorder=0)

        # re‐apply limits so axes don’t auto‐expand
        
        pad = 0.05  # 5% padding
        x_min, x_max = plot_df[x].min(), plot_df[x].max()
        y_min, y_max = plot_df[y].min(), plot_df[y].max()
        ax.set_xlim(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))
        ax.set_ylim(y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min))
        #if type == "model":
        #    ax.set_ylim(y_min, y_max)
        #    ax.set_xlim(x_min, x_max)
        # Label axes and set title
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title + "  N=" + str(N), fontsize=14)
        # Show grid and ensure everything fits nicely
        plt.grid(False)
        sns.set_context("paper")
        # Optionally choose a style you like
        sns.despine()
        plt.tight_layout()
        # Save and close
        plt.savefig(f'{save_path}_{title}.png')
        plt.savefig(f'{save_path}_{title}.svg')
        plt.close()
        
    
    def plot_model(metrics_path: str, title: str):
        # Read the metrics CSV
        metrics_df = pd.read_csv(metrics_path)
        # Read the BDI data CSV
        df = pd.read_csv(data_path)
        # Extract the best model's metrics
        best_model = metrics_df.iloc[0]
        # Extract the y_test and y_pred arrays from the DataFrame
        y_test_str = best_model["y_test"]
        y_pred_str = best_model["y_pred"]
        # Parse the string into a Python list of floats
        try:
            y_test = ast.literal_eval(y_test_str)
            y_pred = ast.literal_eval(y_pred_str)
        except (ValueError, SyntaxError):
            # fallback: strip brackets and split on whitespace
            y_test = list(map(float, y_test_str.strip("[]").split()))
            y_pred = list(map(float, y_pred_str.strip("[]").split()))
        N = len(y_test)
        # Build your plotting DataFrame
        plot_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })
        r, p = best_model["r2"], best_model["p_value"]
        plot_regression(plot_df, r, p, save_path, title)
    
    def plot_linear_regression(data_path: str, title: str, xlabel: str = "BDI Pre Score", ylabel: str = "BDI Ratio"):
        # Read the BDI data CSV
        df = pd.read_csv(data_path)
        # Extract the y_test and y_pred arrays from the DataFrame
        if quest == "BDI":
            plot_df = pd.DataFrame({
                "Pre": df["BDI_sum_pre"],
                "Ratio": df["BDI_ratio"]
            })
        elif quest == "MoCA":
            plot_df = pd.DataFrame({
                "Pre": df["MoCA_sum_pre"],
                "Ratio": df["MoCA_ratio"]
            })
        else:
            raise ValueError("Invalid questionnaire type. Choose 'BDI' or 'MoCA'.")
        # Fit a linear regression model
        _, _, r, p, std_err = linregress(plot_df["Pre"], plot_df["Ratio"])

        
        plot_regression(plot_df, r, p, save_path, title, xlabel=xlabel, ylabel=ylabel, type="linear")

        
    plot_model(metrics_path_full, "Model Predicted vs. Actual - Full Model")
    plot_model(metrics_path_best, "Model Predicted vs. Actual - Best Model")
    #plot_linear_regression(data_path, "Pre vs. Ratio")


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
    bdi_df = pd.read_csv(data_path)

    shap_values = np.load(shap_data_path)      # shape = (n_samples, n_features_remaining)
    

    # Drop unnecessary columns from bdi_df
    bdi_df = bdi_df.drop(columns=["BDI_diff", "BDI_ratio", "Pat_ID"], errors="ignore")
    original_features = [col for col in bdi_df.columns if col != "BDI_diff"]
    n_original = len(original_features)
    n_shap_cols = shap_values.shape[1]
    n_removed_before = n_original - n_shap_cols
    
    if n_removed_before < 0 or n_removed_before > len(removals):
        raise ValueError(
            f"Calculated removed_count = {n_removed_before} is invalid. "
            f"Check that shap_values and removal_list correspond."
        )

    # Take exactly the first n_removed_before entries from the removal history
    removed_up_to_now = removals.iloc[:n_removed_before, 0].astype(str).tolist()

    # Form the list of features that remain at the time SHAP values were computed
    remaining_features = [f for f in original_features if f not in removed_up_to_now]

    #if feature_name not in remaining_features:
    #    raise ValueError(f"Feature '{feature_name}' was already removed in the ablation history; no SHAP values available.")
    for feature_name in remaining_features:

        feature = bdi_df[feature_name].values
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

        fig, ax = plt.subplots(figsize=(10, 6))

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
    
    bdi_df = pd.read_csv(data_path)
    all_shap_values = np.load(shap_data_path)
    foldwise_abs_shaps = np.mean(np.abs(all_shap_values), axis=1)       # shape = (n_samples, n_features_remaining)
    abs_shaps = np.mean(np.abs(np.mean(all_shap_values, axis=0)), axis=0)  # Mean absolute SHAP values across all samples
    # Drop unnecessary columns from bdi_df
    bdi_df = bdi_df.drop(columns=["BDI_diff", "BDI_ratio", "Pat_ID"], errors="ignore")
    original_features = [col for col in bdi_df.columns if col != "BDI_diff"]
    n_original = len(original_features)
    n_shap_cols = abs_shaps.shape[0]
    n_removed_before = n_original - n_shap_cols

    if n_removed_before < 0 or n_removed_before > len(removals):
        raise ValueError(
            f"Calculated removed_count = {n_removed_before} is invalid. "
            f"Check that shap_values and removal_list correspond."
        )

    # Take exactly the first n_removed_before entries from the removal history
    removed_up_to_now = removals.iloc[:n_removed_before, 0].astype(str).tolist()

    # Form the list of features that remain at the time SHAP values were computed
    remaining_features = [f for f in original_features if f not in removed_up_to_now]

    # The column index within shap_values for feature_name:
    sorted_shap_indices = np.argsort(abs_shaps)[::-1]  # Sort indices by absolute SHAP values in descending order
    x_labels = [remaining_features[i] for i in sorted_shap_indices]
    x_labels = [feature_name_mapping.get(label, label) for label in x_labels]  # Map feature names if available
    x_values = abs_shaps[sorted_shap_indices]
    
    sorted_shap_matrix = foldwise_abs_shaps[:, sorted_shap_indices]

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


if __name__ == "__main__":
    root_dir = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    #root_dir = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    #root_dir = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction"
    #/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/results/level2/level2/NGBoost/BDI_ablation_history.csv
    #visualize_demographics("BDI", root_dir)
    metrics_path_best = root_dir + "/results/Paper_runs/MoCA/level2/NGBoost/ablation/ablation_step[9]/MoCA_ratio_metrics.csv"
    metrics_path_full = root_dir + "/results/Paper_runs/MoCA/level2/NGBoost/MoCA_ratio_metrics.csv"
    moca_data_path = root_dir + "/data/MoCA/level2/moca_df.csv"
    moca_data_folder_path = root_dir + "/data/MoCA/level2"
    original_features = ['MoCA_sum_pre', 'AGE_AT_OP', 'TimeSinceDiag', 'X_L', 'Y_L', 'Z_L', 'X_R',
       'Y_R', 'Z_R', 'Left_1_mA', 'Right_1_mA', 'LEDD_ratio']
    
    feature_names_for_plotting = ['MoCA Sum Pre', 'Age at Operation', 'Time passed Since Diagnosis', 'X Left', 'Y Left', 'Z Left', 'X Right',
       'Y Right', 'Z Right', 'Left mA', 'Right mA', 'LEDD Reduction Ratio']
    
    feature_name_mapping = dict(zip(original_features, feature_names_for_plotting))

    #visualize_demographics("MoCA", root_dir)
    visualize_demographics(
        moca_data_folder_path, 
        "MoCA", 
        save_path = root_dir + "/figures/MoCA/")
    
    regression_figures(
        metrics_path_best,
        metrics_path_full,
        moca_data_path,
        save_path = root_dir + "/figures/MoCA/",
        quest="MoCA")
    
    #threshold_figure(
    #    feature_name_mapping,
    #    data_path=moca_data_path,
    #    shap_data_path=root_dir + "/results/Paper_runs/MoCA/level2/NGBoost/ablation/ablation_step[9]/MoCA_ratio_all_shap_values(mu).npy",
    #    removal_list_path=root_dir + "/results/Paper_runs/MoCA/level2/NGBoost/ablation/MoCA_ablation_history.csv",
    #    save_path=root_dir + "/figures/MoCA/moca_threshold_figure"
    #)
#
    #shap_importance_histo_figure(
    #    feature_name_mapping,
    #    data_path=moca_data_path,
    #    shap_data_path=root_dir +  "/results/Paper_runs/MoCA/level2/NGBoost/ablation/ablation_step[9]/MoCA_ratio_all_shap_values(mu).npy",
    #    removal_list_path=root_dir + "/results/Paper_runs/MoCA/level2/NGBoost/ablation/MoCA_ablation_history.csv",
    #    save_path=root_dir + "/figures/MoCA/moca_abs_importance_figure"
    #)