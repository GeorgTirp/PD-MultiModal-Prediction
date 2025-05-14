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
    plt.savefig(safe_path + modality_name + "_raincloud_plot.png")
    plt.show()
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
    
    
def histoplot(input_path: str , save_path: str) -> None:
    """ Plot the demographic data before and after the treatment as raincloud plot"""
    # Load the data
    data = pd.read_csv(input_path)
    sns.set_theme(style="white", context="paper")
    sns.set_palette("deep")
    # Create a figure with two subplots
    plt.figure(figsize=(10, 6))
    print(data['TimeSinceSurgery'].mean())
    print(data['TimeSinceSurgery'].std())
    sns.histplot(data['TimeSinceSurgery'], kde=True, bins=30, color='blue')
    plt.title('Distribution of Time Since Surgery', fontsize=16)
    plt.xlabel('Time Since Surgery (years)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(save_path + "time_since_surgery_histoplot.png")
    plt.show()
    plt.close()



def visualize_demographics(questionnaire, root_dir):
    # Example usage
    root_dir = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    #root_dir = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction"
    data_path = root_dir + "/data/" + questionnaire 
    stim_path = data_path + "/stim_positions.csv"
    save_path = root_dir + "/results/data_analysis/" + questionnaire + "/"
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
    histoplot(quest, save_path)

    # Load data
    #data = pd.read_csv(data_path)
    #X = data.drop(columns=['BDI_ratio','Pat_ID'])
    #stim_positions = pd.read_csv(stim_path).drop(columns=['OP_DATUM'])
    # Perform PCA
    #pca_result = pca(X, n_components=10, safe_path=save_path)
    #print("PCA safed into {}".format(save_path))
    #
    # Calculate and plot covariance matrix
    #cov_matrix = covariance_matrix(X, safe_path=save_path)
    #print("Covariance Matrix safed into {}".format(save_path))
    
    
    # Plot stimulation positions
    #plot_stim_positions(stim_positions, safe_path=save_path)
    #print("Stimulation positions plotted and safed into {}".format(save_path))


def regression_plot(metrics_path: str, save_path: str) -> None:
    """ Plot predicted vs. actual values """
    # Use a context suitable for publication-quality figures
    sns.set_context("paper")
    # Optionally choose a style you like
    sns.set_style("whitegrid")
    # Create a wider (landscape) figure
    plt.figure(figsize=(10, 6))
    # Create a DataFrame for Seaborn
    colors = {
        "deterioration": "04E762",
        "improvement": "FF5714",
        "line": "grey",
        "scatter": "grey",
        "ideal_line": "black",
    }
    # Read your metrics CSV:
    metrics_df = pd.read_csv(metrics_path)

    # Suppose the first row has your arrays:
    y_test_str = metrics_df.loc[0, "y_test"]
    y_pred_str = metrics_df.loc[0, "y_pred"]

    # 1) Parse the string into a Python list of floats
    #    ast.literal_eval is safest if they’re valid Python lists: "[1.0, 2.0, 3.0]"
    #    If your arrays omit commas (e.g. "[1.0 2.0 3.0]"), you can do a manual split instead:
    try:
        y_test = ast.literal_eval(y_test_str)
        y_pred = ast.literal_eval(y_pred_str)
    except (ValueError, SyntaxError):
        # fallback: strip brackets and split on whitespace
        y_test = list(map(float, y_test_str.strip("[]").split()))
        y_pred = list(map(float, y_pred_str.strip("[]").split()))
    N = len(y_test)
    # 2) Build your plotting DataFrame
    plot_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    r, p = metrics_df["r2"].iloc[0], metrics_df["p_value"].iloc[0]
    # Scatter plot only (no regression line)
    sns.scatterplot(
        x='Actual', 
        y='Predicted', 
        data=plot_df, 
        alpha=0.7
    )
   
    
    # Plot a reference line with slope = 1
    min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
    max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color=colors["ideal_line"], alpha=0.5, linestyle='--')
    # Fit a regression line
    sns.regplot(
        x='Actual', 
        y='Predicted', 
        data=plot_df, 
        scatter=False, 
        color=colors["line"], 
        line_kws={'label': 'Regression Line'}
    )
    # Plot confidence intervals
    ci = 95  # Confidence interval percentage
    sns.regplot(
        x='Actual', 
        y='Predicted', 
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
    # Color the background: left of y=0 as improvement, right as deterioration
    ax = plt.gca()
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
    ax.set_ylim(min_y, max_y)
    # Label axes and set title
    plt.xlabel(f'Actual BDI ratio', fontsize=12)
    plt.ylabel(f'Predicted BDI ratio', fontsize=12)
    plt.title("Predictions vs. Actual BDI Ratio" + "  N=" + str(N), fontsize=14)
    # Show grid and ensure everything fits nicely
    plt.grid(True)
    plt.tight_layout()
    # Save and close
    plt.savefig(f'{save_path}BDI_ratio_actual_vs_predicted.png')
    plt.close()
    
if __name__ == "__main__":
    #root_dir = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction"
    root_dir = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    #visualize_demographics("BDI", root_dir)
    regression_plot(metrics_path = root_dir + "/results/level3/NGBoost/BDI_metrics.csv", save_path = root_dir + "/figures/")