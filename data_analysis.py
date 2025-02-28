import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D 

def pca(X: pd.DataFrame, n_components: int, safe_path: str = "") -> pd.DataFrame:
    """ Perform PCA on the data"""
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.title('Explained Variance Ratio by Principal Components')
    plt.savefig(safe_path + 'explained_variance_ratio.png')
    plt.close()
    return principalDf


def covariance_matrix(X: pd.DataFrame, safe_path: str = "") -> pd.DataFrame:
    """ Calculate the covariance matrix of the data and visualize it as a heatmap"""
    cov_matrix = X.cov()
    plt.figure(figsize=(14, 12), dpi=300)  # Increase figure size and resolution
    plt.title('Covariance Matrix Heatmap', fontsize=16)
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='viridis', annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(safe_path + "covariance_matrix.png", dpi=300)  # Save with higher resolution
    plt.close()
    return cov_matrix

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

def raincloud_plot(data: pd.DataFrame, modality_name: str, features_list: list, 
                   safe_path: str = "", violin_positions: list = None) -> None:
    """Create a refined raincloud plot with customizable violin positions."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Prepare data
    data_list = [data[col].dropna().values for col in data.columns]
    x_positions = np.arange(1, len(data_list) + 1)
    
    # Default violin positions to left if not provided
    if violin_positions is None:
        violin_positions = ['left'] * len(data.columns)
    
    # Define colors based on modality
    if modality_name == "UPDRS":
        boxplots_colors = ['yellowgreen', 'olivedrab', 'darkolivegreen']
        violin_colors = ['thistle', 'orchid', 'purple']
        scatter_colors = ['tomato', 'darksalmon', 'firebrick']
    else:
        boxplots_colors = ['yellowgreen', 'olivedrab']
        violin_colors = ['thistle', 'orchid']
        scatter_colors = ['tomato', 'darksalmon']
    
    # Violin plot (half-violin) aligned to specified side
    for idx, (x, values) in enumerate(zip(x_positions, data_list)):
        pos = violin_positions[idx]
        hue = np.zeros(len(values))  # Dummy hue for split
        
        # Adjust x position and hue order based on desired side
        if pos == 'left':
            adjusted_x = x - 0.2
            hue_order = [0, 1]
        else:
            adjusted_x = x + 0.2
            hue_order = [1, 0]
        
        x_values = np.full(len(values), adjusted_x)
        sns.violinplot(
            x=x_values,
            y=values,
            hue=hue,
            hue_order=hue_order,
            split=True,
            inner=None,
            cut=0,
            linewidth=0,
            color=violin_colors[idx],
            ax=ax,
            width=0.6,
            dodge=False,
            palette={0: violin_colors[idx]},
            legend=False
        )
    
    # Boxplot (slimmer)
    bp = sns.boxplot(
        data=data,
        width=0.2,
        showcaps=True,
        whiskerprops={'color': 'black'},
        medianprops={'color': 'black'},
        flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red', 'alpha': 0.5},
        ax=ax
    )
    for patch, color in zip(bp.artists, boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    # Scatter plot
    for idx, (x, values) in enumerate(zip(x_positions, data_list)):
        x_jitter = np.random.uniform(low=-0.05, high=0.05, size=len(values)) + x
        ax.scatter(x_jitter, values, s=10, color=scatter_colors[idx], alpha=0.7)
    
    # Labels and aesthetics
    ax.set_xticks(x_positions)
    ax.set_xticklabels(features_list)
    ax.set_ylabel(modality_name)
    ax.set_title(f"{modality_name} Raincloud Plot")
    
    # BDI legend
    if modality_name == "BDI":
        line_colors = {"increase": "blue", "decrease": "red", "no change": "grey"}
        legend_elements = [
            Line2D([0], [0], color=line_colors["increase"], lw=2, label="Increase (Post > Pre)"),
            Line2D([0], [0], color=line_colors["decrease"], lw=2, label="Decrease (Post < Pre)")
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    
    plt.savefig(safe_path + modality_name + "_raincloud_plot.png")
    plt.show()
    plt.close()
    
def raincloud_plot(data: pd.DataFrame, modality_name: str, features_list: list, safe_path: str = "") -> None:
    """Create a refined raincloud plot using seaborn for better aesthetics."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Prepare data
    data_list = [data[col].dropna().values for col in data.columns]
    x_positions = np.arange(1, len(data_list) + 1)
    
    # Define colors based on modality
    if modality_name == "UPDRS":
        boxplots_colors = ['#FFCA3A', '#8AC926', '#1982C4']
        violin_colors = ['#FFCA3A', '#8AC926', '#1982C4']
        scatter_colors = ['#FF595E', '#FF595E', '#FF595E']
    else:
        boxplots_colors = ['#FFCA3A', '#8AC926']
        violin_colors = ['#FFCA3A', '#8AC926', ]
        scatter_colors = ['#FF595E', '#FF595E',]
    
    # Violin plot (half-violin) aligned to boxplot
    for idx, (x, values) in enumerate(zip(x_positions, data_list)):
        sns.violinplot(
            y=values, 
            inner=None, 
            cut=0, 
            linewidth=0, 
            color=violin_colors[idx], 
            ax=ax,
            width=0.6,
            split=True
        )
    
    # Boxplot (make it slimmer)
    bp = sns.boxplot(
        data=data,
        width=0.2,
        showcaps=True,
        whiskerprops={'color': 'black'},
        medianprops={'color': 'black'},
        flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red', 'alpha': 0.5},
        ax=ax
    )
    for patch, color in zip(bp.artists, boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    # Scatter plot (overlay on boxplot)
    for idx, (x, values) in enumerate(zip(x_positions, data_list)):
        x_jitter = np.random.uniform(low=-0.05, high=0.05, size=len(values)) + x
        ax.scatter(x_jitter, values, s=10, color=scatter_colors[idx], alpha=0.7)
    
    # Labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(features_list)
    ax.set_ylabel(modality_name)
    ax.set_title(f"{modality_name} Raincloud Plot")
    
    # Add legend for BDI
    if modality_name == "BDI":
        line_colors = {"increase": "blue", "decrease": "red", "no change": "grey"}
        legend_elements = [
            Line2D([0], [0], color=line_colors["increase"], lw=2, label="Increase (Post > Pre)"),
            Line2D([0], [0], color=line_colors["decrease"], lw=2, label="Decrease (Post < Pre)")
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    
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
        

    if modality_name == "UPDRS":
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
    if modality_name == "UPDRS":
        raincloud_plot(data, modality_name , ['Pre OFF', 'Pre ON', 'Post ON/Stim On'], save_path)
    else:
        raincloud_plot(data, modality_name , ['Pre', 'Post'], save_path)
    
    


if __name__ == "__main__":
    # Example usage
    root_dir = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    data_path = root_dir + "/data/bdi_df_normalized.csv"
    stim_path = root_dir + "/data/stim_positions.csv"
    save_path = root_dir + "/results/data_analysis/"
    mds_prepost_path = root_dir + "/data/mupdrs3_pre_vs_post.csv"
    ledd_prepost_path = root_dir + "/data/ledd_pre_vs_post.csv"
    bdi_prepost_path = root_dir + "/data/bdi_pre_vs_post.csv"
    op_dates_path = root_dir + "/data/op_dates.csv"

    demographics_pre_post(mds_prepost_path, op_dates_path, "UPDRS", save_path)
    demographics_pre_post(ledd_prepost_path, op_dates_path, "LEDD", save_path)
    demographics_pre_post(bdi_prepost_path, op_dates_path, "BDI", save_path)
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=['BDI_ratio','Pat_ID'])
    stim_positions = pd.read_csv(stim_path).drop(columns=['OP_DATUM'])
    # Perform PCA
    pca_result = pca(X, n_components=10, safe_path=save_path)
    print("PCA safed into {}".format(save_path))
    
    # Calculate and plot covariance matrix
    cov_matrix = covariance_matrix(X, safe_path=save_path)
    print("Covariance Matrix safed into {}".format(save_path))
    
    
    # Plot stimulation positions
    plot_stim_positions(stim_positions, safe_path=save_path)
    print("Stimulation positions plotted and safed into {}".format(save_path))