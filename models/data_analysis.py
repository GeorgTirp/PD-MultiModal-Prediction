import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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

if __name__ == "__main__":
    # Example usage
    root_dir = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    data_path = root_dir + "/data/bdi_df.csv"
    stim_path = root_dir + "/data/stim_positions.csv"
    save_path = root_dir + "/results/data_analysis/"

    
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=['BDI_diff','Pat_ID'])
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