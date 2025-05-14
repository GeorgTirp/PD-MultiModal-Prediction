import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def filter_shaps(data_path, shap_path, save_path, identifier, threshold=0.0, features=None):
    # Load the SHAP values from the .npy file
    shap_values = np.load(shap_path)
    
    # Load the data as a pandas DataFrame
    data = pd.read_csv(data_path)
    # Get indices where the absolute value of 'BDI_diff' column is greater than or equal to 8
    indices = data.index[abs(data['BDI_diff']) >= threshold].tolist()

    # Drop the specified columns from the data
    data = data.drop(columns=['BDI_ratio', 'BDI_diff', 'Pat_ID'])
    if features is not None:
        data = data[features]
    # Filter the SHAP values based on the selected indices
    filtered_shap_values = shap_values[indices, :]
    
    # Replot the beeswarm plot using the filtered SHAP values
    shap.summary_plot(filtered_shap_values, data.iloc[indices], data.columns, show=False)
    plt.title(f'Filtered Summary Plot (Aggregated - Mean)', fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.savefig(f'{save_path}/{identifier}_filter_geq_{threshold}.png')
    plt.close()
    return data, shap_values

def covariance_matrix(data_path, safe_path: str = "") -> pd.DataFrame:
    """ Calculate the covariance matrix of the data and visualize it as a heatmap"""
    # Read the data from the provided path
    X = pd.read_csv(data_path)
    corr_matrix = X.corr(method='pearson')
    plt.figure(figsize=(14, 12), dpi=300)  # Increase figure size and resolution
    plt.title('Covariance Matrix Heatmap', fontsize=16)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='viridis', annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(safe_path + "_covariance_matrix.png", dpi=300)  # Save with higher resolution
    plt.close()
    return corr_matrix


if __name__ == "__main__":
    # Example usage
    data_path2 = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/data/BDI/level2/bdi_df.csv"
    shap_path2 = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/results/level2/NGBoost/BDI_mean_shap_values.npy"

    data_path3 = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/data/BDI/level3/bdi_df.csv"
    shap_path3 = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/results/level3/NGBoost/BDI_ratio_[9]_mean_shap_values.npy"

    save_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/results/"
    # Load the data and SHAP values
    data, shap_values = filter_shaps(data_path2, shap_path2, save_path, "level2", threshold=8.0)
    features = [
        "BDI_sum_pre",
        "AGE_AT_OP",
        "TimeSinceDiag",
        "alpha_L",
        "high-beta_L",
        "high-beta_R",
        "low-beta_L",
        "theta_R",
        "Z_L",
        "Z_R",
    ]
    data, shap_values = filter_shaps(
        data_path3, 
        shap_path3, 
        save_path, 
        "level3_9_features", 
        threshold=8.0,
        features=features)
    

    covariance_matrix(data_path2, "results/level2")
    covariance_matrix(data_path3, "results/level3") 
    
    # Print the shapes of the loaded data and SHAP values
    print("Data shape:", data.shape)
    print("SHAP values shape:", shap_values.shape)