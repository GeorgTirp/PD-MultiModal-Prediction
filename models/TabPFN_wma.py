import os
#from RegressionsModels import TabPFNRegression
from TabPFN import TabPFNRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
#sfrom sklearn.datasets import load_diabetes
from scipy.stats import zscore

def detect_outliers_with_std(df, columns_to_exclude, threshold, num_feat_th=1):
    """
    Remove outliers based on z-scores for all columns except those specified.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns_to_exclude (list): List of column names to exclude from outlier removal.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    # Select columns to apply z-score outlier removal
    columns_to_include = [col for col in df.columns if col not in columns_to_exclude]
    # Calculate z-scores
    z_scores = df[columns_to_include].apply(zscore)
    
    # Determine outliers (absolute z-score > 3)
    #mask = (z_scores.abs() <= threshold).all(axis=1)
    mask = ((z_scores.abs() <= threshold).sum(axis=1)>= len(columns_to_include)-num_feat_th)
    
    # Add a column named 'outlier' 0 if inlier and 1 if outlier
    df['outlier'] = 0
    df.loc[df.index.isin(df[~mask].index), 'outlier'] = 1

    return df


def main(folder_path, data_path, target, identifier, out, folds=10):
    df = pd.read_excel(data_path)
    data_df = df[[target] + [col for col in df.columns if col.startswith('nmf_')] + ['Age'] + ['Sex'] + ['DOMSIDE']]
    data_df['Sex'] = data_df['Sex'].map({'M': 1, 'F': 0})

    data_df = data_df.dropna()
    test_split_size = 0.2
    Feature_Selection = {}
    ### test
    #X, y = load_diabetes(return_X_y=True, as_frame=True)
    #data_df = pd.concat([X, y.rename("target")], axis=1)
    #Feature_Selection['target'] = "target"
    #Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    #safe_path = os.path.join(folder_path, "test/results/TabPFN")
    ### test ende

    Feature_Selection['target'] = target
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = os.path.join(folder_path, out)
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    model = TabPFNRegression(
        data_df, 
        Feature_Selection, 
        target, 
        test_split_size, 
        safe_path, 
        identifier)
    model.fit()
    X, y = model.model_specific_preprocess(data_df)
    metrics = model.evaluate(folds=folds)
    model.plot(f"Actual vs. Prediction (TabPFN) - {identifier}", identifier)
    #importances = model.feature_importance(top_n=19, batch_size=10)
    #_,_, removals= model.feature_ablation()

if __name__ == "__main__":
    possible_targets = ["BDI_efficacy", "MoCA_efficacy"]
    #folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    folder_path = "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Code/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Results/PPMI_White_Matter_Alteration_Analysis/TDDR_PPMI_BASELINE/merged_demographics_features.xlsx", "updrs_totscore", "WMA_diff_nmfs", "results/WMA/TabPFN", 20)
    