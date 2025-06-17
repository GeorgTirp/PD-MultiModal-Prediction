import os
#from RegressionsModels import TabPFNRegression
from TabPFN import TabPFNRegression
import pandas as pd
#sfrom sklearn.datasets import load_diabetes

def main(folder_path, data_path, target, identifier, out, folds=10):
    df = pd.read_excel(data_path)
    data_df = df[[target] + [col for col in df.columns if col.startswith('nmf_')]]
    
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
    folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/frieder/merged_demographics_features.xlsx", "updrs_totscore", "WMA", "results/WMA/TabPFN", -1)
    