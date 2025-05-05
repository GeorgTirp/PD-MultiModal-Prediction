import os
#from RegressionsModels import TabPFNRegression
from TabPFN import TabPFNRegression
import pandas as pd
#sfrom sklearn.datasets import load_diabetes

def main(folder_path, data_path, target, identifier, out, folds=10):
    target_col = identifier + "_" + target
    possible_targets = ["ratio", "diff"] 
    ignored_targets = [t for t in possible_targets if t != target]
    ignored_target_cols = [identifier + "_" + t for t in ignored_targets]
    data_df = pd.read_csv(folder_path + data_path)
    data_df = data_df.drop(columns=['Pat_ID']+ ignored_target_cols)
    test_split_size = 0.2
    Feature_Selection = {}
    ### test
    #X, y = load_diabetes(return_X_y=True, as_frame=True)
    #data_df = pd.concat([X, y.rename("target")], axis=1)
    #Feature_Selection['target'] = "target"
    #Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    #safe_path = os.path.join(folder_path, "test/results/TabPFN")
    ### test ende
    Feature_Selection['target'] = target_col
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
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction/"
    #folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/BDI/level1/bdi_df.csv", "diff", "BDI", "results/level1/TabPFN", -1)
    #main(folder_path, "data/MoCA/level2/moca_df.csv", "diff", "MoCA", "results/level2/TabPFN", -1)
    main(folder_path, "data/BDI/level1/bdi_df.csv", "ratio", "BDI", "results/level1/TabPFN", -1)
    main(folder_path, "data/BDI/level2/bdi_df.csv", "diff", "BDI", "results/level2/TabPFN", -1)
    main(folder_path, "data/BDI/level2/bdi_df.csv", "ratio", "BDI", "results/level2/TabPFN", -1)
    #main(folder_path, "data/MoCA/level2/moca_df.csv", "diff", "MoCA", "results/level2/TabPFN", -1)
    #main(folder_path, "data/BDI/level3/bdi_df.csv", "ratio", "BDI", "results/level3/TabPFN", -1)
    #main(folder_path, "data/MoCA/level2/moca_df.csv", "ratio", "MoCA", "results/level2/TabPFN", -1)
    #main(folder_path, "data/BDI/post/bdi_df.csv", "sum_post", "BDI", "results/post/TabPFN", -1)
    #main(folder_path, "data/BDI/level3/bdi_df.csv", "diff", "BDI", "results/level3/TabPFN", -1)