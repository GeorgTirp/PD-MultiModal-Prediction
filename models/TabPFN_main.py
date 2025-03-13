import os
#from RegressionsModels import TabPFNRegression
from TabPFN import TabPFNRegression
import pandas as pd


def main(folder_path, data_path, target, identifier, folds=10):
    target_col = identifier + "_" + target
    possible_targets = ["efficacy", "ratio", "diff"] 
    ignored_targets = [t for t in possible_targets if t != target]
    ignored_target_cols = [identifier + "_" + t for t in ignored_targets]
    data_df = pd.read_csv(folder_path + data_path)
    data_df = data_df.drop(columns=['Pat_ID']+ ignored_target_cols)
    test_split_size = 0.2
    Feature_Selection = {}
    Feature_Selection['target'] = target_col
    print(Feature_Selection['target'])
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = os.path.join(folder_path, "results/TabPFN")
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

if __name__ == "__main__":
    possible_targets = ["BDI_efficacy", "MoCA_efficacy"]
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/BDI/bdi_df(2).csv", "diff", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df.csv", "diff", "MoCA", -1)
    main(folder_path, "data/BDI/bdi_df(2).csv", "ratio", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df.csv", "ratio", "MoCA", -1)
    main(folder_path, "data/BDI/bdi_df(2).csv", "efficacy", "BDI", -1)
    main(folder_path, "data/MoCA/moca_df.csv", "efficacy", "MoCA", -1)
