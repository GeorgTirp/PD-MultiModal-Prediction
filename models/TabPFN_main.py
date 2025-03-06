import os
from models.TabPFN import TabPFNRegression
import pandas as pd
def bdi_main():
       #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    data_df = pd.read_csv(folder_path + "/data/BDI/bdi_df.csv")
    data_df = data_df.drop(columns=['Pat_ID'])
    test_split_size = 0.2
    Feature_Selection = {}
    Feature_Selection['target'] = 'BDI_efficacy'
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = folder_path + "/results/TabPFN"
    identifier = "BDI"
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    model = TabPFNRegression(data_df, Feature_Selection, test_split_size, safe_path, identifier)
    model.fit()
    X, y = model.model_specific_preprocess(data_df, Feature_Selection)
    preds = model.predict(X, save_results=True)
    metrics = model.evaluate(n_splits=10)
    model.plot("Actual vs. Prediction (TabPFN)")
    #importances = model.feature_importance(batch_size=10)

def moca_main():
       #folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    data_df = pd.read_csv(folder_path + "/data/MOCA/bdi_df.csv")
    data_df = data_df.drop(columns=['Pat_ID'])
    test_split_size = 0.2
    Feature_Selection = {}
    Feature_Selection['target'] = 'MOCA_efficacy'
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = folder_path + "/results/TabPFN"
    identifier = "MoCA"
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    model = TabPFNRegression(data_df, Feature_Selection, test_split_size, safe_path, identifier)
    model.fit()
    X, y = model.model_specific_preprocess(data_df, Feature_Selection)
    preds = model.predict(X, save_results=True)
    metrics = model.evaluate(n_splits=10)
    model.plot("Actual vs. Prediction (TabPFN)")
    #importances = model.feature_importance(batch_size=10)

if __name__ == "__main__":
    moca_main()
    bdi_main()