import os
from TabPFN import TabPFNRegression
import pandas as pd
def main(folder_path, data_path, target, identifier):
    data_df = pd.read_csv(folder_path + data_path)
    data_df = data_df.drop(columns=['Pat_ID'])
    test_split_size = 0.2
    Feature_Selection = {}
    Feature_Selection['target'] = target
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = os.path.join(folder_path, "results/TabPFN")
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    model = TabPFNRegression(data_df, Feature_Selection, test_split_size, safe_path, identifier)
    model.fit()
    X, y = model.model_specific_preprocess(data_df, Feature_Selection)
    metrics = model.evaluate(n_splits=10)
    model.plot(f"Actual vs. Prediction (TabPFN) - {identifier}", identifier)

if __name__ == "__main__":
    folder_path = "/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Neuromodulation/PD-MultiModal-Prediction/"
    main(folder_path, "data/BDI/bdi_df.csv", "BDI_efficacy", "BDI")
    main(folder_path, "data/MoCA/moca_df.csv", "MoCA_efficacy", "MoCA")
