import pandas as pd
import sys
from typing import Tuple, Dict
from XGBoost_pipeline import run_XGBoost_pipeline


#######-- Set the parameters for the analysis --#######
folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
# Preprocessed data
data = folder_path + "/data/bdi_df.csv"

#
external_tuned_hparams = pd.read_csv(folder_path +"/results/bdi_haparams/XGBoost_hparams.csv").to_dict(orient='records')[0]

# Name of the variable to predict in the data table
target = 'BDI_diff'


# Name of the variables to use for the prediction
features = []  # Emtpy list means all the variables except the target

# Outlier removal?
outlier_removal = False

# Number of cross-validation folds
cv = 5

# Correlation threshold for correlated features
correlation_threshold = 0.9

# Save the results?
save_results = True

# Safe path
safe_path = folder_path + "/results"
    

# Identifier
identifier = "XGBoost_bdi"

# Random state
random_state = 42

#######----------------------------------------#######


### Run the pipeline with the specified paramters
run_XGBoost_pipeline(
    data=data, 
    target=target, 
    features=features, 
    outlier_removal=outlier_removal, 
    cv=cv, 
    correlation_threshold=correlation_threshold,
    save_results=True, 
    save_path=safe_path, 
    identifier=identifier, 
    random_state=random_state,
    external_tuned_hparams=external_tuned_hparams)
