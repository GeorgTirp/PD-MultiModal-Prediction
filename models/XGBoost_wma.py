import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
from RegressionsModels import XGBoostRegressionModel
import pandas as pd
from faster_evidential_boost import NormalInverseGamma, NIGLogScore
from ngboost.distns.normal import Normal, NormalCRPScore, NormalLogScore
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
# Logging
from utils.my_logging import Logging
from utils.messages import Messages
from pprint import pformat


# --- Dynamic Tobit bound functions ---
## Calculate wma specific scores
def calculate_pigd_td(updrs3, updrs2):
    tremor_items_study_mds_updrs_part_3 = ['NP3PTRML', 'NP3PTRMR', 'NP3KTRML', 'NP3KTRMR', 'NP3RTALJ', 'NP3RTALL', 'NP3RTALU', 'NP3RTARL', 'NP3RTARU', 'NP3RTCON']
    tremor_items_study_mds_updrs_part_2 = ['NP2TRMR']
    
    PIGD_items_mds_updrs_3 = ['NP3GAIT', 'NP3FRZGT', 'NP3PSTBL']
    PIGD_items_mds_updrs_2 = ['NP2WALK', 'NP2FREZ']

    updrs = pd.merge(updrs3, updrs2, on='PATNO')

    updrs_pigd = updrs[PIGD_items_mds_updrs_3 + PIGD_items_mds_updrs_2]
    updrs_tremor = updrs[tremor_items_study_mds_updrs_part_3 + tremor_items_study_mds_updrs_part_2]

    # Calculate the PIGD and TD scores
    updrs['PIGD_avg'] = updrs_pigd[PIGD_items_mds_updrs_3 + PIGD_items_mds_updrs_2].mean(axis=1, skipna=False)
    updrs['TD_avg'] = updrs_tremor[tremor_items_study_mds_updrs_part_3 + tremor_items_study_mds_updrs_part_2].mean(axis=1, skipna=False)
    updrs['TDPIGD'] = (updrs['TD_avg'] + 1)/(updrs['PIGD_avg'] + 1)

    return updrs

def load_updrs_subscores(path_to_updrs3, path_to_updrs2, demographics, set_of_subjects, clinical_score, logfile=None):
    # All UPDRS3 score columns and their descriptions
    # Load MDS-UPDRS_Part_III_10Jun2024.csv data and filter out the PATNOs that are in the demographics data only
    updrs3 = pd.read_csv(path_to_updrs3)
    updrs2 = pd.read_csv(path_to_updrs2)
    updrs3 = updrs3[updrs3['PATNO'].isin(demographics['PATNO'])]
    updrs2 = updrs2[updrs2['PATNO'].isin(demographics['PATNO'])]

    # Only keep the rows where PDTRTMNT = 0, which means the patient is not on treatment
    updrs3 = updrs3[(updrs3['PDTRTMNT'] == 0) | (pd.isna(updrs3['PDTRTMNT']) | (updrs3['PDSTATE'] == 'OFF'))]

    # Filter only EVENT_ID = BL
    if set_of_subjects == 'baseline':
        updrs3 = updrs3[updrs3['EVENT_ID'] == 'BL']
        updrs2 = updrs2[updrs2['EVENT_ID'] == 'BL']
    elif set_of_subjects == 'longitudonal':
        updrs3 = updrs3[updrs3['EVENT_ID'] == 'V06']
    else:
        raise ValueError('set_of_subjects must be either baseline or longitudonal')

    if logfile:
        logfile.write("UPDRS subscores loaded and processed.\n")

    updrs3 = calculate_pigd_td(updrs3, updrs2)
    
    return updrs3


def main(folder_path, data_path, target, identifier, out, folds=10):
    # --- DISPLAY WELCOME MESSAGE ---
    Messages().welcome_message()

    # --- SETUP PATHS ---
    safe_path = os.path.join(folder_path, out)
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
        os.makedirs(os.path.join(safe_path, 'log'))

    # --- SETUP LOGGING ---
    logging = Logging(f'{safe_path}/log').get_logger()

    # --- Print all selected parameters ---
    logging.info('-------------------------------------------')
    logging.info(f"Folder Path: {folder_path}")
    logging.info(f"Data Path: {data_path}")
    logging.info(f"Target: {target}")
    logging.info(f"Identifier: {identifier}")
    logging.info(f"Output Path: {out}")
    logging.info(f"Folds: {folds}")
    logging.info('-------------------------------------------\n')

    ###### WMA specific preprocessing ######
    df = pd.read_excel(data_path)

    # 1. Preprocess the specific columns
    updrs_subscores = load_updrs_subscores(path_to_updrs3='data/MDS-UPDRS_Part_III_05Oct2024.csv',
                            path_to_updrs2='data/MDS_UPDRS_Part_II__Patient_Questionnaire_05Oct2024.csv',
                            demographics=df,
                            set_of_subjects='baseline',
                            clinical_score=target)
    data_df = pd.merge(df, updrs_subscores[['PATNO', 'TDPIGD', 'TD_avg', 'PIGD_avg']], on='PATNO', how='left')
    data_df = data_df[[target] + [col for col in data_df.columns if col.startswith('nmf_')] + ['Age'] + ['Sex'] + ['DOMSIDE'] + ['duration_yrs'] + ['moca'] + ['td_pigd']]
    data_df['Sex'] = data_df['Sex'].map({'M': 1, 'F': 0})
    #

    # 2. Standard scaling for numerical stability
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    feature_cols = [col for col in data_df.columns if col != target]
    data_df[feature_cols] = scaler_X.fit_transform(data_df[feature_cols])
    data_df[target] = scaler_y.fit_transform(data_df[[target]])
    # Shuffle the target variable
    #shuffled_target = data_df[target].sample(frac=1, random_state=42).reset_index(drop=True)
    #data_df[target] = shuffled_target
    
    # 3. Remove NaN values
    data_df = data_df.dropna()
    #
    ### WMA specific preprocessing end ###


    Feature_Selection = {}
    Feature_Selection['target'] = target
    Feature_Selection['features'] = [col for col in data_df.columns if col != Feature_Selection['target']]
    safe_path = os.path.join(folder_path, out)
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)



    test_split_size = 0.2
    # XGBoost hyperparameter grid
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 6],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.6, 0.8, 1.0],
#        'reg_alpha': [0, 0.1, 0.5],
#        'reg_lambda': [1, 2, 5]
    }
    logging.info(f"----- Parameter grid for XGBoost ----- \n" + pformat(param_grid_xgb) + "\n")

    # Default XGBoost hyperparameters
    XGB_Hparams = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 2,
        'random_state': 42,
        'verbosity': 0
    }
    logging.info(f"----- XGBoost Hyperparameters ----- \n" + pformat(XGB_Hparams) + "\n")

    model = XGBoostRegressionModel(
        data_df, 
        Feature_Selection, 
        target,
        XGB_Hparams, 
        test_split_size, 
        safe_path, 
        identifier,
        -1,
        param_grid_xgb, 
        logging=logging)
    metrics = model.evaluate(
        folds=folds, 
        tune=False, 
        nested=True, 
        tune_folds=20, 
        get_shap=True,
        uncertainty=False)
    
    # Log the metrics
    logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}")
    _,_, removals= model.feature_ablation(folds=folds, tune=True, tune_folds=5, features_per_step=2, threshold_to_one_fps=10)
        
        

if __name__ == "__main__":
    folder_path = "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Code/PD-MultiModal-Prediction/"

    main(folder_path, "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Results/PPMI_White_Matter_Alteration_Analysis/TDDR_PPMI_BASELINE/merged_demographics_features_diff_20.xlsx", "updrs_totscore", "WMA", "results/WMA_refined/XGBoost_updrs_totscore_tuned", 10)
    