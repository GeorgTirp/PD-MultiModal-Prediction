import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
from model_classes.XGBoostRegressionModel import XGBoostRegressionModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Logging
from utils.my_logging import Logging
from utils.messages import Messages
from pprint import pformat


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
    updrs1_patient = pd.read_csv('data/MDS-UPDRS_Part_I_Patient_Questionnaire_24Jun2025.csv')
    updrs1_examiner = pd.read_csv('data/MDS-UPDRS_Part_I_24Jun2025.csv')
    updrs3 = updrs3[updrs3['PATNO'].isin(demographics['PATNO'])]
    updrs2 = updrs2[updrs2['PATNO'].isin(demographics['PATNO'])]
    updrs1_patient = updrs1_patient[updrs1_patient['PATNO'].isin(demographics['PATNO'])]
    updrs1_examiner = updrs1_examiner[updrs1_examiner['PATNO'].isin(demographics['PATNO'])]

    # Only keep the rows where PDTRTMNT = 0, which means the patient is not on treatment
    updrs3 = updrs3[(updrs3['PDTRTMNT'] == 0) | (pd.isna(updrs3['PDTRTMNT']) | (updrs3['PDSTATE'] == 'OFF'))]

    # Filter only EVENT_ID = BL
    if set_of_subjects == 'baseline':
        updrs3 = updrs3[updrs3['EVENT_ID'] == 'BL']
        updrs2 = updrs2[updrs2['EVENT_ID'] == 'BL']
        updrs1_patient = updrs1_patient[updrs1_patient['EVENT_ID'] == 'BL']
        updrs1_examiner = updrs1_examiner[updrs1_examiner['EVENT_ID'] == 'BL']
    elif set_of_subjects == 'longitudonal':
        updrs3 = updrs3[updrs3['EVENT_ID'] == 'V06']
    else:
        raise ValueError('set_of_subjects must be either baseline or longitudonal')

    if logfile:
        logfile.write("UPDRS subscores loaded and processed.\n")

    updrs3 = calculate_pigd_td(updrs3, updrs2)
    
    return updrs3, updrs2, updrs1_patient, updrs1_examiner


def main(folder_path, data_path, target, identifier, out, folds=10):
    # --- DISPLAY WELCOME MESSAGE ---
    Messages().welcome_message()

    # --- SETUP PATHS ---
    safe_path = os.path.join(folder_path, out)
    print(safe_path)
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
    #updrs3_subscores, updrs2, updrs1_patient, updrs1_exam = load_updrs_subscores(path_to_updrs3='data/MDS-UPDRS_Part_III_24Jun2025.csv',
    #                        path_to_updrs2='/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Code/PD-MultiModal-Prediction/data/MDS_UPDRS_Part_II__Patient_Questionnaire_24Jun2025.csv',
    #                        demographics=df,
    #                        set_of_subjects='baseline',
    #                        clinical_score=target)
    #if target == 'updrs1_pat_score':
    #    updrs1_subscores = updrs1_patient[['PATNO', 'NP1PTOT']]
    #    updrs1_subscores.rename(columns={'NP1PTOT': 'updrs1_pat_score'}, inplace=True)
    #    updrs3_subscores = pd.merge(updrs3_subscores, updrs1_subscores, on='PATNO', how='left')
    #elif target == 'updrs1_exam_score':
    #    updrs1_subscores = updrs1_exam[['PATNO', 'NP1RTOT']]
    #    updrs1_subscores.rename(columns={'NP1RTOT': 'updrs1_exam_score'}, inplace=True)
    #    updrs3_subscores = pd.merge(updrs3_subscores, updrs1_subscores, on='PATNO', how='left')
    #data_df = pd.merge(df, updrs3_subscores, on='PATNO', how='left')
    data_df = df[[target] + [col for col in df.columns if col.startswith('nmf_')]] #+ ['Age'] + ['Sex'] + ['DOMSIDE'] + ['duration_yrs'] + ['moca'] + ['td_pigd']]
    # Convert 'Sex' column to 0 (female) and 1 (male)
    if 'Sex' in data_df.columns:
        data_df['Sex'] = data_df['Sex'].map({'F': 0, 'M': 1})
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
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.9],
    'colsample_bytree': [0.8, 1],
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [1, 2, 5],
    'random_state': [42],
    'verbosity': [0]
}
    logging.info(f"----- Parameter grid for XGBoost ----- \n" + pformat(param_grid_xgb) + "\n")

    # Default XGBoost hyperparameters
    XGB_Hparams = {
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 0,
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
    #metrics = model.evaluate(
    #    folds=folds, 
    #    tune=True, 
    #    nested=True, 
    #    tune_folds=20, 
    #    get_shap=True,
    #    uncertainty=False)
    
    # Log the metrics
    #logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    #logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    #model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}",  save_path=safe_path + '/model_evaluation/')
    _,_, removals= model.feature_ablation(folds=folds, tune=False, tune_folds=20, features_per_step=1, threshold_to_one_fps=10)
        
        

if __name__ == "__main__":
    folder_path = "/home/ubuntu/STN_WMA/PD-MultiModal-Prediction/"

#    main(folder_path, "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Results/PPMI_White_Matter_Alteration_Analysis/TDDR_PPMI_BASELINE/merged_demographics_features_diff_20.xlsx", "updrs1_pat_score", "WMA", "results/Paper_runs/XGBoost_updrs1_pat", -1)
#    main(folder_path, "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Results/PPMI_White_Matter_Alteration_Analysis/TDDR_PPMI_BASELINE/merged_demographics_features_diff_20.xlsx", "updrs1_exam_score", "WMA", "results/Paper_runs/XGBoost_updrs1_exam", -1)
#    main(folder_path, "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Results/PPMI_White_Matter_Alteration_Analysis/TDDR_PPMI_BASELINE/merged_demographics_features_diff_20.xlsx", "updrs2_score", "WMA", "results/Paper_runs/XGBoost_updrs2_nodem_shuffeled", -1)
    main(folder_path, "data/merged_demographics_features_diff_20.xlsx", "updrs3_score", "WMA", "results/Paper_runs/XGBoost_updrs3_test", 5)
#    main(folder_path, "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Results/PPMI_White_Matter_Alteration_Analysis/TDDR_PPMI_BASELINE/merged_demographics_features_diff_20.xlsx", "moca", "WMA", "results/Paper_runs/XGBoost_moca_nodem", -1)
    