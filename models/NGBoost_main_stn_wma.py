import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
from RegressionsModels import NGBoostRegressionModel
import pandas as pd
from faster_evidential_boost import NormalInverseGamma, NIGLogScore
from ngboost.distns.normal import Normal, NormalCRPScore, NormalLogScore
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

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

    ###### WMA specific preprocessing ######
    df = pd.read_excel(data_path)

    # 1. Preprocess the specific columns
    updrs_subscores = load_updrs_subscores(path_to_updrs3='data/MDS-UPDRS_Part_III_05Oct2024.csv',
                            path_to_updrs2='data/MDS_UPDRS_Part_II__Patient_Questionnaire_05Oct2024.csv',
                            demographics=df,
                            set_of_subjects='baseline',
                            clinical_score=target)
    data_df = pd.merge(df, updrs_subscores[['PATNO', 'TDPIGD', 'TD_avg', 'PIGD_avg']], on='PATNO', how='left')
    data_df = data_df[[target] + [col for col in data_df.columns if col.startswith('nmf_')] + ['Age'] + ['Sex'] + ['DOMSIDE']]
    data_df['Sex'] = data_df['Sex'].map({'M': 1, 'F': 0})
    #

    # 2. Standard scaling for numerical stability
    scaler_X = StandardScaler()

    scaler_y = StandardScaler()
    feature_cols = [col for col in data_df.columns if col != target]
    data_df[feature_cols] = scaler_X.fit_transform(data_df[feature_cols])
    data_df[target] = scaler_y.fit_transform(data_df[[target]])
    #

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


    param_grid_ngb = {
    'n_estimators': [300, 400, 500, 600],
    'learning_rate': [0.05, 0.1],
    'Base__max_depth': [3, 4],
    'Score__evid_strength': [0.05, 0.1, 0.2],
    'Score__kl_strength': [0.01, 0.05, 0.1],
    }   
    
    # BEST ONES: 600, 0.1 and for regs 0.1 and 0.001
    NGB_Hparams = {
        'Dist': Normal, #NormalInverseGamma,
        'Score' : NormalCRPScore ,#NIGLogScore,
        'n_estimators': 600,
        'learning_rate': 0.1,
        'natural_gradient': True,
        #'minibatch_frac': 0.1,
        'verbose': False,
        'Base': DecisionTreeRegressor(max_depth=3)  # specify the depth here
    }

    model = NGBoostRegressionModel(
        data_df, 
        Feature_Selection, 
        target,
        NGB_Hparams, 
        0.2, 
        safe_path, 
        identifier,
        -1,
        param_grid_ngb)
    metrics = model.evaluate(
        folds=folds, 
        tune=False, 
        nested=True, 
        tune_folds=20, 
        get_shap=True,
        uncertainty=False)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log the metrics
    logging.info(f"Aleatoric Uncertainty: {metrics['aleatoric']}")
    logging.info(f"Epistemic Uncertainty: {metrics['epistemic']}")
    model.plot(f"Actual vs. Prediction (NGBoost) - {identifier}")
    _,_, removals= model.feature_ablation(folds=folds, tune=False, tune_folds=10)
    model.calibration_analysis()
    
    
    summands = [0, 0, 1, 0]
    param_names  = ["mu", "lambda", "alpha", "beta"]
    for i in range(len(param_names)):
        if i == 0:
            param = metrics["pred_dist"][i] + summands[i]
        param = np.exp(metrics["pred_dist"][i]) + summands[i]
        print(metrics["pred_dist"][i].shape)
        plt.figure()
        plt.hist(param, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f"Histogram of {param_names[i]} - Sample {i+1}")
        plt.xlabel(f"{param_names[i]} values")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(safe_path, f"histogram_{param_names[i]}_sample.png"))
        plt.close()
        
        

if __name__ == "__main__":
    folder_path = "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Code/PD-MultiModal-Prediction/"

    main(folder_path, "/media/sn/Frieder_Data/Projects/White_Matter_Alterations/STN/Results/PPMI_White_Matter_Alteration_Analysis/TDDR_PPMI_BASELINE/merged_demographics_features.xlsx", "moca", "WMA", "results/WMA/NGBoost_Moca_Gauss", 10)
    