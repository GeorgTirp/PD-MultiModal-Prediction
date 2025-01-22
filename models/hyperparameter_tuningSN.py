import random
from collections import Counter
import math
import pickle
import sys
from io import TextIOWrapper
from typing import Final
import multiprocessing
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import explained_variance_score
from sklearn.cluster import (
    OPTICS,
    KMeans,
    DBSCAN,
    SpectralClustering,
    MeanShift,
    AffinityPropagation,
    BisectingKMeans,
)
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_halving_search_cv
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import xgboost
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut, KFold
from xgboost import XGBRegressor
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, rand
from hyperopt.pyll import scope
from scipy.stats import zscore
import warnings
import os


# Define the objective function for hyperparameter tuning
def objective(params, model, train_x, train_y):
    print("Current hyperparameters:", params)
    print("Current model parameters:", model.get_params())

    # Update model's parameters (except n_estimators)
    # depth_choices = np.arange(1, 9, dtype=int)
    if "reg_alpha" in params:

        # Ensure reg_alpha is within the expected range
        if not 0.0 <= params["reg_alpha"] <= 200:
            print(f"reg_alpha {params['reg_alpha']} is out of bounds.")
            return {"loss": float("inf"), "status": STATUS_OK}

    if "max_depth" in params:
        params["max_depth"] = int(params["max_depth"])
        print("Parameters:", params["max_depth"])
    model.set_params(**params)

    # Inner K-Fold Cross-Validation
    inner_cv = KFold(n_splits=6)
    inner_mse_scores = []
    inner_r2_scores = []
    inner_pearson_scores = []

    for inner_train_index, inner_val_index in inner_cv.split(train_x):
        inner_train_X, inner_val_X = (
            train_x[inner_train_index],
            train_x[inner_val_index],
        )
        inner_train_y, inner_val_y = (
            train_y[inner_train_index],
            train_y[inner_val_index],
        )

        model.fit(inner_train_X, inner_train_y)
        pred = model.predict(inner_val_X)
        mse = mean_squared_error(inner_val_y, pred)
        # Calculate explained variance
        explained_var = r2_score(inner_val_y, pred)
        inner_mse_scores.append(mse)
        inner_r2_scores.append(explained_var)
        # Calculate Pearson correlation coefficient
        pearson_corr, _ = pearsonr(inner_val_y, pred)
        inner_pearson_scores.append(pearson_corr)

    avg_mse = np.mean(inner_mse_scores)
    avg_pearson = np.mean(inner_pearson_scores)
    avg_r2 = np.mean(inner_r2_scores)

    combined_score = (0.5 * (1 / (1 + np.sqrt(avg_mse)))) + (0.5 * avg_r2)

    return {
        "loss": avg_mse,
        "status": STATUS_OK,
        "avg_mse": avg_mse,
        "avg_pearson": avg_pearson,
        # "avg_r2": avg_r2,
    }


def prune_trees(X, y, pat_ids, init_params, output_path):
    """
    Function to determine the optimal number of trees for XGBoost using Leave-One-Out Cross-Validation.

    Parameters:
    X (array-like): Feature dataset.
    y (array-like): Target variable.
    pat_ids (array-like): Patient IDs corresponding to each row in X and y.
    init_params (dict): Initial parameters for the XGBoost model.
    output_path (str): Path to save the CSV file with the best parameters.

    Returns:
    DataFrame: A DataFrame containing the best parameters for each iteration.
    """

    loo = LeaveOneOut()
    best_params_list = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        left_out_pat_id = pat_ids[test_index][0]

        if isinstance(
            init_params, pd.DataFrame
        ):  # Retrieve previously optimized parameters for this LOOCV iteration
            init_params_temp = init_params.iloc[i].to_dict()
            init_params_temp.pop("left_out_PatId")  # Remove non-model parameter
        else:
            init_params_temp = init_params

        if "max_depth" in init_params_temp:
            init_params_temp["max_depth"] = int(init_params_temp["max_depth"])

        # Convert DataFrame to DMatrix (XGBoost's internal data structure)
        dtrain = xgb.DMatrix(train_X, label=train_y)

        # Perform XGBoost cross-validation to determine the best n_estimators
        cvresult = xgb.cv(
            init_params_temp,
            dtrain,
            num_boost_round=1000,
            nfold=6,
            early_stopping_rounds=50,
            verbose_eval=True,
            seed=42,
        )

        # Update n_estimators
        best_n_estimators = cvresult.shape[0]
        print(f"Best number of estimators: {best_n_estimators}")

        # Store the best number of estimators along with the left-out PatId
        best_params = init_params_temp.copy()
        best_params["n_estimators"] = best_n_estimators
        best_params["left_out_PatId"] = left_out_pat_id
        best_params_list.append(best_params)

    # Save the final best parameters to a CSV file
    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.to_csv(output_path, index=False)

    return best_params_df


def tune_nested_cross_validation(
    X,
    y,
    pat_ids,
    initial_best_params_df,
    model,
    max_evals,
    space,
    objective,
    output_path,
):
    """
    Function to perform nested cross-validation for hyperparameter tuning of an XGBoost model.

    Parameters:
    X (array-like): Feature dataset.
    y (array-like): Target variable.
    pat_ids (array-like): Patient IDs corresponding to each row in X and y.
    initial_best_params_df (DataFrame): DataFrame containing initial best parameters for each LOOCV iteration.
    model (XGBModel): Pre-initialized XGBoost model.
    space (dict): Hyperparameter space for optimization.
    objective (function): Objective function for hyperparameter optimization.
    output_path (str): Path to save the CSV file with the best parameters.

    Returns:
    float: Average MSE across all LOOCV splits.
    DataFrame: A DataFrame containing the best parameters for each iteration.
    """

    loo = LeaveOneOut()
    loo_mse_scores = []
    best_params_list = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        left_out_pat_id = pat_ids[test_index][0]  # ID of the left-out patient

        # Retrieve previously optimized parameters for this LOOCV iteration
        previous_params = initial_best_params_df.iloc[i].to_dict()
        previous_loss = previous_params.get("mse", 2)
        previous_params.pop("left_out_PatId")  # Remove non-model parameter
        previous_params.pop("mse", None)
        previous_params.pop("pearson", None)

        if "max_depth" in previous_params:
            previous_params["max_depth"] = int(previous_params["max_depth"])

        model.set_params(**previous_params)

        current_params = model.get_params()

        trials = Trials()

        # Create a trial document with the necessary structure
        trial_doc = {
            "tid": 0,
            "state": 0,  # JOB_STATE_DONE
            "result": {
                "loss": previous_loss,
                "status": STATUS_OK,
                "avg_mse": previous_loss,
                "avg_pearson": 0.1,
            },  # Use an appropriate loss value
            "misc": {
                "vals": {key: [current_params[key]] for key in space.keys()},
                "tid": 0,
                "cmd": ("domain_attachment", "FMinIter_Domain"),
                "workdir": None,
                "idxs": {key: [0] for key in space.keys()},
            },
            "exp_key": None,
            "owner": None,
            "version": 0,
            "book_time": None,
            "refresh_time": None,
            "spec": None,  # Including the 'spec' key with a value of None
        }

        # Add the trial document to the trials object
        trials.insert_trial_doc(trial_doc)
        trials.refresh()

        best = fmin(
            fn=lambda params: objective(params, model, train_X, train_y),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals + 1,
            trials=trials,
            rstate=np.random.default_rng(42),
        )

        # for trial in trials.trials:
        # print(trial['result'])

        # Find the trial with the best result
        best_trial = trials.best_trial

        # Extract MSE and Pearson from the best trial
        best_mse = best_trial["result"]["avg_mse"]
        best_pearson = best_trial["result"]["avg_pearson"]

        # Convert best parameters to actual values (if necessary)
        best_params = {k: v[0] if isinstance(v, list) else v for k, v in best.items()}

        # Add MSE and Pearson to the best parameters
        best_params["mse"] = best_mse
        best_params["pearson"] = best_pearson
        best_params_with_id = best_params.copy()
        best_params_with_id["left_out_PatId"] = left_out_pat_id
        best_params.pop("mse", None)
        best_params.pop("pearson", None)

        best_params_list.append(best_params_with_id)

        if "max_depth" in best_params:
            best_params["max_depth"] = int(best_params["max_depth"])

        # Train model with the best parameters (except n_estimators)
        model.set_params(**best_params)
        model.fit(train_X, train_y)

        # Evaluate on the test set
        final_pred_y = model.predict(test_X)
        final_mse = mean_squared_error(test_y, final_pred_y)
        loo_mse_scores.append(final_mse)

    # Average MSE across all LOOCV splits
    average_mse = np.mean(loo_mse_scores)
    print("Average MSE:", average_mse)

    # Save the best parameters and left-out PatId to a CSV file
    best_params_df = pd.DataFrame(best_params_list)
    # Columns to exclude (keys from the space dictionary)
    exclude_columns = list(space.keys())
    # Add 'mse' and 'pearson' to exclude_columns if not already present
    if "mse" not in exclude_columns:
        exclude_columns.append("mse")
    if "pearson" not in exclude_columns:
        exclude_columns.append("pearson")
    filtered_initial_best_params_df = initial_best_params_df.drop(
        columns=exclude_columns, errors="ignore"
    )
    best_params_df = pd.merge(
        best_params_df, filtered_initial_best_params_df, on="left_out_Pat_Id"
    )
    best_params_df.to_csv(output_path, index=False)

    return average_mse, best_params_df


if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=RuntimeWarning
        )  # Replace RuntimeWarning with the specific category
        folder_path = "/home/georg-tirpitz/Documents/PD-MultiModal-Prediction"
        #folder_path = "/home/georg/Documents/Neuromodulation/PD-MultiModal-Prediction"
        decode = "/bdi_hparams"
        result_path = folder_path + "/results"
        target = "BDI_diff"
        IDcol = "Pat_ID"
        # Create the result_path + decode folder if it does not exist
        os.makedirs(result_path + decode, exist_ok=True)
        m_df = pd.read_excel(folder_path + "/data/bdi_df.xlsx")
        # print df size before removal
        # print(m_df.shape)

        # find max value of MoCA_pre
        # max_value = m_df["MoCA_sum_pre"].max()
        # remove the highest score values in MoCA_pre
        # m_df = m_df[~(m_df["MoCA_sum_pre"] == max_value)]
        # print(m_df.shape)

        # Choose all predictors except target & IDcols
        predictors = [x for x in m_df.columns if x not in [target, IDcol]]
        xgb1 = xgboost.XGBRegressor(
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.8,
            gamma=0,
            colsample_bytree=0.8,
            max_depth=4,
            random_state=44,
        )

        # have initial parametrs in a dictionary
        init_params = {
            "learning_rate": 0.05,
            "n_estimators": 3000,
            "min_child_weight": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "rmse",
            "reg_alpha": 0,
        }

        X = m_df[predictors].values
        y = m_df[target].values
        pat_ids = m_df["Pat_ID"].values  # Extracting PatId
        ITERATIONS = 300
        FIRST_PRUNING_PATH = result_path + decode + "/first_pruning.csv"

        # start with tree pruning
        initial_best_params_df = prune_trees(
            X, y, pat_ids, init_params, FIRST_PRUNING_PATH
        )

        #################

        # Load the previously saved best parameters
        initial_best_params_df = pd.read_csv(FIRST_PRUNING_PATH)
        # Define the space over which to search
        space1 = {
            "min_child_weight": hp.uniform("min_child_weight", 1, 12),
            #'max_depth': hp.choice('max_depth', np.arange(1, 10, dtype=int)),
            "max_depth": hp.uniformint("max_depth", 1, 10, q=1),
        }
        FIRST_BEST_PARAM = result_path + decode + "/first_best_param.csv"
        # initial_best_params_df = pd.read_csv('/home/enricoferrea/Documents/python_scripts/ddbm/example/res/'+decode+'/final_best_parameters_with_patid.csv')
        _, first_best_params_df = tune_nested_cross_validation(
            X,
            y,
            pat_ids,
            initial_best_params_df,
            xgb1,
            ITERATIONS,
            space1,
            objective,
            FIRST_BEST_PARAM,
        )

        ##### tune second parameter

        # Load the previously saved best parameters
        first_best_params_df = pd.read_csv(FIRST_BEST_PARAM)

        # Define the new space for hyperparameter search
        space2 = {
            "gamma": hp.uniform("gamma", 0.0, 1),
        }
        SECOND_BEST_PARAM = result_path + decode + "/second_best_param.csv"
        _, second_best_params_df = tune_nested_cross_validation(
            X,
            y,
            pat_ids,
            first_best_params_df,
            xgb1,
            ITERATIONS,
            space2,
            objective,
            SECOND_BEST_PARAM,
        )

        ##### tune third parameter

        # Load the previously saved best parameters
        second_best_params_df = pd.read_csv(SECOND_BEST_PARAM)

        # Define the new space for hyperparameter search
        space3 = {
            "subsample": hp.uniform("subsample", 0.3, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
        }
        THIRD_BEST_PARAM = result_path + decode + "/third_best_param.csv"
        _, third_best_params_df = tune_nested_cross_validation(
            X,
            y,
            pat_ids,
            second_best_params_df,
            xgb1,
            ITERATIONS,
            space3,
            objective,
            THIRD_BEST_PARAM,
        )

        third_best_params_df = pd.read_csv(THIRD_BEST_PARAM)

        # Define the new space for hyperparameter search
        space4 = {
            #'reg_alpha': hp.choice('reg_alpha', [0.0001,1e-4,1e-3, 1e-2, 0.1, 1, 10, 100]),
            "reg_alpha": hp.uniform("reg_alpha", 0.0, 2)
            #'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(100))
        }
        FORTH_BEST_PARAM = result_path + decode + "/forth_best_param.csv"
        _, forth_best_params_df = tune_nested_cross_validation(
            X,
            y,
            pat_ids,
            third_best_params_df,
            xgb1,
            ITERATIONS,
            space4,
            objective,
            FORTH_BEST_PARAM,
        )

        forth_best_params_df = pd.read_csv(FORTH_BEST_PARAM)

        # Lower learning rate and perform tree pruning again
        forth_best_params_df["learning_rate"] = (
            forth_best_params_df["learning_rate"] / 2
        )
        FINAL_BEST_PATH = result_path + decode + "/XGBoost_hparams.csv"
        final_best_params_df = prune_trees(
            X, y, pat_ids, forth_best_params_df, FINAL_BEST_PATH
        )

        print("ok")
