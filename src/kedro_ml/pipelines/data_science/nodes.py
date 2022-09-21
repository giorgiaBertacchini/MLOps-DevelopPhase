"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""
from kedro.io import *
import logging

from typing import Dict, Tuple
import yaml, os, json, pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training (60%), test (20%) and validation (20%) sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["Quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_size"], random_state=parameters["random_state"])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=parameters["val_size"], random_state=parameters["random_state"])

    return X_train, X_test, X_val, y_train, y_test, y_val


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Trains the model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = RandomForestRegressor(max_depth=2, random_state=42)
    regressor.fit(X_train, y_train)

    # Report training set score
    train_score = regressor.score(X_train, y_train) * 100

    logger = logging.getLogger(__name__)
    logger.info("Model has a accurancy of %.3f on train data.", train_score)
    
    return regressor


def evaluate_model(regressor: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_val: Valuate data of independent features.
        y_val: Valuate data for quality.

    Returns:
        Values from predict.
    """
    # scores_cross is the accuracy score which is normalized i.e between the value from 0-1, 
    # where 0 means none of the output were accurate and 1 means every prediction was accurate.
    #scores_cross = cross_val_score(regressor, X_val, y_val, cv=5, scoring='neg_root_mean_squared_error')
    scores_cross = cross_val_score(regressor, X_val, y_val, cv=5)

    y_pred = regressor.predict(X_val)
    # MAE to measure errors between the predicted value and the true value.
    mae = metrics.mean_absolute_error(y_val, y_pred)
    # MSE to average squared difference between the predicted value and the true value.
    mse = metrics.mean_squared_error(y_val, y_pred)
    # ME to capture the worst-case error between the predicted value and the true value.
    me = metrics.max_error(y_val, y_pred)
    
    logger = logging.getLogger(__name__)
    logger.info("Model has a accurancy of %.3f on validation data.", scores_cross.mean())
    return {"mean_score(accurancy)": scores_cross.mean(), "standard_deviation": scores_cross.std(), "mean_absolute_error": mae, "mean_squared_error": mse, "max_error": me}


def testing_model(regressor: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> RandomForestRegressor:
    """Diagnose the source issue when they fail. Testing code, data, models.
        Unit testing, integration testing.
        Performing the final “Model Acceptance Test” by using the hold backtest dataset to estimate the generalization error
        compare the model with its previous version

     Args:
        regressor: Trained model.
        X_test: Test data of independent features.
        y_test: Test data for quality.

    Returns:
        Values from testing versions.
    """
    test_scores_cross = cross_val_score(regressor, X_test, y_test, cv=5)
    test_standard_deviation = test_scores_cross.std()
    y_pred = regressor.predict(X_test)
    test_mae = metrics.mean_absolute_error(y_test, y_pred)
    test_mse = metrics.mean_squared_error(y_test, y_pred)
    test_me = metrics.max_error(y_test, y_pred)

    # See older versions data
    change_version = 'current version'
    actual_mean = (test_standard_deviation + test_mae + test_mse + test_me)/4    
    versions_differnce = {'current version': actual_mean}

    for root, dirnames, filenames in os.walk(os.path.join("files", os.getcwd(),'data','09_tracking','metrics.json')):
        for dirname in dirnames:
            with open(os.path.join("files", os.getcwd(),'data','09_tracking','metrics.json', dirname , 'metrics.json'), "r") as f:
                old_data = json.load(f)
                old_mean = old_data['standard_deviation']
                old_mean += old_data['mean_absolute_error']
                old_mean += old_data['mean_squared_error']
                old_mean += old_data['max_error']
                old_mean /= 4
                versions_differnce[dirname] = old_mean

                if (old_mean < actual_mean):
                    actual_mean = old_mean
                    change_version = dirname
                
    if (change_version != 'current version'):
        print("ATTENTION!!!\nCHANGE MODEL VERSION INTO: ", change_version)
        regressor=pickle.load(open(os.path.join("files", os.getcwd(),'data','06_models','regressor.pickle', change_version , 'regressor.pickle'),"rb"))    
    versions_differnce["best_version"] = change_version

    logger = logging.getLogger(__name__)
    logger.info("Best model version is %s.", change_version)

    return versions_differnce


def plot_feature_importance(regressor: RandomForestRegressor, data: pd.DataFrame) -> int:
    """Create plot of feature importance and save into png

     Args:
        regressor: Trained model.
        data: Data containing features and target.
    """
    # Calculate feature importance in random forest
    importances = regressor.feature_importances_
    labels = data.columns
    feature_data = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
    feature_data = feature_data.sort_values(by='importance', ascending=False,)
    
    # image formatting
    axis_fs = 16 #fontsize
    title_fs = 22 #fontsize
    sns.set(style="whitegrid")

    ax = sns.barplot(x="importance", y="feature", data=feature_data)
    ax.set_xlabel('Importance',fontsize = axis_fs) 
    ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
    ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

    plt.tight_layout()
    plt.savefig("feature_importance.png",dpi=120) 
    plt.close()

    return 0


def plot_residuals(regressor: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> int:
    """Create plot of residuals and save into png
    A residual is a measure of how far away a point is vertically from the regression line. 
    Simply, it is the error between a predicted value and the observed actual value.

     Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test) + np.random.normal(0,0.25,len(y_test))
    y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
    res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

    axis_fs = 16 #fontsize
    title_fs = 22 #fontsize

    ax = sns.scatterplot(x="true", y="pred",data=res_df)
    ax.set_aspect('equal')
    ax.set_xlabel('True activities quality',fontsize = axis_fs) 
    ax.set_ylabel('Predicted activities quality', fontsize = axis_fs)#ylabel
    ax.set_title('Residuals', fontsize = title_fs)

    # Make it pretty- square aspect ratio
    #ax.plot([1, 10], [1, 10], 'black', linewidth=1)
    plt.ylim((3,7))
    plt.xlim((-2,12))

    plt.tight_layout()
    plt.savefig("residuals.png",dpi=120) 

    return 0