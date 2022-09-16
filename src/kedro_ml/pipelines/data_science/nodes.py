"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, max_error

import matplotlib.pyplot as plt
import seaborn as sns


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
    

def evaluate_model(regressor: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.

    Returns:
        Values from predict.
    """
    score = regressor.score(X_test, y_test) * 100
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    
    logger = logging.getLogger(__name__)
    logger.info("Model has a accurancy of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}


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
    axis_fs = 18 #fontsize
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