"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""

import pandas as pd
from typing import Tuple, Dict
from kedro.extras.datasets.pandas import CSVDataSet

def _validation(apps: pd.DataFrame) -> pd.DataFrame:
    # Check data Format
    if 'Date' in apps.columns:
        apps['Date'] = pd.to_datetime(apps['Date'])
    # Check value ranges
    for x in apps.index:
        if apps.loc[x, "Distance (km)"] > 30:
            apps.loc[x, "Distance (km)"] = 30
        if apps.loc[x, "Average Speed (km/h)"] > 60:
            apps.loc[x, "Average Speed (km/h)"] = 60
        if apps.loc[x, "Average Heart rate (tpm)"] < 60:
            apps.loc[x, "Average Heart rate (tpm)"] = 60
    apps.drop_duplicates(inplace = True)
    return apps

def _wrangling(apps: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates
    apps.drop_duplicates(inplace = True)

    # Calculate the MEAN, and replace any empty values with it
    x = apps["Average Heart rate (tpm)"].mean()
    apps["Average Heart rate (tpm)"].fillna(x, inplace = True)

    # Clean rows that contain empty cells
    apps.dropna(inplace = True)

    # Rename 'Other' type to 'Walking'
    if 'Type' in apps.columns:
        apps['Type'] = apps['Type'].str.replace('Other', 'Walking')

    return apps


def preprocess_activities(activities: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Preprocesses the data for activities.

    Args:
        activities: Raw data.
    Returns:
        Preprocessed data.
    """
    activities = _validation(activities)
    activities = _wrangling(activities)
    
    return activities, {"columns": activities.columns.tolist(), "data_type": "activities"}


def exploration_activities(activities: pd.DataFrame) -> Dict[str, float]:
    """exploration the data for activities.

    Args:
        activities: Raw data.
    Returns:
        File JSON.
    """
    totalNumber = activities.size
    maxDistance = activities["Distance (km)"].max()
    meanAverageSpeed = activities["Average Speed (km/h)"].mean()
    minAverageHeartRate = activities["Average Heart rate (tpm)"].min()
    
    return {"total number of values":  totalNumber, "max distance": maxDistance, "mean average speed": meanAverageSpeed, "min average heart rate": minAverageHeartRate}


def create_model_input_table(activities: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        activities: Preprocessed data for activities.
        reviews: Raw data for reviews.
    Returns:
        model input table.

    """
    
    # Delete unnecessary columns
    for column in activities.columns:
        if column not in parameters["header"]:
            activities.drop(column, axis=1, inplace=True)

    return activities