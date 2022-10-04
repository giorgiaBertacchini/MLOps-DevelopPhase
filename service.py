import bentoml
import numpy as np
import pandas as pd
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON
from pydantic import BaseModel

class Activity(BaseModel):
    Distance: float = 26.91
    AverageSpeed: float = 11.08
    CaloriesBurned: float = 1266
    Climb: float = 98
    AverageHeartrate: float = 121

model_runner = bentoml.mlflow.get('my_model:latest').to_runner()

svc = bentoml.Service('activities_model', runners=[ model_runner ])

@svc.api(
    input=PandasDataFrame(), 
    output=NumpyNdarray())
def predict(input_data: pd.DataFrame):
    """
    Example of data input: 
    [{"Distance (km)": 26.91, "Average Speed (km/h)": 11.08, "Calories Burned": 1266, "Climb (m)": 98, "Average Heart rate (tpm)":121}]
    """
    return model_runner.predict.run(input_data)