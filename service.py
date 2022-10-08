import bentoml
import numpy as np
import pandas as pd
from bentoml.io import NumpyNdarray, PandasDataFrame

from pydantic import BaseModel

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