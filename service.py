import bentoml
import numpy as np
import pandas as pd

from bentoml.io import NumpyNdarray, PandasDataFrame
from src.kedro_ml.pipelines.data_processing.nodes import preprocess_activities, create_model_input_table

from pydantic import BaseModel


model_runner = bentoml.mlflow.get('my_model:latest').to_runner()
#title = '%s' %bentoml.mlflow.get('my_model:latest').tag
#t = title.replace(":", "-")

#svc = bentoml.Service(t, runners=[ model_runner ])
svc = bentoml.Service('activities_model', runners=[ model_runner ])

@svc.api(
    input=PandasDataFrame(),
    output=NumpyNdarray())
def predict(input_data: pd.DataFrame):
    """
    Example of data input: 

    [{"Distance (km)": 26.91, "Average Speed (km/h)": 11.08, "Calories Burned": 1266, "Climb (m)": 98, "Average Heart rate (tpm)":121}]
    
    or other, because process input data:

    [{"Distance (km)": 26.91, "Example": 43, "Type": "Running", "Average Speed (km/h)": 11.08, "Activity ID": 5, "Date": 6, "Duration": 6, "Calories Burned": 1266, "Climb (m)": 98, "Average Heart rate (tpm)":121}]
    """

    input_data, dict_col = preprocess_activities(input_data)
    print("Terminate the preprocessing of input data.")

    parameters = {
        "header": [
            "Distance (km)",
            "Average Speed (km/h)" ,
            "Calories Burned",
            "Climb (m)",
            "Average Heart rate (tpm)",
            "Quality"]
    }
    input_data = create_model_input_table(input_data, parameters)
    print("Terminate the dropping of useless columns.")
    print("Start the prediction...")
    return model_runner.predict.run(input_data)