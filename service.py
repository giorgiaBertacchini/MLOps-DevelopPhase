import bentoml
import pandas as pd
import json
import os
import yaml

from bentoml.io import NumpyNdarray, PandasDataFrame
from src.kedro_ml.pipelines.data_processing.nodes import preprocess_activities, create_model_input_table


model_runner = bentoml.mlflow.get('my_model:latest').to_runner()
#title = '%s' %bentoml.mlflow.get('activities_model:latest').tag

#stri = bentoml.mlflow.get('my_model:latest').path
#with open('versione_ok.txt', 'w') as f:
#    f.write("what ", title)

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

    #parameters = {
    #    "header": [
    #        "Distance (km)",
    #        "Average Speed (km/h)" ,
    #        "Calories Burned",
    #        "Climb (m)",
    #        "Average Heart rate (tpm)",
    #    ]
    #}

    with open(os.path.join("conf", "base", "parameters", "data_science.yml"), "r") as f:
        configuration = yaml.safe_load(f)    
    with open('temp.json', 'w') as json_file:
        json.dump(configuration, json_file)    
    output = json.load(open('temp.json'))

    #print(output["model_options"]["features"])
    parameters = {"header":output["model_options"]["features"]}
    
    input_data = create_model_input_table(input_data, parameters)

    input_data, dict_col = preprocess_activities(input_data)
    
    print("Start the prediction...")
    return model_runner.predict.run(input_data)