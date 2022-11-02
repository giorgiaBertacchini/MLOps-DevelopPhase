from flask import Flask
import flask
import pandas as pd
import os
import yaml
import json

from run import update_data, run_retrain, bentoml_set
from src.kedro_ml.pipelines.data_processing.nodes import create_model_input_table


app = Flask(__name__)

@app.get("/dvc_file")
def get_dvc_file():
    str = open('data/01_raw/DATA.csv.dvc', 'r').read()
    return str

@app.route("/load_new_data", methods=["POST"])
def load_new_data():
    item = flask.request.json
    df_data = pd.DataFrame.from_dict(item)
    print(df_data)
    update_data(df_data)
    run_retrain()
    return "ok"

@app.get("/retrain")
def retrain():
    run_retrain()
    return "ok"

@app.get("/bento")
def bento():
    bentoml_set()
    return "ok"

@app.get("/header")
def header():    
    with open(os.path.join("conf", "base", "parameters", "data_processing.yml"), "r") as f:
        configuration = yaml.safe_load(f)    
    with open('config.json', 'w') as json_file:
        json.dump(configuration, json_file)    
    output = json.load(open('config.json'))

    return output["table_columns"]