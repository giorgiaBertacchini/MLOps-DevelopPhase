from flask import Flask
import flask
from run import update_data
import pandas as pd

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
    #update_data(url_data)
    return "ok"