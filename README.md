<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#mlops">MLOps</a>
      <ul>        
        <li><a href="#three-level">Three Level</a></li>  
        <li><a href="#principles">Principles</a></li>
      </ul>
    </li>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>        
        <li><a href="#built-with">Built With</a></li>  
        <li><a href="#schema">Schema</a></li>     
      </ul>
    </li>
    <li>
      <a href="#how-it-works">How it works</a>
      <ul>
        <li><a href="#workflow-orchestration">Workflow orchestration</a></li>
        <ul>
          <li><a href="#structure">Structure</a></li>          
          <li><a href="#key-elements">Key elements</a></li>
          <ul>
            <li><a href="#data-catalog">Data Catalog</a></li>
            <li><a href="#node">Node</a></li>
            <li><a href="#pipeline">Pipeline</a></li>
          </ul>
          <li><a href="#kedro-viz">Kedro-Viz</a></li>
        </ul>
        <li><a href="#data-versioning">Data versioning</a></li>
        <ul>
          <li><a href="#structure-dvc">Structure DVC</a></li>          
          <li><a href="#key-elements-dvc">Key elements DVC</a></li>
          <ul>
            <li><a href="#file-dvc">file .dvc</a></li>
            <li><a href="#set-remote-storage">Set remote storage</a></li>
          </ul>
          <li><a href="#commands">Commands</a></li>
          <li><a href="#more">More</a></li>
        </ul>
        <li><a href="#data-analysis-and-manipulation">Data analysis and manipulation</a></li>
        <ul>
          <li><a href="#key-elements-pandas">Key elements Pandas</a></li>
          <li><a href="#commands-pandas">Commands pandas</a></li>
          <li><a href="#code">Code</a></li>
        </ul>
        <li><a href="#model-training">Model training</a></li>
        <ul>
          <li><a href="#structure-sklearn">Structure sklearn</a></li>
          <li><a href="#key-elements-sklearn">Key elements sklearn</a></li>
          <ul>
            <li><a href="#splitting-dataset">Splitting dataset</a></li>
            <li><a href="#estimator-and-fitting-model">Estimator and fitting model</a></li>
            <li><a href="#model-evaluation">Model evaluation</a></li>
            <li><a href="#metrics">Metrics</a></li>
          </ul>
        </ul>
        <li><a href="#experimentation-management">Experimentation management</a></li>   
        <li><a href="#model-packaging-and-serving">Model packaging and serving</a></li>
        <li><a href="#deploying-pipeline">Deploying pipeline</a></li>
      </ul>
    </li>
    <li>
      <a href="#bridge">Bridge</a>
      <ul>
        <li><a href="#interactions-and-communication">Interactions And Communication</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#acknowledgments">Acknowledgments</a>
    </li>
  </ol>
</details>


# MLOps
MLOps is designed to facilitate the installation of ML software in a production environment. 
Machine Learning Operations (MLOps).

The term MLOps is defined as
> “the extension of the DevOps methodology to include Machine Learning and Data Science assets as first-class citizens within the DevOps ecology”

> “the ability to apply DevOps principles to Machine Learning applications”

by [MLOps SIG](https://github.com/cdfoundation/sig-mlops/blob/main/roadmap/2020/MLOpsRoadmap2020.md)

## Three Level
<div align="center">
  <img width="600" alt="kedro logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/cycle.webp">
</div>

MLOps combine machine learning model, application development and operations.

## Principles


# About The Project
This project puts into practice the steps of MLOps and it is complete using the Monitoring step at link [https://my_observability_project.com](https://github.com/giorgiaBertacchini/MLOps/tree/main/MLOps%20-observability).

## Built With

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/tools.png)


## Schema

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/schema.png)


# How it works

## Workflow orchestration

<div align="center">
  <img width="270" alt="kedro logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/kedro_logo.png">
</div>

As Workflow orchestration is used [Kedro](https://kedro.readthedocs.io/en/stable/), an open-source Python framework for creating reproducible, maintainable and modular data science code.
Kedro is a template for new data engineering and data science projects. This tool provide to organize all MLOps steps in a well-defined pipeline.

### Structure

When you installing and initialize a Kedro project, with the commands
```
pip install kedro
```
```
kedro new
```
after are automatic creted needed folders and files. The most important of these are as follows:
* `conf/base/`
  * `catalog.yml` file
  * `parameters/` folder, with the parameters for each pipelines
* `data/` folder, with all data sets and other output, separate for additional named folders, as: `01_raw`, `02_intermediate`, `06_models`, `08_reporting`
* `logs/` folder
* `src/kedro_ml/`
  * `pipeline_registry.py` file, with the pipeline names
  * `pipelines/` folder, with specified each pipelines
* `src/requirements.txt` file, with needed models

### Key elements
* **Data Catalog**
  * It makes the datasets declarative, rather than imperative. So all the informations related to a dataset are highly organized.
* **Node**
  * It is a Python function that accepts input and optionally provides outputs.
* **Pipeline**
  * It is a collection of nodes. It create the Kedro DAG (Directed acyclic graph).

#### Data Catalog
In the project Data Catalog is implemented in `conf/base/catalog.yml`.
Here is define each type, destination filepath and if is versioned about the data sets, in output from the nodes.
Here you can define all your data sets by using simple YAML syntax.
Documentation for this file format can be found at link: [Data Catalog Doc](https://kedro.readthedocs.io/en/stable/data/data_catalog.html)

Examples:

``` yaml
model_input_table:
  type: pandas.CSVDataSet
  filepath: data/04_feature/model_input_table.csv
  versioned: true
```

``` yaml
regressor:
  type: pickle.PickleDataSet
  filepath: data/06_models/regressor.pickle
  versioned: true
```

``` yaml
metrics:
  type: tracking.MetricsDataSet
  filepath: data/09_tracking/metrics.json
  versioned: true
```

The next is more complex, so this plot is showed also in the Kedro interactive visualization platform [Kedro-Viz](https://kedro.readthedocs.io/en/0.17.4/03_tutorial/06_visualise_pipeline.html).

``` yaml
plot_feature_importance_img:
  type: plotly.PlotlyDataSet
  filepath: data/08_reporting/plot_feature_importance_img.json
  versioned: true
  plotly_args:
    type: bar
    fig:
      x: importance
      y: feature
      orientation: h
    layout:
      xaxis_title: importance
      yaxis_title: feature
      title: Importance for feature
```

As you can see:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/plot_in_kedro_viz.png)


#### Node

The node functions are write in `nodes.py` file of the respective pipeline folder.

``` python
def preprocess_activities(activities: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Preprocesses the data for activities.

    Args:
        activities: Raw data.
    Returns:
        Preprocessed data and JSON file.
    """
    activities = _validation(activities)
    activities = _wrangling(activities)
    
    return activities, {"columns": activities.columns.tolist(), "data_type": "activities"}
```

#### Pipeline

The pipeline definition is write in the `pipeline.py` file in the respective pipeline folder.

From example we can see the nodes, with their function, input, output, name. These must match with the nodes implementation (see previously) and the 'name's are repeated in the Data Catalog (see previously).

``` python
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_activities,
                inputs="activities",
                outputs=["preprocessed_activities", "activities_columns"],
                name="preprocess_activities_node",
            ),
            node(
                func=exploration_activities,
                inputs="activities",
                outputs="exploration_activities",
                name="exploration_activities_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_activities", "params:table_columns"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
```

### Kedro-Viz

It is a interactive visualization of the entire pipeline. It is a tool that can be very helpful for explaining what you're doing to people. 

To see the kedro ui:

```
kedro viz
```

To see the kedro ui go to the `270.0.0.1:4141` browser page.

Example:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/kedro-viz.png)
From here we can also see and compare the experiments, that are the versions created runned the kedro project.

Example:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/kedro_experiments.png)
![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/kedro_experiments_0.png)

## Data versioning

<div align="center">
  <img width="280" alt="dvc logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/dvc_logo.png">
</div>

Ad data versioning managenet tool is used [DVC](https://dvc.org/doc). This provide to handle large files, data sets, machine learning models, and metrics.

### Structure DVC

When you installing and initialize dvc setting:

```
pip install dvc
```
```
dvc init
```

is created `.dvc` folder, where the most important file is:

* `config`, with the url to remote destination, where save the data.

### Key elements DVC

#### file .dvc

DVC stores information about the added file in a special `.dvc` file named `data/data.xml.dvc`, this metadata file is a placeholder for the original data.

In this case we would save the original datasets, so `data/01_raw/DATA.csv`.

#### Set remote storage

In this project is used own Google Drive, the code at path `.dvc/config` is:

``` yaml
[core]
    remote = storage
    autostage = true
['remote "storage"']
    url = gdrive://1LMUFVzJn4CNaqVbMsVGZazii4Mdxsanj
```

### Commands

To update data in remote storage, we use the next commands:

```
dvc add data/01_raw/DATA.csv
```
This for update or create the file `.dvc`.
We can see in folder `.dvc/cache/` in the corrispective folder there are the data to save.

```
dvc push data/01_raw/DATA.csv
```

Now in Google Drive appear a new folder with the new data version.

Note: If the data to save is not changed (and also saved), the file `.dvc` is not update and not appear another folder in Google Drive.

### More
DVC is a more usefull tool. As Data manager, we can also create data pipeline and specify the metrics, parameters and plots. DVC is also a Experiment manager, providing comparison and visualize experiment results. 

For more: LINK MIO FILE Part 3


## Data analysis and manipulation

<div align="center">
  <img width="300" alt="pandas logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/pandas_logo.png">
</div>

[pandas](https://pandas.pydata.org/docs/) is a Python package providing fast, flexible, and expressive data structures.

> it has the broader goal of becoming the most powerful and flexible open source data analysis/manipulation tool available in any language. It is already well on its way toward this goal.

by [pandas.pydata.org](https://pandas.pydata.org/docs/getting_started/overview.html)

### Key elements pandas
pandas will help you to explore, clean, and process your data. In pandas, a data table is called a DataFrame. If it is 1-D is called Series.

### Commands pandas
It require the installation, also with conda:
```
pip install pandas
```

### Code
In the code, pandas Dataframes are used in nodes as funciontion input and output. For example:

``` python
def preprocess_activities(activities: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]
```

``` python
def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple
```

## Model training

<div align="center">
  <img width="260" alt="scikitlearn logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/scikitlearn_logo.png">
</div>

[scikit-learn](https://scikit-learn.org/stable/getting_started.html), or sklearn, is an open source machine learning library.

It is a simple and efficient tools for predictive data analysis and it also provides various tools for model fitting, data preprocessing, model selection, model evaluation.

### Structure sklearn

For change the parameters, update `conf/base/parameters/data_science.yml` file with settings use during model management.

``` yaml
model_options:
  test_size: 0.2
  val_size: 0.25
  random_state: 42
  max_depth: 2
  features:
    - Distance (km)
    - Average Speed (km/h)
    - Calories Burned
    - Climb (m)
    - Average Heart rate (tpm)
```


For passing as input this parameters to nodes, this specification is write in file `src/kedro_ml/pipelines/data_science/pipeline.py`. For example:

``` python
node(
  func=split_data,
  inputs=["model_input_table", "params:model_options"],
  name="split_data_node",
  outputs=["X_train", "X_test", "X_val", "y_train", "y_test", "y_val"],
),
```

### Key elements sklearn

Note: the next codes are take from `src/kedro_ml/pipelines/data_science/nodes.py`.

#### Splitting dataset
``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_size"], random_state=parameters["random_state"])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=parameters["val_size"], random_state=parameters["random_state"])

```

#### Estimator and fitting model

``` python
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(max_depth=parameters["max_depth"], random_state=parameters["random_state"])
regressor.fit(X_train, y_train)
```

#### Model evaluation
``` python
from sklearn.model_selection import GridSearchCV

# define search space
space = dict()
space['max_depth'] = [1,2,3]
space['random_state'] = [41,42,43,44]

# define search
search = GridSearchCV(regressor, space, scoring='neg_mean_absolute_error')
# execute search
result = search.fit(X_train, y_train)
```

#### Metrics
``` python
from sklearn import metrics

# MAE to measure errors between the predicted value and the true value.
mae = metrics.mean_absolute_error(y_val, y_pred)
# MSE to average squared difference between the predicted value and the true value.
mse = metrics.mean_squared_error(y_val, y_pred)
# ME to capture the worst-case error between the predicted value and the true value.
me = metrics.max_error(y_val, y_pred)
```

## Experimentation management

<div align="center">
  <img width="280" alt="MLflow logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/mlflow_logo.png">
</div>

[MLflow](https://mlflow.org/docs/latest/index.html) is an open source platform for managing the end-to-end machine learning lifecycle.

MLFlow can be very helpful in terms of tracking metrics over time. We can visualize that and communicate what is the progress over time. MLFlow centralize all of these metrics and also the models generates.

### MLflow and Kedro
MLflow and Kedro are tools complementary and not conflicting:
* Kedro is the foundation of your data science and data engineering project
* MLflow create that centralized repository of metrics and progress over time

<div align="center">
  <img width="550" alt="mlflow and kedro" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/mlflow+kedro.png">
</div>

### Guidelines mlflow

#### Installation
It needs to be installed

```
pip install mlflow
```

#### Logging File

``` python
mlflow.log_artifact(local_path=os.path.join("data", "01_raw", "DATA.csv"))
```

#### Logging Model

``` python
mlflow.sklearn.log_model(sk_model=regressor, artifact_path="model")
```
``` python
mlflow.log_artifact(local_path=os.path.join("data", "04_feature", "model_input_table.csv", dirname ,"model_input_table.csv"))
mlflow.log_artifact(local_path=os.path.join("data", "08_reporting", "feature_importance.png"))
mlflow.log_artifact(local_path=os.path.join("data", "08_reporting", "residuals.png")) 
```

#### Logging key-value param

``` python
mlflow.log_param('test_size', parameters["test_size"])
mlflow.log_param('val_size', parameters["val_size"])
mlflow.log_param('max_depth', parameters["max_depth"])
mlflow.log_param('random_state', parameters["random_state"])
```

#### Logging key-value metric

``` python
mlflow.log_metric("accuracy", score)
mlflow.log_metric("mean_absolute_erro", mae)
mlflow.log_metric("mean_squared_error", mse)
mlflow.log_metric("max_error", me)
```

#### Setting key-value tag 

``` python
mlflow.set_tag("Model Type", "Random Forest")
```

### Structure

For specify more options is used `MLproject` file:

``` yaml
name: kedro mlflow
conda_env: conda.yaml
entry_points:
  main:
    command: "kedro run"
```
So when run mlflow is execute the command `kedro run`.

#### Commands mlflow

### Before activate conda environment

Need Python version 3.7. Using conda:

```
conda create -n env_name python=3.7
```

```
conda activate env_name
```

### How to run mlflow project

You can run mlflow project with:

```
mlflow run . --experiment-name activities-example
```

### How to run mlflow project in Windows

You can run mlflow project with:

```
mlflow run . --experiment-name activities-example --no-conda
```

### How to vizualize mlflow project

You can run ui as follows:

```
mlflow ui
```

To see the mlflow ui go to the `270.0.0.1:5000` browser page.

Example:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/mlflow-ui.png)

From this page we can select a single experiment and see more information about it. Example:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/mlflow_experiment.png)

## Model packaging and serving

<div align="center">
  <img width="350" alt="BentoML logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/bentoml_logo.png">
</div>

[BentoML](https://docs.bentoml.org/en/latest/), on the other hand, focuses on ML in production. By design, BentoML is agnostic to the experimentation platform and the model development environment. BentoML only focuses on serving and deploying trained models.

### BentoML and MLflow

MLFlow focuses on loading and running a model, while BentoML provides an abstraction to build a prediction service, which includes the necessary pre-processing and post-processing logic in addition to the model itself.

BentoML is more feature-rich in terms of serving, it supports many essential model serving features that are missing in MLFlow, including multi-model inference, API server dockerization, built-in Prometheus metrics endpoint and many more.

### Structure bentoml

BentoML stores all packaged model files under the `~/bentoml/repository/{service_name}/{service_version}` directory by default. The BentoML packaged model format contains all the code, files, and configs required to run and deploy the model.

#### Configuration

`bentofile.yaml`

``` yaml
service: "service:svc"
include:
  - "service.py"
  - "src/kedro_ml/pipelines/data_processing/nodes.py"
conda:
  environment_yml: "./conda.yaml"
docker:
  env:
  - BENTOML_PORT=3005
```

### Key elements bentoml

The BentoML basic steps are two:
* save the machine learning model
* create a prediction service

#### Save Model

In `src/pipeline/data_science/nodes.py` file, to save the model is used to help MLflow, included in BentoML module.

``` python
import bentoml

bentoml.mlflow.import_model("my_model", model_uri= os.path.join(os.getcwd(), 'my_model', dirname))
```

#### Prediction Service

`service.py` file

``` python
def predict(input_data: pd.DataFrame):
  with open(os.path.join("conf", "base", "parameters", "data_science.yml"), "r") as f:
    configuration = yaml.safe_load(f)    
  with open('temp.json', 'w') as json_file:
    json.dump(configuration, json_file)    
  output = json.load(open('temp.json'))
  
  parameters = {"header":output["model_options"]["features"]}
  input_data = create_model_input_table(input_data, parameters)
  input_data, dict_col = preprocess_activities(input_data)
  
  print("Start the prediction...")
  return model_runner.predict.run(input_data)
```

#### Deploy Bento

Bento is a file archive with all the source code, models, data files and dependency configurations required for running a user-defined bentoml.Service, packaged into a standardized format. Bento is crete with the command:

```
bentoml build
```

The three most common deployment options with BentoML are:
* 🐳 Generate container images from Bento for custom docker deployment
* 🦄️ Yatai: Model Deployment at scale on Kubernetes
* 🚀 bentoctl: Fast model deployment on any cloud platform

We containerize Bentos as Docker images allows users to easily distribute and deploy bentos. With the command:

```
bentoml containerize activities_model:latest
```

### Guidelines bentoml

#### Installation

BentoML requires installation:

```
pip install bentoml
```

#### View

To see all bento models:

```
bentoml models list
```

To see more about a bento model:

```
bentoml models get <name_model>:<number_version>
```

#### Build a bento model

```
bentoml build
```

#### Start Bento model in production

```
bentoml serve <name_model>:latest --production
```

#### Run Bento Server

If use Windows run this without `-- reload`:

```
bentoml serve service:svc --reload
```

or more general:

```
bentoml serve
```

After you can open a web page `127.0.0.1:3000` to have a model serving. Example:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/bentoml.png)


## Deploying pipeline

<div align="center">
  <h1>kedro-docker</h1>
</div>

[kedro-docker](https://github.com/quantumblacklabs/kedro-docker) is a plugin to create a Docker image and run kedro project in a Docker environment.

### Structure docker

`Dockerfile` file

For set the number port:

``` python
EXPOSE 3030
```

For set the command:

``` python
#CMD ["kedro", "run"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=3030"]
```

### Guidelines

To install, run:
```
pip install kedro-docker
```

For create docker image of Kedro pipeline.

```
kedro docker build --image pipeline-ml
```

To run the docker model:

```
docker run <name_model>
```

or to production:

```
docker run <name_model> serve --production
```

Or to run the project in a Docker environment:

```
kedro docker run --image <image-name>
```


# Bridge

## Interactions And Communication

To interact with pipeline and all step, there is run.py that answer to command line. The avaiable command line are:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/run.png)


For communication between this project and observability step, there is a flask application with avaible API:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/app.png)


# Getting Started

To run flask application: 

```
flask run --host=0.0.0.0 --port=3030
```

## Prerequisites

## Installation


# Usage

# Acknowledgments

* [ml-ops.org](https://ml-ops.org/)
* [neptune.ai](https://neptune.ai/blog/mlops)
* [mlebook.com](http://www.mlebook.com/wiki/doku.php)
* Book "Introducing MLOps How to Scale Machine Learning in the Enterprise (Mark Treveil, Nicolas Omont, Clément Stenac etc.)"
