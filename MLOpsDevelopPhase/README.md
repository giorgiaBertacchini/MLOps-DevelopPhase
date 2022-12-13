<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img width="200" alt="logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/logo.png">
  <h1 align="center">MLOps</h1>
  <h3 align="center">To automate and encourage machine learning in enterprises!</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#mlops-notion">MLOps Notion</a>
      <ul>        
        <li><a href="#three-level">Three Level</a></li>
        <ul> 
          <li><a href="#data-engineering">Data Engineering</a></li>
          <li><a href="#model-engineering">Model Engineering</a></li>
          <li><a href="#model-deployment">Model Deployment</a></li>
        </ul>
        <li><a href="#mlops-people">MLOps People</a></li>
        <li><a href="#principles">Principles</a></li>
        <ul> 
          <li><a href="#iterative-incremental-process">Iterative-Incremental Process</a></li>
          <li><a href="#automation">Automation</a></li>
          <li><a href="#continuous-deployment">Continuous Deployment</a></li>
          <li><a href="#versioning">Versioning</a></li>
          <li><a href="#reproducibility">Reproducibility</a></li>
          <li><a href="#experiments-tracking">Experiments Tracking</a></li>
          <li><a href="#ml-based-software-delivery-metrics">ML-based Software Delivery Metrics</a></li>
        </ul>
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
        <li><a href="#01-workflow-orchestration">01 Workflow orchestration</a></li>
        <ul>
          <li><a href="#01-structure">01 Structure</a></li>          
          <li><a href="#01-key-elements">01 Key Elements</a></li>
          <ul>
            <li><a href="#data-catalog">Data Catalog</a></li>
            <li><a href="#node">Node</a></li>
            <li><a href="#pipeline">Pipeline</a></li>
          </ul>
          <li><a href="#01-guidelines">01 Guidelines</a></li>
        </ul>
        <li><a href="#02-data-versioning">02 Data versioning</a></li>
        <ul>
          <li><a href="#02-structure">02 Structure</a></li>          
          <li><a href="#02-key-elements">02 Key Elements</a></li>
          <ul>
            <li><a href="#file-dvc">file .dvc</a></li>
            <li><a href="#set-remote-storage">Set remote storage</a></li>
          </ul>
          <li><a href="#02-guidelines">02 Guidelines</a></li>
          <li><a href="#02-more">02 More</a></li>
        </ul>
        <li><a href="#03-data-analysis-and-manipulation">03 Data analysis and manipulation</a></li>
        <ul>
          <li><a href="#03-key-elements">03 Key Elements</a></li>
          <li><a href="#03-guidelines">03 Guidelines</a></li>
        </ul>
        <li><a href="#04-model-training">04 Model training</a></li>
        <ul>
          <li><a href="#04-structure">04 Structure</a></li>
          <li><a href="#04-key-elements">04 Key Elements</a></li>
          <ul>
            <li><a href="#splitting-dataset">Splitting dataset</a></li>
            <li><a href="#estimator-and-fitting-model">Estimator and fitting model</a></li>
            <li><a href="#model-evaluation">Model evaluation</a></li>
            <li><a href="#metrics">Metrics</a></li>
          </ul>
        </ul>
        <li><a href="#05-experimentation-management">05 Experimentation management</a></li>   
        <ul>
          <li><a href="#05-collaboration">05 Collaboration</a></li>
          <li><a href="#05-structure">05 Structure</a></li>
          <li><a href="#05-guidelines">05 Guidelines</a></li>
        </ul>      
        <li><a href="#06-model-packaging-and-serving">06 Model packaging and serving</a></li>
        <ul>
          <li><a href="#06-collaboration">06 Collaboration</a></li>          
          <li><a href="#06-structure">06 Structure</a></li>
          <li><a href="#06-key-elements">06 Key Elements</a></li>
          <ul>
            <li><a href="#save-model">Save Model</a></li>          
            <li><a href="#prediction-service">Prediction Service</a></li>
            <li><a href="#deploy-bento">Deploy Bento</a></li>
          </ul>  
          <li><a href="#06-guidelines">06 Guidelines</a></li>
        </ul>  
        <li><a href="#07-deploying-pipeline">07 Deploying pipeline</a></li>
        <ul>
          <li><a href="#07-structure">07 Structure</a></li>
          <li><a href="#07-guidelines">07 Guidelines</a></li>
        </ul>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#kedro-platform">Kedro Platform</a></li>
        <li><a href="#mlflow-platform">MLflow Platform</a></li>
        <li><a href="#bentoml-platform">BentoML Platform</a></li>
      </ul>
    </li>
    <li>
      <a href="#acknowledgments">Acknowledgments</a>
    </li>
  </ol>
</details>


# MLOps Notion
MLOps is designed to facilitate the installation of ML software in a production environment. 
Machine Learning Operations (MLOps).

The term MLOps is defined as
> “the extension of the DevOps methodology to include Machine Learning and Data Science assets as first-class citizens within the DevOps ecology”

> “the ability to apply DevOps principles to Machine Learning applications”

by [MLOps SIG](https://github.com/cdfoundation/sig-mlops/blob/main/roadmap/2020/MLOpsRoadmap2020.md)

## Three Level
<div align="center">
  <img width="600" alt="kedro logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/cycle.webp">
</div>

MLOps combine machine learning model, application development and operations.

MLOps is the result by ModelOps, DataOps and DevOps.

### Data Engineering
It is the step to acquire and prepare the data to be analyzed.
Typically, data is being integrated from various resources and has different formats. Collecting good data sets has a huge impact on the quality and performance of the ML model.
Therefore, the data, which has been used for training of the ML model, indirectly influence the overall performance of the production system.

Data engineering pipeline:
1. **Data Ingestion**, collecting data by using various frameworks and formats, such as as internal/external databases, data marts, OLAP cubes, data warehouses, OLTP systems, Spark, HDFS, CSV, etc.
2. **Exploration and Validation**, data validation operations are user-defined error detection functions, which scan the dataset in order to spot some errors.
3. **Data Wrangling (Cleaning)**, is the process of re-formatting or re-structuring particular attributes and correcting errors in data.
4. **Data Splitting**, splitting the data into training, validation, and test datasets to be used during the core machine learning stages to produce the ML model.

### Model Engineering
The core of the ML workflow is the phase of writing and executing machine learning algorithms to obtain an ML model. 

Issue: model decay, the performance of ML models in production degenerate over time because of changes in the real-life data that has not been seen during the model training.

Model engineering pipeline:
1. **Model Training**, is the process of applying the machine learning algorithm on training data to train an ML model. It also includes feature engineering and the hyperparameter tuning for the model training activity.
2. **Model Evaluation**, validating the trained model to ensure it meets original codified objectives before serving the ML model in production to the end-user.
3. **Model Testing**, performing the final “Model Acceptance Test” by using the hold backtest dataset to estimate the generalization error.
4. **Model Packaging**, is the process of exporting the final ML model into a specific format (e.g. PMML, PFA, or ONNX), which describes the model, in order to be consumed by the business application.

### Model Deployment
Once we trained a machine learning model, we need to deploy it as part of a business application.

This stage includes the following operations:
1. **Model Serving**, is the process of addressing the ML model artifact in a production environment.
2. **Model Performance Monitoring**, is the process of observing the ML model performance based on live and previously unseen data. In particular, we are interested in ML-specific signals, such as prediction deviation from previous model performance. These signals might be used as triggers for model re-training.
3. **Model Performance Logging**, every inference request results in the log-record.

## MLOps People
Afterwards in the picture is represents a machine learning model life cycle inside an average organization today. We can observe that is involves many different people with completely different skill sets and who are often using entirely different tools.

<div align="center">
  <img width="550" alt="people" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/people.png">
</div>

## Principles

### Iterative-Incremental Process 
The complete MLOps process includes three broad phases of “Designing the ML-powered application”, “ML Experimentation and Development”, and “ML Operations”.

All three phases are interconnected and influence each other.

### Automation
The level of automation of the Data, ML Model, and Code pipelines determines the maturity of the ML process. With increased maturity, the velocity for the training of new models is also increased.

The objective of an MLOps team is to automate the deployment of ML models into the core software system or as a service component.
There are three levels of MLOps automation, starting from the initial level with manual model training and deployment, up to running both ML and CI/CD pipelines automatically.

### Continuous Deployment
We are interested in the identity, components, versioning, and dependencies of these ML artifacts. The target destination for an ML artifact may be a (micro-) service or some infrastructure components.

A deployment service provides orchestration, logging, monitoring, and notification to ensure that the ML models, code and data artifacts are stable.

### Versioning
The goal of the versioning is to treat ML training scrips, ML models and data sets by tracking ML models and data sets with version control systems.
With data scientists building, testing, and iterating on several versions of models, they need to be able to keep all the versions straight.

Furthermore, every ML model specification should be versioned in a VCS to make the training of ML models auditable and reproducible.

### Reproducibility

In general, reproducibility in MLOps also involves the ability to easily rerun the exact same experiment. Data scientists may neet to have the ability to go back to different "branches" of the experiments—for example, restoring a previous state of a project.

ML reproducibility must provide relevant metadata and information to reproduce models. Model metadata management includes the type of algorithm, features and transformations, data snapshots, hyperparameters, performance metrics, verifiable code from source code management, and the training environment.

### Experiments Tracking

Experimentation takes place throughout the entire model development process, and usually every important decision or assumption comes with at least some experiment or previous research to justify it.

Data scientists need to be able to quickly iterate through all the possibilities for each of the model building blocks.

### ML-based Software Delivery Metrics
There are  four key metrics to measure and improve ones ML-based software delivery: 
* Deployment Frequency, how often does your organization deploy code to production or release it to end-users?
* Lead Time for Changes, how long does it take to go from code committed to code successfully running in production?
* Mean Time To Restore, how long does it generally take to restore service when a service incident or a defect that impacts users occurs?
* Change Fail Percentage, what percentage of changes to production or released to users result in degraded service  and subsequently require remediation?

These are the same for capture the effectivenes of the software development and delivery of elite/high performing organisations.

# About The Project
This project puts into practice the steps of MLOps and it is complete using the Production phase (Observability phase) at link [https://ProductionPhase_project.com](https://github.com/giorgiaBertacchini/MLOps-ProductionPhase-Slide/tree/main/MLOps%20-observability).

## Built With

<div align="center">
  <img width="800" alt="streamlit logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/tools.png">
</div>


## Schema

The following image illustrates how the Develop phase works. The entire development process is managed by workflow orchestration, which in cycle performs several steps, each of which is executed by a specific tool. 
<div align="center">
  <img width="800" alt="streamlit logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/schema.png">
</div>

# How it works

## 01 Workflow orchestration

<div align="center">
  <img width="270" alt="kedro logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/kedro_logo.png">
</div>

:books: *Theory: This apply CI/CD methodology. The desire in MLOps is to automate the CI/CD pipeline as far as possible.*

As Workflow orchestration is used [Kedro](https://kedro.readthedocs.io/en/stable/), an open-source Python framework for creating reproducible, maintainable and modular data science code.
Kedro is a template for new data engineering and data science projects. This tool provide to organize all MLOps steps in a well-defined pipeline.

### 01 Structure

When you initialize a Kedro project, with the command:
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

### 01 Key Elements
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

![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/kedro_plot.png)


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

### 01 Guidelines

Kedro requires installation:

```
pip install kedro
```

Kedro have also a GUI, called Kedro-Viz.
Kedro-Viz is a interactive visualization of the entire pipeline. It is a tool that can be very helpful for explaining what you're doing to people. 

To see the kedro ui:

```
kedro viz
```

To see the kedro ui go to the `270.0.0.1:4141` browser page.


## 02 Data versioning

<div align="center">
  <img width="280" alt="dvc logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/dvc_logo.png">
</div>

:books: *Theory: versioning is essential to reproduce the experiments. Reproducibility in MLOps also involves the ability to easily rerun the exact same experiment.*

Ad data versioning management tool is used [DVC](https://dvc.org/doc). This provides a way to handle large files, data sets, machine learning models, and metrics.

### 02 Structure

When you initialize dvc setting:

```
dvc init
```

is created `.dvc` folder, where the most important file is:

* `config`, with the url to remote destination, where save the data.

### 02 Key Elements 

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

### 02 Guidelines

DVC needs to be installed:
```
pip install dvc
```

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

### 02 More
DVC is a more usefull tool. As Data manager, we can also create data pipeline and specify the metrics, parameters and plots. DVC is also a Experiment manager, providing comparison and visualize experiment results. 

For more: LINK MIO FILE Part 3


## 03 Data analysis and manipulation

<div align="center">
  <img width="300" alt="pandas logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/pandas_logo.png">
</div>

:books: *Theory: Perform exploratory data analysis (EDA) is when data scientists or analysts consider available data sources to train a ML model*

[pandas](https://pandas.pydata.org/docs/) is a Python package providing fast, flexible, and expressive data structures.

> it has the broader goal of becoming the most powerful and flexible open source data analysis/manipulation tool available in any language. It is already well on its way toward this goal.

by [pandas.pydata.org](https://pandas.pydata.org/docs/getting_started/overview.html)

### 03 Key Elements
pandas will help you to explore, clean, and process your data. In pandas, a data table is called a DataFrame. If it is 1-D is called Series.

### 03 Guidelines

It require the installation, also with conda:
```
pip install pandas
```

In the code, pandas Dataframes are used in nodes as funciontion input and output. For example:

``` python
def preprocess_activities(activities: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]
```

``` python
def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple
```

Examples of pandas methods:

``` python
apps: pd.DataFrame

# Drop duplicates
apps.drop_duplicates(inplace = True)

# Calculate the MEAN, and replace any empty values with it
x = apps["Average Heart rate (tpm)"].mean()
apps["Average Heart rate (tpm)"].fillna(x, inplace = True)

# Clean rows that contain empty cells
apps.dropna(inplace = True)
```

``` python
activities: pd.DataFrame

totalNumber = activities.size
maxDistance = activities["Distance (km)"].max()
meanAverageSpeed = activities["Average Speed (km/h)"].mean()
minAverageHeartRate = activities["Average Heart rate (tpm)"].min()
```

## 04 Model training

<div align="center">
  <img width="260" alt="scikitlearn logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/scikitlearn_logo.png">
</div>

:books: *Theory: The data scientist implements different algorithms with the prepared data to train various ML models. In addition, you subject the implemented algorithms to hyperparameter tuning to get the best performing ML model.*

[scikit-learn](https://scikit-learn.org/stable/getting_started.html), or sklearn, is an open source machine learning library.

It is a simple and efficient tools for predictive data analysis and it also provides various tools for model fitting, data preprocessing, model selection, model evaluation.

### 04 Structure

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

### 04 Key Elements

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

### 04 Guidelines

It needs to be installed:
```
pip install -U scikit-learn
```

## 05 Experimentation management

<div align="center">
  <img width="280" alt="MLflow logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/mlflow_logo.png">
</div>

:books: *Theory: ML experiment steps are orchestrated and done automatically. Experiment environment is used in the preproduction and production environment, which is a key aspect of MLOps practice for unifying DevOps.*

[MLflow](https://mlflow.org/docs/latest/index.html) is an open source platform for managing the end-to-end machine learning lifecycle.

MLFlow can be very helpful in terms of tracking metrics over time. We can visualize that and communicate what is the progress over time. MLFlow centralize all of these metrics and also the models generates.

### 05 Collaboration
MLflow and Kedro are tools complementary and not conflicting:
* Kedro is the foundation of your data science and data engineering project
* MLflow create that centralized repository of metrics and progress over time

<div align="center">
  <img width="550" alt="mlflow and kedro" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/mlflow+kedro.png">
</div>

### 05 Structure

For specify more options is used `MLproject` file:

``` yaml
name: kedro mlflow
conda_env: conda.yaml
entry_points:
  main:
    command: "kedro run"
```
So when run mlflow is execute the command `kedro run`.

### 05 Guidelines

It needs to be installed:

```
pip install mlflow
```

To log file:

``` python
mlflow.log_artifact(local_path=os.path.join("data", "01_raw", "DATA.csv"))
```
``` python
mlflow.log_artifact(local_path=os.path.join("data", "04_feature", "model_input_table.csv", dirname ,"model_input_table.csv"))
mlflow.log_artifact(local_path=os.path.join("data", "08_reporting", "feature_importance.png"))
mlflow.log_artifact(local_path=os.path.join("data", "08_reporting", "residuals.png")) 
```

To log model:

``` python
mlflow.sklearn.log_model(sk_model=regressor, artifact_path="model")
```

To log key-value param:

``` python
mlflow.log_param('test_size', parameters["test_size"])
mlflow.log_param('val_size', parameters["val_size"])
mlflow.log_param('max_depth', parameters["max_depth"])
mlflow.log_param('random_state', parameters["random_state"])
```

To log key-value metric:

``` python
mlflow.log_metric("accuracy", score)
mlflow.log_metric("mean_absolute_erro", mae)
mlflow.log_metric("mean_squared_error", mse)
mlflow.log_metric("max_error", me)
```

To set key-value tag:

``` python
mlflow.set_tag("Model Type", "Random Forest")
```

#### Before activate conda environment

Need Python version 3.7. Using conda:

```
conda create -n env_name python=3.7
```

```
conda activate env_name
```

#### How to run mlflow project

You can run mlflow project with:

```
mlflow run . --experiment-name activities-example
```

To run mlflow project in Windows, you can run mlflow project with:

```
mlflow run . --experiment-name activities-example --no-conda
```

#### How to vizualize mlflow project

You can run ui as follows:

```
mlflow ui
```

To see the mlflow ui go to the `270.0.0.1:5000` browser page.

## 06 Model packaging and serving

<div align="center">
  <img width="350" alt="BentoML logo" src="https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/bentoml_logo.png">
</div>

:books: *Theory: the validated model is deployed to a target environment to serve predictions. In Continuous Deployment (CD) the deploied system should automatically deploy the model prediction service.*

[BentoML](https://docs.bentoml.org/en/latest/), on the other hand, focuses on ML in production. By design, BentoML is agnostic to the experimentation platform and the model development environment. BentoML only focuses on serving and deploying trained models.

### 06 Collaboration

MLFlow focuses on loading and running a model, while BentoML provides an abstraction to build a prediction service, which includes the necessary pre-processing and post-processing logic in addition to the model itself.

BentoML is more feature-rich in terms of serving, it supports many essential model serving features that are missing in MLFlow, including multi-model inference, API server dockerization, built-in Prometheus metrics endpoint and many more.

### 06 Structure

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

### 06 Key Elements

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

### 06 Guidelines

BentoML requires installation:

```
pip install bentoml
```

To see all bento models:

```
bentoml models list
```

To see more about a bento model:

```
bentoml models get <name_model>:<number_version>
```

To build a bento model:

```
bentoml build
```

To start Bento model in production:

```
bentoml serve <name_model>:latest --production
```

If use Windows run Bento Server without `-- reload`:

```
bentoml serve service:svc --reload
```

or more general:

```
bentoml serve
```

After you can open a web page `127.0.0.1:3000` to have a model serving.


## 07 Deploying pipeline

<div align="center">
  <h1>kedro-docker</h1>
</div>

:books: *Theory: A Deployment pipeline is the process of taking code from version control and making it readily available to users quickly and accurately.*

[kedro-docker](https://github.com/quantumblacklabs/kedro-docker) is a plugin to create a Docker image and run kedro project in a Docker environment.

### 07 Structure

The settings are in `Dockerfile` file.

For set the number port:

``` python
EXPOSE 3030
```
Note: by default the bento port is 3000, but it is also the same port as granafa, another tool used in Production phase.

For set the command, one of the commands:

``` python
CMD ["kedro", "run"]
```
``` python
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=3030"]
```
Note: if we want it to be just the kedro pipeline then we use CMD ["kedro", "run"], otherwise if we want it to also be capable of more interactivity with production pahse we use the second CMD, thus activating the apllication.

### 07 Guidelines

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

# Getting Started

To interact with pipeline and all step, there is `run.py` which answer to command lines. It makes easier to do tasks such as opening tool gui, creating a new model and its bento, and updating dataset. The avaiable command lines are:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/run.png)


For communication between this project and production phase, there is a [Flask](https://flask.palletsprojects.com/en/2.2.x/) application with avaible API:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/app.png)

To run Flask application, usefull for Production phase: 

```
flask run --host=0.0.0.0 --port=3030
```

## Prerequisites

## Installation
All necessary installations are present at the `src/requirements.txt`

``` yaml
kedro
kedro[pandas.CSVDataSet, pandas.ExcelDataSet, pandas.ParquetDataSet]
kedro-viz                                                          
scikit-learn
matplotlib
seaborn
numpy
mlflow
bentoml
dvc
kedro-docker
requests
flask

```
# Usage

## Kedro Platform 
At `270.0.0.1:4141` browser page.

![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/kedro-viz.png)

From here we can also see and compare the experiments, that are the versions created runned the kedro project.

![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/kedro_experiments.png)
![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/kedro_experiments_0.png)

## MLflow Platform 

At `270.0.0.1:5000` browser page.

![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/mlflow-ui.png)

From this page we can select a single experiment and see more information about it.

![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/mlflow_experiment.png)

## BentoML Platform

At web page `127.0.0.1:3000`.

![This is an image](https://github.com/giorgiaBertacchini/MLOps-DevelopPhase/blob/main/img_readme/bentoml.png)

# Acknowledgments

* [ml-ops.org](https://ml-ops.org/)
* [neptune.ai](https://neptune.ai/blog/mlops)
* [mlebook.com](http://www.mlebook.com/wiki/doku.php)
* [cloud.google.com about MLOps](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
* [Made With ML](https://madewithml.com/courses/mlops/)
* Book "Introducing MLOps How to Scale Machine Learning in the Enterprise (Mark Treveil, Nicolas Omont, Clément Stenac etc.)"
