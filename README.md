<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#mlops">MLOps</a>
      <ul>        
        <li><a href="#three-level">Three Level</a></li>  
      </ul>
    </li>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>        
        <li><a href="#built-with">Built With</a></li>  
        <li><a href="#schema">Schema</a></li>     
        <li><a href="#interactions-and-communication">Interactions And Communication</a></li>
        <li><a href="#getting-started">Getting Started</a></li>
      </ul>
    </li>
    <li>
      <a href="#How it works">Tools</a>
      <ul>
        <li><a href="#workflow-orchestration">Workflow orchestration</a></li>
        <li><a href="#dvc">DVC</a></li>
        <li><a href="#mlflow">mlflow</a></li>
        <li><a href="#bentoml">Bentoml</a></li>
        <li><a href="#containerize">Containerize</a></li>        
        <li><a href="#kedro-docker">kedro-docker</a></li>
      </ul>
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

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/tools.png)


## Schema

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/schema.png)


## Interactions And Communication

To interact with pipeline and all step, there is run.py that answer to command line. The avaiable command line are:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/run.png)


For communication between this project and observability step, there is a flask application with avaible API:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/app.png)


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

``` python
model_input_table:
  type: pandas.CSVDataSet
  filepath: data/04_feature/model_input_table.csv
  versioned: true
```

``` python
regressor:
  type: pickle.PickleDataSet
  filepath: data/06_models/regressor.pickle
  versioned: true
```

``` python
metrics:
  type: tracking.MetricsDataSet
  filepath: data/09_tracking/metrics.json
  versioned: true
```

The next is more complex, so this plot is showed also in the Kedro interactive visualization platform [Kedro-Viz](https://kedro.readthedocs.io/en/0.17.4/03_tutorial/06_visualise_pipeline.html).

``` python
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

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/kedro-viz.png)

From here we can also see and compare the experiments, that are the versions created runned the kedro project.

Example:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/kedro-viz_exaperiments.png)


## Data versioning

<div align="center">
  <img width="280" alt="dvc logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/dvc_logo.png">
</div>

Ad data versioning managenet tool is used [DVC](https://dvc.org/doc). This provide to handle large files, data sets, machine learning models, and metrics.

### file .dvc

DVC stores information about the added file in a special `.dvc` file named `data/data.xml.dvc`, this metadata file is a placeholder for the original data.

In this case we would save the original datasets, so `data/01_raw/DATA.csv`.

### Structure

When you installing and initialize dvc setting:

```
pip install dvc
```
```
dvc init
```

is created `.dvc` folder, where the most important file is:

* `config`, with the url to remote destination, where save the data.

### Set remote storage

In this project is used own Google Drive, the code at path `.dvc/config` is:

```
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

[pandas](https://pandas.pydata.org/docs/)

* DataFrame

## Model training

<div align="center">
  <img width="260" alt="scikitlearn logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/scikitlearn_logo.png">
</div>

[scikit-learn](https://scikit-learn.org/stable/getting_started.html)

### Set hyperparameters

Set hyperparameter in file `conf/base/parameters/data_science.yml`

``` python
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

## Experimentation management

<div align="center">
  <img width="280" alt="MLflow logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/mlflow_logo.png">
</div>

[MLflow](https://mlflow.org/docs/latest/index.html)

``` python
mlflow.log_artifact(local_path=os.path.join("data", "01_raw", "DATA.csv"))
```

``` python
mlflow.sklearn.log_model(sk_model=regressor, artifact_path="model")
```

``` python
mlflow.log_param('test_size', parameters["test_size"])
mlflow.log_param('val_size', parameters["val_size"])
mlflow.log_param('max_depth', parameters["max_depth"])
mlflow.log_param('random_state', parameters["random_state"])
```

``` python
mlflow.log_metric("accuracy", score)
mlflow.log_metric("mean_absolute_erro", mae)
mlflow.log_metric("mean_squared_error", mse)
mlflow.log_metric("max_error", me)
mlflow.log_param("time of prediction", str(datetime.now()))
mlflow.set_tag("Model Type", "Random Forest")
```

``` python
mlflow.log_artifact(local_path=os.path.join("data", "04_feature", "model_input_table.csv", dirname ,"model_input_table.csv"))
mlflow.log_artifact(local_path=os.path.join("data", "08_reporting", "feature_importance.png"))
mlflow.log_artifact(local_path=os.path.join("data", "08_reporting", "residuals.png"))  
```

### Before activate conda environment

Need Python version 3.7.

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


## Model packaging and serving

<div align="center">
  <img width="350" alt="BentoML logo" src="https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-finally/img_readme/bentoml_logo.png">
</div>

[BentoML](https://docs.bentoml.org/en/latest/)



# Getting Started

To run flask application: 

```
flask run --host=0.0.0.0 --port=3030
```



---



# Tools

## kedro-docker

It is a plugin to create a Docker image and run kedro project in a Docker environment.

To install, run:
```
pip install kedro-docker
```

### Docker image

To create a docker image:

```
kedro docker build --image <image-name>
```

To run the project in a Docker environment:

```
kedro docker run --image <image-name>
```

### Use container registry

Tag your image on your local machine:
```
docker tag <image-name> <DockerID>/<image-name>
```

Push the image to Docker hub:
```
docker push <DockerID>/<image-name>
```

Pull the image from Docker hub onto your production server:
```
docker pull <DockerID>/<image-name>
```


## mlflow

### Rules and guidelines

To install them, run:

```
pip install mlflow
```

### Before activate conda environment

Need Python version 3.7.

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


### Set hyperparameters

Set hyperparameter in file `conf/base/parameters/data_science.yml`


## Bentoml

### Rules and guidelines

To install run:

```
pip install bentoml
```

### See

To see all bento models:

```
bentoml models list
```

To see more about a bento model:

```
bentoml models get <name_model>:<number_version>
```

### Build a bento model

```
bentoml build
```

### Start Bento model in production

```
bentoml serve <name_model>:latest --production
```

### Run Bento Server

If use Windows run this without `-- reload`:

```
bentoml serve service:svc --reload
```

or more general:

```
bentoml serve
```

After you can open a web page `127.0.0.1:3000` to have a model serving.

## Containerize

To containerize:

```
bentoml containerize <name_model>:latest
```

To run the docker model:

```
docker run <name_model>
```

or to production:

```
docker run <name_model> serve --production
```
