<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
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
      <a href="#tools">Tools</a>
      <ul>
        <li><a href="#kedro-docker">kedro-docker</a></li>
        <li><a href="#dvc">DVC</a></li>
        <li><a href="#mlflow">mlflow</a></li>
        <li><a href="#bentoml">Bentoml</a></li>
        <li><a href="#containerize">Containerize</a></li>
      </ul>
    </li>
  </ol>
</details>

# MLOps
MLOps is designed to facilitate the installation of ML software in a production environment. 
Machine Learning Operations (MLOps).

The term MLOps is defined as
> “the extension of the DevOps methodology to include Machine Learning and Data Science assets as first-class citizens within the DevOps ecology”
by [MLOps SIG](https://github.com/cdfoundation/sig-mlops/blob/main/roadmap/2020/MLOpsRoadmap2020.md)

## Three Level
![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/cycle.webp)
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
![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/kedro_logo.png)
As Workflow orchestration is used Kedro, an open-source Python framework for creating reproducible, maintainable and modular data science code.
Kedro is a template for new data engineering and data science projects. This tool provide to organize all MLOps steps in a well-defined pipeline.

### Key elements
* Data Catalog
  * It makes the datasets declarative, rather than imperative. So all the informations related to a dataset are highly organized.
* Node
  * It is a Python function that accepts input and optionally provides outputs.
* Pipeline
  * It is a collection of nodes. It create the Kedro DAG (Directed acyclic graph).

#### Data Catalog
In the project Data Catalog is implemented in `conf/base/catalog`.
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

# Getting Started

To run flask application: 

```
flask run --host=0.0.0.0 --port=3030
```


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


## DVC

It is for Data Versioning.

To install, run: 

```
pip install dvc
```

To update data:

```
dvc remove data/01_raw/DATA.csv.dvc 
```

```
dvc add data/01_raw/DATA.csv
```

We can see in folder .dvc/cache/ in the corrispective folder there are the data .dvc

```
dvc push data/01_raw/DATA.csv
```

Now in drive or data updated or there are a new folder with the new data version.

### Set remote storage

Set the `url` in the file `.dvc/config`


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

## Kedro

### Rules and guidelines

To install run:

```
pip install kedro
```

### How to vizualize kedro pipeline

To see the kedro ui:

```
kedro viz
```

To see the mlflow ui go to the `270.0.0.1:4141` browser page.

Example:

![This is an image](https://github.com/giorgiaBertacchini/MLOps-kedro-auto/blob/experiment-dockerize/img_readme/kedro-viz.png)


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
