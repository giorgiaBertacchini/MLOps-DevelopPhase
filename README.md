# kedro_ml, DVC, mlflow and Bentoml project

## Overview

This is new Kedro project with DVC, mlflow, Bentoml and kedro-docker.

## test:
* bentoml containerize...

# kedro-docker

It is a plugin to create a Docker image and run kedro project in a Docker environment.

To install, run:
```
pip install kedro-docker
```

## Docker image

To create a docker image:

```
kedro docker build --image <image-name>
```

To run the project in a Docker environment:

```
kedro docker run --image <image-name>
```

## Use container registry

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


# DVC

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

## Set remote storage

Set the `url` in the file `.dvc/config`


# mlflow

## Rules and guidelines

To install them, run:

```
pip install mlflow
```

## Before activate conda environment

Need Python version 3.7.

```
conda create -n env_name python=3.7
```

```
conda activate env_name
```

## How to run mlflow project

You can run mlflow project with:

```
mlflow run . --experiment-name activities-example
```

## How to run mlflow project in Windows

You can run mlflow project with:

```
mlflow run . --experiment-name activities-example --no-conda
```

## How to vizualize mlflow project

You can run ui as follows:

```
mlflow ui
```

To see the mlflow ui go to the `270.0.0.1:5000` browser page.

## Set hyperparameters

Set hyperparameter in file `conf/base/parameters/data_science.yml`


# Bentoml

## Rules and guidelines

To install run:

```
pip install bentoml
```

## See

To see all bento models:

```
bentoml models list
```

To see more about a bento model:

```
bentoml models get <name_model>:<number_version>
```

## Build a bento model

```
bentoml build
```

## Start Bento model in production

```
bentoml serve <name_model>:latest --production
```

## Run Bento Server

If use Windows run this without `-- reload`:

```
bentoml serve service:svc --reload
```

or more general:

```
bentoml serve
```

After you can open a web page `127.0.0.1:3000` to have a model serving.

# Containerize

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
