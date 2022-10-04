# kedro_ml, mlflow and Bentoml project

## Overview

This is new Kedro project with mlflow and Bentoml.

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

## Run Bento Server

If use Windows run this without `-- reload`:

```
bentoml serve servize:svc --reload
```

or more general:

```
bentoml server
```

After you can open a web page `127.0.0.1:3000` to have a model serving.

# Containerize

To containerize:

```
bentoml containerize <name_model>
```

To run the docker model:

```
docker run <name_model> serve --production
```
