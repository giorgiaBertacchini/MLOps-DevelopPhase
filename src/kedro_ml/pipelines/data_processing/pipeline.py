"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, preprocess_activities

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
                func=create_model_input_table,
                inputs="preprocessed_activities",
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
