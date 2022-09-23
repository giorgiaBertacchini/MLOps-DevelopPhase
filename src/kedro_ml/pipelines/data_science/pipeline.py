"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, testing_model, plot_feature_importance, plot_residuals


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "X_val", "y_train", "y_test", "y_val"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_val", "y_val"],
                name="evaluate_model_node",
                outputs="metrics",
            ),
            node(
                func=testing_model,
                inputs=["regressor", "X_test", "y_test"],
                name="testing_model_node",
                outputs="test_difference",
            ),
            node(
                func=plot_feature_importance,
                inputs=["regressor", "model_input_table"],
                name="plot_feature_importance_node", 
                outputs="plot_feature_importance_img",
            ),
            node(
                func=plot_residuals,
                inputs=["regressor", "X_test", "y_test"],
                name="plot_residuals_node", 
                outputs="plot_residuals_img",
            ),
        ]
    )