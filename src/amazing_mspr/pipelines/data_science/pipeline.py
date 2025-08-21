from kedro.pipeline import Node, Pipeline
from .nodes import pca_transformation, search_hyperparameters, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=pca_transformation,
                inputs="scaled_dataset",
                outputs="pca_dataset",
                name="pca_transformation_node",
            ),
            Node(
                func=search_hyperparameters,
                inputs="pca_dataset",
                outputs="eps_value",
                name="search_hyperparameters_node",
            ),
            Node(
                func=train_model,
                inputs=["pca_dataset", "eps_value"],
                outputs="dbscan_model",
                name="train_model_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["dbscan_model", "pca_dataset", "eps_value", "params:seed"],
                outputs="evaluation_results",
                name="evaluate_model_node",
            ),
        ]
    )
