from kedro.pipeline import Node, Pipeline

from .nodes import pca_transformation, search_hyperparameters


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=pca_transformation,
                inputs=["scaled_dataset"],
                outputs=["pca_dataset"],
                name="pca_transformation_node",
            ),
            Node(
                func=search_hyperparameters,
                inputs=["scaled", "params:model_options"],
                outputs=["eps_values"],
                name="search_hyperparameters_node",
            ),
        ]
    )
