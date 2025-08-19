from kedro.pipeline import Node, Pipeline

from src.amazing_mspr.pipelines.data_processing.nodes import group_events_by_client, add_categories, normalise_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=group_events_by_client,
                inputs="events_dataset",
                outputs="users_dataset",
                name="group_events_by_client_node",
            ),
            Node(
                func=add_categories,
                inputs=["users_dataset", "params:nb_categories"],
                outputs="completed_dataset",
                name="add_categories_node",
            ),
            Node(
                func=normalise_dataset,
                inputs=["completed_dataset"],
                outputs="scaled_dataset",
                name="normalise_dataset_node",
            ),
        ]
    )
