from kedro.pipeline import Pipeline

# importe chaque pipeline à la main
from .pipelines.data_processing import create_pipeline as dp_pipeline
from .pipelines.data_science import create_pipeline as ds_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines manually."""
    return {
        "data_processing": dp_pipeline(),
        "data_science": ds_pipeline(),
        "__default__": dp_pipeline() + ds_pipeline()
    }
