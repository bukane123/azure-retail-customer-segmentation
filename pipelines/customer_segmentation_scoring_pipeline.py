"""
Azure ML batch scoring pipeline for retail customer segmentation.

The pipeline accepts raw transaction data, reproduces the validated
data preparation and feature-engineering workflow, and applies the
registered customer segmentation model to generate segment predictions.
"""

from azure.ai.ml import MLClient, Input, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import CommandComponent
from azure.identity import DefaultAzureCredential


# ------------------------------------------------------------------
# Connect to Azure ML workspace
# ------------------------------------------------------------------

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential()
)

# ------------------------------------------------------------------
# Define customer scoring component
# ------------------------------------------------------------------

scoring_component = CommandComponent(
    name="online_retail_score_customers",
    version="1",
    description=(
        "Apply the registered customer segmentation model and its "
        "fitted preprocessing artefacts to engineered customer features."
    ),
    inputs={
        "customer_features": {
            "type": "uri_file",
        },
        "model_dir": {
            "type": AssetTypes.CUSTOM_MODEL,
        },
    },
    outputs={
        "customer_segments": {
            "type": "uri_file",
        },
    },
    code="src/components",
    command=(
        "python score_customers.py "
        '--customer_features "${{inputs.customer_features}}" '
        '--model_dir "${{inputs.model_dir}}" '
        '--output_path "${{outputs.customer_segments}}"'
    ),
    environment="azureml:online-retail-clustering-env:2",
)

# ------------------------------------------------------------------
# Register scoring component
# ------------------------------------------------------------------

registered_scoring_component = (
    ml_client.components.create_or_update(
        scoring_component
    )
)

print(
    "Scoring component:",
    registered_scoring_component.name,
    registered_scoring_component.version,
)

# ------------------------------------------------------------------
# Load validated reusable pipeline components
# ------------------------------------------------------------------

ingestion_step = ml_client.components.get(
    name="online_retail_ingest_validate",
    version="1",
)

data_cleaning_step = ml_client.components.get(
    name="online_retail_data_cleaning",
    version="3",
)

feature_engineering_step = ml_client.components.get(
    name="online_retail_feature_engineering",
    version="2",
)

scoring_step = ml_client.components.get(
    name="online_retail_score_customers",
    version="1",
)

# ------------------------------------------------------------------
# Define batch scoring pipeline
# ------------------------------------------------------------------

@dsl.pipeline(
    description=(
        "Batch customer segmentation scoring pipeline from raw "
        "transaction data through segment prediction."
    )
)
def customer_segmentation_scoring_pipeline(
    raw_data,
    model_dir,
):
    """Score customers from raw business transaction data."""

    # Validate the incoming Excel or CSV file and convert it into
    # the standard internal Parquet format used by downstream steps.
    ingestion_job = ingestion_step(
        raw_data=raw_data
    )

    # Apply the same transaction-cleaning rules used during training.
    cleaning_job = data_cleaning_step(
        validated_data=(
            ingestion_job.outputs.validated_data
        )
    )

    # Recreate the customer-level behavioural features expected
    # by the registered segmentation model.
    feature_job = feature_engineering_step(
        paid_purchases=(
            cleaning_job.outputs.paid_purchases
        ),
        returns=(
            cleaning_job.outputs.returns
        ),
    )

    # Apply the fitted preprocessing artefacts and trained K-Means
    # model. No scaler fitting or model training occurs here.
    scoring_job = scoring_step(
        customer_features=(
            feature_job.outputs.customer_features
        ),
        model_dir=model_dir,
    )

    return {
        "customer_segments": (
            scoring_job.outputs.customer_segments
        )
    }


    # ------------------------------------------------------------------
# Prepare scoring pipeline inputs
# ------------------------------------------------------------------

# Use the original raw transaction file for the first end-to-end
# scoring validation. Once validated, this input can be replaced
# with a new incoming business Excel or CSV file.
raw_data_asset = ml_client.data.get(
    name="online-retail-raw",
    version="1",
)

# Retrieve the self-contained model produced by the validated
# end-to-end training pipeline.
registered_model = ml_client.models.get(
    name="online-retail-kmeans-segmentation",
    version="3",
)


# ------------------------------------------------------------------
# Create scoring pipeline job
# ------------------------------------------------------------------

pipeline_job = customer_segmentation_scoring_pipeline(
    raw_data=Input(
        type="uri_file",
        path=raw_data_asset.path,
    ),
    model_dir=Input(
        type=AssetTypes.CUSTOM_MODEL,
        path=(
            f"azureml:{registered_model.name}:"
            f"{registered_model.version}"
        ),
    ),
)

# Run all scoring components on the existing Azure ML compute instance.
pipeline_job.settings.default_compute = (
    "bukola-compute-instance"
)

pipeline_job.settings.default_compute = (
    "bukola-compute-instance"
)

# ------------------------------------------------------------------
# Submit scoring pipeline
# ------------------------------------------------------------------

submitted_pipeline = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="online-retail-customer-segmentation-scoring",
)

print(
    "Scoring pipeline job:",
    submitted_pipeline.name,
)

print(
    "Scoring pipeline status:",
    submitted_pipeline.status,
)