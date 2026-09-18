"""
Azure ML customer segmentation pipeline.

Orchestrates the validated feature engineering, preprocessing and
model-training components for the Online Retail segmentation project.
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import dsl, Input
from azure.core.exceptions import ResourceNotFoundError


# ------------------------------------------------------------------
# Connect to the Azure Machine Learning workspace
# ------------------------------------------------------------------

# Use the existing workspace configuration and authentication approach
# already validated throughout this project.
ml_client = MLClient.from_config(
    credential=DefaultAzureCredential()
)


# ------------------------------------------------------------------
# Component registration helper
# ------------------------------------------------------------------

def get_or_register_component(component):
    """
    Reuse an existing immutable Azure ML component version when it
    already exists. Register it only when that version is new.

    If component code changes intentionally, increase its version
    number before running the pipeline.
    """

    try:
        existing_component = ml_client.components.get(
            name=component.name,
            version=component.version,
        )

        print(
            f"Reusing component: "
            f"{existing_component.name}:{existing_component.version}"
        )

        return existing_component

    except ResourceNotFoundError:
        registered_component = (
            ml_client.components.create_or_update(component)
        )

        print(
            f"Registered component: "
            f"{registered_component.name}:{registered_component.version}"
        )

        return registered_component
# ------------------------------------------------------------------
# Define shared component environment
# ------------------------------------------------------------------

# All pipeline components will use the same Python environment.
# Package dependencies are defined separately in the Conda YAML file.
component_environment = Environment(
    name="online-retail-clustering-env",
    version="2",
    description=(
        "Runtime environment for the Online Retail "
        "customer segmentation pipeline."
    ),
    conda_file="config/component_environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
)

# Register the reusable environment in the Azure ML workspace.

registered_environment = ml_client.environments.create_or_update(
    component_environment
)

print("Environment:", registered_environment.name)
print("Version:", registered_environment.version)


# ------------------------------------------------------------------
# Define ingestion and validation component
# ------------------------------------------------------------------

# This component represents the business-facing entry point.
# It accepts the original Excel or CSV transaction extract,
# validates the expected schema and converts it to Parquet for
# consistent downstream processing.

ingestion_component = CommandComponent(
    name="online_retail_ingest_validate",
    version="1",
    display_name="Online Retail Ingestion and Validation",
    description=(
        "Validates the raw Online Retail transaction extract and "
        "converts it to a standardised Parquet dataset."
    ),
    inputs={
        "raw_data": {
            "type": "uri_file"
        },
    },
    outputs={
        "validated_data": {
            "type": "uri_file"
        },
    },
    code="src/components",
    command=(
        'python ingest_validate.py '
        '--input_data "${{inputs.raw_data}}" '
        '--output_path "${{outputs.validated_data}}"'
    ),
    environment=(
        f"azureml:{registered_environment.name}:"
        f"{registered_environment.version}"
    ),
)


# Register the ingestion component so it can be reused
# within the end-to-end Azure ML pipeline.
registered_ingestion_component = (
    get_or_register_component(
        ingestion_component
    )
)

print(
    "Ingestion component:",
    registered_ingestion_component.name
)

print(
    "Ingestion component version:",
    registered_ingestion_component.version
)

# ------------------------------------------------------------------
# Define data-cleaning component
# ------------------------------------------------------------------

# This component applies the validated cleaning rules from model
# development: remove rows without CustomerID, remove exact duplicate
# transactions, identify returns, and separate zero-price transactions.

data_cleaning_component = CommandComponent(
    name="online_retail_data_cleaning",
    version="3",
    display_name="Online Retail Data Cleaning",
    description=(
        "Cleans the validated Online Retail transaction dataset and "
        "produces paid purchases, returns and zero-price transactions."
    ),
    inputs={
        "validated_data": {
            "type": "uri_file"
        },
    },
    outputs={
        "paid_purchases": {
            "type": "uri_file"
        },
        "returns": {
            "type": "uri_file"
        },
        "zero_price_transactions": {
            "type": "uri_file"
        },
    },
    code="src/components",
    command=(
        'python data_cleaning.py '
        '--input_data "${{inputs.validated_data}}" '
        '--paid_purchases_output "${{outputs.paid_purchases}}" '
        '--returns_output "${{outputs.returns}}" '
        '--zero_price_output "${{outputs.zero_price_transactions}}"'
    ),
    environment=(
        f"azureml:{registered_environment.name}:"
        f"{registered_environment.version}"
    ),
)


# Register the reusable data-cleaning component.
registered_data_cleaning_component = (
    get_or_register_component(
        data_cleaning_component
    )
)

print(
    "Data cleaning component:",
    registered_data_cleaning_component.name
)

print(
    "Data cleaning component version:",
    registered_data_cleaning_component.version
)


# ------------------------------------------------------------------
# Define feature engineering component
# ------------------------------------------------------------------

# Define a reusable Azure ML command component that transforms the
# cleaned transaction datasets into customer-level behavioural features.
feature_engineering_component = CommandComponent(
    name="online_retail_feature_engineering",
    version="2",
    display_name="Online Retail Feature Engineering",
    description=(
        "Creates customer-level behavioural features from cleaned "
        "paid-purchase and returns transaction datasets."
    ),

    # Define the datasets expected by the component.
    inputs={
        "paid_purchases": {
            "type": "uri_file"
        },
        "returns": {
            "type": "uri_file"
        },
    },

    # Define the engineered customer feature dataset produced by the component.
    outputs={
        "customer_features": {
            "type": "uri_file"
        },
    },

    # Upload the directory containing the tested feature engineering script.
    code="src/components",

    # Pass Azure ML input and output paths into the Python component script.
    command=(
        "python feature_engineering.py "
        "--paid_purchases ${{inputs.paid_purchases}} "
        "--returns ${{inputs.returns}} "
        "--output_path ${{outputs.customer_features}}"
    ),

    # Run the component using the environment registered earlier.
    environment=(
        f"azureml:{registered_environment.name}:"
        f"{registered_environment.version}"
    ),
)

# Register the reusable feature engineering component in Azure ML.

registered_feature_engineering_component = (
    get_or_register_component(
        feature_engineering_component
    )
)

print(
    "Component:",
    registered_feature_engineering_component.name
)

print(
    "Component version:",
    registered_feature_engineering_component.version
)


# ------------------------------------------------------------------
# Define preprocessing component
# ------------------------------------------------------------------

# Define a reusable Azure ML component that transforms the engineered
# customer features into the final scaled clustering dataset.
preprocessing_component = CommandComponent(
    name="online_retail_preprocessing",
    version="2",
    display_name="Online Retail Feature Preprocessing",
    description=(
        "Applies validated feature transformations, final feature selection "
        "and scaling for customer segmentation modelling."
    ),

    # Consume the customer-level feature dataset produced by the
    # feature engineering component.
    inputs={
        "customer_features": {
            "type": "uri_file"
        },
    },

    # Produce the final clustering dataset and reusable preprocessing artefacts.
    outputs={
        "clustering_input": {
            "type": "uri_file"
        },
        "preprocessing_artifacts": {
            "type": "uri_folder"
        },
    },

    # The preprocessing script is stored in the shared component source folder.
    code="src/components",

    # Pass Azure ML-managed input/output locations into the tested script.
    command=(
        "python preprocessing.py "
        "--customer_features ${{inputs.customer_features}} "
        "--output_path ${{outputs.clustering_input}} "
        "--artifact_dir ${{outputs.preprocessing_artifacts}}"
    ),

    # Reuse the registered component environment.
    environment=(
        f"azureml:{registered_environment.name}:"
        f"{registered_environment.version}"
    ),
)

# Register the preprocessing component only if this version
# does not already exist in Azure ML.
registered_preprocessing_component = (
    get_or_register_component(
        preprocessing_component
    )
)

print(
    "Preprocessing component:",
    registered_preprocessing_component.name
)

print(
    "Preprocessing component version:",
    registered_preprocessing_component.version
)

# ------------------------------------------------------------------
# Define model training component
# ------------------------------------------------------------------

# Define a reusable Azure ML component that trains the final six-cluster
# K-Means model and produces customer segment assignments.
training_component = CommandComponent(
    name="online_retail_train_segmentation_model",
    version="5",
    display_name="Online Retail Segmentation Model Training",
    description=(
        "Trains the validated six-cluster K-Means customer segmentation "
        "model and produces final customer segment assignments."
    ),

    # Consume the scaled clustering dataset produced by preprocessing.
    inputs={
    # Scaled feature dataset used to train the K-Means model.
    "clustering_input": {
        "type": "uri_file"
    },

    # Fitted preprocessing artefacts from the preprocessing step.
    # These are packaged with the model for future scoring.
    "preprocessing_artifacts": {
        "type": "uri_folder"
    },
},

    # Produce the trained model artefacts and final customer assignments.
    outputs={
        "model_artifacts": {
            "type": "uri_folder"
        },
        "customer_segments": {
            "type": "uri_file"
        },
    },

    # The tested training script is stored in the shared component folder.
    code="src/components",

    # Pass Azure ML-managed input and output paths into the training script.
    command=(
    "python train_model.py "
    "--clustering_input ${{inputs.clustering_input}} "
    "--preprocessing_artifacts ${{inputs.preprocessing_artifacts}} "
    "--model_output_dir ${{outputs.model_artifacts}} "
    "--segment_output_path ${{outputs.customer_segments}}"
),

    # Reuse the shared registered environment.
    environment=(
        f"azureml:{registered_environment.name}:"
        f"{registered_environment.version}"
    ),
)

# Register the reusable model training component in Azure ML.

registered_training_component = (
    get_or_register_component(
        training_component
    )
)

print(
    "Training component:",
    registered_training_component.name
)

print(
    "Training component version:",
    registered_training_component.version
)

# ------------------------------------------------------------------
# Load registered components for pipeline orchestration
# ------------------------------------------------------------------

# Retrieve the exact registered component versions used by this
# end-to-end training pipeline.

ingestion_step = ml_client.components.get(
    name="online_retail_ingest_validate",
    version="1"
)

data_cleaning_step = ml_client.components.get(
    name="online_retail_data_cleaning",
    version="3"
)

feature_engineering_step = ml_client.components.get(
    name="online_retail_feature_engineering",
    version="2"
)

preprocessing_step = ml_client.components.get(
    name="online_retail_preprocessing",
    version="2"
)

training_step = ml_client.components.get(
    name="online_retail_train_segmentation_model",
    version="5"
)

# ------------------------------------------------------------------
# Define end-to-end customer segmentation training pipeline
# ------------------------------------------------------------------

@dsl.pipeline(
    description=(
        "End-to-end Online Retail customer segmentation training pipeline "
        "from raw business transaction data through model training."
    )
)
def customer_segmentation_pipeline(
    raw_data,
):

    # Step 1:
    # Validate the raw Excel/CSV business extract and convert it
    # to the standard internal Parquet format.
    ingestion_job = ingestion_step(
        raw_data=raw_data,
    )

    # Step 2:
    # Apply the validated transaction-cleaning rules and produce
    # the paid-purchase and returns datasets used downstream.
    cleaning_job = data_cleaning_step(
        validated_data=(
            ingestion_job.outputs.validated_data
        )
    )

    # Step 3:
    # Convert transaction-level purchase and return history into
    # customer-level behavioural features.
    feature_job = feature_engineering_step(
        paid_purchases=(
            cleaning_job.outputs.paid_purchases
        ),
        returns=(
            cleaning_job.outputs.returns
        ),
    )

    # Step 4:
    # Apply the validated transformations, select the final
    # eight modelling features and fit the StandardScaler.
    preprocessing_job = preprocessing_step(
        customer_features=(
            feature_job.outputs.customer_features
        )
    )

    # Step 5:
    # Train the validated six-cluster K-Means segmentation model.
    training_job = training_step(
    # Scaled feature dataset used to train K-Means.
    clustering_input=(
        preprocessing_job.outputs.clustering_input
    ),

    # Pass the fitted scaler and preprocessing configuration into
    # training so they can be packaged with the trained model.
    preprocessing_artifacts=(
        preprocessing_job.outputs.preprocessing_artifacts
    ),
)

    # Expose important intermediate and final outputs for validation,
    # lineage and future downstream scoring workflows.
    return {
        "validated_data": (
            ingestion_job.outputs.validated_data
        ),
        "paid_purchases": (
            cleaning_job.outputs.paid_purchases
        ),
        "returns": (
            cleaning_job.outputs.returns
        ),
        "zero_price_transactions": (
            cleaning_job.outputs.zero_price_transactions
        ),
        "customer_features": (
            feature_job.outputs.customer_features
        ),
        "clustering_input": (
            preprocessing_job.outputs.clustering_input
        ),
        "preprocessing_artifacts": (
            preprocessing_job.outputs.preprocessing_artifacts
        ),
        "model_artifacts": (
            training_job.outputs.model_artifacts
        ),
        "customer_segments": (
            training_job.outputs.customer_segments
        ),
    }

# ------------------------------------------------------------------
# Retrieve raw business input dataset
# ------------------------------------------------------------------

# The end-to-end training pipeline now starts from the original
# business-style transaction extract rather than pre-cleaned data.
raw_data_asset = ml_client.data.get(
    name="online-retail-raw",
    version="1"
)


# ------------------------------------------------------------------
# Create end-to-end pipeline job
# ------------------------------------------------------------------

pipeline_job = customer_segmentation_pipeline(
    raw_data=Input(
        type="uri_file",
        path=raw_data_asset.path,
    ),
)


# ------------------------------------------------------------------
# Configure pipeline compute
# ------------------------------------------------------------------

# Use the existing Azure ML compute instance for every component job.
pipeline_job.settings.default_compute = (
    "bukola-compute-instance"
)

# ------------------------------------------------------------------
# Submit end-to-end pipeline
# ------------------------------------------------------------------

# Submit the complete customer segmentation workflow to Azure ML.
submitted_pipeline = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="online-retail-customer-segmentation"
)

print("Pipeline job name:", submitted_pipeline.name)
print("Pipeline status:", submitted_pipeline.status)