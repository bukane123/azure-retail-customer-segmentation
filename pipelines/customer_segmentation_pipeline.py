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


# ------------------------------------------------------------------
# Connect to the Azure Machine Learning workspace
# ------------------------------------------------------------------

# Use the existing workspace configuration and authentication approach
# already validated throughout this project.
ml_client = MLClient.from_config(
    credential=DefaultAzureCredential()
)

# ------------------------------------------------------------------
# Define shared component environment
# ------------------------------------------------------------------

# All pipeline components will use the same Python environment.
# Package dependencies are defined separately in the Conda YAML file.
component_environment = Environment(
    name="online-retail-clustering-env",
    version="1",
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
# Define feature engineering component
# ------------------------------------------------------------------

# Define a reusable Azure ML command component that transforms the
# cleaned transaction datasets into customer-level behavioural features.
feature_engineering_component = CommandComponent(
    name="online_retail_feature_engineering",
    version="1",
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
    ml_client.components.create_or_update(
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
    version="1",
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

# Register the reusable preprocessing component in Azure ML.

registered_preprocessing_component = (
    ml_client.components.create_or_update(
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
    version="1",
    display_name="Online Retail Segmentation Model Training",
    description=(
        "Trains the validated six-cluster K-Means customer segmentation "
        "model and produces final customer segment assignments."
    ),

    # Consume the scaled clustering dataset produced by preprocessing.
    inputs={
        "clustering_input": {
            "type": "uri_file"
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
    ml_client.components.create_or_update(
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

# Retrieve the registered component versions from the Azure ML workspace.
# Using registered components keeps the pipeline definition independent
# of the local component objects used during registration.

feature_engineering_step = ml_client.components.get(
    name="online_retail_feature_engineering",
    version="1"
)

preprocessing_step = ml_client.components.get(
    name="online_retail_preprocessing",
    version="1"
)

training_step = ml_client.components.get(
    name="online_retail_train_segmentation_model",
    version="1"
)

# ------------------------------------------------------------------
# Define end-to-end customer segmentation pipeline
# ------------------------------------------------------------------

@dsl.pipeline(
    description=(
        "End-to-end Online Retail customer segmentation pipeline "
        "covering feature engineering, preprocessing and model training."
    )
)
def customer_segmentation_pipeline(
    paid_purchases,
    returns,
):

    # Step 1:
    # Transform cleaned transaction-level datasets into one
    # customer-level engineered feature dataset.
    feature_job = feature_engineering_step(
        paid_purchases=paid_purchases,
        returns=returns,
    )

    # Step 2:
    # Apply the validated transformations, feature selection and scaling.
    preprocessing_job = preprocessing_step(
        customer_features=(
            feature_job.outputs.customer_features
        )
    )

    # Step 3:
    # Train the validated six-cluster K-Means model and generate
    # final customer segment assignments.
    training_job = training_step(
        clustering_input=(
            preprocessing_job.outputs.clustering_input
        )
    )

    # Expose important intermediate and final artefacts as pipeline outputs.
    return {
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
# Retrieve registered pipeline input datasets
# ------------------------------------------------------------------

paid_purchases_asset = ml_client.data.get(
    name="online-retail-paid-purchases-parquet",
    version="1"
)

returns_asset = ml_client.data.get(
    name="online-retail-returns-parquet",
    version="1"
)

# ------------------------------------------------------------------
# Create pipeline job
# ------------------------------------------------------------------

pipeline_job = customer_segmentation_pipeline(
    paid_purchases=Input(
        type="uri_file",
        path=paid_purchases_asset.path,
    ),
    returns=Input(
        type="uri_file",
        path=returns_asset.path,
    ),
)
print(pipeline_job)

# ------------------------------------------------------------------
# Configure pipeline compute
# ------------------------------------------------------------------

# Use the existing Azure ML compute instance as the default compute
# for all three pipeline component jobs.
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