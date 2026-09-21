from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


# ------------------------------------------------------------------
# Connect to the Azure ML workspace
# ------------------------------------------------------------------

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential()
)


# ------------------------------------------------------------------
# Register the model produced by the successful pipeline training job
# ------------------------------------------------------------------

# Register directly from the Azure ML job output so the registered
# model remains associated with the pipeline run that produced it.
training_job_name = "4af2c230-005e-4c6c-bfa7-24b743b1b6c7"

model_path = (
    f"azureml://jobs/{training_job_name}"
    "/outputs/model_artifacts"
)

pipeline_model = Model(
    name="online-retail-kmeans-segmentation",
    version="3",
    path=model_path,
    type=AssetTypes.CUSTOM_MODEL,
    description=(
        "Six-cluster K-Means customer segmentation model produced "
        "by the end-to-end Azure ML training pipeline. The model "
        "package includes the fitted StandardScaler and preprocessing "
        "configuration required for reproducible batch scoring."
    ),
    tags={
        "algorithm": "KMeans",
        "clusters": "6",
        "pipeline_trained": "true",
        "self_contained": "true",
        "supports_batch_scoring": "true",
    },
)

registered_model = ml_client.models.create_or_update(
    pipeline_model
)

print("Model:", registered_model.name)
print("Version:", registered_model.version)
print("Type:", registered_model.type)