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
training_job_name = "4374b932-9c11-47f2-b107-d91ef25f2932"

model_path = (
    f"azureml://jobs/{training_job_name}"
    "/outputs/model_artifacts"
)

pipeline_model = Model(
    name="online-retail-kmeans-segmentation",
    version="2",
    path=model_path,
    type=AssetTypes.CUSTOM_MODEL,
    description=(
        "Six-cluster K-Means customer segmentation model produced "
        "by the Azure ML customer segmentation pipeline."
    ),
    tags={
        "algorithm": "KMeans",
        "clusters": "6",
        "pipeline_trained": "true",
    },
)

registered_model = ml_client.models.create_or_update(
    pipeline_model
)

print("Model:", registered_model.name)
print("Version:", registered_model.version)
print("Type:", registered_model.type)
