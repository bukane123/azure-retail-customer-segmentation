"""
Customer segmentation model training component.

Trains the validated six-cluster K-Means customer segmentation model using the
preprocessed customer feature matrix.

Input
-----
clustering_input:
    CustomerID plus the eight transformed and scaled modelling features.

Outputs
-------
model_output_dir:
    Trained K-Means model and supporting model metadata.

segment_output_path:
    Customer-to-segment assignment dataset.
"""
import shutil
import os
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def parse_args():
    """Parse command-line inputs and outputs."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clustering_input",
        type=str,
        required=True,
        help="Path to the preprocessed clustering-input Parquet dataset.",
    )

    parser.add_argument(
        "--model_output_dir",
        type=str,
        required=True,
        help="Directory where the trained model and metadata will be saved.",
    )

    parser.add_argument(
        "--segment_output_path",
        type=str,
        required=True,
        help="Path where final customer segment assignments will be saved.",
    )

    # Fitted preprocessing artefacts produced during training.
    # These are packaged with the model so future scoring uses
    # the exact same fitted preprocessing configuration.
    parser.add_argument(
        "--preprocessing_artifacts",
        type=str,
        required=True,
        help="Directory containing fitted preprocessing artefacts.",
    )

    return parser.parse_args()


def main():
    """Train and persist the final customer segmentation model."""
   

    args = parse_args()

    # ------------------------------------------------------------------
    # Load and validate modelling dataset
    # ------------------------------------------------------------------

    df_clustering = pd.read_parquet(
        args.clustering_input
    )

    df_clustering["CustomerID"] = (
        df_clustering["CustomerID"]
        .astype("Int64")
    )

    # Preserve the exact validated feature order used during
    # model development.
    expected_features = [
        "Recency",
        "Frequency",
        "MonetaryValue",
        "UniqueProducts",
        "CustomerTenureDays",
        "AverageQuantityPerOrder",
        "PostageInvoiceShare",
        "ObservedMerchandiseReturnValueRate",
    ]

    expected_columns = [
        "CustomerID",
        *expected_features,
    ]

    # Fail early if the upstream preprocessing output does not match
    # the schema expected by the trained clustering workflow.
    if list(df_clustering.columns) != expected_columns:
        raise ValueError(
            "Unexpected clustering-input schema. "
            f"Expected {expected_columns}, "
            f"received {list(df_clustering.columns)}."
        )

    customer_ids = (
        df_clustering["CustomerID"]
        .copy()
    )

    X_model = (
        df_clustering[expected_features]
        .copy()
    )

    # Ensure the model receives only complete finite numeric values.
    if X_model.isna().sum().sum() > 0:
        raise ValueError(
            "Clustering input contains missing values."
        )

    if np.isinf(X_model.to_numpy()).any():
        raise ValueError(
            "Clustering input contains infinite values."
        )

    # ------------------------------------------------------------------
    # Train final K-Means model
    # ------------------------------------------------------------------

    # Six clusters were selected during model development based on
    # statistical quality, stability, interpretability and suitability
    # for assigning future customers to existing segments.
    final_kmeans = KMeans(
        n_clusters=6,
        random_state=42,
        n_init=50,
    )

    final_cluster_labels = (
        final_kmeans.fit_predict(
            X_model
        )
    )

    # ------------------------------------------------------------------
    # Evaluate final fitted model
    # ------------------------------------------------------------------

    silhouette = silhouette_score(
        X_model,
        final_cluster_labels,
    )

    calinski_harabasz = (
        calinski_harabasz_score(
            X_model,
            final_cluster_labels,
        )
    )

    davies_bouldin = (
        davies_bouldin_score(
            X_model,
            final_cluster_labels,
        )
    )

    # ------------------------------------------------------------------
    # Apply validated business segment labels
    # ------------------------------------------------------------------

    # These names were derived from the behavioural profiling performed
    # during model development for this validated six-cluster solution.
    segment_names = {
        0: "Active Regular Customers",
        1: "Lapsed Low-Value Customers",
        2: "Postage-Heavy Occasional Customers",
        3: "High-Value Loyal Customers",
        4: "High-Return Customers",
        5: "Larger-Basket Low-Frequency Customers",
    }

    customer_segments = pd.DataFrame({
        "CustomerID": customer_ids.values,
        "Cluster": final_cluster_labels,
    })

    customer_segments["SegmentName"] = (
        customer_segments["Cluster"]
        .map(segment_names)
    )

    # Every customer must receive exactly one valid segment.
    if customer_segments["SegmentName"].isna().any():
        raise ValueError(
            "One or more cluster labels could not be mapped "
            "to a segment name."
        )

    if (
        customer_segments["CustomerID"].nunique()
        != len(customer_segments)
    ):
        raise ValueError(
            "Duplicate customer identifiers found in final assignments."
        )

    # ------------------------------------------------------------------
    # Persist trained model
    # ------------------------------------------------------------------

    model_output_dir = Path(
        args.model_output_dir
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_output_dir
        / "final_kmeans_model.joblib"
    )

    joblib.dump(
        final_kmeans,
        model_path,
    )

        # --------------------------------------------------------------
    # Package fitted preprocessing artefacts with the trained model
    # --------------------------------------------------------------
    # New customer data must use the exact StandardScaler and feature
    # configuration fitted during training. Copy those artefacts into
    # the model output so the registered model is self-contained.

    preprocessing_files = [
        "standard_scaler.joblib",
        "feature_order.json",
        "preprocessing_metadata.json",
    ]

    for file_name in preprocessing_files:
        source_path = os.path.join(
            args.preprocessing_artifacts,
            file_name,
        )

        destination_path = os.path.join(
            args.model_output_dir,
            file_name,
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

    print(
        "Preprocessing artefacts packaged with trained model."
    )

    # ------------------------------------------------------------------
    # Persist model metadata
    # ------------------------------------------------------------------

    cluster_counts = (
        customer_segments["Cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_sizes = {
        str(int(cluster)): int(count)
        for cluster, count in cluster_counts.items()
    }

    model_metadata = {
        "model_type": "KMeans",
        "n_clusters": 6,
        "random_state": 42,
        "n_init": 50,
        "training_customers": len(X_model),
        "model_features": expected_features,
        "silhouette_score": float(silhouette),
        "calinski_harabasz_score": float(
            calinski_harabasz
        ),
        "davies_bouldin_score": float(
            davies_bouldin
        ),
        "cluster_sizes": cluster_sizes,
        "segment_names": {
            str(key): value
            for key, value in segment_names.items()
        },
    }

    metadata_path = (
        model_output_dir
        / "model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_metadata,
            file,
            indent=4,
        )

    # ------------------------------------------------------------------
    # Persist customer segment assignments
    # ------------------------------------------------------------------

    segment_output_path = Path(
        args.segment_output_path
    )

    segment_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    customer_segments.to_parquet(
        segment_output_path,
        engine="pyarrow",
        index=False,
    )

    # ------------------------------------------------------------------
    # Component execution summary
    # ------------------------------------------------------------------

    print(
        f"Customers segmented: "
        f"{len(customer_segments):,}"
    )

    print(
        f"Silhouette Score: "
        f"{silhouette:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: "
        f"{calinski_harabasz:.4f}"
    )

    print(
        f"Davies-Bouldin Score: "
        f"{davies_bouldin:.4f}"
    )

    print(
        f"Model artefacts written to: "
        f"{model_output_dir}"
    )

    print(
        f"Segment assignments written to: "
        f"{segment_output_path}"
    )


if __name__ == "__main__":
    main()