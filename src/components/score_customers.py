"""
Customer segmentation scoring component.

Applies the preprocessing configuration fitted during model training and
assigns customer segments using the trained K-Means model.

Important:
This component never fits preprocessing artefacts or retrains the model.
It only applies artefacts produced by the validated training pipeline.

Inputs
------
customer_features:
    Engineered customer-level behavioural features.

model_dir:
    Registered model package containing the trained K-Means model,
    fitted StandardScaler and preprocessing metadata.

Output
------
customer_segments:
    CustomerID, predicted cluster and business segment name.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args():
    """Parse command-line inputs and outputs."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--customer_features",
        type=str,
        required=True,
        help="Path to engineered customer feature dataset.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the registered segmentation model package.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where customer segment predictions will be saved.",
    )

    return parser.parse_args()


def load_json(path):
    """Load a JSON configuration file."""

    with open(path, "r") as file:
        return json.load(file)


def validate_required_columns(df, required_columns):
    """Ensure all features required for scoring are available."""

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Customer feature dataset is missing required columns: "
            f"{missing_columns}"
        )


def main():
    """Apply fitted preprocessing and predict customer segments."""

    args = parse_args()

    model_dir = Path(args.model_dir)

    # ------------------------------------------------------------------
    # Load trained model and fitted preprocessing artefacts
    # ------------------------------------------------------------------

    model = joblib.load(
        model_dir / "final_kmeans_model.joblib"
    )

    scaler = joblib.load(
        model_dir / "standard_scaler.joblib"
    )

    feature_order = load_json(
        model_dir / "feature_order.json"
    )

    preprocessing_metadata = load_json(
        model_dir / "preprocessing_metadata.json"
    )

    model_metadata = load_json(
        model_dir / "model_metadata.json"
    )

    # ------------------------------------------------------------------
    # Validate model package consistency
    # ------------------------------------------------------------------

    # The feature order stored by preprocessing, the scaler input and
    # the trained model must all represent the same modelling features.
    if (
        preprocessing_metadata["final_model_features"]
        != feature_order
    ):
        raise ValueError(
            "Feature order does not match preprocessing metadata."
        )

    if model_metadata["model_features"] != feature_order:
        raise ValueError(
            "Feature order does not match model metadata."
        )

    if scaler.n_features_in_ != len(feature_order):
        raise ValueError(
            "Saved scaler does not match the expected feature count."
        )

    if model.n_features_in_ != len(feature_order):
        raise ValueError(
            "Saved K-Means model does not match the expected feature count."
        )

    # ------------------------------------------------------------------
    # Load engineered customer features
    # ------------------------------------------------------------------

    df_customers = pd.read_parquet(
        args.customer_features
    )

    validate_required_columns(
        df_customers,
        ["CustomerID"],
    )

    df_customers["CustomerID"] = (
        df_customers["CustomerID"]
        .astype("Int64")
    )

    model_feature_candidates = (
        preprocessing_metadata[
            "model_feature_candidates"
        ]
    )

    validate_required_columns(
        df_customers,
        model_feature_candidates,
    )

    # Work only with the modelling feature candidates that were used
    # during the validated training preprocessing workflow.
    X = df_customers[
        model_feature_candidates
    ].copy()

    # All model inputs must be numeric and complete.
    X = X.apply(
        pd.to_numeric,
        errors="raise",
    )

    if X.isna().sum().sum() > 0:
        raise ValueError(
            "Customer feature dataset contains missing modelling values."
        )

    # ------------------------------------------------------------------
    # Apply the same transformations used during training
    # ------------------------------------------------------------------

    log_transform_features = (
        preprocessing_metadata[
            "log_transform_features"
        ]
    )

    X_transformed = X.copy()

    for feature in log_transform_features:
        X_transformed[feature] = np.log1p(
            X_transformed[feature]
        )

    # Scoring must stop if transformation creates invalid values.
    if X_transformed.isna().sum().sum() > 0:
        raise ValueError(
            "Log transformation produced missing values."
        )

    if np.isinf(
        X_transformed.to_numpy()
    ).sum() > 0:
        raise ValueError(
            "Log transformation produced infinite values."
        )

    # ------------------------------------------------------------------
    # Enforce the exact feature order used during model training
    # ------------------------------------------------------------------

    X_final = X_transformed[
        feature_order
    ].copy()

    # ------------------------------------------------------------------
    # Apply fitted StandardScaler
    # ------------------------------------------------------------------

    # transform() is deliberately used rather than fit_transform().
    # New customer data must use the scaling parameters learned from
    # the original training population.
    X_scaled = scaler.transform(
        X_final
    )

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Scaled scoring features contain invalid values."
        )

    # ------------------------------------------------------------------
    # Predict customer clusters
    # ------------------------------------------------------------------

    predicted_clusters = model.predict(
        X_scaled
    )

    # Use the business segment labels stored with the trained model
    # rather than maintaining a separate mapping in scoring code.
    segment_names = model_metadata[
        "segment_names"
    ]

    predicted_segment_names = [
        segment_names[str(int(cluster))]
        for cluster in predicted_clusters
    ]

    # ------------------------------------------------------------------
    # Build scoring output
    # ------------------------------------------------------------------

    customer_segments = pd.DataFrame(
        {
            "CustomerID": df_customers["CustomerID"],
            "Cluster": predicted_clusters,
            "SegmentName": predicted_segment_names,
        }
    )

    # Each engineered customer must receive exactly one segment.
    if (
        len(customer_segments)
        != customer_segments["CustomerID"].nunique()
    ):
        raise ValueError(
            "Customer segment output contains duplicate CustomerIDs."
        )

    # ------------------------------------------------------------------
    # Persist predictions
    # ------------------------------------------------------------------

    output_path = Path(
        args.output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    customer_segments.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
    )

    print(
        f"Scored {len(customer_segments):,} customers."
    )

    print(
        "\nSegment distribution:"
    )

    print(
        customer_segments[
            "SegmentName"
        ].value_counts()
    )


if __name__ == "__main__":
    main()