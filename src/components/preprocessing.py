"""
Customer feature preprocessing component.

Prepares engineered customer-level behavioural features for clustering by
applying the transformations, feature selection and scaling decisions validated
during model development.

Input
-----
customer_features:
    Engineered customer-level feature dataset.

Outputs
-------
clustering_input:
    CustomerID plus the final eight scaled clustering features.

preprocessing_artifacts:
    Fitted StandardScaler and preprocessing metadata required for
    reproducible future scoring.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args():
    """Parse command-line inputs and outputs."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--customer_features",
        type=str,
        required=True,
        help="Path to the engineered customer feature Parquet dataset.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the scaled clustering-input dataset will be saved.",
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        required=True,
        help="Directory where fitted preprocessing artefacts will be saved.",
    )

    return parser.parse_args()


def main():
    """Run the validated customer feature preprocessing workflow."""

    args = parse_args()

    # ------------------------------------------------------------------
    # Load customer features
    # ------------------------------------------------------------------

    df_customers = pd.read_parquet(
        args.customer_features
    )

    # CustomerID is retained for traceability but is never supplied
    # to the clustering algorithm as a modelling feature.
    df_customers["CustomerID"] = (
        df_customers["CustomerID"]
        .astype("Int64")
    )

    # ------------------------------------------------------------------
    # Define preprocessing feature set
    # ------------------------------------------------------------------

    # AverageDaysBetweenPurchases is deliberately excluded because it
    # contains structural missingness for single-purchase customers and
    # largely duplicates Frequency and CustomerTenureDays information.
    model_feature_columns = [
        "Recency",
        "Frequency",
        "MonetaryValue",
        "AverageOrderValue",
        "UniqueProducts",
        "CustomerTenureDays",
        "TotalQuantity",
        "AverageQuantityPerOrder",
        "PostageSpend",
        "PostageInvoices",
        "PostageInvoiceShare",
        "ReturnInvoices",
        "MerchandiseReturnValue",
        "ObservedMerchandiseReturnValueRate",
    ]

    X = df_customers[
        model_feature_columns
    ].copy()

    # The selected modelling candidates must be complete before
    # transformation begins.
    assert X.isna().sum().sum() == 0

    # ------------------------------------------------------------------
    # Apply validated log transformations
    # ------------------------------------------------------------------

    # These features showed substantial positive skewness during
    # exploratory preprocessing and are transformed using log1p.
    # log1p safely preserves genuine zero-valued observations.
    log_transform_features = [
        "Recency",
        "Frequency",
        "MonetaryValue",
        "AverageOrderValue",
        "UniqueProducts",
        "TotalQuantity",
        "AverageQuantityPerOrder",
        "PostageSpend",
        "PostageInvoices",
        "ReturnInvoices",
        "MerchandiseReturnValue",
        "ObservedMerchandiseReturnValueRate",
    ]

    X_transformed = X.copy()

    for feature in log_transform_features:
        X_transformed[feature] = np.log1p(
            X_transformed[feature]
        )

    # Validate that transformation has not introduced invalid values.
    assert X_transformed.isna().sum().sum() == 0
    assert np.isinf(
        X_transformed.to_numpy()
    ).sum() == 0

    # ------------------------------------------------------------------
    # Final clustering feature selection
    # ------------------------------------------------------------------

    # The final eight features were selected after post-transformation
    # redundancy analysis to represent distinct customer behaviours
    # without giving highly correlated measures excessive influence.
    final_model_features = [
        "Recency",
        "Frequency",
        "MonetaryValue",
        "UniqueProducts",
        "CustomerTenureDays",
        "AverageQuantityPerOrder",
        "PostageInvoiceShare",
        "ObservedMerchandiseReturnValueRate",
    ]

    X_final = X_transformed[
        final_model_features
    ].copy()

    # ------------------------------------------------------------------
    # Scale final modelling features
    # ------------------------------------------------------------------

    # StandardScaler places all clustering features on comparable scales.
    # This is essential because K-Means uses Euclidean distance.
    scaler = StandardScaler()

    X_scaled_array = scaler.fit_transform(
        X_final
    )

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=final_model_features,
        index=X_final.index,
    )

    # Confirm that the final modelling matrix is complete and finite.
    assert X_scaled.isna().sum().sum() == 0
    assert np.isinf(
        X_scaled.to_numpy()
    ).sum() == 0

    # ------------------------------------------------------------------
    # Reattach customer identifier
    # ------------------------------------------------------------------

    # CustomerID is included in the persisted output for traceability,
    # but downstream model training must exclude it from the feature matrix.
    clustering_input = pd.concat(
        [
            df_customers.loc[
                X_scaled.index,
                ["CustomerID"],
            ],
            X_scaled,
        ],
        axis=1,
    )

    # Ensure each customer appears exactly once.
    assert (
        len(clustering_input)
        == clustering_input["CustomerID"].nunique()
    )

    # ------------------------------------------------------------------
    # Persist clustering input
    # ------------------------------------------------------------------

    output_path = Path(args.output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clustering_input.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
    )

    # ------------------------------------------------------------------
    # Persist reusable preprocessing artefacts
    # ------------------------------------------------------------------

    artifact_dir = Path(args.artifact_dir)

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve the fitted means and standard deviations learned from
    # the training customer population.
    scaler_path = (
        artifact_dir
        / "standard_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    # Store the exact modelling feature order expected by the scaler
    # and clustering model.
    feature_order_path = (
        artifact_dir
        / "feature_order.json"
    )

    with open(
        feature_order_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_model_features,
            file,
            indent=4,
        )

    # Store transformation configuration so future scoring applies
    # exactly the same feature preparation as model training.
    preprocessing_metadata = {
        "model_feature_candidates": model_feature_columns,
        "log_transform_features": log_transform_features,
        "final_model_features": final_model_features,
        "scaler": "StandardScaler",
        "scaler_file": "standard_scaler.joblib",
    }

    metadata_path = (
        artifact_dir
        / "preprocessing_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            preprocessing_metadata,
            file,
            indent=4,
        )

    print(
        f"Clustering input created: "
        f"{clustering_input.shape}"
    )

    print(
        f"Clustering input written to: "
        f"{output_path}"
    )

    print(
        f"Preprocessing artefacts written to: "
        f"{artifact_dir}"
    )


if __name__ == "__main__":
    main()