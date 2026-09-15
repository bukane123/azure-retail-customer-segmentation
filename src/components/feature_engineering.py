"""
Customer feature engineering component.

Transforms cleaned Online Retail transaction datasets into one customer-level
record containing purchasing, tenure, postage and return behavioural features.

Inputs
------
paid_purchases:
    Cleaned positive-value paid transaction dataset.

returns:
    Cleaned cancelled/returned transaction dataset.

Output
------
customer_features:
    Customer-level feature dataset used by the preprocessing component.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    """Parse command-line inputs supplied by Azure ML or local execution."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paid_purchases",
        type=str,
        required=True,
        help="Path to cleaned paid-purchase Parquet dataset.",
    )

    parser.add_argument(
        "--returns",
        type=str,
        required=True,
        help="Path to cleaned returns Parquet dataset.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the engineered customer feature dataset will be saved.",
    )

    return parser.parse_args()


def main():
    """Run the customer feature engineering workflow."""

    args = parse_args()

    # ------------------------------------------------------------------
    # Load and validate source datasets
    # ------------------------------------------------------------------

    df_paid = pd.read_parquet(args.paid_purchases)
    df_returns = pd.read_parquet(args.returns)

    # Standardise key data types so feature calculations behave
    # consistently across local and Azure ML execution environments.
    df_paid["CustomerID"] = df_paid["CustomerID"].astype("Int64")
    df_returns["CustomerID"] = df_returns["CustomerID"].astype("Int64")

    df_paid["InvoiceDate"] = pd.to_datetime(df_paid["InvoiceDate"])
    df_returns["InvoiceDate"] = pd.to_datetime(df_returns["InvoiceDate"])

    # ------------------------------------------------------------------
    # Core merchandise behaviour
    # ------------------------------------------------------------------

    # Core RFM and product features use merchandise transactions only.
    # Postage and other non-merchandise activity are engineered separately.
    df_merchandise = df_paid.loc[
        df_paid["TransactionCategory"] == "Merchandise"
    ].copy()

    # Use one day after the final observed transaction as the fixed
    # reference date for customer recency.
    reference_date = (
        df_merchandise["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    customer_recency = (
        df_merchandise
        .groupby("CustomerID")["InvoiceDate"]
        .max()
        .reset_index(name="LastPurchaseDate")
    )

    customer_recency["Recency"] = (
        reference_date
        - customer_recency["LastPurchaseDate"]
    ).dt.days

    customer_frequency = (
        df_merchandise
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index(name="Frequency")
    )

    customer_monetary = (
        df_merchandise
        .groupby("CustomerID")["LineValue"]
        .sum()
        .reset_index(name="MonetaryValue")
    )

    customer_features = (
        customer_recency
        .merge(
            customer_frequency,
            on="CustomerID",
            how="inner",
        )
        .merge(
            customer_monetary,
            on="CustomerID",
            how="inner",
        )
    )

    # Average spend generated per merchandise order.
    customer_features["AverageOrderValue"] = (
        customer_features["MonetaryValue"]
        / customer_features["Frequency"]
    )

    # Number of distinct merchandise products purchased.
    customer_product_diversity = (
        df_merchandise
        .groupby("CustomerID")["StockCode"]
        .nunique()
        .reset_index(name="UniqueProducts")
    )

    customer_features = customer_features.merge(
        customer_product_diversity,
        on="CustomerID",
        how="left",
    )

    # ------------------------------------------------------------------
    # Customer tenure
    # ------------------------------------------------------------------

    customer_first_purchase = (
        df_merchandise
        .groupby("CustomerID")["InvoiceDate"]
        .min()
        .reset_index(name="FirstPurchaseDate")
    )

    customer_features = customer_features.merge(
        customer_first_purchase,
        on="CustomerID",
        how="left",
    )

    customer_features["CustomerTenureDays"] = (
        customer_features["LastPurchaseDate"]
        - customer_features["FirstPurchaseDate"]
    ).dt.days

    # ------------------------------------------------------------------
    # Purchase quantity behaviour
    # ------------------------------------------------------------------

    customer_quantity = (
        df_merchandise
        .groupby("CustomerID")["Quantity"]
        .sum()
        .reset_index(name="TotalQuantity")
    )

    customer_features = customer_features.merge(
        customer_quantity,
        on="CustomerID",
        how="left",
    )

    customer_features["AverageQuantityPerOrder"] = (
        customer_features["TotalQuantity"]
        / customer_features["Frequency"]
    )

    # ------------------------------------------------------------------
    # Purchase interval behaviour
    # ------------------------------------------------------------------

    # Collapse merchandise lines to one record per customer invoice before
    # calculating intervals between successive purchases.
    customer_invoices = (
        df_merchandise
        .groupby(
            ["CustomerID", "InvoiceNo"]
        )["InvoiceDate"]
        .min()
        .reset_index()
        .sort_values(
            ["CustomerID", "InvoiceDate"]
        )
    )

    customer_invoices["DaysSincePreviousPurchase"] = (
        customer_invoices
        .groupby("CustomerID")["InvoiceDate"]
        .diff()
        .dt.total_seconds()
        / 86400
    )

    customer_purchase_interval = (
        customer_invoices
        .groupby("CustomerID")["DaysSincePreviousPurchase"]
        .mean()
        .reset_index(
            name="AverageDaysBetweenPurchases"
        )
    )

    customer_features = customer_features.merge(
        customer_purchase_interval,
        on="CustomerID",
        how="left",
    )

    # Binary interpretation field retained for downstream profiling.
    customer_features["RepeatCustomer"] = (
        customer_features["Frequency"] > 1
    ).astype(int)

    # ------------------------------------------------------------------
    # Postage behaviour
    # ------------------------------------------------------------------

    df_postage = df_paid.loc[
        df_paid["TransactionCategory"] == "Postage"
    ].copy()

    customer_postage = (
        df_postage
        .groupby("CustomerID")
        .agg(
            PostageSpend=("LineValue", "sum"),
            PostageInvoices=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    customer_features = customer_features.merge(
        customer_postage,
        on="CustomerID",
        how="left",
    )

    # No matching postage transaction represents genuine zero activity.
    customer_features[
        ["PostageSpend", "PostageInvoices"]
    ] = customer_features[
        ["PostageSpend", "PostageInvoices"]
    ].fillna(0)

    customer_paid_invoices = (
        df_paid
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index(name="PaidInvoices")
    )

    customer_features = customer_features.merge(
        customer_paid_invoices,
        on="CustomerID",
        how="left",
    )

    customer_features["PostageInvoiceShare"] = (
        customer_features["PostageInvoices"]
        / customer_features["PaidInvoices"]
    )

    customer_features["HasPostage"] = (
        customer_features["PostageInvoices"] > 0
    ).astype(int)

    # ------------------------------------------------------------------
    # Return and cancellation behaviour
    # ------------------------------------------------------------------

    customer_returns = (
        df_returns
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index(name="ReturnInvoices")
    )

    customer_features = customer_features.merge(
        customer_returns,
        on="CustomerID",
        how="left",
    )

    customer_features["ReturnInvoices"] = (
        customer_features["ReturnInvoices"]
        .fillna(0)
        .astype(int)
    )

    # Convert negative cancellation quantities into positive monetary
    # magnitudes for customer return-behaviour measurement.
    df_returns["ReturnLineValue"] = (
        df_returns["Quantity"]
        * df_returns["UnitPrice"]
    ).abs()

    customer_return_value = (
        df_returns
        .groupby("CustomerID")["ReturnLineValue"]
        .sum()
        .reset_index(name="ReturnValue")
    )

    customer_features = customer_features.merge(
        customer_return_value,
        on="CustomerID",
        how="left",
    )

    customer_features["ReturnValue"] = (
        customer_features["ReturnValue"]
        .fillna(0)
    )

    # Exclude known non-merchandise cancellation types from the
    # merchandise-specific return measure.
    non_merchandise_return_categories = {
        "M": "Manual transaction",
        "POST": "Postage",
        "CRUK": "Commission",
        "D": "Discount",
    }

    df_returns["ReturnCategory"] = (
        df_returns["StockCode"]
        .map(non_merchandise_return_categories)
        .fillna("Merchandise")
    )

    df_merchandise_returns = df_returns.loc[
        df_returns["ReturnCategory"] == "Merchandise"
    ].copy()

    customer_merchandise_return_value = (
        df_merchandise_returns
        .groupby("CustomerID")["ReturnLineValue"]
        .sum()
        .reset_index(name="MerchandiseReturnValue")
    )

    customer_features = customer_features.merge(
        customer_merchandise_return_value,
        on="CustomerID",
        how="left",
    )

    customer_features["MerchandiseReturnValue"] = (
        customer_features["MerchandiseReturnValue"]
        .fillna(0)
    )

    # Compare merchandise returns with merchandise purchases observed
    # within the available dataset period.
    customer_features[
        "ObservedMerchandiseReturnValueRate"
    ] = (
        customer_features["MerchandiseReturnValue"]
        / customer_features["MonetaryValue"]
    )

    customer_features["HasMerchandiseReturn"] = (
        customer_features["MerchandiseReturnValue"] > 0
    ).astype(int)

    # ------------------------------------------------------------------
    # Final validation and schema
    # ------------------------------------------------------------------

    customer_features["CustomerID"] = (
        customer_features["CustomerID"]
        .astype("Int64")
    )

    customer_features["PostageInvoices"] = (
        customer_features["PostageInvoices"]
        .astype("int64")
    )

    # Preserve the exact validated feature-engineering schema from
    # the development notebook.
    output_columns = [
        "CustomerID",
        "LastPurchaseDate",
        "Recency",
        "Frequency",
        "MonetaryValue",
        "AverageOrderValue",
        "UniqueProducts",
        "FirstPurchaseDate",
        "CustomerTenureDays",
        "TotalQuantity",
        "AverageQuantityPerOrder",
        "AverageDaysBetweenPurchases",
        "RepeatCustomer",
        "PostageSpend",
        "PostageInvoices",
        "PaidInvoices",
        "PostageInvoiceShare",
        "HasPostage",
        "ReturnInvoices",
        "ReturnValue",
        "MerchandiseReturnValue",
        "ObservedMerchandiseReturnValueRate",
        "HasMerchandiseReturn",
    ]

    customer_features = customer_features[
        output_columns
    ].copy()

    # Each output row must represent exactly one customer.
    assert (
        len(customer_features)
        == customer_features["CustomerID"].nunique()
    )

    # All fields except AverageDaysBetweenPurchases should be complete.
    unexpected_missing = (
        customer_features
        .drop(
            columns=["AverageDaysBetweenPurchases"]
        )
        .isna()
        .sum()
        .sum()
    )

    assert unexpected_missing == 0

    # ------------------------------------------------------------------
    # Persist component output
    # ------------------------------------------------------------------

    output_path = Path(args.output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    customer_features.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
    )

    print(
        f"Customer features created: "
        f"{customer_features.shape}"
    )

    print(
        f"Output written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()