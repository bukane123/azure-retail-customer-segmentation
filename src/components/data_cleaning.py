import argparse
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Required transaction schema
# ------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def validate_input(df: pd.DataFrame) -> None:
    """
    Confirm that the standardised transaction dataset contains all
    fields required by the cleaning and downstream feature pipeline.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cleaning input is missing required columns: "
            f"{missing_columns}"
        )


def save_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Write a dataframe to an Azure ML-managed Parquet output path.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        path,
        index=False,
    )

# ------------------------------------------------------------------
# Transaction category classification
# ------------------------------------------------------------------

def classify_transaction_category(stock_code: str) -> str:
    """
    Classify transaction lines using the same StockCode rules
    established during model development.
    """

    stock_code = str(stock_code).strip().upper()

    if stock_code in {"POST", "DOT"}:
        return "Postage"

    if stock_code == "M":
        return "Manual transaction"

    if stock_code == "BANK CHARGES":
        return "Bank charge"

    if stock_code == "PADS":
        return "Nominal-value item"

    return "Merchandise"

# ------------------------------------------------------------------
# Main cleaning workflow
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Validated transaction dataset in Parquet format.",
    )

    parser.add_argument(
        "--paid_purchases_output",
        type=str,
        required=True,
        help="Output path for paid purchase transactions.",
    )

    parser.add_argument(
        "--returns_output",
        type=str,
        required=True,
        help="Output path for return/cancellation transactions.",
    )

    parser.add_argument(
        "--zero_price_output",
        type=str,
        required=True,
        help="Output path for zero-price transactions retained for audit.",
    )

    args = parser.parse_args()


    # --------------------------------------------------------------
    # Load validated transactions
    # --------------------------------------------------------------

    df = pd.read_parquet(
        args.input_data
    )

    validate_input(df)

    print("Input rows:", len(df))


    # --------------------------------------------------------------
    # Remove transactions without an identifiable customer
    # --------------------------------------------------------------
    # Customer-level segmentation cannot be created where CustomerID
    # is missing, so these transactions are excluded from modelling.

    df_customer = df[
        df["CustomerID"].notna()
    ].copy()

    print(
        "Rows after removing missing CustomerID:",
        len(df_customer)
    )


    # --------------------------------------------------------------
    # Remove exact duplicate transaction rows
    # --------------------------------------------------------------
    # Exact duplicates were identified during development across all
    # transaction fields. Retaining only the first occurrence prevents
    # duplicated source records from overstating customer behaviour.

    duplicate_count = int(
        df_customer.duplicated().sum()
    )

    df_customer = (
        df_customer
        .drop_duplicates(keep="first")
        .copy()
    )

    print(
        "Exact duplicate rows removed:",
        duplicate_count
    )

    print(
        "Rows after duplicate removal:",
        len(df_customer)
    )

    # --------------------------------------------------------------
    # Categorise transaction types
    # --------------------------------------------------------------
    # Feature engineering relies on TransactionCategory to distinguish
    # merchandise from postage and other non-merchandise transactions.
    # These are the same StockCode rules used during model development.

    df_customer["TransactionCategory"] = (
        df_customer["StockCode"]
        .apply(classify_transaction_category)
    )

    # --------------------------------------------------------------
    # Calculate transaction line value
    # --------------------------------------------------------------
    # LineValue represents the monetary value of each transaction
    # line and is required by downstream feature engineering.
    # For returns, negative Quantity naturally produces a negative
    # LineValue, preserving the original transaction direction.

    df_customer["LineValue"] = (
        df_customer["Quantity"]
        * df_customer["UnitPrice"]
    )

    print(
        "Transaction categories created:",
        df_customer["TransactionCategory"]
        .value_counts()
        .to_dict()
    )


    # --------------------------------------------------------------
    # Identify returns / cancellations
    # --------------------------------------------------------------


    # --------------------------------------------------------------
    # Identify returns / cancellations
    # --------------------------------------------------------------

    cancellation_mask = (
        df_customer["InvoiceNo"]
        .astype("string")
        .str.startswith("C", na=False)
    )

    negative_quantity_mask = (
        df_customer["Quantity"] < 0
    )

    mismatch_mask = (
        cancellation_mask
        != negative_quantity_mask
    )

    if mismatch_mask.any():
        raise ValueError(
            "Return classification validation failed: "
            f"{int(mismatch_mask.sum())} transaction(s) have "
            "inconsistent cancellation and negative-quantity indicators."
        )

    # --------------------------------------------------------------
    # Identify returns / cancellations
    # --------------------------------------------------------------

    returns = df_customer[
        cancellation_mask
    ].copy()

    # Keep the returns dataset consistent with the schema used during
    # model development. The downstream return-feature logic expects
    # the original eight transaction fields.
    returns = returns[
        REQUIRED_COLUMNS
    ].copy()


    # --------------------------------------------------------------
    # Identify purchase transactions
    # --------------------------------------------------------------

    purchases = df_customer[
        ~cancellation_mask
    ].copy()


    # --------------------------------------------------------------
    # Separate zero-price transactions
    # --------------------------------------------------------------
    # Zero-price transactions are retained for traceability but are
    # excluded from the core paid-purchase dataset used for modelling.

    zero_price_transactions = purchases[
        purchases["UnitPrice"] == 0
    ].copy()

    paid_purchases = purchases[
        (purchases["Quantity"] > 0)
        & (purchases["UnitPrice"] > 0)
    ].copy()




    # --------------------------------------------------------------
    # Save cleaned datasets
    # --------------------------------------------------------------

    save_parquet(
        paid_purchases,
        args.paid_purchases_output,
    )

    save_parquet(
        returns,
        args.returns_output,
    )

    save_parquet(
        zero_price_transactions,
        args.zero_price_output,
    )


    # --------------------------------------------------------------
    # Cleaning summary
    # --------------------------------------------------------------

    print("\n=== CLEANING SUMMARY ===")
    print("Original rows:", len(df))
    print(
        "Missing CustomerID removed:",
        int(df["CustomerID"].isna().sum())
    )
    print("Rows after duplicate removal:", len(df_customer))
    print("Returns:", len(returns))
    print(
        "Zero-price transactions:",
        len(zero_price_transactions)
    )
    print("Paid purchases:", len(paid_purchases))


if __name__ == "__main__":
    main()
