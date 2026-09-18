import argparse
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Expected business input schema
# ------------------------------------------------------------------

# These are the columns the customer-segmentation workflow requires
# from the raw business transaction extract.
EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def load_raw_data(input_path: str) -> pd.DataFrame:
    """
    Load the raw business transaction file.

    The production workflow supports the business supplying either
    an Excel workbook or CSV file with the agreed transaction schema.
    """

    file_path = Path(input_path)
    file_suffix = file_path.suffix.lower()

    if file_suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    if file_suffix == ".csv":
        return pd.read_csv(file_path)

    # Azure ML normally preserves the original filename, but this
    # fallback makes the component more robust if the mounted path
    # does not expose a recognised extension.
    try:
        return pd.read_excel(file_path)
    except Exception:
        try:
            return pd.read_csv(file_path)
        except Exception as exc:
            raise ValueError(
                "Unable to read the input file. "
                "Supported formats are Excel (.xlsx/.xls) and CSV."
            ) from exc


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the incoming business file contains every column
    required by the downstream segmentation workflow.
    """

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Incoming dataset is missing required columns: "
            f"{missing_columns}"
        )


def standardise_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise critical data types without applying any business
    cleaning rules. Cleaning remains the responsibility of the
    downstream data-cleaning component.
    """

    df = df.copy()

    # Identifier fields may contain both numeric-looking and alphanumeric
    # values. Store them consistently as strings rather than numbers.
    df["InvoiceNo"] = df["InvoiceNo"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")

    # Text fields are also standardised explicitly while preserving
    # missing values for the downstream cleaning component to handle.
    df["Description"] = df["Description"].astype("string")
    df["Country"] = df["Country"].astype("string")

    # Invoice dates must be valid dates because recency and tenure
    # calculations later depend on this field.
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="raise",
    )

    # Quantity and UnitPrice must be numeric for transaction-value
    # calculations and purchase/return classification.
    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="raise",
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="raise",
    )

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Raw Excel or CSV transaction file.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the standardised Parquet dataset.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load and validate the business input
    # --------------------------------------------------------------

    df = load_raw_data(args.input_data)

    print("Raw input shape:", df.shape)

    validate_schema(df)

    # Keep only the agreed business schema and preserve its order.
    df = df[EXPECTED_COLUMNS]

    df = standardise_data_types(df)

    # --------------------------------------------------------------
    # Convert to internal Parquet format
    # --------------------------------------------------------------

    output_path = Path(args.output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print("Validated dataset shape:", df.shape)
    print("Validated columns:", df.columns.tolist())
    print("Standardised Parquet written to:", output_path)


if __name__ == "__main__":
    main()
