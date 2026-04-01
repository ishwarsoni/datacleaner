"""Run a realistic end-to-end validation of the datacleaner library."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

import pandas as pd

# Ensure local src package is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datacleaner import analyze, clean


def build_messy_dataset() -> pd.DataFrame:
    """Create a realistic messy dataset for validation."""
    data = {
        "customer_id": [
            "C001", "C002", "C003", "C004", "C005", "C006", "C007", "C008", "C009", "C010",
            "C011", "C012", "C013", "C014", "C015", "C016", "C017", "C018", "C019", "C020",
        ],
        "age": [
            "25", "31", "29", None, "42", "38", "35", "unknown", "41", "27",
            "33", "29", None, "36", "120", "28", "30", "32", "26", "300",
        ],
        "income": [
            "50000", "62000", "58000", "54000", None, "71000", "68000", "65000", "60000", "56000",
            "59000", "61000", "62000", "63000", "1000000", "57000", "58000", "59000", "60000", "-50000",
        ],
        "city": [
            "New York", "Chicago", None, "Houston", "Seattle", "Chicago", "Austin", "Austin", "Miami", None,
            "Denver", "Denver", "Seattle", "Houston", "Austin", "Chicago", "New York", None, "Miami", "Miami",
        ],
        "signup_date": [
            "2024-01-10", "2024/02/11", "2024-03-12", "not_a_date", "2024-05-01", None, "2024-07-15", "2024-08-20", "2024-09-10", "2024-10-05",
            "2024-11-11", "2024-12-12", None, "2025-01-01", "2025-02-02", "2025-03-03", "2025-04-04", "2025-05-05", "2025-06-06", "2025-07-07",
        ],
        "notes": [
            None, None, "late payment", None, None, "vip", None, None, None, "call back",
            None, None, None, None, None, None, "prefers email", None, None, None,
        ],
    }

    df = pd.DataFrame(data)

    # Add exact duplicate rows to validate duplicate removal.
    df = pd.concat([df, df.iloc[[1, 4, 7]]], ignore_index=True)

    return df


def print_divider(title: str) -> None:
    """Print a readable section divider for console output."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def main() -> None:
    """Execute the real-data validation workflow."""
    df = build_messy_dataset()

    rows_before, cols_before = df.shape

    print_divider("Initial Dataset")
    print(f"Shape: {df.shape}")
    print("Dtypes:")
    print(df.dtypes)
    print("Preview:")
    print(df.head(5))

    print_divider("Analyze Before Cleaning")
    analysis_report = analyze(df)
    pprint(analysis_report, sort_dicts=False)

    print_divider("Run Cleaning")
    cleaned_df, cleaning_report = clean(df, verbose=True)

    rows_after, cols_after = cleaned_df.shape

    print_divider("Cleaned Dataset")
    print(f"Shape: {cleaned_df.shape}")
    print("Dtypes:")
    print(cleaned_df.dtypes)
    print("Preview:")
    print(cleaned_df.head(5))

    print_divider("Cleaning Report")
    pprint(cleaning_report, sort_dicts=False)

    print_divider("Before vs After")
    print(f"Rows: {rows_before} -> {rows_after} (change: {rows_after - rows_before})")
    print(f"Columns: {cols_before} -> {cols_after} (change: {cols_after - cols_before})")

    rows_removed = rows_before - rows_after
    columns_dropped = cols_before - cols_after

    row_loss_pct = (rows_removed / rows_before * 100) if rows_before else 0.0
    column_loss_pct = (columns_dropped / cols_before * 100) if cols_before else 0.0

    print(f"Row loss: {row_loss_pct:.2f}%")
    print(f"Column loss: {column_loss_pct:.2f}%")

    if row_loss_pct > 30:
        print("WARNING: More than 30% of rows were removed during cleaning.")
    if column_loss_pct > 30:
        print("WARNING: More than 30% of columns were dropped during cleaning.")


if __name__ == "__main__":
    main()
