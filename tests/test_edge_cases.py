"""Stress-test script for datacleaner edge cases."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys
import traceback

import pandas as pd

# Ensure local package import works when run directly from tests/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datacleaner import analyze, clean


def section(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 18 + f" {title} " + "=" * 18)


def print_shape_change(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    """Print before/after shape and row-column deltas."""
    before_rows, before_cols = before_df.shape
    after_rows, after_cols = after_df.shape
    print(f"Before shape: {before_df.shape}")
    print(f"After shape:  {after_df.shape}")
    print(f"Rows change:   {after_rows - before_rows}")
    print(f"Cols change:   {after_cols - before_cols}")


def run_case(case_name: str, case_func) -> None:
    """Run a case and keep the full script alive if an error occurs."""
    section(case_name)
    try:
        case_func()
    except Exception as exc:  # pragma: no cover - script-style safety
        print(f"Case failed with error: {exc}")
        traceback.print_exc()


def case_extreme_missing_values() -> None:
    """Case 1: Verify columns with >70% missing are dropped."""
    df = pd.DataFrame(
        {
            "mostly_missing": [None, None, None, None, None, None, None, "x", None, None],
            "value": [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
            "category": ["a", "a", None, "b", "b", "b", None, "a", "b", "a"],
        }
    )

    cleaned_df, report = clean(df)

    print_shape_change(df, cleaned_df)
    print("Columns dropped:", report["steps"]["missing_values"].get("columns_dropped", []))
    print("Is 'mostly_missing' dropped?", "mostly_missing" not in cleaned_df.columns)


def case_dirty_string_values() -> None:
    """Case 2: Observe behavior with placeholder dirty string values."""
    dirty_values = ["N/A", "na", "-", "", "unknown", "??", "valid", None, "na", "N/A"]
    df = pd.DataFrame(
        {
            "raw_text": dirty_values,
            "status": ["ok", "ok", "ok", "ok", None, "bad", "ok", "ok", "bad", "ok"],
            "amount": ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"],
        }
    )

    cleaned_df, report = clean(df)

    print_shape_change(df, cleaned_df)
    print("Unique dirty placeholders before:", sorted({v for v in dirty_values if isinstance(v, str)}))
    print("Values filled per column:", report["steps"]["missing_values"].get("values_filled_per_column", {}))
    print("Method used:", report["steps"]["missing_values"].get("method_used", {}))


def case_mixed_numeric_strings() -> None:
    """Case 3: Verify safe conversion for mixed numeric strings."""
    df = pd.DataFrame(
        {
            "mixed_num": ["100", "200", "abc", "300", None, "500"],
            "other": [1, 2, 3, 4, 5, 6],
        }
    )

    cleaned_df, report = clean(df)

    print_shape_change(df, cleaned_df)
    print("Original dtype:", df["mixed_num"].dtype)
    print("Cleaned dtype:", cleaned_df["mixed_num"].dtype)
    print("Columns converted:", report["steps"]["datatypes"].get("columns_converted", {}))
    print("Sample values after cleaning:", cleaned_df["mixed_num"].head(6).tolist())


def case_already_clean_dataset() -> None:
    """Case 4: Ensure a clean dataset is not unnecessarily modified."""
    df = pd.DataFrame(
        {
            "age": [22, 25, 31, 28, 35, 40],
            "income": [45000, 52000, 61000, 58000, 73000, 81000],
            "city": ["A", "B", "C", "D", "E", "F"],
            "signup_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
            ),
        }
    )

    cleaned_df, report = clean(df)

    print_shape_change(df, cleaned_df)
    print("DataFrame unchanged by value?", cleaned_df.equals(df))
    print("Actions summary:")
    pprint(report.get("summary", {}).get("actions_summary", {}), sort_dicts=False)


def case_small_dataset_outliers() -> None:
    """Case 5: Ensure outlier handling does not remove all rows on tiny datasets."""
    df = pd.DataFrame(
        {
            "score": [10, 11, 12, 13, 999],
            "age": [20, 21, 22, 23, 24],
            "city": ["X", "Y", "Z", "X", "Y"],
        }
    )

    cleaned_df, report = clean(df, outlier_method="remove")

    print_shape_change(df, cleaned_df)
    rows_removed = report["steps"]["outliers"].get("rows_removed", 0)
    print("Outlier rows removed:", rows_removed)
    print("Rows remaining:", len(cleaned_df))
    print("All rows removed?", len(cleaned_df) == 0)


def case_high_cardinality_analysis() -> None:
    """Case 6: Verify analyze flags high-cardinality and possible ID columns."""
    row_count = 120
    df = pd.DataFrame(
        {
            "session_id": [f"SID_{i:04d}" for i in range(row_count)],
            "email": [f"user{i}@example.com" for i in range(row_count)],
            "group": ["A" if i % 2 == 0 else "B" for i in range(row_count)],
            "score": [i % 10 for i in range(row_count)],
        }
    )

    analysis_report = analyze(df)
    warnings = analysis_report.get("warnings", {})

    print("Shape:", df.shape)
    print("High cardinality columns:", warnings.get("high_cardinality_columns", []))
    print("Possible ID columns:", warnings.get("possible_id_columns", []))


def main() -> None:
    """Execute all edge-case stress tests with readable output."""
    print("Running datacleaner edge-case stress tests...")

    run_case("1) Extreme Missing Values", case_extreme_missing_values)
    run_case("2) Dirty String Values", case_dirty_string_values)
    run_case("3) Mixed Numeric Strings", case_mixed_numeric_strings)
    run_case("4) Already Clean Dataset", case_already_clean_dataset)
    run_case("5) Small Dataset Outliers", case_small_dataset_outliers)
    run_case("6) High Cardinality Analysis", case_high_cardinality_analysis)

    section("Done")
    print("All cases executed. Check outputs above for behavior and edge-case handling.")


if __name__ == "__main__":
    main()
