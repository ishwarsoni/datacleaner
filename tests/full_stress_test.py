"""Comprehensive stress test for the datacleaner library."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
import io
import sys

import numpy as np
import pandas as pd

# Ensure local src package is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datacleaner import analyze, clean


def divider(title: str) -> None:
    """Print a readable section separator."""
    print("\n" + "=" * 25)
    print(title)
    print("=" * 25)


def print_info(df: pd.DataFrame, label: str) -> None:
    """Print DataFrame info in a consistent readable format."""
    print(f"\n{label} shape: {df.shape}")
    print(f"\n{label} head:")
    print(df.head())

    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    print(f"\n{label} info:")
    print(info_buffer.getvalue())

    print(f"\n{label} describe(include='all'):")
    print(df.describe(include="all"))


def build_messy_dataset(row_count: int = 200) -> pd.DataFrame:
    """Create a realistic messy dataset with multiple quality issues."""
    rng = np.random.default_rng(42)

    age = rng.normal(loc=35, scale=10, size=row_count)
    income = rng.normal(loc=65000, scale=15000, size=row_count)
    city = rng.choice(["New York", "Chicago", "Austin", "Seattle", "Miami"], size=row_count)

    # Mixed numeric strings with invalid values.
    mixed_numeric = rng.choice(["100", "200", "abc", "300", "??"], size=row_count)

    # Dirty string placeholders.
    notes = rng.choice(["N/A", "na", "-", "", "unknown", "??", "ok", "needs_review"], size=row_count)

    # High-cardinality column (almost unique).
    high_cardinality = [f"REC_{i:04d}" for i in range(row_count)]
    for i in range(5):
        high_cardinality[-(i + 1)] = high_cardinality[i]

    df = pd.DataFrame(
        {
            "record_id": high_cardinality,
            "age": age.round(0),
            "income": income.round(2),
            "city": city,
            "dirty_text": notes,
            "mixed_numeric": mixed_numeric,
        }
    )

    # Missing values in numeric and categorical columns.
    missing_numeric_idx = rng.choice(row_count, size=30, replace=False)
    missing_categorical_idx = rng.choice(row_count, size=25, replace=False)
    df.loc[missing_numeric_idx, "income"] = np.nan
    df.loc[missing_categorical_idx, "city"] = np.nan

    # Outliers.
    df.loc[0, "income"] = 10000
    df.loc[1, "income"] = -9999
    df.loc[2, "age"] = 150

    # Intentional duplicates while keeping total rows at 200.
    df.iloc[180:185] = df.iloc[0:5].to_numpy()

    return df


def run_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run clean and normalize return shape for this test script."""
    result = clean(df, verbose=True)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def robustness_case(name: str, df: pd.DataFrame) -> bool:
    """Run one robustness test and return pass/fail."""
    try:
        _ = clean(df)
        print(f"{name}: PASSED")
        return True
    except Exception as exc:
        print(f"{name}: FAILED ({exc})")
        return False


def main() -> None:
    """Execute full stress test workflow."""
    warnings_triggered: list[str] = []
    total_tests_run = 0

    divider("STEP 1: CREATE MESSY DATASET")
    df = build_messy_dataset(row_count=200)

    divider("STEP 2: PRINT INITIAL STATE")
    rows_before, cols_before = df.shape
    dtypes_before = df.dtypes.copy()
    print_info(df, "Raw dataset")

    divider("STEP 3: RUN ANALYZE")
    analysis = analyze(df)
    print("Analysis output:")
    pprint(analysis, sort_dicts=False)

    divider("STEP 4: RUN CLEANING")
    df_clean, cleaning_report = run_clean(df)

    divider("STEP 5: PRINT FINAL STATE")
    rows_after, cols_after = df_clean.shape
    dtypes_after = df_clean.dtypes.copy()
    print_info(df_clean, "Cleaned dataset")
    print("\nCleaning report:")
    pprint(cleaning_report, sort_dicts=False)

    divider("STEP 6: DATA LOSS CHECK")
    rows_removed = rows_before - rows_after
    cols_dropped = cols_before - cols_after
    row_loss_pct = (rows_removed / rows_before * 100) if rows_before else 0.0
    col_loss_pct = (cols_dropped / cols_before * 100) if cols_before else 0.0

    print(f"Rows before vs after: {rows_before} -> {rows_after}")
    print(f"Columns before vs after: {cols_before} -> {cols_after}")
    print(f"Row loss percentage: {row_loss_pct:.2f}%")
    print(f"Column loss percentage: {col_loss_pct:.2f}%")

    total_tests_run += 2
    if row_loss_pct > 30:
        warning = "WARNING: More than 30% rows removed."
        print(warning)
        warnings_triggered.append(warning)
    if col_loss_pct > 30:
        warning = "WARNING: More than 30% columns dropped."
        print(warning)
        warnings_triggered.append(warning)

    divider("STEP 7: DATATYPE VALIDATION")
    print("Dtypes before cleaning:")
    print(dtypes_before)
    print("\nDtypes after cleaning:")
    print(dtypes_after)

    unexpected_dtype_changes: list[str] = []
    shared_columns = [col for col in dtypes_before.index if col in dtypes_after.index]
    for col in shared_columns:
        before_dtype = str(dtypes_before[col])
        after_dtype = str(dtypes_after[col])
        if before_dtype != after_dtype:
            print(f"Changed dtype: {col}: {before_dtype} -> {after_dtype}")
            # Heuristic: object/string to typed conversions are usually expected.
            if not (before_dtype in {"object", "string"} and after_dtype != "object"):
                unexpected_dtype_changes.append(f"{col}: {before_dtype} -> {after_dtype}")

    total_tests_run += 1
    if unexpected_dtype_changes:
        warning = "WARNING: Unexpected dtype changes detected."
        print(warning)
        for item in unexpected_dtype_changes:
            print(f"  - {item}")
        warnings_triggered.append(warning)

    divider("STEP 8: ROBUSTNESS TESTS")
    robustness_results = []

    robustness_results.append(
        robustness_case("1) Empty dataframe", pd.DataFrame())
    )

    robustness_results.append(
        robustness_case(
            "2) All-null dataframe",
            pd.DataFrame({"a": [None, None, None], "b": [None, None, None]}),
        )
    )

    robustness_results.append(
        robustness_case("3) Single-column dataframe", pd.DataFrame({"only_col": [1, 2, None, 4, 5]}))
    )

    robustness_results.append(
        robustness_case(
            "4) Random garbage dataframe",
            pd.DataFrame(
                {
                    "x": [1, "two", None, {"k": "v"}, [1, 2], 3.14],
                    "y": ["??", "N/A", "", "unknown", 999, -1],
                }
            ),
        )
    )

    total_tests_run += 4
    failed_robustness = sum(1 for passed in robustness_results if not passed)

    divider("STEP 9: FINAL SUMMARY")
    print(f"Total tests run: {total_tests_run}")
    print(f"Warnings triggered: {len(warnings_triggered)}")
    if warnings_triggered:
        for warning in warnings_triggered:
            print(f"- {warning}")

    if failed_robustness == 0 and not warnings_triggered:
        print("Final verdict: Library looks stable")
    else:
        print("Final verdict: Library needs fixes")


if __name__ == "__main__":
    main()
