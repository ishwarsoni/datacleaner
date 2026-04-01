"""Stress-test datacleaner on sklearn datasets with simulated messy conditions."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

import numpy as np
import pandas as pd
from sklearn import datasets

# Ensure local package import works when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datacleaner import analyze, clean


RNG = np.random.default_rng(42)


def divider(title: str) -> None:
    """Print a readable separator."""
    print("\n" + "=" * 24)
    print(title)
    print("=" * 24)


def load_sklearn_dataframes() -> list[tuple[str, pd.DataFrame]]:
    """Load required sklearn datasets as pandas DataFrames."""
    iris = datasets.load_iris(as_frame=True)
    wine = datasets.load_wine(as_frame=True)
    diabetes = datasets.load_diabetes(as_frame=True)

    return [
        ("iris", iris.data.copy()),
        ("wine", wine.data.copy()),
        ("diabetes", diabetes.data.copy()),
    ]


def make_messy(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic data quality issues into a DataFrame."""
    messy_df = df.copy()

    # 1) Add missing values to ~10-15% of cells (numeric columns only).
    numeric_columns = messy_df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        missing_ratio = float(RNG.uniform(0.10, 0.15))
        total_numeric_cells = len(messy_df) * len(numeric_columns)
        missing_count = int(total_numeric_cells * missing_ratio)

        if missing_count > 0:
            row_idx = RNG.integers(0, len(messy_df), size=missing_count)
            col_idx = RNG.integers(0, len(numeric_columns), size=missing_count)
            for r, c in zip(row_idx, col_idx):
                messy_df.at[r, numeric_columns[c]] = np.nan

    # 2) Add duplicates (5-10 random rows).
    duplicate_count = int(RNG.integers(5, 11))
    if len(messy_df) > 0 and duplicate_count > 0:
        duplicate_indices = RNG.choice(messy_df.index.to_numpy(), size=duplicate_count, replace=True)
        duplicate_rows = messy_df.loc[duplicate_indices]
        messy_df = pd.concat([messy_df, duplicate_rows], ignore_index=True)

    # 3) Add fake categorical dirty column.
    fake_categories = np.array(["Male", "male", "M", "??", "unknown"], dtype=object)
    messy_df["fake_category"] = RNG.choice(fake_categories, size=len(messy_df), replace=True)

    # 4) Convert one numeric column to mixed strings.
    numeric_columns = messy_df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        target_col = numeric_columns[0]
        as_strings = messy_df[target_col].astype("string")
        mixed_idx = RNG.choice(np.arange(len(messy_df)), size=max(1, len(messy_df) // 12), replace=False)
        replacement_pool = np.array(["100", "200", "abc", "300", "??"], dtype=object)
        replacement_values = RNG.choice(replacement_pool, size=len(mixed_idx), replace=True)
        for i, value in zip(mixed_idx, replacement_values):
            as_strings.iat[i] = value
        messy_df[target_col] = as_strings

    # 5) Add outliers in another numeric column if available.
    numeric_columns = messy_df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        outlier_col = numeric_columns[-1]
        outlier_idx = RNG.choice(np.arange(len(messy_df)), size=max(1, len(messy_df) // 20), replace=False)
        if len(outlier_idx) > 0:
            messy_df.loc[outlier_idx, outlier_col] = messy_df.loc[outlier_idx, outlier_col].fillna(1) * 10
        if len(messy_df) > 1:
            messy_df.loc[messy_df.index[0], outlier_col] = 99999
            messy_df.loc[messy_df.index[1], outlier_col] = -99999

    return messy_df


def run_dataset_test(dataset_name: str, original_df: pd.DataFrame) -> bool:
    """Run analyze/clean on one dataset and report results."""
    divider(f"DATASET: {dataset_name}")

    messy_df = make_messy(original_df)
    rows_before, cols_before = messy_df.shape

    print(f"Original shape: {messy_df.shape}")

    try:
        analysis = analyze(messy_df)
        print("\nAnalyze output:")
        pprint(analysis, sort_dicts=False)

        clean_result = clean(messy_df, verbose=True)
        if isinstance(clean_result, tuple):
            cleaned_df, report = clean_result
        else:
            cleaned_df, report = clean_result, {}

        rows_after, cols_after = cleaned_df.shape

        print("\nCleaned shape:", cleaned_df.shape)
        print("\nCleaning report:")
        pprint(report, sort_dicts=False)

        print("\nValidation checks:")
        print(f"Rows: {rows_before} -> {rows_after} (change: {rows_after - rows_before})")
        print(f"Columns: {cols_before} -> {cols_after} (change: {cols_after - cols_before})")

        row_loss_pct = ((rows_before - rows_after) / rows_before * 100) if rows_before else 0.0
        col_loss_pct = ((cols_before - cols_after) / cols_before * 100) if cols_before else 0.0

        print(f"Row loss: {row_loss_pct:.2f}%")
        print(f"Column loss: {col_loss_pct:.2f}%")

        if row_loss_pct > 30:
            print("WARNING: More than 30% of rows removed.")
        if col_loss_pct > 30:
            print("WARNING: More than 30% of columns dropped.")

        print("\nRobustness: PASSED")
        return True

    except Exception as exc:
        print(f"\nRobustness: FAILED with error: {exc}")
        return False


def main() -> None:
    """Execute sklearn-based stress tests for datacleaner."""
    divider("STEP 1: LOAD DATASETS")
    datasets_to_test = load_sklearn_dataframes()
    print("Loaded datasets:", ", ".join(name for name, _ in datasets_to_test))

    passed_count = 0

    divider("STEP 2-5: MESSY DATA + PIPELINE + VALIDATION + ROBUSTNESS")
    for dataset_name, dataset_df in datasets_to_test:
        if run_dataset_test(dataset_name, dataset_df):
            passed_count += 1

    divider("STEP 6: FINAL SUMMARY")
    total_datasets = len(datasets_to_test)
    print(f"Total datasets tested: {total_datasets}")
    print(f"Passed without errors: {passed_count}")

    if passed_count == total_datasets:
        print("Overall verdict: Library stable on sklearn datasets")
    else:
        print("Overall verdict: Issues detected")


if __name__ == "__main__":
    main()
