"""Outliers cleaning module."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Compute IQR-based lower and upper bounds for a numeric pandas Series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def handle_outliers(
    df: pd.DataFrame,
    method: str = "cap",
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Handle IQR-based outliers in numeric columns.

    Args:
        df: Input pandas DataFrame.
        method: Outlier strategy. Supported values are "cap" and "remove".

    Returns:
        A tuple containing the cleaned DataFrame and outlier-handling metadata.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If method is not one of the supported options.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if method not in {"cap", "remove"}:
        raise ValueError("method must be either 'cap' or 'remove'")

    cleaned_df = df.copy()
    numeric_columns = cleaned_df.select_dtypes(include="number").columns.tolist()
    processed_numeric_columns = [
        column_name for column_name in numeric_columns if target_column is None or column_name != target_column
    ]

    if not processed_numeric_columns:
        changes: dict[str, Any] = {"columns_analyzed": processed_numeric_columns}
        if method == "remove":
            changes["rows_removed"] = 0
        else:
            changes["values_capped"] = 0
            changes["capped_percentage_per_column"] = {}
            changes["high_capping_columns"] = []
            changes["warnings"] = []
        return cleaned_df, changes

    if method == "remove":
        outlier_row_mask = pd.Series(False, index=cleaned_df.index)
        for column_name in processed_numeric_columns:
            lower_bound, upper_bound = _iqr_bounds(cleaned_df[column_name])

            column_outlier_mask = (cleaned_df[column_name] < lower_bound) | (cleaned_df[column_name] > upper_bound)
            outlier_row_mask = outlier_row_mask | column_outlier_mask.fillna(False)

        rows_removed = int(outlier_row_mask.sum())
        cleaned_df = cleaned_df.loc[~outlier_row_mask].copy()

        changes = {
            "columns_analyzed": processed_numeric_columns,
            "rows_removed": rows_removed,
        }
        return cleaned_df, changes

    values_capped = 0
    capped_percentage_per_column: dict[str, float] = {}
    high_capping_columns: list[str] = []

    for column_name in processed_numeric_columns:
        lower_bound, upper_bound = _iqr_bounds(cleaned_df[column_name])

        lower_mask = cleaned_df[column_name] < lower_bound
        upper_mask = cleaned_df[column_name] > upper_bound
        capped_count = int(lower_mask.fillna(False).sum() + upper_mask.fillna(False).sum())
        values_capped += capped_count

        non_null_count = int(cleaned_df[column_name].notna().sum())
        capped_pct = (capped_count / non_null_count * 100) if non_null_count > 0 else 0.0
        capped_percentage_per_column[column_name] = capped_pct
        if capped_pct > 20.0:
            high_capping_columns.append(column_name)

        cleaned_df[column_name] = cleaned_df[column_name].clip(lower=lower_bound, upper=upper_bound)

    changes = {
        "columns_analyzed": processed_numeric_columns,
        "values_capped": values_capped,
        "capped_percentage_per_column": capped_percentage_per_column,
        "high_capping_columns": high_capping_columns,
        "warnings": [
            f"High capping rate detected in column '{column_name}' (>20%)."
            for column_name in high_capping_columns
        ],
    }
    return cleaned_df, changes


def clean_outliers(
    df: pd.DataFrame,
    method: str = "cap",
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible wrapper for handle_outliers."""
    return handle_outliers(df, method=method, target_column=target_column)
