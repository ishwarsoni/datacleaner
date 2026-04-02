"""Skewness handling utilities for datacleaner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def handle_skewness(
    df: pd.DataFrame,
    threshold: float = 1.0,
    method: str = "log",
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply optional skewness correction to highly skewed numeric columns.

    Args:
        df: Input pandas DataFrame.
        threshold: Absolute skewness threshold for transformation.
        method: Transformation method, either "log" or "sqrt".

    Returns:
        A tuple containing transformed DataFrame and skewness metadata.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cleaned_df = df.copy()

    try:
        safe_threshold = float(threshold)
    except Exception:
        safe_threshold = 1.0
    if safe_threshold < 0:
        safe_threshold = abs(safe_threshold)

    safe_method = method if isinstance(method, str) else "log"
    safe_method = safe_method.lower()
    if safe_method not in {"log", "sqrt"}:
        safe_method = "log"

    numeric_columns = cleaned_df.select_dtypes(include="number").columns.tolist()
    transformed_columns: list[str] = []
    original_skewness: dict[str, float] = {}
    skipped_columns: dict[str, str] = {}
    transformed_skewness: dict[str, float] = {}
    method_used_per_column: dict[str, str] = {}
    details: dict[str, dict[str, Any]] = {}

    for column_name in numeric_columns:
        try:
            if target_column is not None and column_name == target_column:
                continue

            numeric_series = pd.to_numeric(cleaned_df[column_name], errors="coerce")
            if numeric_series.notna().sum() < 2:
                continue

            skew_value = float(numeric_series.dropna().skew())
            if np.isnan(skew_value):
                continue

            if abs(skew_value) <= safe_threshold:
                continue

            original_skewness[column_name] = skew_value

            if safe_method == "log":
                # Signed log handles positive, negative, and mixed values safely.
                transformed_series = np.sign(numeric_series) * np.log1p(np.abs(numeric_series))
            else:
                transformed_series = np.sign(numeric_series) * np.sqrt(np.abs(numeric_series))

            finite_mask = np.isfinite(transformed_series) | transformed_series.isna()
            if not bool(finite_mask.all()):
                skipped_columns[column_name] = "non_finite_values_generated"
                continue

            transformed_skew = float(pd.to_numeric(transformed_series, errors="coerce").dropna().skew())
            if not np.isnan(transformed_skew) and abs(transformed_skew) >= abs(skew_value):
                skipped_columns[column_name] = "skew_not_improved"
                continue

            cleaned_df[column_name] = transformed_series
            transformed_columns.append(column_name)
            if not np.isnan(transformed_skew):
                transformed_skewness[column_name] = transformed_skew
            method_used_per_column[column_name] = safe_method
            details[column_name] = {
                "before": skew_value,
                "after": transformed_skew,
                "method": safe_method,
            }
        except Exception as exc:
            skipped_columns[column_name] = f"transformation_failed: {exc}"
            continue

    changes = {
        "method": safe_method,
        "threshold": safe_threshold,
        "columns_transformed": transformed_columns,
        "original_skewness": original_skewness,
        "transformed_skewness": transformed_skewness,
        "method_used_per_column": method_used_per_column,
        "details": details,
        "skipped_columns": skipped_columns,
    }
    return cleaned_df, changes
