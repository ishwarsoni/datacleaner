"""Skewness handling utilities for datacleaner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def handle_skewness(
    df: pd.DataFrame,
    threshold: float = 1.0,
    method: str = "log",
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

    for column_name in numeric_columns:
        try:
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
                # log1p requires values >= 0
                if (numeric_series.dropna() < 0).any():
                    skipped_columns[column_name] = "negative_values_not_allowed_for_log"
                    continue
                transformed_series = np.log1p(numeric_series)
            else:
                # sqrt requires values >= 0
                if (numeric_series.dropna() < 0).any():
                    skipped_columns[column_name] = "negative_values_not_allowed_for_sqrt"
                    continue
                transformed_series = np.sqrt(numeric_series)

            cleaned_df[column_name] = transformed_series
            transformed_columns.append(column_name)
        except Exception as exc:
            skipped_columns[column_name] = f"transformation_failed: {exc}"
            continue

    changes = {
        "method": safe_method,
        "threshold": safe_threshold,
        "columns_transformed": transformed_columns,
        "original_skewness": original_skewness,
        "skipped_columns": skipped_columns,
    }
    return cleaned_df, changes
