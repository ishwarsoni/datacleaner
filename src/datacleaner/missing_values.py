"""Missing values cleaning module."""

from __future__ import annotations

from typing import Any
import warnings

import pandas as pd


DIRTY_PLACEHOLDERS = {"n/a", "na", "-", "", "unknown", "??"}


def _normalize_placeholders(series: pd.Series) -> pd.Series:
    """Replace common dirty placeholder strings with missing values."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    normalized = series.copy()
    as_text = normalized.astype("string").str.strip().str.lower()
    placeholder_mask = as_text.isin(DIRTY_PLACEHOLDERS)
    return normalized.mask(placeholder_mask, pd.NA)


def _safe_mode(series: pd.Series) -> Any:
    """Return a robust mode value for mixed-type categorical series."""
    non_null = series.dropna()
    if non_null.empty:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            mode_series = non_null.mode(dropna=True)
        if not mode_series.empty:
            return mode_series.iloc[0]
    except Exception:
        pass

    try:
        normalized = non_null.map(lambda value: value if isinstance(value, (str, int, float, bool)) else str(value))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fallback_mode = normalized.mode(dropna=True)
        if not fallback_mode.empty:
            return fallback_mode.iloc[0]
    except Exception:
        return None

    return None


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Handle missing values with column drop and median/mode fill rules.

    Args:
        df: Input pandas DataFrame.

    Returns:
        A tuple containing the cleaned DataFrame and a dictionary of changes.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cleaned_df = df.copy()
    for column_name in cleaned_df.select_dtypes(include=["object", "string"]).columns:
        try:
            cleaned_df[column_name] = _normalize_placeholders(cleaned_df[column_name])
        except Exception:
            continue

    missing_ratio = cleaned_df.isna().mean()
    columns_dropped = missing_ratio[missing_ratio > 0.40].index.tolist()
    if columns_dropped:
        cleaned_df = cleaned_df.drop(columns=columns_dropped)

    numeric_cols = cleaned_df.select_dtypes(include="number").columns
    categorical_cols = cleaned_df.columns.difference(numeric_cols)

    values_filled_per_column: dict[str, int] = {}
    method_used: dict[str, str] = {}

    for column_name in numeric_cols:
        before_missing = int(cleaned_df[column_name].isna().sum())
        if before_missing == 0:
            continue

        fill_value = cleaned_df[column_name].median()
        cleaned_df[column_name] = cleaned_df[column_name].fillna(fill_value)

        after_missing = int(cleaned_df[column_name].isna().sum())
        values_filled_per_column[column_name] = before_missing - after_missing
        method_used[column_name] = "median"

    for column_name in categorical_cols:
        before_missing = int(cleaned_df[column_name].isna().sum())
        if before_missing == 0:
            continue

        fill_value = _safe_mode(cleaned_df[column_name])
        if fill_value is None:
            continue
        cleaned_df[column_name] = cleaned_df[column_name].fillna(fill_value)

        after_missing = int(cleaned_df[column_name].isna().sum())
        values_filled_per_column[column_name] = before_missing - after_missing
        method_used[column_name] = "mode"

    changes = {
        "columns_dropped": columns_dropped,
        "values_filled_per_column": values_filled_per_column,
        "method_used": method_used,
    }
    return cleaned_df, changes


def clean_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible wrapper for handle_missing."""
    return handle_missing(df)
