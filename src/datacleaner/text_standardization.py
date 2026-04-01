"""Text standardization module for datacleaner."""

from __future__ import annotations

from typing import Any

import pandas as pd


CATEGORY_MAP = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
}


def _normalize_text_value(value: Any) -> Any:
    """Normalize a single text-like value while preserving non-text values."""
    if pd.isna(value):
        return value
    if not isinstance(value, str):
        return value

    normalized = value.strip().lower()
    return CATEGORY_MAP.get(normalized, normalized)


def standardize_text(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize text values in object/string columns.

    Operations:
    - Lowercase text values.
    - Strip leading/trailing spaces.
    - Apply a small category map for common male/female variants.

    Args:
        df: Input pandas DataFrame.

    Returns:
        A tuple containing the updated DataFrame and a dictionary of changes.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cleaned_df = df.copy()
    object_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns

    values_changed_per_column: dict[str, int] = {}

    for column_name in object_columns:
        try:
            original_series = cleaned_df[column_name]
            standardized_series = original_series.map(_normalize_text_value)

            changed_count = int((original_series != standardized_series).fillna(False).sum())
            if changed_count > 0:
                cleaned_df[column_name] = standardized_series
                values_changed_per_column[column_name] = changed_count
            else:
                cleaned_df[column_name] = standardized_series
        except Exception:
            continue

    changes = {
        "columns_processed": object_columns.tolist(),
        "values_standardized_per_column": values_changed_per_column,
    }
    return cleaned_df, changes
