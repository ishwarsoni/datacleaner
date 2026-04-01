"""Duplicates cleaning module."""

from __future__ import annotations

from typing import Any

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove exact duplicate rows from a DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        A tuple containing the cleaned DataFrame and duplicate-removal metadata.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    original_row_count = len(df)
    cleaned_df = df.drop_duplicates().copy()
    removed_row_count = original_row_count - len(cleaned_df)

    changes = {"duplicates_removed": removed_row_count}
    return cleaned_df, changes


def clean_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible wrapper for remove_duplicates."""
    return remove_duplicates(df)
