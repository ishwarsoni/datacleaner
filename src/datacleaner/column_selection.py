"""Column selection and pruning utilities for datacleaner."""

from __future__ import annotations

from typing import Any

import pandas as pd


MIN_ROWS_FOR_RATIO_DROP = 30


def _safe_nunique(series: pd.Series, dropna: bool) -> int:
    """Return unique count safely for mixed/unhashable values."""
    try:
        return int(series.nunique(dropna=dropna))
    except Exception:
        normalized = series.map(
            lambda value: value if isinstance(value, (str, int, float, bool)) else str(value)
        )
        return int(normalized.nunique(dropna=dropna))


def drop_useless_columns(
    df: pd.DataFrame,
    unique_ratio_threshold: float = 0.98,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop low-information columns using simple, safe heuristics.

    Rules:
    - Drop constant columns (only one unique value including null behavior).
    - Drop likely ID columns when unique_ratio > unique_ratio_threshold.

    Safeguards:
    - Ratio-based ID dropping is disabled on very small datasets.

    Args:
        df: Input pandas DataFrame.
        unique_ratio_threshold: Threshold for likely ID detection.

    Returns:
        A tuple with cleaned DataFrame and a dictionary of dropped columns with reasons.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    safe_threshold = unique_ratio_threshold if isinstance(unique_ratio_threshold, (int, float)) else 0.95
    safe_threshold = min(max(float(safe_threshold), 0.0), 1.0)

    cleaned_df = df.copy()
    total_rows = len(cleaned_df)

    dropped_columns: list[str] = []
    dropped_reasons: dict[str, str] = {}

    for column_name in cleaned_df.columns.tolist():
        try:
            if target_column is not None and column_name == target_column:
                continue

            series = cleaned_df[column_name]
            unique_count_including_null = _safe_nunique(series, dropna=False)

            if unique_count_including_null <= 1:
                dropped_columns.append(column_name)
                dropped_reasons[column_name] = "constant_column"
                continue

            if total_rows < MIN_ROWS_FOR_RATIO_DROP or total_rows == 0:
                continue

            unique_count_excluding_null = _safe_nunique(series, dropna=True)
            unique_ratio = unique_count_excluding_null / total_rows
            if unique_ratio > safe_threshold:
                dropped_columns.append(column_name)
                dropped_reasons[column_name] = "likely_id_column"
        except Exception:
            continue

    if dropped_columns:
        cleaned_df = cleaned_df.drop(columns=dropped_columns)

    changes = {
        "dropped_columns": dropped_columns,
        "drop_reasons": dropped_reasons,
        "unique_ratio_threshold": safe_threshold,
        "small_dataset_ratio_drop_skipped": total_rows < MIN_ROWS_FOR_RATIO_DROP,
    }
    return cleaned_df, changes
