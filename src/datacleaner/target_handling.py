"""Target-column handling utilities for optional pre-cleaning workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_mode(series: pd.Series) -> Any:
    """Return mode for non-null values, or None if unavailable."""
    non_null = series.dropna()
    if non_null.empty:
        return None

    mode_series = non_null.mode(dropna=True)
    if mode_series.empty:
        return None
    return mode_series.iloc[0]


def handle_target(
    df: pd.DataFrame,
    target_column: str,
    strategy: str = "auto",
    threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Safely handle missing values in a target column before calling clean.

    Rules:
    - If no target missing values, return unchanged.
    - If target missing ratio is above threshold, drop rows with missing target.
    - If target missing ratio is at or below threshold:
      - strategy='fill': fill numeric with median, categorical with mode.
      - strategy='drop': drop rows with missing target.
      - strategy='auto': default to drop rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError("target_column must be a non-empty string")
    if target_column not in df.columns:
        raise KeyError(f"target column '{target_column}' not found")

    try:
        safe_threshold = float(threshold)
    except Exception as exc:
        raise ValueError("threshold must be a numeric value") from exc
    safe_threshold = min(max(safe_threshold, 0.0), 1.0)

    safe_strategy = strategy if isinstance(strategy, str) else "auto"
    safe_strategy = safe_strategy.lower().strip()
    if safe_strategy not in {"auto", "drop", "fill"}:
        raise ValueError("strategy must be one of: 'auto', 'drop', 'fill'")

    cleaned_df = df.copy()
    target_series = cleaned_df[target_column]
    missing_ratio = float(target_series.isna().mean())

    if missing_ratio == 0.0:
        return cleaned_df, {
            "target_missing_ratio": missing_ratio,
            "action": "none",
            "rows_removed": 0,
        }

    rows_before = len(cleaned_df)
    missing_count = int(target_series.isna().sum())

    # Missing target above threshold is always dropped.
    if missing_ratio > safe_threshold:
        cleaned_df = cleaned_df.loc[~target_series.isna()].copy()
        return cleaned_df, {
            "target_missing_ratio": missing_ratio,
            "action": "rows_dropped",
            "rows_removed": int(rows_before - len(cleaned_df)),
        }

    resolved_strategy = "drop" if safe_strategy == "auto" else safe_strategy
    if resolved_strategy == "drop":
        cleaned_df = cleaned_df.loc[~target_series.isna()].copy()
        return cleaned_df, {
            "target_missing_ratio": missing_ratio,
            "action": "rows_dropped",
            "rows_removed": int(rows_before - len(cleaned_df)),
        }

    # Fill strategy applies only for low missing target ratios.
    if pd.api.types.is_numeric_dtype(target_series):
        fill_value = target_series.median()
    else:
        fill_value = _safe_mode(target_series)
        if fill_value is None:
            # If no safe categorical fill exists, fallback to drop for robustness.
            cleaned_df = cleaned_df.loc[~target_series.isna()].copy()
            return cleaned_df, {
                "target_missing_ratio": missing_ratio,
                "action": "rows_dropped",
                "rows_removed": int(rows_before - len(cleaned_df)),
            }

    cleaned_df[target_column] = cleaned_df[target_column].fillna(fill_value)
    return cleaned_df, {
        "target_missing_ratio": missing_ratio,
        "action": "filled",
        "rows_removed": 0,
        "fill_value": fill_value,
        "filled_count": missing_count,
    }
