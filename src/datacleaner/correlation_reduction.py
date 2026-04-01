"""Correlation-based feature reduction for datacleaner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MAX_DROP_FRACTION = 0.5


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop highly correlated numeric features using a simple threshold rule.

    Args:
        df: Input pandas DataFrame.
        threshold: Absolute correlation threshold for feature dropping.

    Returns:
        A tuple containing the cleaned DataFrame and feature-removal metadata.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cleaned_df = df.copy()
    numeric_columns = cleaned_df.select_dtypes(include="number").columns.tolist()

    if len(numeric_columns) < 2 or len(cleaned_df) < 2:
        return cleaned_df, {
            "threshold": float(threshold),
            "numeric_columns_analyzed": numeric_columns,
            "removed_features": [],
            "correlated_pairs": [],
            "max_correlation_observed": 0.0,
            "pairs_checked": 0,
            "max_allowed_drops": 0,
            "drop_limit_reached": False,
        }

    try:
        safe_threshold = float(threshold)
    except Exception:
        safe_threshold = 0.9
    safe_threshold = min(max(safe_threshold, 0.0), 1.0)

    numeric_df = cleaned_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    pairs_checked = int(np.isfinite(upper_triangle.to_numpy()).sum())
    max_correlation_observed = float(np.nanmax(upper_triangle.to_numpy())) if pairs_checked > 0 else 0.0

    correlated_pairs: list[dict[str, Any]] = []
    candidate_drops: list[str] = []
    for column_name in upper_triangle.columns:
        correlated_with = upper_triangle.index[upper_triangle[column_name] > safe_threshold].tolist()
        for other_column in correlated_with:
            correlated_pairs.append(
                {
                    "feature_a": other_column,
                    "feature_b": column_name,
                    "correlation": float(upper_triangle.at[other_column, column_name]),
                }
            )
        if correlated_with:
            candidate_drops.append(column_name)

    max_allowed_drops = max(0, int(len(numeric_columns) * MAX_DROP_FRACTION))
    if len(numeric_columns) > 1:
        max_allowed_drops = max(1, max_allowed_drops)

    removed_features = candidate_drops[:max_allowed_drops]
    if removed_features:
        cleaned_df = cleaned_df.drop(columns=removed_features)

    changes = {
        "threshold": safe_threshold,
        "numeric_columns_analyzed": numeric_columns,
        "removed_features": removed_features,
        "correlated_pairs": correlated_pairs,
        "max_correlation_observed": max_correlation_observed,
        "pairs_checked": pairs_checked,
        "max_allowed_drops": max_allowed_drops,
        "drop_limit_reached": len(candidate_drops) > max_allowed_drops,
    }
    return cleaned_df, changes
