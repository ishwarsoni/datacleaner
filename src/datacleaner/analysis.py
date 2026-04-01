"""Dataset analysis utilities for datacleaner."""

from __future__ import annotations

from typing import Any
import numpy as np

import pandas as pd


def _safe_nunique(series: pd.Series) -> int:
    """Return unique count even when values are unhashable."""
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        normalized = series.dropna().map(
            lambda value: value if isinstance(value, (str, int, float, bool)) else str(value)
        )
        return int(normalized.nunique(dropna=True))


def _safe_nunique_including_null(series: pd.Series) -> int:
    """Return unique count including null values for ratio calculations."""
    try:
        return int(series.nunique(dropna=False))
    except Exception:
        normalized = series.map(
            lambda value: value if isinstance(value, (str, int, float, bool)) else str(value)
        )
        return int(normalized.nunique(dropna=False))


def _column_kind(series: pd.Series) -> str:
    """Return a lightweight semantic column kind."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_string_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return "text"
        numeric_candidate = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = numeric_candidate.notna().sum() / len(non_null)
        if numeric_ratio > 0.7:
            return "numeric_candidate"
        return "text"
    if pd.api.types.is_object_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return "text"
        try:
            numeric_candidate = pd.to_numeric(non_null, errors="coerce")
            numeric_ratio = numeric_candidate.notna().sum() / len(non_null)
            if numeric_ratio > 0.7:
                return "numeric_candidate"
            text_ratio = non_null.map(lambda value: isinstance(value, str)).mean()
            return "text" if text_ratio >= 0.8 else "categorical"
        except Exception:
            return "categorical"
    return "categorical"


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze a dataset before cleaning.

    The function is read-only and does not modify the input DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        A dictionary with column metadata, missing-value percentages,
        unique-value counts, and basic data quality warnings.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    column_names = df.columns.tolist()
    data_types = df.dtypes.astype(str).to_dict()

    missing_percentage_series = (df.isna().mean() * 100).round(2)
    missing_percentage = {
        column_name: float(percentage)
        for column_name, percentage in missing_percentage_series.to_dict().items()
    }

    unique_values_series = pd.Series({column_name: _safe_nunique(df[column_name]) for column_name in df.columns})
    unique_values = {
        column_name: int(count)
        for column_name, count in unique_values_series.to_dict().items()
    }

    total_rows = len(df)
    if total_rows == 0:
        unique_ratio = pd.Series(0.0, index=df.columns)
    else:
        unique_including_null = pd.Series(
            {column_name: _safe_nunique_including_null(df[column_name]) for column_name in df.columns}
        )
        unique_ratio = unique_including_null / total_rows

    high_missing_values = missing_percentage_series[missing_percentage_series > 40.0].index.tolist()
    high_cardinality_columns = unique_ratio[(unique_ratio > 0.5) & (unique_values_series > 50)].index.tolist()
    object_like_columns = df.select_dtypes(include=["object", "string"]).columns
    possible_id_columns = [
        column_name for column_name in object_like_columns if float(unique_ratio[column_name]) > 0.9
    ]
    constant_columns = [
        column_name for column_name in df.columns if _safe_nunique_including_null(df[column_name]) <= 1
    ]

    column_insights: dict[str, Any] = {}
    numeric_candidate_columns: list[str] = []
    for column_name in df.columns:
        series = df[column_name]
        kind = _column_kind(series)

        insight: dict[str, Any] = {
            "column_type": kind,
            "missing_percentage": float(missing_percentage_series[column_name]),
            "unique_ratio": float(unique_ratio[column_name]),
        }

        if kind == "numeric_candidate":
            non_null = series.dropna()
            if len(non_null) > 0:
                conversion_ratio = float(pd.to_numeric(non_null, errors="coerce").notna().sum() / len(non_null))
            else:
                conversion_ratio = 0.0
            insight["numeric_conversion_ratio"] = conversion_ratio
            numeric_candidate_columns.append(column_name)

        if kind == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            insight["basic_statistics"] = {
                "mean": float(numeric_series.mean()) if numeric_series.notna().any() else None,
                "median": float(numeric_series.median()) if numeric_series.notna().any() else None,
                "std": float(numeric_series.std()) if numeric_series.notna().sum() > 1 else None,
                "min": float(numeric_series.min()) if numeric_series.notna().any() else None,
                "max": float(numeric_series.max()) if numeric_series.notna().any() else None,
            }

        column_insights[column_name] = insight

    # Lightweight high-correlation candidate detection on numeric columns only.
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    high_correlation_candidates: list[dict[str, Any]] = []
    if len(numeric_columns) >= 2 and len(df) >= 2:
        numeric_df = df[numeric_columns]
        valid_numeric_columns = [
            column_name
            for column_name in numeric_columns
            if pd.to_numeric(numeric_df[column_name], errors="coerce").notna().sum() > 1
        ]
        if len(valid_numeric_columns) >= 2:
            corr_matrix = numeric_df[valid_numeric_columns].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            for column_name in upper.columns:
                for other_column in upper.index[upper[column_name] > 0.9].tolist():
                    high_correlation_candidates.append(
                        {
                            "feature_a": other_column,
                            "feature_b": column_name,
                            "correlation": float(upper.at[other_column, column_name]),
                        }
                    )

    warnings = {
        "high_missing_values": high_missing_values,
        "high_cardinality_columns": high_cardinality_columns,
        "constant_columns": constant_columns,
        "high_correlation_candidates": high_correlation_candidates,
        "possible_id_columns": possible_id_columns,
        "numeric_candidate_columns": numeric_candidate_columns,
    }

    return {
        "column_names": column_names,
        "detected_data_types": data_types,
        "missing_values_percentage": missing_percentage,
        "unique_values_per_column": unique_values,
        "column_insights": column_insights,
        "warnings": warnings,
    }
