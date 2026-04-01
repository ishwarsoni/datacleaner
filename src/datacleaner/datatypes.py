"""Datatypes cleaning module."""

from __future__ import annotations

from typing import Any
import re
import warnings

import pandas as pd


DATE_LIKE_PATTERN = re.compile(
    r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$"
)
MISSING_TEXT_VALUES = {"", "na", "n/a", "null", "none"}


def _build_numeric_candidate(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series, bool, bool, bool, bool]:
    """Build a temporary numeric candidate plus normalization indicators."""
    normalized = series.astype("string").str.strip()
    normalized = normalized.where(~normalized.str.lower().isin(MISSING_TEXT_VALUES), pd.NA)

    has_comma = bool(normalized.str.contains(",", regex=False, na=False).any())
    has_currency = bool(normalized.str.contains(r"[$₹€]", regex=True, na=False).any())
    symbol_mask = normalized.str.contains(r",|[$₹€]|%", regex=True, na=False)
    non_missing_mask = normalized.notna()
    all_non_missing_are_formatted = bool(symbol_mask[non_missing_mask].all())
    percentage_mask = normalized.str.contains("%", regex=False, na=False)
    has_percentage = bool(percentage_mask.any())

    normalized = normalized.str.replace(",", "", regex=False)
    normalized = normalized.str.replace(r"[$₹€]", "", regex=True)
    normalized = normalized.str.replace("%", "", regex=False)

    numeric_candidate = pd.to_numeric(normalized, errors="coerce")
    if has_percentage:
        numeric_candidate = numeric_candidate.astype("Float64")
        numeric_candidate.loc[percentage_mask] = numeric_candidate.loc[percentage_mask] / 100.0

    return (
        numeric_candidate,
        normalized,
        has_percentage,
        has_currency,
        has_comma,
        all_non_missing_are_formatted,
    )


def _looks_date_like(series: pd.Series) -> bool:
    """Heuristically detect whether a column looks like datetime text."""
    non_null = series.dropna()
    if non_null.empty:
        return False

    as_text = non_null.astype("string").str.strip()
    sample_size = min(len(as_text), 50)
    sample = as_text.iloc[:sample_size]
    match_ratio = sample.map(lambda value: bool(DATE_LIKE_PATTERN.match(value))).mean()
    return bool(match_ratio >= 0.6)


def fix_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Safely convert numeric-like and datetime-like text columns.

    Args:
        df: Input pandas DataFrame.

    Returns:
        A tuple containing the cleaned DataFrame and conversion metadata.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cleaned_df = df.copy()
    converted_columns: dict[str, str] = {}
    numeric_candidate_columns: list[str] = []
    percentage_columns: list[str] = []
    currency_stripped_columns: list[str] = []
    comma_normalized_columns: list[str] = []

    object_like_cols = cleaned_df.select_dtypes(include=["object", "string"]).columns

    for column_name in object_like_cols:
        try:
            source_series = cleaned_df[column_name]
            (
                numeric_candidate,
                normalized_text,
                has_percentage,
                has_currency,
                has_comma,
                all_non_missing_are_formatted,
            ) = _build_numeric_candidate(source_series)
            non_null_values = normalized_text.dropna()

            if has_percentage:
                percentage_columns.append(column_name)
            if has_currency:
                currency_stripped_columns.append(column_name)
            if has_comma:
                comma_normalized_columns.append(column_name)

            if non_null_values.empty:
                continue

            numeric_success_count = int(numeric_candidate.notna().sum())
            numeric_success_ratio = numeric_success_count / len(non_null_values)
            if numeric_success_ratio > 0.7:
                numeric_candidate_columns.append(column_name)
            if (
                numeric_success_ratio >= 0.85 and numeric_success_count >= 10
            ) or (
                numeric_success_ratio >= 0.95 and numeric_success_count >= 3
            ) or (
                all_non_missing_are_formatted
                and numeric_success_ratio >= 0.98
                and numeric_success_count >= 2
            ):
                cleaned_df[column_name] = numeric_candidate
                converted_columns[column_name] = str(cleaned_df[column_name].dtype)
                continue

            if not _looks_date_like(source_series):
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                datetime_candidate = pd.to_datetime(source_series, errors="coerce")

            datetime_success_count = int(datetime_candidate.notna().sum())
            datetime_success_ratio = datetime_success_count / len(non_null_values)
            if datetime_success_ratio >= 0.6 and datetime_success_count >= 2:
                cleaned_df[column_name] = datetime_candidate
                converted_columns[column_name] = str(cleaned_df[column_name].dtype)
        except Exception:
            continue

    changes = {
        "columns_converted": converted_columns,
        "numeric_candidate_columns": sorted(set(numeric_candidate_columns)),
        "percentage_columns": sorted(set(percentage_columns)),
        "currency_stripped_columns": sorted(set(currency_stripped_columns)),
        "comma_normalized_columns": sorted(set(comma_normalized_columns)),
    }
    return cleaned_df, changes


def clean_datatypes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible wrapper for fix_types."""
    return fix_types(df)
