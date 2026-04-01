"""Reporting module for datacleaner."""

from __future__ import annotations

from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    """Return value when it is a dictionary, otherwise return an empty dictionary."""
    if isinstance(value, dict):
        return value
    return {}


def generate_report(report_dict: dict[str, Any]) -> dict[str, Any]:
    """Generate a simple report summary from pipeline step outputs.

    Args:
        report_dict: Dictionary keyed by step names.

    Returns:
        A user-friendly report dictionary.

    Raises:
        TypeError: If report_dict is not a dictionary.
    """
    if not isinstance(report_dict, dict):
        raise TypeError("report_dict must be a dictionary")

    missing_report = _as_dict(report_dict.get("missing_values", {}))
    duplicates_report = _as_dict(report_dict.get("duplicates", {}))
    datatypes_report = _as_dict(report_dict.get("datatypes", {}))
    correlation_reduction_report = _as_dict(report_dict.get("correlation_reduction", {}))
    column_selection_report = _as_dict(report_dict.get("column_selection", {}))
    outliers_report = _as_dict(report_dict.get("outliers", {}))
    analysis_report = _as_dict(report_dict.get("analysis", {}))
    analysis_warnings = _as_dict(analysis_report.get("warnings", {}))

    columns_dropped = missing_report.get("columns_dropped", [])
    values_filled_per_column = missing_report.get("values_filled_per_column", {})
    method_used = missing_report.get("method_used", {})

    missing_summary = {
        "columns_dropped": columns_dropped,
        "values_filled_per_column": values_filled_per_column,
        "method_used": method_used,
    }

    duplicates_removed = duplicates_report.get("duplicates_removed", 0)
    type_conversions = datatypes_report.get("columns_converted", {})
    correlated_features_removed = correlation_reduction_report.get("removed_features", [])
    useless_columns_dropped = column_selection_report.get("dropped_columns", [])
    outlier_summary = {
        "columns_analyzed": outliers_report.get("columns_analyzed", []),
        "rows_removed": outliers_report.get("rows_removed", 0),
        "values_capped": outliers_report.get("values_capped", 0),
        "capped_percentage_per_column": outliers_report.get("capped_percentage_per_column", {}),
        "high_capping_columns": outliers_report.get("high_capping_columns", []),
    }

    numeric_candidate_columns = analysis_warnings.get("numeric_candidate_columns", [])
    if not numeric_candidate_columns:
        numeric_candidate_columns = datatypes_report.get("numeric_candidate_columns", [])

    diagnostics = {
        "max_correlation": correlation_reduction_report.get("max_correlation_observed", 0.0),
        "pairs_checked": correlation_reduction_report.get("pairs_checked", 0),
        "high_outlier_capping_columns": outlier_summary.get("high_capping_columns", []),
        "numeric_candidate_columns": numeric_candidate_columns,
    }

    missing_percentage_map = _as_dict(analysis_report.get("missing_values_percentage", {}))
    high_missing_percentage_columns = {
        column_name: percentage
        for column_name, percentage in missing_percentage_map.items()
        if isinstance(percentage, (int, float)) and percentage > 40
    }

    warnings_and_insights = {
        "columns_dropped_due_to_high_missing_values": columns_dropped,
        "potential_identifier_columns": analysis_warnings.get("possible_id_columns", []),
        "columns_with_high_missing_percentages": high_missing_percentage_columns,
        "outlier_overcorrection_columns": outlier_summary.get("high_capping_columns", []),
    }

    actions_summary = {
        "columns_dropped": len(columns_dropped) + len(useless_columns_dropped),
        "columns_dropped_missing_values": len(columns_dropped),
        "columns_dropped_useless": len(useless_columns_dropped),
        "missing_values_filled": sum(values_filled_per_column.values()),
        "duplicates_removed": duplicates_removed,
        "type_conversions": len(type_conversions),
        "correlated_features_removed": len(correlated_features_removed),
        "outlier_rows_removed": outlier_summary.get("rows_removed", 0),
        "outlier_values_capped": outlier_summary.get("values_capped", 0),
    }

    return {
        "missing_values_summary": missing_summary,
        "duplicates_removed": duplicates_removed,
        "type_conversions": type_conversions,
        "correlation_reduction_summary": {
            "removed_features": correlated_features_removed,
            "threshold": correlation_reduction_report.get("threshold", 0.9),
            "max_correlation_observed": correlation_reduction_report.get("max_correlation_observed", 0.0),
            "pairs_checked": correlation_reduction_report.get("pairs_checked", 0),
        },
        "column_selection_summary": {
            "dropped_columns": useless_columns_dropped,
            "drop_reasons": column_selection_report.get("drop_reasons", {}),
        },
        "outlier_handling_summary": outlier_summary,
        "diagnostics": diagnostics,
        "warnings_and_insights": warnings_and_insights,
        "actions_summary": actions_summary,
    }


def build_report(changes: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible wrapper for generate_report."""
    return generate_report(changes)
