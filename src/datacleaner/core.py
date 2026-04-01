"""Core orchestration module for datacleaner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .column_selection import drop_useless_columns
from .correlation_reduction import remove_correlated_features
from .datatypes import fix_types
from .duplicates import remove_duplicates
from .missing_values import handle_missing
from .outliers import handle_outliers
from .reporting import generate_report
from .text_standardization import standardize_text


def clean(
    df: pd.DataFrame,
    return_report: bool = True,
    outlier_method: str = "cap",
    verbose: bool = False,
    safe_mode: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]] | pd.DataFrame:
    """Run the full cleaning pipeline.

    Args:
        df: Input pandas DataFrame.
        return_report: When True, include a structured report in the return value.
        outlier_method: Strategy passed to outlier handling.
        verbose: When True, print concise step-by-step cleaning logs.
        safe_mode: When True, rollback aggressive steps if data-loss thresholds are exceeded.

    Returns:
        Cleaned DataFrame, or a tuple of cleaned DataFrame and report.

    Notes:
        The pipeline is defensive and avoids crashing on malformed input. If a step
        fails, remaining steps continue with the current DataFrame.
    """
    safe_return_report = return_report if isinstance(return_report, bool) else True
    safe_verbose = verbose if isinstance(verbose, bool) else False
    safe_outlier_method = outlier_method if isinstance(outlier_method, str) else "cap"
    safe_safe_mode = safe_mode if isinstance(safe_mode, bool) else True

    if isinstance(df, pd.DataFrame):
        cleaned_df = df.copy()
    else:
        try:
            cleaned_df = pd.DataFrame(df)
        except Exception:
            cleaned_df = pd.DataFrame()

    initial_rows, initial_cols = cleaned_df.shape

    safety_warnings: list[str] = []
    safety_rollbacks: list[str] = []

    def _loss_pct(current_df: pd.DataFrame) -> tuple[float, float]:
        if initial_rows > 0:
            row_loss_pct = (initial_rows - len(current_df)) / initial_rows * 100
        else:
            row_loss_pct = 0.0
        if initial_cols > 0:
            col_loss_pct = (initial_cols - len(current_df.columns)) / initial_cols * 100
        else:
            col_loss_pct = 0.0
        return row_loss_pct, col_loss_pct

    def _guard_step(step_name: str, previous_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
        row_loss_pct, col_loss_pct = _loss_pct(candidate_df)
        current_cols = len(candidate_df.columns)
        enforce_col_guard = initial_cols > 5 and current_cols > 1
        exceeded = (row_loss_pct > 30) or (enforce_col_guard and col_loss_pct > 30)
        if not exceeded:
            return candidate_df

        warning = (
            f"Safety warning after {step_name}: row loss {row_loss_pct:.2f}%, "
            f"column loss {col_loss_pct:.2f}%."
        )
        print(warning)
        safety_warnings.append(warning)

        if safe_safe_mode:
            rollback_msg = f"Safe mode rollback: reverted {step_name} to prevent excessive data loss."
            print(rollback_msg)
            safety_rollbacks.append(step_name)
            return previous_df

        return candidate_df

    missing_values_report: dict[str, Any] = {}
    text_standardization_report: dict[str, Any] = {}
    duplicates_report: dict[str, Any] = {}
    datatypes_report: dict[str, Any] = {}
    correlation_reduction_report: dict[str, Any] = {}
    column_selection_report: dict[str, Any] = {}
    outliers_report: dict[str, Any] = {}

    try:
        previous_df = cleaned_df
        candidate_df, missing_values_report = handle_missing(cleaned_df)
        cleaned_df = _guard_step("missing_values", previous_df, candidate_df)
        if cleaned_df is previous_df and candidate_df is not previous_df:
            missing_values_report = {"rolled_back": True, "reason": "excessive_data_loss"}
    except Exception as exc:
        missing_values_report = {"error": str(exc)}

    if safe_verbose:
        dropped_columns_count = len(missing_values_report.get("columns_dropped", []))
        filled_values_count = sum(missing_values_report.get("values_filled_per_column", {}).values())
        print(
            f"Missing values: dropped {dropped_columns_count} columns, filled {filled_values_count} values."
        )

    try:
        cleaned_df, text_standardization_report = standardize_text(cleaned_df)
    except Exception as exc:
        text_standardization_report = {"error": str(exc)}

    if safe_verbose:
        standardized_values = sum(text_standardization_report.get("values_standardized_per_column", {}).values())
        print(f"Text standardization: updated {standardized_values} values.")

    try:
        previous_df = cleaned_df
        candidate_df, duplicates_report = remove_duplicates(cleaned_df)
        cleaned_df = _guard_step("duplicates", previous_df, candidate_df)
        if cleaned_df is previous_df and candidate_df is not previous_df:
            duplicates_report = {"rolled_back": True, "reason": "excessive_data_loss"}
    except Exception as exc:
        duplicates_report = {"error": str(exc)}

    if safe_verbose:
        duplicates_removed = duplicates_report.get("duplicates_removed", 0)
        print(f"Duplicates: removed {duplicates_removed} rows.")

    try:
        cleaned_df, datatypes_report = fix_types(cleaned_df)
    except Exception as exc:
        datatypes_report = {"error": str(exc)}

    if safe_verbose:
        conversion_count = len(datatypes_report.get("columns_converted", {}))
        print(f"Datatypes: converted {conversion_count} columns.")

    try:
        previous_df = cleaned_df
        candidate_df, correlation_reduction_report = remove_correlated_features(cleaned_df)
        cleaned_df = _guard_step("correlation_reduction", previous_df, candidate_df)
        if cleaned_df is previous_df and candidate_df is not previous_df:
            correlation_reduction_report = {"rolled_back": True, "reason": "excessive_data_loss"}
    except Exception as exc:
        correlation_reduction_report = {"error": str(exc)}

    if safe_verbose:
        correlated_removed = len(correlation_reduction_report.get("removed_features", []))
        print(f"Correlation reduction: removed {correlated_removed} correlated features.")

    try:
        previous_df = cleaned_df
        candidate_df, column_selection_report = drop_useless_columns(cleaned_df)
        cleaned_df = _guard_step("column_selection", previous_df, candidate_df)
        if cleaned_df is previous_df and candidate_df is not previous_df:
            column_selection_report = {
                **column_selection_report,
                "rolled_back": True,
                "reason": "excessive_data_loss",
            }
    except Exception as exc:
        column_selection_report = {"error": str(exc)}

    if safe_verbose:
        dropped_count = len(column_selection_report.get("dropped_columns", []))
        print(f"Column selection: dropped {dropped_count} low-information columns.")

    try:
        previous_df = cleaned_df
        candidate_df, outliers_report = handle_outliers(cleaned_df, method=safe_outlier_method)
        cleaned_df = _guard_step("outliers", previous_df, candidate_df)
        if cleaned_df is previous_df and candidate_df is not previous_df:
            outliers_report = {"rolled_back": True, "reason": "excessive_data_loss"}
    except Exception:
        try:
            previous_df = cleaned_df
            candidate_df, outliers_report = handle_outliers(cleaned_df, method="cap")
            cleaned_df = _guard_step("outliers", previous_df, candidate_df)
            if cleaned_df is previous_df and candidate_df is not previous_df:
                outliers_report = {"rolled_back": True, "reason": "excessive_data_loss"}
        except Exception as exc:
            outliers_report = {"error": str(exc)}

    if safe_verbose:
        if "rows_removed" in outliers_report:
            print(f"Outliers ({safe_outlier_method}): removed {outliers_report.get('rows_removed', 0)} rows.")
        else:
            print(f"Outliers ({safe_outlier_method}): capped {outliers_report.get('values_capped', 0)} values.")

    step_reports = {
        "missing_values": missing_values_report,
        "text_standardization": text_standardization_report,
        "duplicates": duplicates_report,
        "datatypes": datatypes_report,
        "correlation_reduction": correlation_reduction_report,
        "column_selection": column_selection_report,
        "outliers": outliers_report,
    }

    report = {
        "steps": step_reports,
        "summary": generate_report(step_reports),
        "safety": {
            "safe_mode": safe_safe_mode,
            "row_loss_percentage": _loss_pct(cleaned_df)[0],
            "column_loss_percentage": _loss_pct(cleaned_df)[1],
            "warnings": safety_warnings,
            "rollbacks": safety_rollbacks,
        },
    }

    if safe_return_report:
        return cleaned_df, report
    return cleaned_df
