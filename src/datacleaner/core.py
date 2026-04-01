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
    target_column: str | None = None,
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
    safe_target_column = target_column if isinstance(target_column, str) else None

    if isinstance(df, pd.DataFrame):
        cleaned_df = df.copy()
    else:
        try:
            cleaned_df = pd.DataFrame(df)
        except Exception:
            cleaned_df = pd.DataFrame()

    original_input_df = cleaned_df.copy()

    initial_rows, initial_cols = cleaned_df.shape
    original_columns = cleaned_df.columns.tolist()
    original_dtypes = cleaned_df.dtypes.to_dict()
    original_target_series = (
        cleaned_df[safe_target_column].copy()
        if safe_target_column is not None and safe_target_column in cleaned_df.columns
        else None
    )

    safety_warnings: list[str] = []
    safety_rollbacks: list[str] = []
    integrity_warnings: list[str] = []

    def _is_dtype_equivalent(original_dtype: Any, new_dtype: Any) -> bool:
        if str(original_dtype) == str(new_dtype):
            return True

        original_is_text = pd.api.types.is_object_dtype(original_dtype) or pd.api.types.is_string_dtype(original_dtype)
        new_is_text = pd.api.types.is_object_dtype(new_dtype) or pd.api.types.is_string_dtype(new_dtype)
        if original_is_text and new_is_text:
            return True

        if pd.api.types.is_numeric_dtype(original_dtype) and pd.api.types.is_numeric_dtype(new_dtype):
            return True
        if pd.api.types.is_bool_dtype(original_dtype) and pd.api.types.is_bool_dtype(new_dtype):
            return True

        return False

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
        candidate_df, missing_values_report = handle_missing(
            cleaned_df,
            target_column=safe_target_column,
        )
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
        if safe_target_column is not None and safe_target_column in cleaned_df.columns:
            target_position = cleaned_df.columns.get_loc(safe_target_column)
            target_series = cleaned_df[safe_target_column].copy()
            text_input_df = cleaned_df.drop(columns=[safe_target_column])
            standardized_df, text_standardization_report = standardize_text(text_input_df)
            cleaned_df = standardized_df
            cleaned_df.insert(target_position, safe_target_column, target_series.reindex(cleaned_df.index))
        else:
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
        candidate_df, correlation_reduction_report = remove_correlated_features(
            cleaned_df,
            target_column=safe_target_column,
        )
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
        candidate_df, column_selection_report = drop_useless_columns(
            cleaned_df,
            target_column=safe_target_column,
        )
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
        candidate_df, outliers_report = handle_outliers(
            cleaned_df,
            method=safe_outlier_method,
            target_column=safe_target_column,
        )
        cleaned_df = _guard_step("outliers", previous_df, candidate_df)
        if cleaned_df is previous_df and candidate_df is not previous_df:
            outliers_report = {"rolled_back": True, "reason": "excessive_data_loss"}
    except Exception:
        try:
            previous_df = cleaned_df
            candidate_df, outliers_report = handle_outliers(
                cleaned_df,
                method="cap",
                target_column=safe_target_column,
            )
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

    # Final hard guards: target presence, valid schema, and non-empty column set.
    if safe_target_column is not None and original_target_series is not None:
        assert safe_target_column in cleaned_df.columns, f"Target column '{safe_target_column}' was removed"

    if cleaned_df.shape[1] == 0 and initial_cols > 0:
        integrity_warnings.append("All columns were removed; returning original input to preserve schema.")
        cleaned_df = original_input_df.copy()

    if initial_cols > 0:
        assert cleaned_df.shape[1] > 0, "cleaned DataFrame must contain at least one column"
    assert cleaned_df.columns.is_unique, "cleaned DataFrame has duplicate column names"

    # Non-blocking integrity checks for traceability.
    dropped_columns = set(missing_values_report.get("columns_dropped", []))
    dropped_columns.update(column_selection_report.get("dropped_columns", []))
    dropped_columns.update(correlation_reduction_report.get("removed_features", []))

    for column_name in cleaned_df.columns:
        if cleaned_df[column_name].isna().all() and column_name not in dropped_columns:
            integrity_warnings.append(
                f"Column '{column_name}' is entirely NaN after cleaning."
            )

    intentional_dtype_conversions = set(datatypes_report.get("columns_converted", {}).keys())
    for column_name in cleaned_df.columns:
        if column_name not in original_dtypes:
            continue

        original_dtype = original_dtypes[column_name]
        new_dtype = cleaned_df[column_name].dtype
        if column_name in intentional_dtype_conversions:
            continue
        if not _is_dtype_equivalent(original_dtype, new_dtype):
            integrity_warnings.append(
                f"Unexpected dtype change for '{column_name}': {original_dtype} -> {new_dtype}."
            )

    expected_column_order = [column_name for column_name in original_columns if column_name in cleaned_df.columns]
    if cleaned_df.columns.tolist() != expected_column_order:
        integrity_warnings.append("Column order changed unexpectedly relative to retained input columns.")

    if not cleaned_df.index.is_unique:
        integrity_warnings.append("Cleaned DataFrame index is not unique.")

    if safe_target_column is not None and original_target_series is not None and safe_target_column in cleaned_df.columns:
        current_target_series = cleaned_df[safe_target_column]
        if not _is_dtype_equivalent(original_target_series.dtype, current_target_series.dtype):
            integrity_warnings.append(
                f"Target dtype changed: {original_target_series.dtype} -> {current_target_series.dtype}."
            )

        target_aligned = original_target_series.reindex(cleaned_df.index)
        if not target_aligned.equals(current_target_series):
            integrity_warnings.append("Target values changed for retained rows.")

    final_rows, final_cols = cleaned_df.shape
    rows_removed_pct = ((initial_rows - final_rows) / initial_rows * 100) if initial_rows > 0 else 0.0
    columns_removed_pct = ((initial_cols - final_cols) / initial_cols * 100) if initial_cols > 0 else 0.0

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
        "final_shape": (final_rows, final_cols),
        "rows_removed_pct": rows_removed_pct,
        "columns_removed_pct": columns_removed_pct,
        "integrity_warnings": integrity_warnings,
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
