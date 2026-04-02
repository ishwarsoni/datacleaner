from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datacleaner import clean, handle_target


RNG = np.random.default_rng(20260402)
RESULT_PATH = PROJECT_ROOT / "tests" / "validation_50_results.json"


@dataclass
class DatasetCase:
    name: str
    category: str
    df: pd.DataFrame
    target: str
    strategy: str
    threshold: float


def _inject_target_missing(df: pd.DataFrame, target: str, ratio: float) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out
    missing_count = max(1, int(len(out) * ratio))
    missing_idx = RNG.choice(out.index.to_numpy(), size=min(missing_count, len(out)), replace=False)
    out.loc[missing_idx, target] = np.nan
    return out


def _messy_variant(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Add duplicate rows.
    if len(out) > 10:
        dup_idx = RNG.choice(out.index.to_numpy(), size=max(2, len(out) // 25), replace=False)
        out = pd.concat([out, out.loc[dup_idx]], ignore_index=True)

    # Add dirty placeholders in object columns.
    object_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    placeholders = np.array(["N/A", "na", "-", "", "unknown", "??"], dtype=object)
    for col in object_cols[:2]:
        if len(out) == 0:
            continue
        idx = RNG.choice(np.arange(len(out)), size=max(1, len(out) // 12), replace=False)
        vals = RNG.choice(placeholders, size=len(idx), replace=True)
        for i, v in zip(idx, vals):
            out.at[i, col] = v

    # Add numeric-formatting noise to one numeric column.
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col = numeric_cols[0]
        s = out[col]
        noisy = s.astype("string")
        idx = RNG.choice(np.arange(len(out)), size=max(2, len(out) // 15), replace=False)
        for i in idx:
            value = s.iloc[i]
            if pd.isna(value):
                continue
            style = int(RNG.integers(0, 3))
            if style == 0:
                noisy.iat[i] = f"{float(value):,.2f}"
            elif style == 1:
                noisy.iat[i] = f"₹{float(value):,.1f}"
            else:
                noisy.iat[i] = f"{float(value) % 100:.0f}%"
        out[col] = noisy

    # Inject outliers in another numeric column.
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2 and len(out) > 5:
        col = numeric_cols[-1]
        out_idx = RNG.choice(np.arange(len(out)), size=max(1, len(out) // 30), replace=False)
        out.loc[out_idx, col] = pd.to_numeric(out.loc[out_idx, col], errors="coerce") * 20

    return out


def _build_real_cases() -> list[DatasetCase]:
    cases: list[DatasetCase] = []

    sklearn_loaders = [
        ("iris", datasets.load_iris),
        ("wine", datasets.load_wine),
        ("breast_cancer", datasets.load_breast_cancer),
        ("diabetes", datasets.load_diabetes),
        ("digits", datasets.load_digits),
        ("linnerud", datasets.load_linnerud),
    ]

    for name, loader in sklearn_loaders:
        bunch = loader(as_frame=True)
        df = bunch.frame.copy()
        target = "target" if "target" in df.columns else df.columns[-1]

        low_missing = _inject_target_missing(df, target, 0.03)
        high_missing = _inject_target_missing(_messy_variant(df), target, 0.2)

        cases.append(DatasetCase(f"real_{name}_low_missing", "real", low_missing, target, "fill", 0.05))
        cases.append(DatasetCase(f"real_{name}_high_missing", "real", high_missing, target, "fill", 0.05))

    csv_path = PROJECT_ROOT / "tests" / "tmp_messy_realworld.csv"
    csv_df = pd.read_csv(csv_path)
    csv_target = "SalePrice"
    cases.append(
        DatasetCase(
            "real_local_housing_csv",
            "real",
            _inject_target_missing(_messy_variant(csv_df), csv_target, 0.1),
            csv_target,
            "fill",
            0.05,
        )
    )
    cases.append(
        DatasetCase(
            "real_local_housing_csv_drop",
            "real",
            _inject_target_missing(csv_df, csv_target, 0.25),
            csv_target,
            "drop",
            0.2,
        )
    )
    cases.append(
        DatasetCase(
            "real_local_housing_csv_fill",
            "real",
            _inject_target_missing(csv_df, csv_target, 0.03),
            csv_target,
            "fill",
            0.05,
        )
    )

    return cases[:15]


def _build_synthetic_cases() -> list[DatasetCase]:
    cases: list[DatasetCase] = []
    for i in range(20):
        n = int(RNG.integers(120, 280))
        x1 = RNG.normal(0, 1, n)
        x2 = x1 * 0.95 + RNG.normal(0, 0.1, n)
        x3 = RNG.lognormal(mean=1.5, sigma=1.2, size=n)
        text = RNG.choice(["NY", "  ny", "N/A", "unknown", "", "LA", "M", "F", "??"], size=n)
        amount = RNG.normal(50000, 12000, n)
        amount_str = pd.Series(amount).map(lambda v: f"₹{v:,.2f}")
        percent = pd.Series(RNG.uniform(0, 100, n)).map(lambda v: f"{v:.1f}%")
        y = (0.8 * x1 + 0.2 * x3 + RNG.normal(0, 0.5, n) > np.median(x3)).astype(float)

        df = pd.DataFrame(
            {
                "id_col": [f"ID_{i}_{j}" for j in range(n)],
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "city": text,
                "amount": amount_str,
                "progress": percent,
                "target": y,
            }
        )

        # Missing values and duplicates.
        for col in ["x1", "city", "amount"]:
            idx = RNG.choice(np.arange(n), size=max(1, n // 10), replace=False)
            df.loc[idx, col] = np.nan

        dup_idx = RNG.choice(np.arange(n), size=max(2, n // 18), replace=False)
        df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

        # Outliers and mixed numeric text.
        oi = RNG.choice(np.arange(len(df)), size=max(1, len(df) // 25), replace=False)
        df.loc[oi, "x3"] = df.loc[oi, "x3"].fillna(1) * 30
        mix_idx = RNG.choice(np.arange(len(df)), size=max(2, len(df) // 20), replace=False)
        for idx in mix_idx:
            df.at[idx, "amount"] = RNG.choice(["abc", "--", "₹1,23,456", "10%", "??"])

        ratio = 0.02 if i % 2 == 0 else 0.18
        strategy = "fill" if i % 3 != 0 else "drop"
        threshold = 0.05
        df = _inject_target_missing(df, "target", ratio)

        cases.append(DatasetCase(f"synthetic_{i:02d}", "synthetic", df, "target", strategy, threshold))

    return cases


def _make_edge_df(kind: int) -> pd.DataFrame:
    n = 160
    base = pd.DataFrame(
        {
            "feature_a": RNG.normal(0, 1, n),
            "feature_b": RNG.normal(5, 2, n),
            "text": RNG.choice(["A", "B", "unknown", "", "??", "M", "F"], size=n),
            "target": RNG.integers(0, 2, size=n).astype(float),
        }
    )

    if kind == 0:
        # Highly positively skewed.
        base["skew_col"] = RNG.lognormal(mean=2.5, sigma=1.4, size=n)
    elif kind == 1:
        # Highly negatively skewed via mirrored lognormal values.
        base["skew_col"] = -(RNG.lognormal(mean=2.2, sigma=1.2, size=n))
    elif kind == 2:
        # Mixed values around zero.
        left = -RNG.lognormal(mean=1.8, sigma=1.1, size=n // 2)
        right = RNG.lognormal(mean=2.0, sigma=1.1, size=n - n // 2)
        mixed = np.concatenate([left, right])
        RNG.shuffle(mixed)
        base["skew_col"] = mixed
    elif kind == 3:
        mask = RNG.random(n) < 0.85
        values = pd.Series(RNG.choice(["x", "y"], size=n), dtype="object")
        values.loc[mask] = pd.NA
        base["mostly_missing"] = values
    elif kind == 4:
        base["constant_col"] = "CONST"
    else:
        base["money"] = pd.Series(RNG.normal(35000, 7000, n)).map(lambda v: f"₹{v:,.0f}")
        base["pct"] = pd.Series(RNG.uniform(0, 100, n)).map(lambda v: f"{v:.0f}%")

    # Inject target missingness pattern.
    miss_ratio = 0.03 if kind % 2 == 0 else 0.2
    base = _inject_target_missing(base, "target", miss_ratio)

    return base


def _build_edge_cases() -> list[DatasetCase]:
    cases: list[DatasetCase] = []
    for i in range(15):
        kind = i % 6
        df = _make_edge_df(kind)
        strategy = "fill"
        threshold = 0.05
        cases.append(DatasetCase(f"edge_{i:02d}", "edge", df, "target", strategy, threshold))
    return cases


def _safe_skew(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 3:
        return None
    value = float(values.skew())
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _validate_report_structure(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required_top = {"steps", "summary", "final_shape", "rows_removed_pct", "columns_removed_pct", "safety"}
    missing_top = sorted([k for k in required_top if k not in report])
    if missing_top:
        issues.append(f"report_missing_top_fields:{','.join(missing_top)}")

    steps = report.get("steps", {}) if isinstance(report, dict) else {}
    required_steps = {
        "missing_values",
        "text_standardization",
        "duplicates",
        "datatypes",
        "correlation_reduction",
        "column_selection",
        "outliers",
        "skewness",
    }
    missing_steps = sorted([k for k in required_steps if k not in steps])
    if missing_steps:
        issues.append(f"report_missing_steps:{','.join(missing_steps)}")

    return issues


def run_case(case: DatasetCase) -> dict[str, Any]:
    before_shape = case.df.shape
    issues: list[str] = []

    if case.target not in case.df.columns:
        return {
            "name": case.name,
            "category": case.category,
            "before_shape": before_shape,
            "after_shape": before_shape,
            "passed": False,
            "target_status": "missing_target_column",
            "issues": ["target_column_missing"],
            "skewness": {},
            "features": {},
        }

    original_target = case.df[case.target].copy()

    handled_df, target_report = handle_target(
        case.df,
        target_column=case.target,
        strategy=case.strategy,
        threshold=case.threshold,
    )

    expected_action = "filled" if (original_target.isna().mean() <= case.threshold and case.strategy == "fill") else "rows_dropped"
    if original_target.isna().mean() == 0:
        expected_action = "none"
    action = target_report.get("action")
    if action != expected_action:
        issues.append(f"target_action_mismatch:expected_{expected_action}_got_{action}")

    pre_skew: dict[str, float] = {}
    post_skew: dict[str, float] = {}
    handled_numeric = handled_df.select_dtypes(include="number")
    for col in handled_numeric.columns:
        if col == case.target:
            continue
        skew = _safe_skew(handled_df[col])
        if skew is not None:
            pre_skew[col] = skew

    handled_shape = handled_df.shape
    cleaned_df, report = clean(handled_df, target_column=case.target, return_report=True)
    after_shape = cleaned_df.shape

    # Target safety: target retained and unchanged for retained rows.
    if case.target not in cleaned_df.columns:
        issues.append("target_removed_in_clean")
    else:
        aligned_before = handled_df[case.target].reindex(cleaned_df.index)
        if not aligned_before.equals(cleaned_df[case.target]):
            issues.append("target_changed_in_clean")

    # Safety guards.
    row_loss = ((handled_shape[0] - after_shape[0]) / handled_shape[0] * 100) if handled_shape[0] else 0.0
    col_loss = ((handled_shape[1] - after_shape[1]) / handled_shape[1] * 100) if handled_shape[1] else 0.0
    if after_shape[0] == 0:
        issues.append("empty_dataframe_after_clean")
    if row_loss > 30:
        issues.append(f"row_loss_gt_30:{row_loss:.2f}")
    if col_loss > 30:
        issues.append(f"col_loss_gt_30:{col_loss:.2f}")

    # Report structure and warnings quality.
    issues.extend(_validate_report_structure(report))
    for warning in report.get("integrity_warnings", []):
        if isinstance(warning, str) and warning.strip() == "":
            issues.append("empty_warning_message")

    # Pipeline feature checks.
    steps = report.get("steps", {})
    if "rows_removed" not in steps.get("duplicates", {}) and "error" not in steps.get("duplicates", {}):
        # duplicates module reports duplicates_removed, ensure key exists.
        if "duplicates_removed" not in steps.get("duplicates", {}):
            issues.append("duplicates_report_missing_key")

    dtype_report = steps.get("datatypes", {})
    if "columns_converted" not in dtype_report:
        issues.append("datatype_report_missing_conversions")

    # Numeric parsing feature evidence.
    converted_cols = dtype_report.get("columns_converted", {})
    numeric_parsing_hit = any(
        key in set(dtype_report.get("currency_stripped_columns", []))
        or key in set(dtype_report.get("percentage_columns", []))
        or key in set(dtype_report.get("comma_normalized_columns", []))
        for key in converted_cols.keys()
    )

    # Skewness checks.
    skew_step = steps.get("skewness", {})
    transformed_cols = skew_step.get("columns_transformed", []) if isinstance(skew_step, dict) else []
    skew_threshold = float(skew_step.get("threshold", 1.0)) if isinstance(skew_step, dict) else 1.0
    for col in cleaned_df.select_dtypes(include="number").columns:
        if col == case.target:
            continue
        skew = _safe_skew(cleaned_df[col])
        if skew is not None:
            post_skew[col] = skew

    if case.target in transformed_cols:
        issues.append("target_transformed_by_skewness")

    for col in transformed_cols:
        if col in pre_skew and abs(pre_skew[col]) <= skew_threshold:
            issues.append(f"non_skewed_column_transformed:{col}:{pre_skew[col]:.4f}")

    for col in transformed_cols:
        if col in pre_skew and col in post_skew:
            if abs(post_skew[col]) > abs(pre_skew[col]) + 1e-6:
                issues.append(f"skew_not_reduced:{col}:{pre_skew[col]:.4f}->{post_skew[col]:.4f}")

    for col in cleaned_df.columns:
        if col == case.target:
            continue
        if col not in transformed_cols and col in pre_skew and col in post_skew:
            # unchanged columns should not shift much from skew step itself.
            pass

    numeric_df = cleaned_df.select_dtypes(include="number")
    if np.isinf(numeric_df.to_numpy(dtype=float, copy=True)).any():
        issues.append("inf_introduced")
    if numeric_df.isna().any().any() and len(numeric_df.columns) > 0:
        # allow NaN if explicitly retained; flag only when newly introduced in transformed cols.
        for col in transformed_cols:
            if col in numeric_df.columns and numeric_df[col].isna().any() and not handled_df[col].isna().any():
                issues.append(f"nan_introduced_in_skew_col:{col}")

    passed = len(issues) == 0
    target_status = "unchanged" if "target_changed_in_clean" not in issues and "target_removed_in_clean" not in issues else "changed"

    feature_flags = {
        "numeric_parsing": bool(numeric_parsing_hit),
        "outliers": "outliers" in steps and isinstance(steps.get("outliers"), dict),
        "skew_handling": "skewness" in steps,
        "target_safety": target_status == "unchanged",
        "report_correctness": len([i for i in issues if i.startswith("report_") or "warning" in i]) == 0,
    }

    return {
        "name": case.name,
        "category": case.category,
        "before_shape": before_shape,
        "handled_shape": handled_shape,
        "after_shape": after_shape,
        "passed": passed,
        "target_status": target_status,
        "target_report": target_report,
        "issues": issues,
        "skewness": {
            "columns_transformed": transformed_cols,
            "before": {k: round(v, 4) for k, v in pre_skew.items()},
            "after": {k: round(v, 4) for k, v in post_skew.items()},
        },
        "features": feature_flags,
    }


def main() -> None:
    real_cases = _build_real_cases()
    synthetic_cases = _build_synthetic_cases()
    edge_cases = _build_edge_cases()

    all_cases = real_cases + synthetic_cases + edge_cases
    assert len(all_cases) == 50, f"Expected 50 cases, got {len(all_cases)}"

    results = [run_case(case) for case in all_cases]

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    feature_summary = {
        "numeric_parsing": sum(1 for r in results if r["features"]["numeric_parsing"]),
        "outliers": sum(1 for r in results if r["features"]["outliers"]),
        "skew_handling": sum(1 for r in results if r["features"]["skew_handling"]),
        "target_safety": sum(1 for r in results if r["features"]["target_safety"]),
        "report_correctness": sum(1 for r in results if r["features"]["report_correctness"]),
    }

    payload = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "feature_summary": feature_summary,
        "results": results,
    }

    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Total datasets tested: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Results written to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
