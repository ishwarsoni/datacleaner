# datacleaner

Production-grade, explainable data cleaning for messy real-world datasets.

`datacleaner` provides a conservative default pipeline that improves data quality while minimizing risky transformations.

## Install

```bash
pip install datacleanr
```

- PyPI package name: `datacleanr`
- Import name: `datacleaner`
- PyPI: https://pypi.org/project/datacleanr/

## Why datacleaner

- Safe defaults that avoid aggressive data mutation
- Transparent step-wise reporting for auditability
- Defensive behavior for noisy, mixed-quality production data
- Drop-in usage with pandas DataFrames

## Core Pipeline

The default `clean(df)` flow runs in this order:

1. Missing value handling
2. Duplicate removal
3. Datatype inference and conversion
4. Outlier treatment
5. Text standardization
6. Low-information column selection
7. Correlation reduction
8. Safety checks and optional rollbacks

## Quick Start

```python
from datacleaner import analyze, clean

# Optional read-only diagnostics before cleaning
analysis = analyze(df)

# Production-safe cleaning with report output
cleaned_df, report = clean(
    df,
    return_report=True,
    outlier_method="cap",  # or "remove"
    verbose=False,
    safe_mode=True,
)

print(df.shape, "->", cleaned_df.shape)
print(report["summary"]["actions_summary"])
```

## API

### clean(df, return_report=True, outlier_method="cap", verbose=False, safe_mode=True)

Runs the full cleaning pipeline.

- Input: pandas DataFrame (or DataFrame-like object)
- Output when `return_report=True`: `(cleaned_df, report)`
- Output when `return_report=False`: `cleaned_df`

### analyze(df)

Performs pre-cleaning analysis without modifying data.

- Returns column metadata, missing-value percentages, uniqueness, and quality warnings.

## Reporting

When `return_report=True`, the report includes:

- `steps`: per-step operation details
- `summary`: consolidated actions and diagnostics
- `safety`: data-loss metrics, warnings, and rollback details

This makes cleaning behavior traceable and easier to review in pipelines and model governance workflows.

## Design Principles

- Conservative conversion thresholds for mixed data
- Explainability over black-box transformations
- Graceful handling of malformed inputs
- Stability under stress-tested messy data scenarios

## Contributing

Contributions are welcome.

1. Open an issue for bug reports or major changes.
2. Add tests for behavioral changes.
3. Keep changes backward-compatible with current API contracts.

## License

MIT License. See [LICENSE](LICENSE).
