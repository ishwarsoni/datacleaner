# datacleaner

ML-safe data cleaning library for real-world datasets.

## 🚀 Overview

A production-grade Python library for cleaning messy datasets with safe, explainable defaults.

- Prevents over-cleaning
- Preserves target column
- Provides transparent transformations
- Tested on 50 datasets

## Why this library?

- Many cleaners over-clean data and can accidentally corrupt labels.
- `datacleaner` is designed for ML-safe preprocessing with conservative defaults.
- Target handling is explicit and separated from general cleaning.
- Every major action is reported for control and traceability.

## 🔥 Key Features

- Full cleaning pipeline
- Target column NEVER modified in clean()
- Explicit target handling via handle_target()
- Smart datatype conversion (₹, %, commas)
- Outlier handling with safeguards
- Skewness handling (only when beneficial)
- Conservative column selection (no aggressive drops)
- Correlation reduction (no cascade deletion)
- Safety guards (row/column loss control)
- Detailed report output
- Validated on 50 datasets

## 📦 Installation

```bash
pip install datacleanr
```

PyPI package name: `datacleanr`

Import name in Python code: `datacleaner`

The package is published as `datacleanr` on PyPI, but imported as `datacleaner` in code.

## ⚡ Quick Example

```python
from datacleaner import clean, handle_target

# handle_target() processes missing target labels safely and returns an action report.
df, target_report = handle_target(df, "target", strategy="auto")

# clean() returns the cleaned dataframe plus a structured transformation report.
cleaned_df, report = clean(df, target_column="target", return_report=True)
```

## 🎯 Target Handling (IMPORTANT)

The `clean()` function **never modifies the target column**.

This is intentional to ensure:

- no label corruption
- safe usage in ML pipelines
- predictable behavior

### Why?

In real-world ML workflows:

- filling or modifying target values can introduce bias
- automatic changes to labels are unsafe

### How to handle target values?

Use the dedicated function:

```python
from datacleaner import handle_target

df, target_report = handle_target(df, "target", strategy="auto")
```

### Behavior:

- If missing values are small -> optional fill (controlled)
- If missing values are large -> rows are dropped
- Fully transparent (reports actions taken)

### ⚠️ Important

If you skip `handle_target()`:

- target column will remain unchanged
- missing values in target will NOT be handled

This design ensures full user control over label processing.

## 📊 Skew Handling

- Applied ONLY when it improves distribution
- Safe for negative values
- Never applied to target
- Fully transparent

Example:

```json
{
    "feature": {
        "before": 1.14,
        "after": 0.05,
        "method": "log"
    }
}
```

## 🧠 Feature Selection

- Avoids dropping useful columns
- Prevents cascade correlation deletion
- Keeps column loss controlled, with validation showing no dataset above 25% clean-stage column loss

## 📈 Validation (IMPORTANT)

- Tested across 50 datasets (classification and regression)
- Includes real datasets, synthetic noisy datasets, and edge cases (including small datasets)
- 0 failures across 50 datasets
- 0 target corruption
- Stable across all scenarios

Validation artifact: [tests/validation_50_results.json](tests/validation_50_results.json)

## 🎯 Example Datasets Tested

- Ames Housing
- Iris
- Breast Cancer
- Synthetic noisy datasets
- Small edge-case datasets

## 📷 Screenshots

- Before vs After cleaning
![Before vs After cleaning](assets/before_after.png)

- Cleaning report output
![Cleaning report output](assets/report.png)

- Skew transformation example
![Skew transformation example](assets/skew.png)

## 📄 Report Example

This is a simplified example of the structured report returned by `clean()`:

```json
{
    "final_shape": [rows, columns],
    "rows_removed_pct": 2.5,
    "columns_removed_pct": 8.3,
    "integrity_warnings": [],
    "skewness_summary": {
        "columns_transformed": ["feature_x"],
        "details": {
            "feature_x": {
                "before": 1.14,
                "after": 0.05,
                "method": "log"
            }
        }
    }
}
```

## 🧠 When to Use / When NOT to Use

Use when:

- datasets are messy and real-world
- you need safe preprocessing before ML training
- you want transparent, structured cleaning reports

Avoid when:

- data is already clean and standardized
- you need heavily customized, domain-specific cleaning rules

## ⚙️ Design Philosophy

- Conservative over aggressive
- Transparency over automation
- Safety over convenience

## 🏷 Version

v0.1.4

## License

MIT License. See [LICENSE](LICENSE).
