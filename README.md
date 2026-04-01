# datacleaner - ML-safe data cleaning library

datacleaner cleans messy real-world tabular datasets with safe defaults, explicit behavior, and explainable outputs.
It is designed for ML workflows where target integrity matters: the main clean pipeline does not modify the target column when target_column is provided.

## Key Features

- Safe cleaning pipeline with defensive guards
- Target column protection across all cleaning steps
- Explicit target handling via handle_target()
- Real-world numeric parsing support (currency symbols, percentages, commas)
- Outlier handling with cap/remove strategies
- Column reduction with safety rollbacks
- Detailed report with shape and integrity metadata
- Validated on 30+ datasets across classification and regression

## Installation

    pip install datacleanr

Package name on PyPI: datacleanr
Import name in code: datacleaner

## Quick Example

    from datacleaner import handle_target, clean

    df, target_meta = handle_target(df, "target")
    cleaned_df, report = clean(df, target_column="target", return_report=True)

## Target Handling

Why target is not modified automatically:
- In supervised ML, target values are labels. Silent mutation can corrupt training targets.
- The main clean function is intentionally conservative and avoids target rewrites.

How to use handle_target:
- Use handle_target before clean when target contains missing values.
- This function is explicit and returns transparent metadata about what it did.

Safe defaults:
- strategy="auto" defaults to dropping rows with missing target
- If target missing ratio is above threshold, rows are dropped regardless of strategy
- Fill is only used for low-missing targets when explicitly requested

Returned metadata fields:
- target_missing_ratio
- action: none | rows_dropped | filled
- rows_removed
- fill_value (only when action is filled)
- filled_count (only when action is filled)

## Pipeline Overview

clean runs this sequence:

1. missing value handling
2. duplicate removal
3. datatype cleaning
4. outlier handling
5. text standardization
6. column selection
7. correlation reduction
8. safety checks and reporting

## Example Report Output

    {
      "final_shape": [1200, 24],
      "rows_removed_pct": 2.5,
      "columns_removed_pct": 8.3,
      "integrity_warnings": []
    }

## Validation

- Tested on 30+ datasets from multiple domains
- Includes classification and regression datasets
- Includes noisy variants with missing values, mixed text/number fields, currency, percentages, and comma-formatted numerics
- No target corruption observed with target_column protection and explicit handle_target preprocessing
- No crash regressions in stress validation

## Design Philosophy

- Conservative over aggressive
- Transparency over automation
- Safety over convenience

## Future Improvements

- Configurable policy profiles per domain
- More advanced locale-aware numeric parsing
- Benchmarking against common sklearn preprocessing baselines

## License

MIT License. See LICENSE.
