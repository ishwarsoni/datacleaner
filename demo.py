from pathlib import Path

import pandas as pd
from datacleaner import clean, handle_target

# -------------------------------
# STEP 1: Load dataset
# -------------------------------
url = "https://raw.githubusercontent.com/selva86/datasets/master/AmesHousing.csv"
local_csv = Path(__file__).resolve().parent / "tests" / "tmp_messy_realworld.csv"

# Prefer the local demo dataset so the script works offline and avoids dead URLs.
if local_csv.exists():
    df = pd.read_csv(local_csv)
else:
    df = pd.read_csv(url)

print("\n=== ORIGINAL DATA ===")
print(df.head())

# -------------------------------
# STEP 2: Select target
# -------------------------------
target_column = "SalePrice"

# -------------------------------
# STEP 3: Handle target
# -------------------------------
df, target_report = handle_target(df, target_column, strategy="auto")

print("\n=== TARGET REPORT ===")
print(target_report)

# -------------------------------
# STEP 4: Clean data
# -------------------------------
cleaned_df, report = clean(df, target_column=target_column, return_report=True)

print("\n=== CLEANED DATA ===")
print(cleaned_df.head())

# -------------------------------
# STEP 5: Report
# -------------------------------
print("\n=== CLEANING REPORT ===")
for k, v in report.items():
    print(f"{k}: {v}")

# -------------------------------
# STEP 6: Skew check (important)
# -------------------------------
print("\n=== SKEW SUMMARY ===")
print(report.get("skewness_summary", {}))

# -------------------------------
# STEP 7: Summary
# -------------------------------
print("\n=== SUMMARY ===")
print("Original shape:", df.shape)
print("Cleaned shape:", cleaned_df.shape)