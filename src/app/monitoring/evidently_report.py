import pandas as pd
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric

print("--- Evidently Report Script Starting ---\n")

# Paths
TRAIN_SET_PATH = "data/processed/train_set.csv"
TEST_SET_PATH = "data/processed/test_set.csv"
RAW_DATA_PATH = "data/raw/Top_Selling_Product_Data.csv"

REPORT_DIR = "reports"
TABULAR_REPORT_PATH = os.path.join(REPORT_DIR, "data_and_target_drift.html")
TEXT_REPORT_PATH = os.path.join(REPORT_DIR, "retrieval_corpus_drift.html")

os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------------------------
# [1/2] Tabular Data Drift (D1) – already working
# ---------------------------------------------------------
print("[1/2] Generating Tabular Drift Report (D1)...")
try:
    train_df = pd.read_csv(TRAIN_SET_PATH)
    test_df = pd.read_csv(TEST_SET_PATH)

    tabular_report = Report(
        metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ]
    )

    tabular_report.run(reference_data=train_df, current_data=test_df)
    tabular_report.save_html(TABULAR_REPORT_PATH)
    print(f"-> Saved to {TABULAR_REPORT_PATH}\n")

except Exception as e:
    print(f"Warning: Tabular report failed: {e}\n")

# ---------------------------------------------------------
# [2/2] Retrieval Corpus Drift (D4)
# ---------------------------------------------------------
print("[2/2] Generating Retrieval Corpus Drift Report (D4)...")
try:
    raw_df = pd.read_csv(RAW_DATA_PATH)

    # Choose the best text-like column (Title is usually the richest text)
    if "Title" in raw_df.columns:
        text_column = "Title"
    elif "Description" in raw_df.columns:
        text_column = "Description"
    else:
        text_column = "Category"

    print(f"   Using column for drift analysis: '{text_column}'")

    # Clean NaNs in the chosen column
    raw_df = raw_df.dropna(subset=[text_column]).reset_index(drop=True)
    print(f"   Rows after dropping NaNs in '{text_column}': {len(raw_df)}")

    if len(raw_df) < 10:
        raise ValueError("Not enough data after cleaning")

    # Split into reference (first half) and current (second half)
    mid = len(raw_df) // 2
    reference_corpus = raw_df.iloc[:mid].copy()
    current_corpus = raw_df.iloc[mid:].copy()

    print(
        f"   Reference rows: {len(reference_corpus)}, Current rows: {len(current_corpus)}"
    )

    # Use simple, robust drift detection that works on any column type
    text_report = Report(
        metrics=[
            DataDriftPreset(),  # catches drift in all columns (including text/category)
            ColumnDriftMetric(
                column_name=text_column
            ),  # explicit drift test on the chosen column
        ]
    )

    text_report.run(reference_data=reference_corpus, current_data=current_corpus)

    text_report.save_html(TEXT_REPORT_PATH)
    print(f"-> Saved to {TEXT_REPORT_PATH}\n")

except Exception as e:
    print(f"Error generating D4 report: {e}")
    import traceback

    traceback.print_exc()
