import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import os

print("--- Evidently Report Script Starting ---")

# Define File Paths
TRAIN_SET_PATH = "data/processed/train_set.csv"
TEST_SET_PATH = "data/processed/test_set.csv"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "data_and_target_drift.html")

# Create the 'reports' directory if it doesn't exist
os.makedirs(REPORT_DIR, exist_ok=True)

# Load Datasets
try:
    train_df = pd.read_csv(TRAIN_SET_PATH)
    test_df = pd.read_csv(TEST_SET_PATH)
    print("Successfully loaded training and test datasets.")
except FileNotFoundError as e:
    print("Error: Could not find dataset files.")
    print(f"Details: {e}")
    print("Please run 'python train.py' first to generate the datasets.")
    exit()

# Generate the Report
print("Generating Evidently drift report...")

# Initialize the report and add the monitoring "metrics"
data_drift_report = Report(
    metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ]
)

# Run the report
data_drift_report.run(
    current_data=test_df,
    reference_data=train_df,
    column_mapping=None,
)

# Save the Report
data_drift_report.save_html(REPORT_PATH)

print(f"*** Success! Report saved to {REPORT_PATH} ***")
print("--- Script Finished ---")
