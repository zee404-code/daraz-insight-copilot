import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import os
import json
import mlflow
import mlflow.sklearn

print("--- Script Starting (v3.4: Adding MLflow) ---")

# Define File Paths
DATA_PATH = "data/raw/Top_Selling_Product_Data.csv"
PROCESSED_DATA_DIR = "data/processed"
TRAIN_SET_PATH = os.path.join(PROCESSED_DATA_DIR, "train_set.csv")
TEST_SET_PATH = os.path.join(PROCESSED_DATA_DIR, "test_set.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")  # Still save locally too
MODEL_COLS_PATH = os.path.join(MODEL_DIR, "model_columns.json")
REPORT_DIR = "reports"  # For saving metrics plot

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Default local MLflow server URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Daraz Product Success")  # Experiment name

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
print("Directories ready.")


try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data from {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

# Prepare Data
numeric_features = [
    "Original Price",
    "Discount Price",
    "Number of Ratings",
    "Positive Seller Ratings",
    "Ship On Time",
    "Chat Response Rate",
    "No. of products to be sold",
]
categorical_features = ["Category", "Delivery Type", "Flagship Store"]
target_column = "Sell percentage to increase"

all_features = numeric_features + categorical_features
df_clean = df.dropna(subset=all_features + [target_column])
y_raw = df_clean[target_column]  # Keep raw target for potential analysis later
y = y_raw.clip(lower=0, upper=100).reset_index(drop=True)  # Clipped target for training

X_numeric = df_clean[numeric_features].reset_index(drop=True)
X_categorical = df_clean[categorical_features].reset_index(drop=True)
X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)
X = pd.concat([X_numeric, X_categorical_encoded], axis=1)
print(f"Final features shape (X): {X.shape}")

# Train Model & Log with MLflow
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# NEW: Start MLflow Run
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    mlflow.log_param("data_path", DATA_PATH)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Define and Log Model Parameters
    n_estimators = 10
    max_depth = 5
    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
        "n_jobs": -1,
    }
    mlflow.log_params(model_params)

    model = RandomForestRegressor(**model_params)

    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate and Log Metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    mlflow.log_metrics(metrics)
    print("Logged metrics to MLflow:")
    print(metrics)

    # Log Model with MLflow
    print("Logging model to MLflow...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        registered_model_name="daraz-product-success-predictor",
    )
    print("Model logged and registered.")

    # Log Artifacts like plots
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs. Predicted Success Score")
        plot_path = os.path.join(REPORT_DIR, "actual_vs_predicted.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print(f"Logged prediction plot artifact: {plot_path}")
    except ImportError:
        print("Matplotlib not found, skipping plot artifact logging.")

# Save Local Artifacts
print("Saving local model artifact...")
joblib.dump(model, MODEL_PATH)
print(f"Local model saved to {MODEL_PATH}")

print("Saving local model columns...")
model_columns = list(X.columns)
with open(MODEL_COLS_PATH, "w") as f:
    json.dump(model_columns, f)
print(f"Local model columns saved to {MODEL_COLS_PATH}")

# Saving train/test
print("Saving local train and test sets...")
train_df = pd.concat(
    [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
)
train_df.to_csv(TRAIN_SET_PATH, index=False)
test_df.to_csv(TEST_SET_PATH, index=False)
print(f"Local Train/Test sets saved to {PROCESSED_DATA_DIR}")

print("--- Script Finished ---")
