import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import os
import json
import mlflow
import mlflow.sklearn

print("--- Training Script Starting (Light Version) ---")

# -------------------------
# PATHS
# -------------------------
DATA_PATH = "data/raw/Top_Selling_Product_Data.csv"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"

MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODEL_COLS_PATH = os.path.join(MODEL_DIR, "model_columns.json")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# -------------------------
# MLflow Setup
# -------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Daraz Product Success")


# -------------------------
# Load Data
# -------------------------
try:
    df = pd.read_csv(DATA_PATH)
    print("Loaded training data.")
except FileNotFoundError:
    print(f"ERROR: Could not find {DATA_PATH}")
    exit()


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


# -------------------------
# Clean Data
# -------------------------
df_clean = df.dropna(subset=numeric_features + categorical_features + [target_column])
y_raw = df_clean[target_column]

# Clip target safely (avoid extreme outliers)
y = y_raw.reset_index(drop=True)


# -------------------------
# Feature Engineering
# -------------------------
X_numeric = df_clean[numeric_features].reset_index(drop=True)
X_categorical = df_clean[categorical_features].reset_index(drop=True)

# Scale numeric
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(X_numeric), columns=X_numeric.columns
)

# Encode categoricals
X_cat_encoded = pd.get_dummies(X_categorical, drop_first=True)

# Final data
X = pd.concat([X_numeric_scaled, X_cat_encoded], axis=1)
model_columns = list(X.columns)

print(f"Feature matrix shape: {X.shape}")


# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Start MLflow Run
# -------------------------
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 80)
    mlflow.log_param("max_depth", 12)

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    mlflow.log_metrics(metrics)

    print("Metrics:", metrics)

    # Log model to MLflow
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="daraz-product-success",
    )

# -------------------------
# Save Local Artifacts
# -------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

with open(MODEL_COLS_PATH, "w") as f:
    json.dump(model_columns, f)

print("Saved model.joblib, scaler.joblib, model_columns.json")


# -------------------------
# Save Processed Train/Test Sets
# -------------------------
train_df = pd.concat(
    [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
)

train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_set.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_set.csv"), index=False)

print("Training completed successfully!")
print("--- End ---")
