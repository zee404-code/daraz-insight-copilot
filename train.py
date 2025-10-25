import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Added more metrics
import numpy as np  # Added numpy for sqrt
import joblib
import os

print("--- Script Starting (v3.1: Corrected) ---")

# --- 1. Define File Paths ---
DATA_PATH = "data/raw/Top_Selling_Product_Data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

# --- 2. Create 'models' Directory ---
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Directory '{MODEL_DIR}' is ready.")

# --- 3. Load the Data ---
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded data from {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

# --- 4. Prepare Data for Modeling (Feature Engineering) ---

numeric_features = [
    'Original Price',
    'Discount Price',
    'Number of Ratings',
    'Positive Seller Ratings',
    'Ship On Time',
    'Chat Response Rate',
    'No. of products to be sold'
]

categorical_features = [
    'Category',
    'Delivery Type',
    'Flagship Store'
]

target_column = 'Sell percentage to increase'

print(f"Numeric Features: {numeric_features}")
print(f"Categorical Features: {categorical_features}")

# Define all_features by combining the two lists
all_features = numeric_features + categorical_features

df_clean = df.dropna(subset=all_features + [target_column])
print(f"Data shape after cleaning: {df_clean.shape}")

X_numeric = df_clean[numeric_features]
X_categorical = df_clean[categorical_features]
y = df_clean[target_column]

print("Applying One-Hot Encoding...")
X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)

# We must align the indices before concatenating
X_numeric = X_numeric.reset_index(drop=True)
X_categorical_encoded = X_categorical_encoded.reset_index(drop=True)

X = pd.concat([X_numeric, X_categorical_encoded], axis=1)

print(f"Final features shape (X): {X.shape}")

# --- 5. Train the Model ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

print("Initializing tuned model...")

model = RandomForestRegressor(
    criterion="friedman_mse",
    n_estimators=1800,
    max_depth=36,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.55,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- Model Performance Section ---
print("\n--- Model Performance on Test Set ---")
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"R-squared (R²):   {r2:.4f}")
print(f"Mean Absolute Error (MAE):  {mae:.4f}")
print(f"Mean Squared Error (MSE):   {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("-----------------------------------")

# --- 6. Save the Model ---
joblib.dump(model, MODEL_PATH)

print(f"*** Success! Model saved to {MODEL_PATH} ***")
print("--- Script Finished ---")
