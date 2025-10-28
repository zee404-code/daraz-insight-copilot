import joblib
import pandas as pd
import json
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
from .instrumentation import setup_instrumentation, observe_prediction

# Initialize API and Load Artifacts ---
app = FastAPI(title="Daraz Product Success Predictor")

# Setup instrumentation
setup_instrumentation(app)

# Load the trained model.
# This happens once when the app starts.
try:
    model = joblib.load("models/model.joblib")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.joblib not found. Make sure it's in the 'models' folder.")
    model = None  # Set model to None to handle error gracefully

# Load the list of model columns (features)
# This is crucial for matching the API input to the model's expected input.
try:
    with open("models/model_columns.json", "r") as f:
        model_columns = json.load(f)
    print("Model columns loaded successfully.")
except FileNotFoundError:
    print("Error: model_columns.json not found. Run train.py to create it.")
    model_columns = []  # Set to empty list


# Define Input Data Shape (Pydantic BaseModel) ---
# This defines what the API expects in a /predict request.
# It must match the *raw* features you used for training.
class ProductFeatures(BaseModel):
    # Numeric features
    Original_Price: float
    Discount_Price: float
    Number_of_Ratings: int
    Positive_Seller_Ratings: float
    Ship_On_Time: float
    Chat_Response_Rate: float
    No_of_products_to_be_sold: float  # Pydantic uses Python-friendly names

    # Categorical features
    Category: str
    Delivery_Type: str
    Flagship_Store: str

    # This is Pydantic's way to show an example in the /docs
    class Config:
        schema_extra = {
            "example": {
                "Original_Price": 1650,
                "Discount_Price": 725,
                "Number_of_Ratings": 31,
                "Positive_Seller_Ratings": 86,
                "Ship_On_Time": 0,
                "Chat_Response_Rate": 93,
                "No_of_products_to_be_sold": 113.79,
                "Category": "Watches, Bags, Jewellery",
                "Delivery_Type": "Free Delivery",
                "Flagship_Store": "No",
            }
        }


# Define Output Data Shape ---
class PredictionOut(BaseModel):
    predicted_success_score: float


# Create API Endpoints ---
@app.get("/")
def home():
    return {"message": "API is running. Go to /docs to see the endpoints."}


# Health check endpoint (required by your instructor)
@app.get("/health")
def health():
    return {"status": "ok", "canary": os.getenv("CANARY", "false")}


# Prediction endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(features: ProductFeatures):

    # Convert Pydantic input to a single-row DataFrame
    # We must rename columns to match the *original CSV* for pd.get_dummies
    data_dict = features.dict()
    data_dict_renamed = {
        "Original Price": data_dict["Original_Price"],
        "Discount Price": data_dict["Discount_Price"],
        "Number of Ratings": data_dict["Number_of_Ratings"],
        "Positive Seller Ratings": data_dict["Positive_Seller_Ratings"],
        "Ship On Time": data_dict["Ship_On_Time"],
        "Chat Response Rate": data_dict["Chat_Response_Rate"],
        "No. of products to be sold": data_dict["No_of_products_to_be_sold"],
        "Category": data_dict["Category"],
        "Delivery Type": data_dict["Delivery_Type"],
        "Flagship Store": data_dict["Flagship_Store"],
    }
    input_df = pd.DataFrame([data_dict_renamed])

    # Apply One-Hot Encoding
    # This will only create columns for the categories in *this single request*
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align Columns with the Training Data
    # This is the most important MLOps trick:
    # - It adds all the missing one-hot columns (and fills them with 0)
    # - It drops any new, unseen category columns from this request
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Make the prediction
    # Ensure model and columns were loaded
    if model is None or not model_columns:
        return {"error": "Model or columns not loaded. Check server logs."}

    prediction = np.clip(model.predict(input_df_aligned)[0], 1, 100)
    # --- Add monitoring ---
    observe_prediction()

    # Return the result
    # model.predict() returns a numpy array, so we take the first item [0]
    return {"predicted_success_score": float(prediction)}
