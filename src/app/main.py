# src/app/main.py
import joblib
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import numpy as np
import os
from typing import List
from dotenv import load_dotenv

# NEW Imports for Deliverable 3
from .instrumentation import (
    setup_instrumentation,
    observe_prediction,
    log_guardrail_event,
)
from .guardrails import CustomGuardrails

# Force reload from the current directory
load_dotenv()

key = os.getenv("GROQ_API_KEY")
print(f"DEBUG: API Key Loaded? {key is not None}")

# D2 RAG Import (safe)
try:
    from src.rag.query import ask_rag

    RAG_READY = True
    print("RAG system loaded successfully!")
except ImportError as e:
    print(f"RAG not ready: {e} — Run 'make rag' first")
    RAG_READY = False

# Initialize API and Load Artifacts ---
app = FastAPI(title="Daraz Product Success Predictor")

# Setup instrumentation
setup_instrumentation(app)

# Initialize Guardrails Engine (New)
guardrails = CustomGuardrails()

# Load the trained model.
try:
    model = joblib.load("models/model.joblib")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.joblib not found.")
    model = None

# Load the list of model columns (features)
try:
    with open("models/model_columns.json", "r") as f:
        model_columns = json.load(f)
    print("Model columns loaded successfully.")
except FileNotFoundError:
    print("Error: model_columns.json not found.")
    model_columns = []


# Define Input Data Shape (Pydantic BaseModel) ---
class ProductFeatures(BaseModel):
    # Numeric features
    Original_Price: float
    Discount_Price: float
    Number_of_Ratings: int
    Positive_Seller_Ratings: float
    Ship_On_Time: float
    Chat_Response_Rate: float
    No_of_products_to_be_sold: float

    # Categorical features
    Category: str
    Delivery_Type: str
    Flagship_Store: str

    # FIX: ConfigDict is now correctly indented inside the class
    model_config = ConfigDict(
        json_schema_extra={
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
    )


# Define Output Data Shape ---
class PredictionOut(BaseModel):
    predicted_success_score: float


# D2 RAG Schema
class AskQuery(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    latency_seconds: float


# Create API Endpoints ---
@app.get("/")
def home():
    return {
        "message": "Daraz Insight Copilot — Milestone 3 Complete",
        "endpoints": {
            "D1": "POST /predict → Product Success Score",
            "D2": "POST /ask → RAG Chatbot with Guardrails",
            "Health": "GET /health",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "canary": os.getenv("CANARY", "false"),
        "d1_model": model is not None,
        "d2_rag": RAG_READY,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(features: ProductFeatures):
    data_dict = features.model_dump()
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
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    if model is None or not model_columns:
        return {"error": "Model or columns not loaded."}

    prediction = np.clip(model.predict(input_df_aligned)[0], 1, 100)
    observe_prediction()

    return {"predicted_success_score": float(prediction)}


# D2 RAG Chatbot Endpoint (Updated with Guardrails)
@app.post("/ask", response_model=RAGResponse)
def ask(query: AskQuery):
    question = query.question.strip()

    # --- GUARDRAIL 1: INPUT VALIDATION ---
    # We check PII and Injection before touching the RAG system
    is_safe, reason = guardrails.check_input(question)
    if not is_safe:
        log_guardrail_event("input_validation", "blocked")
        print(f"GUARDRAIL ALERT: {reason}")
        raise HTTPException(status_code=400, detail=f"Request blocked: {reason}")

    if not RAG_READY:
        raise HTTPException(status_code=503, detail="RAG not ready — run: make rag")

    try:
        # Get answer from RAG
        result = ask_rag(question)

        # --- GUARDRAIL 2: OUTPUT MODERATION ---
        # We check the 'answer' field of the result
        is_safe_out, reason_out = guardrails.check_output(result["answer"])
        if not is_safe_out:
            log_guardrail_event("output_moderation", "blocked")
            print(f"GUARDRAIL ALERT: {reason_out}")
            # We override the answer but keep the sources so the user knows we tried
            result["answer"] = "I cannot answer this due to safety guidelines."

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")
