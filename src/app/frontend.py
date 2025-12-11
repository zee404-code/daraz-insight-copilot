# src/app/frontend.py

import streamlit as st
import requests
from streamlit_chat import message
from typing import Dict, Any

# ← Move imports to module level (critical for patching!)
# requests is now available as app.frontend.requests

import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


def make_prediction(payload: Dict[str, Any]) -> str:
    """Helper to call prediction endpoint"""
    try:
        response = requests.post(f"{FASTAPI_URL}/predict", json=payload)

        if response.status_code == 200:
            score = response.json()["predicted_success_score"]
            return f"Predicted Success Score: {score:.2f}"
        else:
            return f"Error: {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"API Error: {e}"


def ask_question(question: str) -> str:
    """Helper to call RAG endpoint"""
    try:
        response = requests.post(
            response=requests.post(f"{FASTAPI_URL}/ask", json={"question": question})
        )

        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return "Error fetching answer."
    except Exception as e:
        return f"API Error: {e}"


def main():
    st.set_page_config(page_title="Daraz Insight Copilot", layout="wide")

    st.markdown(
        """
    <style>
    /* your CSS here */
    </style>
    """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio("Navigation", ["Prediction", "Chatbot"])

    if page == "Prediction":
        st.title("Product Success Predictor")

        with st.form("prediction_form"):
            # ... all your inputs ...
            Original_Price = st.number_input("Original Price", value=1650)
            Discount_Price = st.number_input("Discount Price", value=725)
            Number_of_Ratings = st.number_input("Number of Ratings", value=31)
            Positive_Seller_Ratings = st.number_input(
                "Positive Seller Ratings", value=86
            )
            Ship_On_Time = st.number_input("Ship On Time", value=0)
            Chat_Response_Rate = st.number_input("Chat Response Rate", value=93)
            No_of_products_to_be_sold = st.number_input(
                "No. of products to be sold", value=113.79
            )
            Category = st.text_input("Category", value="Watches, Bags, Jewellery")
            Delivery_Type = st.text_input("Delivery Type", value="Free Delivery")
            Flagship_Store = st.text_input("Flagship Store", value="No")
            # ... etc ...

            submitted = st.form_submit_button("Predict Success Score")

            if submitted:
                payload = {
                    "Original_Price": Original_Price,
                    "Discount_Price": Discount_Price,
                    "Number_of_Ratings": Number_of_Ratings,
                    "Positive_Seller_Ratings": Positive_Seller_Ratings,
                    "Ship_On_Time": Ship_On_Time,
                    "Chat_Response_Rate": Chat_Response_Rate,
                    "No_of_products_to_be_sold": No_of_products_to_be_sold,
                    "Category": Category,
                    "Delivery_Type": Delivery_Type,
                    "Flagship_Store": Flagship_Store,
                }
                result = make_prediction(payload)
                if "Success Score" in result:
                    st.success(result)
                else:
                    st.error(result)

    elif page == "Chatbot":
        st.title("Daraz RAG Chatbot")
        if not hasattr(st.session_state, "messages"):
            st.session_state.messages = []

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question about product reviews:")
            submitted = st.form_submit_button("Send")

            if submitted and user_input:
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )
                answer = ask_question(user_input)
                st.session_state.messages.append({"role": "bot", "content": answer})

        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}")
            else:
                message(msg["content"], is_user=False, key=f"bot_{i}")


if __name__ == "__main__":
    main()
