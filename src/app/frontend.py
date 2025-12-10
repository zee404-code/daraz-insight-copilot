# frontend/app.py
import streamlit as st
import requests
from streamlit_chat import message  # pip install streamlit-chat


def main():
    # ---------------------------
    # Page Config
    # ---------------------------
    st.set_page_config(page_title="Daraz Insight Copilot", layout="wide")

    # ---------------------------
    # Custom CSS for Purple + Orange Theme
    # ---------------------------
    st.markdown(
        """
    <style>
    body {
        background: linear-gradient(to right, #5D3FD3, #FF7F50);
        color: #fff;
    }
    .stButton>button {
        background: linear-gradient(to right, #FF7F50, #5D3FD3);
        color: #fff;
        border-radius: 10px;
        height: 40px;
        width: 100%;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        padding: 10px;
    }
    h1, h2, h3 {
        text-align: center;
        color: #fff;
    }
    .chat-bubble {
        background: linear-gradient(to right, #5D3FD3, #FF7F50);
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 8px;
        max-width: 70%;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------------------------
    # Sidebar Navigation
    # ---------------------------
    page = st.sidebar.radio("Navigation", ["Prediction", "Chatbot"])

    # ---------------------------
    # Prediction Page
    # ---------------------------
    if page == "Prediction":
        st.title("Product Success Predictor")

        with st.form("prediction_form"):
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
                try:
                    response = requests.post(
                        "http://localhost:8000/predict", json=payload
                    )
                    if response.status_code == 200:
                        st.success(
                            f"Predicted Success Score: {response.json()['predicted_success_score']:.2f}"
                        )
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"API Error: {e}")

    # ---------------------------
    # Chatbot Page
    # ---------------------------
    elif page == "Chatbot":
        st.title("Daraz RAG Chatbot")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question about product reviews:")
            submitted = st.form_submit_button("Send")
            if submitted and user_input:
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )
                try:
                    response = requests.post(
                        "http://localhost:8000/ask", json={"question": user_input}
                    )
                    if response.status_code == 200:
                        answer = response.json()["answer"]
                        st.session_state.messages.append(
                            {"role": "bot", "content": answer}
                        )
                    else:
                        st.session_state.messages.append(
                            {"role": "bot", "content": "Error fetching answer."}
                        )
                except Exception as e:
                    st.session_state.messages.append(
                        {"role": "bot", "content": f"API Error: {e}"}
                    )

        # Display messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=msg["content"] + "_user")
            else:
                message(msg["content"], key=msg["content"] + "_bot")


if __name__ == "__main__":
    main()
