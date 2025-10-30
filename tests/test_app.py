from fastapi.testclient import TestClient
from app.main import app

# Create a 'client' that can make fake requests to our app
client = TestClient(app)


def test_health_check():
    """
    Tests the /health endpoint.
    It should return a 200 OK status and the correct JSON.
    """
    print("Testing /health endpoint...")
    response = client.get("/health")

    # Check 1: Was the request successful (Status 200)?
    assert response.status_code == 200

    # We check for CANARY=false
    assert response.json() == {"status": "ok", "canary": "false"}


def test_prediction():
    """
    Tests the /predict endpoint with a valid sample.
    It should return a 200 OK status and a valid success score.
    """
    print("Testing /predict endpoint...")

    # valid sample payload
    test_payload = {
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

    # Make the POST request
    response = client.post("/predict", json=test_payload)

    # Was the request successful (Status 200)?
    assert response.status_code == 200

    # Did we get a JSON response?
    data = response.json()
    assert "predicted_success_score" in data

    # Is the score a number (float)?
    score = data["predicted_success_score"]
    assert isinstance(score, float)

    # Is the score within 0-100 clamped range?
    assert 0.0 <= score <= 100.0


def test_bad_prediction_payload():
    """
    Tests the /predict endpoint with a missing field.
    FastAPI should automatically catch this and return a 422 error.
    """
    print("Testing /predict with bad payload...")

    # This payload is missing "Category"
    bad_payload = {
        "Original_Price": 1650,
        "Discount_Price": 725,
        "Number_of_Ratings": 31,
        "Positive_Seller_Ratings": 86,
        "Ship_On_Time": 0,
        "Chat_Response_Rate": 93,
        "No_of_products_to_be_sold": 113.79,
        # "Category": "Watches, Bags, Jewellery", <-- MISSING
        "Delivery_Type": "Free Delivery",
        "Flagship_Store": "No",
    }

    response = client.post("/predict", json=bad_payload)

    # Check: Did we get a 422 "Unprocessable Entity" error?
    assert response.status_code == 422
