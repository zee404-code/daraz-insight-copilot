from unittest.mock import patch
import app.frontend as frontend


def test_frontend_renders(monkeypatch):
    # Mock the API call inside frontend
    with patch("app.frontend.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"predicted_success_score": 42.0}
        frontend.run()  # or the main function that starts Streamlit
        # Assert code paths, or just that it runs without exceptions
