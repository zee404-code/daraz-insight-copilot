# tests/test_frontend.py
import pytest
from unittest.mock import patch, MagicMock
import app.frontend as frontend


class FakeSessionState:
    def __init__(self):
        self.messages = []


@pytest.mark.parametrize("page", ["Prediction", "Chatbot"])
def test_frontend_renders_without_crashing(page, monkeypatch):
    # --- Mock Streamlit ---
    monkeypatch.setattr("streamlit.set_page_config", lambda **kw: None)
    monkeypatch.setattr("streamlit.markdown", lambda *a, **kw: None)
    monkeypatch.setattr("streamlit.sidebar.radio", lambda *a, **kw: page)
    monkeypatch.setattr("streamlit.title", lambda x: None)
    monkeypatch.setattr("streamlit.success", lambda x: None)
    monkeypatch.setattr("streamlit.error", lambda x: None)
    monkeypatch.setattr("streamlit.text_input", lambda *a, **kw: "test")
    monkeypatch.setattr("streamlit.number_input", lambda *a, **kw: 1000.0)

    # Mock st.form context manager
    mock_form = MagicMock()
    mock_form.__enter__.return_value = mock_form
    monkeypatch.setattr("streamlit.form", lambda *a, **kw: mock_form)

    # Simulate form submission only on Prediction page
    monkeypatch.setattr(
        "streamlit.form_submit_button", lambda *a, **kw: page == "Prediction"
    )

    # Proper session_state with attribute access
    monkeypatch.setattr("streamlit.session_state", FakeSessionState())

    # THIS IS THE FIX: patch the imported function in your module, not the package
    monkeypatch.setattr("app.frontend.message", lambda *a, **kw: None)

    # --- Mock requests ---
    with patch("app.frontend.requests.post") as mock_post:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "predicted_success_score": 95.0,
            "answer": "Mocked answer",
        }
        mock_post.return_value = mock_resp

        # This will now run cleanly
        frontend.main()

    # Assert API was called only on Prediction page
    if page == "Prediction":
        mock_post.assert_called_once()
    else:
        mock_post.assert_not_called()
