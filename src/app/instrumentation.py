from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter
import time

# --- Custom Metrics ---
# We will create a custom metric to count the number of predictions made
PREDICTIONS_COUNTER = Counter(
    "api_predictions_total",
    "Total number of predictions made",
    labelnames=("model_version",) # You can add labels like model version
)

# --- Standard Instrumentation ---
def setup_instrumentation(app):
    """
    Sets up the standard Prometheus instrumentation.
    This creates default metrics like /metrics, request latency, etc.
    """
    print("Setting up Prometheus instrumentation...")
    instrumentator = Instrumentator(
        excluded_handlers=["/metrics"], # Don't monitor the metrics endpoint itself
    )

    # Add the standard instrumentation to the app
    instrumentator.instrument(app)
    instrumentator.expose(app)

    print("Instrumentation setup complete.")


# --- Helper function to be called from main.py ---
def observe_prediction():
    """
    Call this function from your /predict endpoint
    to increment the prediction counter.
    """
    # We'll just hardcode the model version for now
    PREDICTIONS_COUNTER.labels(model_version="v1.0").inc()