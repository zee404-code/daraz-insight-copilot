from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# --- Custom Metrics ---
PREDICTIONS_COUNTER = Counter(
    "api_predictions_total",
    "Total number of predictions made",
    labelnames=("model_version",),
)

# NEW: Guardrail Counter for Deliverable 3
GUARDRAIL_COUNTER = Counter(
    "guardrail_events_total",
    "Total number of guardrail triggers",
    ["type", "action"],  # Labels: type (input/output), action (blocked/flagged)
)


# --- Standard Instrumentation ---
def setup_instrumentation(app):
    """
    Sets up the standard Prometheus instrumentation.
    """
    print("Setting up Prometheus instrumentation...")
    instrumentator = Instrumentator(
        excluded_handlers=["/metrics"],
    )
    instrumentator.instrument(app)
    instrumentator.expose(app)
    print("Instrumentation setup complete.")


# --- Helper functions ---
def observe_prediction():
    """Increment prediction counter."""
    PREDICTIONS_COUNTER.labels(model_version="v1.0").inc()


def log_guardrail_event(event_type: str, action: str):
    """
    Logs a guardrail event to Prometheus.
    Usage: log_guardrail_event("input_validation", "blocked")
    """
    GUARDRAIL_COUNTER.labels(type=event_type, action=action).inc()
