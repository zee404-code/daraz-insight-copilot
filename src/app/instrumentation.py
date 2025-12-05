from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# --- Custom Metrics ---

# D1: Model Predictions
PREDICTIONS_COUNTER = Counter(
    "api_predictions_total",
    "Total number of predictions made",
    labelnames=("model_version",),
)

# D3: Guardrails
GUARDRAIL_COUNTER = Counter(
    "guardrail_events_total", "Total number of guardrail triggers", ["type", "action"]
)

# D4: LLM Metrics (New)
RAG_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency of RAG /ask endpoint",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

TOKEN_COUNTER = Counter(
    "llm_token_usage_total",
    "Total LLM tokens used",
    ["type"],  # Labels: input, output
)

COST_COUNTER = Counter("llm_cost_total", "Total estimated cost in USD", ["model"])


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
    PREDICTIONS_COUNTER.labels(model_version="v1.0").inc()


def log_guardrail_event(event_type: str, action: str):
    GUARDRAIL_COUNTER.labels(type=event_type, action=action).inc()


def log_llm_metrics(latency: float, input_tokens: int, output_tokens: int):
    """
    Logs RAG metrics: Latency, Tokens, and Cost.
    """
    # 1. Observe Latency
    RAG_LATENCY.observe(latency)

    # 2. Count Tokens
    TOKEN_COUNTER.labels(type="input").inc(input_tokens)
    TOKEN_COUNTER.labels(type="output").inc(output_tokens)

    # 3. Estimate Cost (Hypothetical pricing: $0.50/1M input, $1.50/1M output)
    input_cost = (input_tokens / 1_000_000) * 0.50
    output_cost = (output_tokens / 1_000_000) * 1.50
    total_cost = input_cost + output_cost

    COST_COUNTER.labels(model="llama3-8b").inc(total_cost)
