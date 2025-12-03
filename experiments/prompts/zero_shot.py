# experiments/prompts/zero_shot.py   ← REPLACE YOUR FILE WITH THIS EXACT CODE
import json
import os
from groq import Groq
import mlflow
from sklearn.metrics import accuracy_score, f1_score
import time

# MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("prompt_engineering_d1")

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# PROMPT — NO { } THAT CAN BREAK .format()
PROMPT = """Classify this Daraz review as positive, negative, or neutral.
Return ONLY this exact JSON format, nothing else:

{{"sentiment": "positive", "reason": "short reason"}}
{{"sentiment": "negative", "reason": "short reason"}}
{{"sentiment": "neutral", "reason": "short reason"}}

Review: {review}

Your answer (JSON only):"""

# Safe path
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval.jsonl")

# Load data
with open(DATA_PATH, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

results = []
print(f"Starting classification on {len(data)} items...")

for i, item in enumerate(data):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            # The 'PROMPT' variable here automatically uses whatever you defined
            # at the top of the specific file (Zero-shot or Few-shot)
            messages=[
                {"role": "user", "content": PROMPT.format(review=item["review"])}
            ],
            temperature=0.0,
            max_tokens=100,
        )
        output = response.choices[0].message.content.strip()

        # Robust JSON parsing
        start = output.find("{")
        end = output.rfind("}") + 1

        if start != -1 and end > 0:
            json_str = output[start:end]
            parsed = json.loads(json_str)
        else:
            print(f"Row {i} Parse Warning: No JSON found. Output: {output[:50]}...")
            parsed = {"sentiment": "neutral", "reason": "parsing_error"}

    except Exception as e:
        print(f"Row {i} Error: {e}")
        parsed = {"sentiment": "neutral", "reason": "api_error"}

    results.append(parsed)

    # Sleep to prevent hitting Rate Limits (429 Errors)
    time.sleep(1)


# Calculate metrics
pred = [r.get("sentiment", "neutral") for r in results]
true = [item["ground_truth"] for item in data]
acc = accuracy_score(true, pred)
f1 = f1_score(true, pred, average="weighted")

# Log to MLflow
with mlflow.start_run(run_name="zero_shot"):
    mlflow.log_params({"strategy": "zero_shot", "model": "llama3-8b-8192"})
    mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
    mlflow.log_artifact(DATA_PATH)

# Save results
os.makedirs("../results", exist_ok=True)
with open("../results/zero_shot.json", "w", encoding="utf-8") as f:
    json.dump(
        {"accuracy": round(acc, 4), "f1": round(f1, 4), "results": results}, f, indent=2
    )

print(f"ZERO-SHOT DONE → Accuracy: {acc:.4f} | F1: {f1:.4f}")
