# experiments/prompts/cot.py
import json
import os
import time  # <--- Essential for rate limiting
from groq import Groq
import mlflow
from sklearn.metrics import accuracy_score, f1_score

# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("prompt_engineering_d1")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- PROMPT ---
# We use double braces {{ }} for the JSON examples so Python doesn't crash.
PROMPT = """You are an expert at understanding mixed Urdu/English Daraz reviews.
Think step-by-step and classify the sentiment as positive, negative, or neutral.

Step 1: Read the review carefully
Step 2: Look for positive words (good, acha, zabardast, satisfied, recommended, etc.)
Step 3: Look for negative words (bakwas, fake, waste, damaged, return, etc.)
Step 4: Check if it's mixed/neutral (theek hai lekin, good but, etc.)

Review: {review}

Now think step-by-step and finally return ONLY this JSON:

{{"sentiment": "positive", "reason": "short explanation in English"}}
{{"sentiment": "negative", "reason": "short explanation in English"}}
{{"sentiment": "neutral", "reason": "short explanation in English"}}

Your final answer (JSON only):"""

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval.jsonl")

# Load Data
with open(DATA_PATH, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

results = []
print(f"Starting Chain-of-Thought classification on {len(data)} items...")

# --- CLASSIFICATION LOOP ---
for i, item in enumerate(data):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated Model
            messages=[
                {"role": "user", "content": PROMPT.format(review=item["review"])}
            ],
            temperature=0.0,
            max_tokens=150,  # Increased slightly for CoT reasoning
        )
        output = response.choices[0].message.content.strip()

        # --- SMART PARSING LOGIC ---
        # 1. Reverse Find (rfind): Look for the LAST opening brace '{'
        #    This ignores the "thinking" steps at the start and grabs the final JSON at the end.
        start = output.rfind("{")
        end = output.rfind("}") + 1

        if start != -1 and end > 0:
            json_str = output[start:end]
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # If the smart parse fails, log it as an error but keep going
                print(f"Row {i} JSON Decode Error. Content: {json_str}")
                parsed = {"sentiment": "neutral", "reason": "parsing_error"}
        else:
            print(f"Row {i} Warning: No JSON braces found. Output: {output[:50]}...")
            parsed = {"sentiment": "neutral", "reason": "parsing_error"}

    except Exception as e:
        print(f"Row {i} API Error: {e}")
        parsed = {"sentiment": "neutral", "reason": "api_error"}

    results.append(parsed)

    # Sleep 1 second to respect Groq rate limits
    time.sleep(1)

# --- METRICS & LOGGING ---
pred = [r.get("sentiment", "neutral") for r in results]
true = [item["ground_truth"] for item in data]
acc = accuracy_score(true, pred)
f1 = f1_score(true, pred, average="weighted")

with mlflow.start_run(run_name="chain_of_thought"):
    mlflow.log_params({"strategy": "chain_of_thought", "model": "llama-3.1-8b-instant"})
    mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
    mlflow.log_artifact(DATA_PATH)

# Save Results
os.makedirs("../results", exist_ok=True)
with open("../results/cot.json", "w", encoding="utf-8") as f:
    json.dump(
        {"accuracy": round(acc, 4), "f1": round(f1, 4), "results": results}, f, indent=2
    )

print(f"Chain-of-Thought → Accuracy: {acc:.4f} | F1: {f1:.4f}")
