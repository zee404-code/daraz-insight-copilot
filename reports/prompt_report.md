# D1 Prompt Engineering Report
**Daraz Insight Copilot — Sentiment Analysis on Code-Mixed Urdu/English Reviews**

**Student**: Farah
**Model**: `llama-3.1-8b-instant` (Groq)
**Dataset**: 51 real Daraz reviews (`data/eval.jsonl`) — English ground truth
**Temperature**: 0.0
**Date**: December 2025

### Final Results Table

| Strategy              | Accuracy | F1 (weighted) | Improvement |
|-----------------------|----------|---------------|-------------|
| Zero-Shot             | 0.9216   | 0.9093        | Baseline    |
| Few-Shot (k=3)        | **0.9412** | **0.9364**  | **+2.0%**   |
| Few-Shot (k=5)        | 0.9020   | 0.8940        | -2.0%       |
| Chain-of-Thought      | 0.9216   | 0.Driver9205        | +0.0%       |

**Winner**: **Few-Shot (k=3)** — 94.12% accuracy on real-world code-mixed reviews

### MLflow Experiment Dashboard
<img src="assets/MLflow_s1.png" alt="MLflow Overview" width="500">

### Individual Run Screenshots

<img src="assets/zeroshot.png" alt="Zero-Shot Run" width="500">
<img src="assets/fewshotk3.png" alt="Few-Shot k=3 Run" width="500">
<img src="assets/fewshotk5.png" alt="Few-Shot k=5 Run" width="500">
<img src="assets/cot.png" alt="Chain-of-Thought Run" width="500">

### Key Findings
- Few-shot prompting gives the **highest accuracy** on Roman Urdu + English reviews
- **Quality > Quantity**: 3 perfect examples beat 5 noisy ones
- Chain-of-Thought adds no benefit for simple classification
- All scripts are robust: rate limiting, JSON parsing, Windows-compatible paths
- Evaluation uses two metrics: **Accuracy + F1-weighted**

**MLflow Link**: http://localhost:5000/#/experiments/263845935945026779
**All artifacts logged** (eval.jsonl, parameters, metrics)

**Conclusion**
**Few-Shot (k=3)** is selected as the production strategy for the Daraz Insight Copilot.

**D1 Complete — Ready for D2 RAG**
