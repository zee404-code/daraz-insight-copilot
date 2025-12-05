# Security Policy

## Prompt Injection Defenses

This application processes user input through Large Language Models (LLMs). To mitigate **Prompt Injection** attacks (where malicious input attempts to override system instructions), we have implemented multiple layers of defense:

1. **System Prompt Encapsulation**
   - All user input is strictly encapsulated using clear delimiters (e.g., `"""User Query"""`) before being passed to the LLM.
   - The system prompt explicitly instructs the model to respond **only** based on the provided retrieved context and Daraz product information.

2. **Context Grounding (RAG-based)**
   - The model is forbidden from hallucinating or using external knowledge.
   - Responses must cite specific reviews or product data retrieved from the `faiss_index`.
   - If the retrieved context is irrelevant or insufficient, the model falls back to a safe response such as “I cannot answer that.”

3. **Input Sanitization**
   - All incoming API requests are validated using Pydantic models to enforce correct data types, formats, and length limits, preventing injection via malformed payloads.

## Data Privacy & Handling

We prioritize user privacy and minimize data retention:

1. **No Permanent Storage of Queries**
   - Queries sent to the `/ask` endpoint are processed entirely in memory.
   - User queries are **never** persisted to databases, S3 buckets, or permanent logs. Server logs are ephemeral and automatically rotated.

2. **PII Redaction**
   - Current dataset (`reviews.csv`) contains only public product reviews; any potential PII in the source data is considered public domain.
   - **Planned**: Automated detection and redaction of PII in incoming user queries in future releases.

3. **API Key Security**
   - LLM provider API keys (Groq, Gemini, etc.) are injected exclusively as environment variables at runtime.
   - Keys are **never** hardcoded or committed to the repository.

## Reporting Security Vulnerabilities

If you discover a security vulnerability, a bypass of the implemented guardrails, or any other security issue:

**Please do not open a public GitHub issue.**

Instead, report it privately by emailing the repository maintainer directly. Responsible disclosures will be acknowledged and addressed promptly.

Thank you for helping keep this project secure!
