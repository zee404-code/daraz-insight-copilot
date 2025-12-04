.PHONY: all lint test dev docker run-docker

all: lint test

lint: @echo "Linting..." ruff check src tests black --check src tests

test: @echo "Testing..." pytest tests/

dev: @echo "Starting dev server..." uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload

docker: @echo "Building Docker image..." docker build -t daraz-predictor:latest .

run-docker: @echo "Running Docker container..." docker run -p 8000:8000 daraz-predictor:latest

rag:
	@echo "Running RAG ingestion pipeline (FAISS + LlamaIndex)..."
	python src/ingest.py
	@echo "RAG pipeline complete! FAISS index saved to ./faiss_index"
	@echo "You can now use the /ask endpoint"
