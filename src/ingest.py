# src/ingest.py
import os
import pandas as pd
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter

# 1. Setup Hugging Face cache & Embedding
hf_cache_dir = os.getenv("HF_HOME", "models/hf_cache")
os.makedirs(hf_cache_dir, exist_ok=True)

embed_model_name = os.getenv(
    "EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"
)
print(f"Using embedding model: {embed_model_name}")

Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

# 2. Load CSV
file_path = os.path.join("data", "raw", "daraz-code-mixed-product-reviews.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"CSV not found at {file_path}. Please move it there or update the path."
    )

df = pd.read_csv(file_path)
documents = []

print(f"Processing {len(df)} rows...")
for _, row in df.iterrows():
    review_text = str(row["Reviews"])
    sentiment_value = str(row.get("Sentiments", "unknown"))

    text = f"Review: {review_text}\nSentiment: {sentiment_value}"
    metadata = {"sentiment": sentiment_value}
    documents.append(Document(text=text, metadata=metadata))

# 3. Chunking
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

# 4. Build & Save Index
index = VectorStoreIndex(nodes, show_progress=True)
faiss_dir = "faiss_index"
os.makedirs(faiss_dir, exist_ok=True)
index.storage_context.persist(persist_dir=faiss_dir)

print(f"SUCCESS: Index built and saved to '{faiss_dir}'")
