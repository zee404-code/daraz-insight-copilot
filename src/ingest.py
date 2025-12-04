# src/ingest.py
import os
import pandas as pd
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter

# 1. Setup
print("Starting Ingestion...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

# 2. Load Data
# CHECK THIS PATH: Make sure your CSV is actually inside data/raw/
file_path = os.path.join("data", "raw", "daraz-code-mixed-product-reviews.csv")

if not os.path.exists(file_path):
    print(f" Error: File not found at {file_path}")
    print("   Please move your CSV to 'data/raw/' or update the path.")
    exit()

df = pd.read_csv(file_path)
documents = []

print(f"   Processing {len(df)} rows...")
for _, row in df.iterrows():
    review_text = str(row["Reviews"])  # Ensure string
    sentiment_value = str(row.get("Sentiments", "unknown"))

    text = f"Review: {review_text}\nSentiment: {sentiment_value}"

    # Metadata helps the LLM filter if needed
    metadata = {
        "sentiment": sentiment_value,
    }
    documents.append(Document(text=text, metadata=metadata))

# 3. Chunking
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

# 4. Build & Save (Default LlamaIndex Storage - Compatible with your query.py)
print("Building Index...")
index = VectorStoreIndex(nodes, show_progress=True)

print("Saving to 'faiss_index'...")
index.storage_context.persist(persist_dir="faiss_index")

print("SUCCESS: Index built! Now run 'make run' or 'python main.py'")
