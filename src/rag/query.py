# src/rag/query.py
import os
import time
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        print("Loading RAG engine...")

        # 1️⃣ Hugging Face caching & token
        hf_cache_dir = os.getenv("HF_HOME", "models/hf_cache")
        os.makedirs(hf_cache_dir, exist_ok=True)
        hf_model_name = os.getenv(
            "EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=hf_model_name, cache_folder=hf_cache_dir
        )
        Settings.llm = Groq(
            model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")
        )

        # 2️⃣ Load existing FAISS index
        persist_dir = "faiss_index"
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(
                f"FAISS index not found at '{persist_dir}'. Run 'ingest.py' first."
            )
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

        # 3️⃣ Convert to query engine
        _engine = index.as_query_engine(similarity_top_k=5)
        print("✅ RAG engine ready!")

    return _engine


def ask_rag(question: str) -> dict:
    start = time.time()
    engine = get_engine()
    response = engine.query(question)
    latency = time.time() - start

    sources = [node.node.get_text()[:200] + "..." for node in response.source_nodes]

    return {
        "answer": str(response),
        "sources": sources,
        "latency_seconds": round(latency, 2),
    }
