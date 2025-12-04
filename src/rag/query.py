# src/rag/query.py

import os

# 1. FORCE OFFLINE MODE
os.environ["HF_HUB_OFFLINE"] = "1"
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import os
import time

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        print("Loading RAG engine...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.llm = Groq(
            model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")
        )

        storage_context = StorageContext.from_defaults(persist_dir="faiss_index")
        index = load_index_from_storage(storage_context)
        _engine = index.as_query_engine(similarity_top_k=5)
        print("RAG engine ready!")
    return _engine


def ask_rag(question: str) -> dict:
    start = time.time()
    response = get_engine().query(question)
    latency = time.time() - start
    sources = [node.node.get_text()[:200] + "..." for node in response.source_nodes]
    return {
        "answer": str(response),
        "sources": sources,
        "latency_seconds": round(latency, 2),
    }
