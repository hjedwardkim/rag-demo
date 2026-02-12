"""Configuration module. Loads environment variables and defines constants."""

import os

from dotenv import load_dotenv

load_dotenv()

# Embedding
EMBEDDING_API_BASE = os.environ["EMBEDDING_API_BASE"]
EMBEDDING_API_KEY = os.environ["EMBEDDING_API_KEY"]
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"
)

# LLM
LLM_API_BASE = os.environ["LLM_API_BASE"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "cm-llm")

# Reranker
RERANKER_API_BASE = os.environ["RERANKER_API_BASE"]
RERANKER_API_KEY = os.environ["RERANKER_API_KEY"]
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ChromaDB
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "kb_articles"

# Retrieval defaults
DEFAULT_TOP_K = 5
RRF_K = 60  # RRF constant
