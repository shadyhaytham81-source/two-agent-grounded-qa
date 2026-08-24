"""
Central configuration. Everything sensitive or environment-specific is
loaded from environment variables (via .env locally) — nothing is hardcoded.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # no-op in production if you set real env vars instead of a .env file


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


# ---- LLM (both agents) ----
# Two free-tier providers are supported; pick one with LLM_PROVIDER and supply
# only that provider's key. Neither requires a credit card to sign up.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}
if LLM_PROVIDER not in _DEFAULT_MODELS:
    raise RuntimeError(
        f"LLM_PROVIDER must be one of {sorted(_DEFAULT_MODELS)}, got {LLM_PROVIDER!r}."
    )

LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS[LLM_PROVIDER]
GROQ_API_KEY = _require("GROQ_API_KEY") if LLM_PROVIDER == "groq" else os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = _require("GEMINI_API_KEY") if LLM_PROVIDER == "gemini" else os.getenv("GEMINI_API_KEY")

# ---- Qdrant (remote cluster) ----
QDRANT_URL = _require("QDRANT_URL")
QDRANT_API_KEY = _require("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "langchain_qdrant_docs")

# ---- Embeddings (run locally — no API key, no cost) ----
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2; update if you swap embedding models

# ---- Corpus: the two documentation sites that get ingested ----
CORPUS_NAME = "the LangChain and Qdrant documentation"

# Each entry: sitemap to enumerate, the path prefix that marks a docs page, and
# the label stored on every chunk so answers can cite which product a passage
# came from. Both sites serve a clean Markdown version of every page, which is
# what the ingester actually downloads (see ingestion/ingest_docs.py).
DOC_SOURCES = [
    {
        "source": "langchain",
        "sitemap": "https://docs.langchain.com/sitemap.xml",
        "include_prefix": "https://docs.langchain.com/oss/python/",
    },
    {
        "source": "qdrant",
        "sitemap": "https://qdrant.tech/sitemap.xml",
        "include_prefix": "https://qdrant.tech/documentation/",
    },
]

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---- Retrieval / grounding behavior ----
TOP_K_PASSAGES = int(os.getenv("TOP_K_PASSAGES", "5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))
MAX_REVIEWER_RETRIES = int(os.getenv("MAX_REVIEWER_RETRIES", "1"))
