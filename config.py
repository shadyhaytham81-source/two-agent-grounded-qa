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


# ---- Anthropic ----
ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

# ---- Qdrant (remote cluster) ----
QDRANT_URL = _require("QDRANT_URL")
QDRANT_API_KEY = _require("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rich_dad_poor_dad")

# ---- Embeddings ----
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2; update if you swap embedding models

# ---- Corpus (local PDF — never committed to git, see .gitignore) ----
CORPUS_NAME = os.getenv("CORPUS_NAME", "Rich Dad Poor Dad")
BOOK_PDF_PATH = os.getenv("BOOK_PDF_PATH", "corpus/rich_dad_poor_dad.pdf")

# ---- Retrieval / grounding behavior ----
TOP_K_PASSAGES = int(os.getenv("TOP_K_PASSAGES", "5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))
MAX_REVIEWER_RETRIES = int(os.getenv("MAX_REVIEWER_RETRIES", "1"))
