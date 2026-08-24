"""
Thin wrapper around the Qdrant client + local embedding model, shared by the
Researcher agent. Kept separate from agents/ so both the graph and any
future tooling can reuse it without circular imports.
"""
from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def retrieve_passages(query: str, top_k: int = None) -> list[dict]:
    """
    Embeds `query` and searches the remote Qdrant collection.
    Returns a list of dicts: {text, title, page, chunk_index, score},
    filtered to results at or above config.MIN_RELEVANCE_SCORE.
    """
    top_k = top_k or config.TOP_K_PASSAGES
    client = get_qdrant_client()
    embedder = get_embedder()

    query_vector = embedder.encode(query).tolist()

    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=config.MIN_RELEVANCE_SCORE,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "title": hit.payload.get("title", config.CORPUS_NAME),
            "page": hit.payload.get("page"),
            "chunk_index": hit.payload.get("chunk_index"),
            "score": round(hit.score, 3),
        }
        for hit in response.points
    ]
