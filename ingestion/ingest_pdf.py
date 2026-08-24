"""
Ingests a local PDF (the book) into a REMOTE Qdrant collection.

Run once (or whenever you want to re-ingest):
    python -m ingestion.ingest_pdf

Pipeline:
  1. Extract text page-by-page from the local PDF (pdfplumber)
  2. Split each page into overlapping chunks, keeping the page number as metadata
  3. Embed chunks locally with sentence-transformers (no external API key needed)
  4. Upsert into the remote Qdrant collection with source metadata (title + page)

Note on copyright: this script only ever touches a copy of the PDF that YOU
provide locally (see corpus/README.md). It is never uploaded anywhere except
as vector embeddings + short text chunks in your own private Qdrant cluster.
Do not commit the PDF to git or otherwise redistribute it.
"""
import os
import sys
import uuid

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

sys.path.append(".")  # allow running as `python -m ingestion.ingest_pdf` from repo root
import config


def build_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION_NAME in existing:
        print(f"Collection '{config.QDRANT_COLLECTION_NAME}' already exists — reusing it.")
        return
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=config.EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
    )
    print(f"Created collection '{config.QDRANT_COLLECTION_NAME}'.")


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Returns [(page_number, text), ...] for pages with extractable text."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((i, text))
    return pages


def main():
    if not os.path.exists(config.BOOK_PDF_PATH):
        raise FileNotFoundError(
            f"No PDF found at '{config.BOOK_PDF_PATH}'. Place your own legally-obtained "
            f"copy there (see corpus/README.md) or update BOOK_PDF_PATH in .env."
        )

    print(f"Connecting to Qdrant at {config.QDRANT_URL} ...")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    build_collection(client)

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' (runs locally)...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    print(f"Extracting text from {config.BOOK_PDF_PATH} ...")
    pages = extract_pages(config.BOOK_PDF_PATH)
    if not pages:
        raise RuntimeError(
            "No extractable text found in the PDF. It may be a scanned/image-only PDF — "
            "OCR it first (see the pdf-reading approach for scanned documents)."
        )
    print(f"  {len(pages)} pages with extractable text.")

    points: list[qmodels.PointStruct] = []
    for page_num, page_text in pages:
        chunks = splitter.split_text(page_text)
        for i, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "title": config.CORPUS_NAME,
                        "page": page_num,
                        "chunk_index": i,
                    },
                )
            )

    print(f"Upserting {len(points)} chunks into Qdrant (in batches of 64)...")
    for i in range(0, len(points), 64):
        client.upsert(collection_name=config.QDRANT_COLLECTION_NAME, points=points[i : i + 64])

    print("Done.")
    print(f"  Pages ingested: {len(pages)}")
    print(f"  Chunks stored:  {len(points)}")


if __name__ == "__main__":
    main()
