"""
Ingests a local PDF (the book) into a REMOTE Qdrant collection.

Run once (or whenever you want to re-ingest):
    python -m ingestion.ingest_pdf

Pipeline:
  1. Extract text page-by-page from the local PDF (pdfplumber)
  1b. Strip running headers/footers — lines that repeat on most pages (page
      furniture, watermarks). Left in, they start nearly every chunk with
      identical text, which blurs the embeddings and hurts retrieval.
  2. Split each page into overlapping chunks, keeping the page number as metadata
  3. Embed chunks locally with sentence-transformers (no external API key needed)
  4. Upsert into the remote Qdrant collection with source metadata (title + page)

Note on copyright: this script only ever touches a copy of the PDF that YOU
provide locally (see corpus/README.md). It is never uploaded anywhere except
as vector embeddings + short text chunks in your own private Qdrant cluster.
Do not commit the PDF to git or otherwise redistribute it.
"""
import argparse
import os
import sys
import uuid
from collections import Counter

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

sys.path.append(".")  # allow running as `python -m ingestion.ingest_pdf` from repo root
import config


def build_collection(client: QdrantClient, recreate: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION_NAME in existing:
        if not recreate:
            print(f"Collection '{config.QDRANT_COLLECTION_NAME}' already exists — reusing it.")
            return
        client.delete_collection(config.QDRANT_COLLECTION_NAME)
        print(f"Dropped existing collection '{config.QDRANT_COLLECTION_NAME}'.")
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=config.EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
    )
    print(f"Created collection '{config.QDRANT_COLLECTION_NAME}'.")


def strip_repeated_lines(pages: list[tuple[int, str]], threshold: float = 0.5):
    """
    Drops lines that appear on more than `threshold` of the pages.

    Running headers, footers and watermarks repeat on nearly every page. They
    carry no information, but they would otherwise be prepended to almost
    every chunk and pull all the embeddings toward each other.
    """
    if not pages:
        return pages, []

    counts = Counter()
    for _, text in pages:
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            counts[line] += 1

    cutoff = len(pages) * threshold
    boilerplate = {line for line, n in counts.items() if n > cutoff}

    cleaned = []
    for page_num, text in pages:
        kept = [ln for ln in text.splitlines() if ln.strip() not in boilerplate]
        body = "\n".join(kept).strip()
        if body:
            cleaned.append((page_num, body))
    return cleaned, sorted(boilerplate)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Drop the collection first")
    args = parser.parse_args()

    if not os.path.exists(config.BOOK_PDF_PATH):
        raise FileNotFoundError(
            f"No PDF found at '{config.BOOK_PDF_PATH}'. Place your own legally-obtained "
            f"copy there (see corpus/README.md) or update BOOK_PDF_PATH in .env."
        )

    print(f"Connecting to Qdrant at {config.QDRANT_URL} ...")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    build_collection(client, recreate=args.recreate)

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' (runs locally)...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    print(f"Extracting text from {config.BOOK_PDF_PATH} ...")
    pages = extract_pages(config.BOOK_PDF_PATH)
    if not pages:
        raise RuntimeError(
            "No extractable text found in the PDF. It may be a scanned/image-only PDF — "
            "OCR it first (see the pdf-reading approach for scanned documents)."
        )
    print(f"  {len(pages)} pages with extractable text.")

    pages, boilerplate = strip_repeated_lines(pages)
    if boilerplate:
        print(f"  Stripped {len(boilerplate)} running header/footer line(s):")
        for line in boilerplate[:5]:
            print(f"    - {line[:90]!r}")
    print(f"  {len(pages)} pages remain after cleaning.")

    texts: list[str] = []
    payloads: list[dict] = []
    for page_num, page_text in pages:
        for i, chunk in enumerate(splitter.split_text(page_text)):
            texts.append(chunk)
            payloads.append(
                {
                    "text": chunk,
                    "title": config.CORPUS_NAME,
                    "page": page_num,
                    "chunk_index": i,
                }
            )

    print(f"Embedding {len(texts)} chunks locally...")
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    points = [
        qmodels.PointStruct(id=str(uuid.uuid4()), vector=vector.tolist(), payload=payload)
        for vector, payload in zip(vectors, payloads)
    ]

    print(f"Upserting {len(points)} chunks into Qdrant (batches of 128)...")
    for i in range(0, len(points), 128):
        client.upsert(collection_name=config.QDRANT_COLLECTION_NAME, points=points[i : i + 128])

    print("Done.")
    print(f"  Pages ingested: {len(pages)}")
    print(f"  Chunks stored:  {len(points)}")


if __name__ == "__main__":
    main()
