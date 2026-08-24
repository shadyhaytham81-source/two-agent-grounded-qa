"""
Ingests the LangChain and Qdrant documentation into a REMOTE Qdrant collection.

Run once (or whenever you want to refresh the corpus):
    python -m ingestion.ingest_docs

Pipeline:
  1. Read each site's sitemap.xml and keep the URLs under the documentation
     path prefix configured in config.DOC_SOURCES
  2. Download the Markdown version of every page. Both sites publish one:
     docs.langchain.com serves "<url>.md" and qdrant.tech serves
     "<url>index.md". Markdown avoids HTML-stripping heuristics entirely and
     keeps headings/code blocks intact, which makes for much cleaner chunks.
  3. Split each page into overlapping chunks, keeping the page URL and title
     as metadata so the Researcher can cite real sources
  4. Embed chunks locally with sentence-transformers (no API key, no cost)
  5. Upsert into the remote Qdrant collection

Flags:
    --limit N          only ingest the first N pages per source (quick test)
    --recreate         drop and rebuild the collection instead of reusing it
"""
import argparse
import re
import sys
import time
import uuid
import xml.etree.ElementTree as ET

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

sys.path.append(".")  # allow running as `python -m ingestion.ingest_docs` from repo root
import config

HEADERS = {"User-Agent": "grounded-qa-assistant/1.0 (docs ingestion)"}
TIMEOUT = 30

# Both sites prepend a couple of "> ..." advisory lines (links to their llms.txt
# index) to every Markdown page. They are identical on every page, so they add
# nothing but noise to the embeddings.
_LEADING_QUOTE_LINES = re.compile(r"\A(?:>[^\n]*\n)+")


def sitemap_urls(sitemap_url: str, include_prefix: str) -> list[str]:
    response = requests.get(sitemap_url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    # sitemap.xml namespaces the <loc> elements, so match on the local name
    locs = [el.text.strip() for el in root.iter() if el.tag.endswith("}loc") and el.text]
    return sorted({u for u in locs if u.startswith(include_prefix)})


def markdown_url(page_url: str) -> str:
    """Both docs sites expose a Markdown twin of every page at this path."""
    return page_url + "index.md" if page_url.endswith("/") else page_url + ".md"


def fetch_markdown(page_url: str) -> str | None:
    try:
        response = requests.get(markdown_url(page_url), headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    text = response.text
    # A site that doesn't have a .md twin for this page answers with the HTML
    # page instead of a 404 — skip those rather than embedding raw markup.
    if text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        return None
    return _LEADING_QUOTE_LINES.sub("", text).strip()


def page_title(markdown: str, page_url: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    # Fall back to the last path segment, e.g. ".../text-filtering/" -> "Text Filtering"
    slug = page_url.rstrip("/").rsplit("/", 1)[-1]
    return slug.replace("-", " ").replace("_", " ").title()


def build_collection(client: QdrantClient, recreate: bool):
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION_NAME in existing:
        if not recreate:
            print(f"Collection '{config.QDRANT_COLLECTION_NAME}' already exists — reusing it.")
            return
        client.delete_collection(config.QDRANT_COLLECTION_NAME)
        print(f"Dropped existing collection '{config.QDRANT_COLLECTION_NAME}'.")
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=config.EMBEDDING_DIM, distance=qmodels.Distance.COSINE
        ),
    )
    print(f"Created collection '{config.QDRANT_COLLECTION_NAME}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max pages per source")
    parser.add_argument("--recreate", action="store_true", help="Drop the collection first")
    args = parser.parse_args()

    print(f"Connecting to Qdrant at {config.QDRANT_URL} ...")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    build_collection(client, recreate=args.recreate)

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' (runs locally)...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    total_pages = 0
    total_chunks = 0
    skipped = 0

    for source in config.DOC_SOURCES:
        name = source["source"]
        print(f"\n=== {name} ===")
        urls = sitemap_urls(source["sitemap"], source["include_prefix"])
        if args.limit:
            urls = urls[: args.limit]
        print(f"{len(urls)} documentation pages found in the sitemap.")

        texts: list[str] = []
        payloads: list[dict] = []

        for i, url in enumerate(urls, start=1):
            markdown = fetch_markdown(url)
            if not markdown:
                skipped += 1
                print(f"  [{i}/{len(urls)}] SKIP (no markdown) {url}")
                continue

            title = page_title(markdown, url)
            chunks = splitter.split_text(markdown)
            for chunk_index, chunk in enumerate(chunks):
                texts.append(chunk)
                payloads.append(
                    {
                        "text": chunk,
                        "title": title,
                        "url": url,
                        "source": name,
                        "chunk_index": chunk_index,
                    }
                )
            total_pages += 1
            print(f"  [{i}/{len(urls)}] {len(chunks):>3} chunks  {title}")
            time.sleep(0.1)  # stay polite to the docs sites

        if not texts:
            print(f"  Nothing ingested for {name}.")
            continue

        print(f"Embedding {len(texts)} chunks locally...")
        vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)

        points = [
            qmodels.PointStruct(id=str(uuid.uuid4()), vector=vector.tolist(), payload=payload)
            for vector, payload in zip(vectors, payloads)
        ]

        print(f"Upserting {len(points)} chunks into Qdrant (batches of 128)...")
        for i in range(0, len(points), 128):
            client.upsert(
                collection_name=config.QDRANT_COLLECTION_NAME, points=points[i : i + 128]
            )
        total_chunks += len(points)

    stored = client.count(config.QDRANT_COLLECTION_NAME).count
    print("\nDone.")
    print(f"  Pages ingested:   {total_pages}")
    print(f"  Pages skipped:    {skipped}")
    print(f"  Chunks upserted:  {total_chunks}")
    print(f"  Collection total: {stored}")


if __name__ == "__main__":
    main()
