"""
Pre-flight check. Run this after filling in .env and before ingesting or
running the app / test suite:

    python check_setup.py

It verifies, in order:
  1. .env is present and every required variable is set
  2. the book PDF exists locally and has an extractable text layer
  3. the Anthropic API key works (one tiny request)
  4. the remote Qdrant cluster is reachable, and reports how many points
     the collection currently holds

Nothing here writes data — it is safe to run at any time.
"""
import os
import sys


def ok(msg):
    print(f"  [ OK ] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def main() -> int:
    problems = 0

    print("\n1. Environment variables")
    if not os.path.exists(".env"):
        fail("No .env file found. Run: cp .env.example .env  then fill it in.")
        return 1
    try:
        import config
    except RuntimeError as exc:
        fail(str(exc))
        return 1
    ok(f"ANTHROPIC_MODEL       = {config.ANTHROPIC_MODEL}")
    ok(f"QDRANT_URL            = {config.QDRANT_URL}")
    ok(f"QDRANT_COLLECTION_NAME= {config.QDRANT_COLLECTION_NAME}")
    ok(f"BOOK_PDF_PATH         = {config.BOOK_PDF_PATH}")

    print("\n2. Corpus PDF")
    if not os.path.exists(config.BOOK_PDF_PATH):
        fail(f"No PDF at '{config.BOOK_PDF_PATH}'. See corpus/README.md.")
        problems += 1
    else:
        import pdfplumber

        with pdfplumber.open(config.BOOK_PDF_PATH) as pdf:
            total = len(pdf.pages)
            sample = min(total, 10)
            with_text = sum(1 for p in pdf.pages[:sample] if (p.extract_text() or "").strip())
        if with_text == 0:
            fail(
                f"{total} pages, but no extractable text in the first {sample} — "
                f"this looks like a scanned/image-only PDF. It needs OCR first."
            )
            problems += 1
        else:
            ok(f"{total} pages, text layer present ({with_text}/{sample} sampled pages had text)")

    print("\n3. Anthropic API")
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        ok(f"{config.ANTHROPIC_MODEL} responded: {resp.content[0].text.strip()!r}")
    except Exception as exc:  # noqa: BLE001
        fail(f"{type(exc).__name__}: {exc}")
        problems += 1

    print("\n4. Qdrant Cloud")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        names = [c.name for c in client.get_collections().collections]
        ok(f"connected. Collections: {names or '(none yet)'}")
        if config.QDRANT_COLLECTION_NAME in names:
            count = client.count(config.QDRANT_COLLECTION_NAME).count
            if count == 0:
                fail(
                    f"Collection '{config.QDRANT_COLLECTION_NAME}' exists but is empty — "
                    f"run: python -m ingestion.ingest_pdf"
                )
                problems += 1
            else:
                ok(f"Collection '{config.QDRANT_COLLECTION_NAME}' holds {count} chunks.")
        else:
            print(
                f"  [NOTE] Collection '{config.QDRANT_COLLECTION_NAME}' doesn't exist yet — "
                f"it gets created by: python -m ingestion.ingest_pdf"
            )
    except Exception as exc:  # noqa: BLE001
        fail(f"{type(exc).__name__}: {exc}")
        problems += 1

    print()
    if problems:
        print(f"{problems} problem(s) found — fix the [FAIL] lines above, then re-run.")
        return 1
    print("All checks passed. Next: python -m ingestion.ingest_pdf  (if not ingested yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
