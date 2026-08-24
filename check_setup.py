"""
Pre-flight check. Run after filling in .env, before ingesting or running the
app / test suite:

    python check_setup.py

Verifies, in order:
  1. .env is present and every required variable is set
  2. the LLM provider key works (one tiny request)
  3. the remote Qdrant cluster is reachable, and how many chunks the
     collection currently holds

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
        fail("No .env file found. Run: cp .env.example .env   then fill it in.")
        return 1
    try:
        import config
    except RuntimeError as exc:
        fail(str(exc))
        return 1
    ok(f"LLM_PROVIDER           = {config.LLM_PROVIDER}")
    ok(f"LLM_MODEL              = {config.LLM_MODEL}")
    ok(f"QDRANT_URL             = {config.QDRANT_URL}")
    ok(f"QDRANT_COLLECTION_NAME = {config.QDRANT_COLLECTION_NAME}")

    print(f"\n2. LLM provider ({config.LLM_PROVIDER})")
    try:
        import llm

        reply = llm.complete(
            system="You are a connectivity test. Reply with one word.",
            user="Reply with the single word: ready",
            max_tokens=16,
        )
        ok(f"{config.LLM_MODEL} responded: {reply!r}")
    except Exception as exc:  # noqa: BLE001
        fail(f"{type(exc).__name__}: {exc}")
        problems += 1

    print("\n3. Qdrant Cloud")
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
                    f"run: python -m ingestion.ingest_docs"
                )
                problems += 1
            else:
                ok(f"Collection '{config.QDRANT_COLLECTION_NAME}' holds {count} chunks.")
        else:
            print(
                f"  [NOTE] Collection '{config.QDRANT_COLLECTION_NAME}' doesn't exist yet — "
                f"it is created by: python -m ingestion.ingest_docs"
            )
    except Exception as exc:  # noqa: BLE001
        fail(f"{type(exc).__name__}: {exc}")
        problems += 1

    print()
    if problems:
        print(f"{problems} problem(s) found — fix the [FAIL] lines above, then re-run.")
        return 1
    print("All checks passed. Next: python -m ingestion.ingest_docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
