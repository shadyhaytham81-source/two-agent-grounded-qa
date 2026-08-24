"""
Runs all 100 test questions (tests/test_questions.py) through the live
Researcher/Reviewer pipeline against your real Qdrant collection and LLM
provider, and writes three log files under tests/logs/:

  - test_run_<timestamp>.jsonl   machine-readable, one JSON object per line:
                                  {id, category, question, retrieved_chunks,
                                   draft, revision_count, reviewer_verdict,
                                   reviewer_feedback, unsupported_claims,
                                   final_answer, final_verdict,
                                   elapsed_seconds, error}
  - test_run_<timestamp>.txt     human-readable transcript of the same runs
  - test_run_<timestamp>_summary.txt   per-category counts

Requires a filled-in .env and a collection that has already been ingested via
ingestion/ingest_docs.py.

Usage:
    python -m tests.run_tests
    python -m tests.run_tests --limit 10               # quick smoke test
    python -m tests.run_tests --category out_of_scope
    python -m tests.run_tests --delay 4                # slow down for free-tier rate limits
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(".")
from graph.pipeline import run_pipeline
from tests.test_questions import TEST_QUESTIONS

LOG_DIR = Path("tests/logs")


def run(questions: list[dict], delay: float) -> list[dict]:
    results = []
    for n, q in enumerate(questions, start=1):
        print(f"[{n}/{len(questions)}] ({q['category']}) {q['question']}")
        start = time.time()
        record = {"id": q["id"], "category": q["category"], "question": q["question"]}
        try:
            state = run_pipeline(q["question"])
            record.update(
                {
                    "retrieved_chunks": [
                        {
                            "source": p.get("source"),
                            "title": p.get("title"),
                            "url": p.get("url"),
                            "score": p.get("score"),
                            "text_snippet": p.get("text", "")[:300],
                        }
                        for p in state.get("passages", [])
                    ],
                    "draft": state.get("draft"),
                    "revision_count": state.get("revision_count", 0),
                    "reviewer_verdict": state.get("reviewer_verdict"),
                    "reviewer_feedback": state.get("reviewer_feedback"),
                    "unsupported_claims": state.get("unsupported_claims", []),
                    "final_answer": state.get("final_answer"),
                    "final_verdict": state.get("final_verdict"),
                    "error": None,
                }
            )
            print(f"    -> {record['final_verdict']}  ({len(record['retrieved_chunks'])} chunks)")
        except Exception as exc:  # noqa: BLE001 — log and keep going through all 100
            record.update(
                {
                    "retrieved_chunks": [],
                    "draft": None,
                    "revision_count": None,
                    "reviewer_verdict": None,
                    "reviewer_feedback": None,
                    "unsupported_claims": [],
                    "final_answer": None,
                    "final_verdict": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"    -> ERROR: {exc}")
        record["elapsed_seconds"] = round(time.time() - start, 2)
        results.append(record)
        if delay and n < len(questions):
            time.sleep(delay)
    return results


def write_logs(results: list[dict]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    jsonl_path = LOG_DIR / f"test_run_{timestamp}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    txt_path = LOG_DIR / f"test_run_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Grounded Q&A test run — {timestamp}\n")
        f.write(f"{len(results)} questions\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"[{r['id']}] ({r['category']}) {r['question']}\n")
            if r["error"]:
                f.write(f"  ERROR: {r['error']}\n\n")
                continue
            f.write(f"  Retrieved chunks ({len(r['retrieved_chunks'])}):\n")
            for c in r["retrieved_chunks"]:
                f.write(f"    - [{c['source']}] {c['title']} (score {c['score']})\n")
                f.write(f"      {c['url']}\n")
                f.write(f"      {c['text_snippet']!r}\n")
            f.write(f"  Draft (revisions: {r['revision_count']}): {r['draft']}\n")
            f.write(f"  Reviewer verdict: {r['reviewer_verdict']} — {r['reviewer_feedback']}\n")
            if r["unsupported_claims"]:
                f.write(f"  Unsupported claims flagged: {r['unsupported_claims']}\n")
            f.write(f"  FINAL ANSWER: {r['final_answer']}\n")
            f.write(f"  UI verdict label: {r['final_verdict']}\n")
            f.write(f"  Elapsed: {r['elapsed_seconds']}s\n\n")

    summary_path = LOG_DIR / f"test_run_{timestamp}_summary.txt"
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        stats = by_category.setdefault(
            cat, {"total": 0, "grounded": 0, "caution": 0, "refused": 0, "error": 0}
        )
        stats["total"] += 1
        if r["error"]:
            stats["error"] += 1
            continue
        if not r["retrieved_chunks"] or r["draft"] == "NOT_GROUNDED":
            stats["refused"] += 1
        if r["final_verdict"] and "✅" in r["final_verdict"]:
            stats["grounded"] += 1
        elif r["final_verdict"]:
            stats["caution"] += 1

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Summary — {timestamp}\n\n")
        f.write(
            f"{'category':<24}{'total':>8}{'grounded':>10}{'caution':>10}"
            f"{'refused':>10}{'error':>8}\n"
        )
        for cat, c in by_category.items():
            f.write(
                f"{cat:<24}{c['total']:>8}{c['grounded']:>10}{c['caution']:>10}"
                f"{c['refused']:>10}{c['error']:>8}\n"
            )

    print(f"\nLogs written:\n  {jsonl_path}\n  {txt_path}\n  {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--category", type=str, default=None, help="Only run this category")
    parser.add_argument(
        "--delay",
        type=float,
        default=8.0,
        help="Seconds to wait between questions — free LLM tiers are rate limited",
    )
    args = parser.parse_args()

    questions = TEST_QUESTIONS
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        print("No questions matched the given filters.")
        return

    results = run(questions, delay=args.delay)
    summary_path = write_logs(results)
    print()
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
