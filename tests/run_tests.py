"""
Runs all 100 test questions (tests/test_questions.py) through the live
Researcher/Reviewer pipeline against your real Qdrant collection + Anthropic
API, and writes two log files under tests/logs/:

  - test_run_<timestamp>.jsonl   machine-readable, one JSON object per line:
                                  {id, category, question, retrieved_chunks,
                                   draft, revision_count, reviewer_verdict,
                                   reviewer_feedback, final_answer,
                                   final_verdict, elapsed_seconds, error}
  - test_run_<timestamp>.txt     human-readable transcript of the same runs,
                                  for quick manual review.

Requires a real .env (ANTHROPIC_API_KEY, QDRANT_URL, QDRANT_API_KEY) and a
collection that's already been ingested via ingestion/ingest_pdf.py — this
makes real API calls and costs a small amount of Anthropic usage.

Usage:
    python -m tests.run_tests
    python -m tests.run_tests --limit 10          # quick smoke test
    python -m tests.run_tests --category out_of_scope
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


def run(questions: list[dict]) -> list[dict]:
    results = []
    for n, q in enumerate(questions, start=1):
        print(f"[{n}/{len(questions)}] ({q['category']}) {q['question']}")
        start = time.time()
        record = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
        }
        try:
            state = run_pipeline(q["question"])
            record.update(
                {
                    "retrieved_chunks": [
                        {
                            "title": p["title"],
                            "page": p.get("page"),
                            "score": p["score"],
                            "text_snippet": p["text"][:300],
                        }
                        for p in state.get("passages", [])
                    ],
                    "draft": state.get("draft"),
                    "revision_count": state.get("revision_count", 0),
                    "reviewer_verdict": state.get("reviewer_verdict"),
                    "reviewer_feedback": state.get("reviewer_feedback"),
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
                    "final_answer": None,
                    "final_verdict": None,
                    "error": str(exc),
                }
            )
            print(f"    -> ERROR: {exc}")
        record["elapsed_seconds"] = round(time.time() - start, 2)
        results.append(record)
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
                f.write(f"    - {c['title']} p.{c['page']} (score {c['score']}): {c['text_snippet']!r}\n")
            f.write(f"  Draft (revisions: {r['revision_count']}): {r['draft']}\n")
            f.write(f"  Reviewer verdict: {r['reviewer_verdict']} — {r['reviewer_feedback']}\n")
            f.write(f"  FINAL ANSWER: {r['final_answer']}\n")
            f.write(f"  UI verdict label: {r['final_verdict']}\n")
            f.write(f"  Elapsed: {r['elapsed_seconds']}s\n\n")

    summary_path = LOG_DIR / f"test_run_{timestamp}_summary.txt"
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, {"total": 0, "grounded": 0, "caution": 0, "error": 0})
        by_category[cat]["total"] += 1
        if r["error"]:
            by_category[cat]["error"] += 1
        elif r["final_verdict"] and "✅" in r["final_verdict"]:
            by_category[cat]["grounded"] += 1
        elif r["final_verdict"]:
            by_category[cat]["caution"] += 1

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Summary — {timestamp}\n\n")
        f.write(f"{'category':<24}{'total':>8}{'grounded':>10}{'caution':>10}{'error':>8}\n")
        for cat, counts in by_category.items():
            f.write(
                f"{cat:<24}{counts['total']:>8}{counts['grounded']:>10}"
                f"{counts['caution']:>10}{counts['error']:>8}\n"
            )

    print(f"\nLogs written:\n  {jsonl_path}\n  {txt_path}\n  {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--category", type=str, default=None, help="Only run questions in this category")
    args = parser.parse_args()

    questions = TEST_QUESTIONS
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        print("No questions matched the given filters.")
        return

    results = run(questions)
    write_logs(results)


if __name__ == "__main__":
    main()
