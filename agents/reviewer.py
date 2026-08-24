"""
Agent 2 — Reviewer.

Responsibilities:
  Review the Researcher's drafted answer against the retrieved passages.
  For every factual claim in the draft, check whether it's actually
  supported by the passages. Return a structured verdict the graph can
  branch on: GROUNDED -> finalize, NOT_GROUNDED -> send back to Researcher.
"""
import json
import re

import config
import llm
from agents.researcher import format_passages

REVIEW_SYSTEM_PROMPT = f"""You are the Reviewer agent in a grounded Q&A system \
answering questions about the book "{config.CORPUS_NAME}". You are a strict, \
skeptical fact-checker — your only job is to catch claims that are NOT \
actually supported by the provided passages, even if they sound plausible, \
are well-known about the book, or are generally true in the real world.

You will get the user's question, the numbered passages that were retrieved, \
and a draft answer that cites them like [1], [2].

For each factual claim in the draft, check:
1. Is this specific claim actually stated or directly implied by the cited \
passage(s)? Citation-shopping — citing a passage that is topically related \
but does not support the specific claim — counts as NOT grounded.
2. Does the draft quote more than ~10 words verbatim from any passage, or \
quote the same passage more than once? Flag that as a copyright violation in \
unsupported_claims even if the content itself is accurate — it should be \
paraphrased instead.

Respond with ONLY a JSON object, no markdown fences, no preamble, in exactly \
this shape:
{{
  "verdict": "GROUNDED" or "NOT_GROUNDED",
  "unsupported_claims": ["claim text ...", ...],
  "feedback": "short explanation for the Researcher agent to act on"
}}

If the draft is "NOT_GROUNDED" (a refusal), verdict is "GROUNDED" — a correct \
refusal does not need fixing. If every claim is well-supported and no \
over-quoting occurred, verdict is "GROUNDED" and unsupported_claims is an \
empty list.
"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _safe_parse(raw: str) -> dict:
    cleaned = _FENCE.sub("", raw.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Some models wrap the JSON in a sentence — take the outermost object.
    match = _OBJECT.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fail safe: if the reviewer's own output is malformed, don't silently
    # approve — flag it so the graph treats it as needing another look.
    return {
        "verdict": "NOT_GROUNDED",
        "unsupported_claims": [],
        "feedback": f"Reviewer output could not be parsed as JSON: {raw[:300]}",
    }


def review(query: str, draft: str, passages: list[dict]) -> dict:
    if draft.strip() == "NOT_GROUNDED":
        return {
            "verdict": "GROUNDED",
            "unsupported_claims": [],
            "feedback": "Correct refusal — no passages supported the question.",
        }

    raw = llm.complete(
        system=REVIEW_SYSTEM_PROMPT,
        user=(
            f"Question: {query}\n\n"
            f"Retrieved passages:\n{format_passages(passages)}\n\n"
            f"Draft answer:\n{draft}"
        ),
        max_tokens=1500,
        model=config.REVIEWER_MODEL,
    )
    result = _safe_parse(raw)
    result.setdefault("verdict", "NOT_GROUNDED")
    result.setdefault("unsupported_claims", [])
    result.setdefault("feedback", "")
    if result["verdict"] not in ("GROUNDED", "NOT_GROUNDED"):
        result["verdict"] = "NOT_GROUNDED"
    return result
