"""
Agent 1 — Researcher.

Responsibilities:
  1. Query the remote Qdrant collection and return relevant passages with
     source metadata (retrieve_passages, via retrieval.py).
  2. Draft an answer to the user's question using ONLY those passages,
     with inline [n] citations matching the passage list.

If retrieval comes back empty (nothing cleared the relevance threshold),
the Researcher refuses outright instead of drafting — there's nothing to
ground an answer in.
"""
from anthropic import Anthropic

import config
from retrieval import retrieve_passages

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

REFUSAL_MESSAGE = (
    f"I don't have grounded information to answer that from the ingested "
    f"copy of {config.CORPUS_NAME}. Please rephrase, or ask something the "
    f"book actually covers."
)

DRAFT_SYSTEM_PROMPT = f"""You are the Researcher agent in a grounded Q&A system \
answering questions about "{config.CORPUS_NAME}".

You will be given a user question and a numbered list of passages retrieved \
from the book. Your job is to draft an answer using ONLY information \
contained in those passages.

Rules:
- Every factual claim must be traceable to at least one passage. Cite it \
inline like [1], [2], combining when needed like [1][3].
- Do NOT use outside knowledge about the book, its author, or personal \
finance in general, even if you're confident it's correct. If the passages \
don't fully answer the question, answer only the part they support and say \
what's missing.
- Copyright: paraphrase in your own words. Do not quote more than about 10 \
words verbatim from any single passage, and use at most one such short quote \
per passage. Never reproduce a full paragraph or a long passage verbatim, \
even across multiple turns.
- If NONE of the passages are actually relevant to the question, respond with \
exactly: NOT_GROUNDED
- Be concise and direct. No filler, no "based on the passages provided" \
preambles — just answer with citations.
"""

REVISE_SYSTEM_PROMPT = f"""You are the Researcher agent revising a draft after \
review feedback from the Reviewer agent, who found unsupported claims about \
"{config.CORPUS_NAME}".

You will get: the original question, the numbered passages, your previous \
draft, and the reviewer's feedback listing what wasn't grounded.

Rewrite the answer so every remaining claim is grounded in the passages. \
Remove or rephrase anything the reviewer flagged that you cannot support. It \
is fine for the revised answer to be shorter or more hedged than the \
original — accuracy matters more than completeness. Keep the [n] citation \
style and the same copyright rule (paraphrase; no verbatim quote over ~10 \
words, at most one per passage). If nothing can be salvaged, respond with \
exactly: NOT_GROUNDED
"""


def _format_passages(passages: list[dict]) -> str:
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] (source: {p['title']}, page {p['page']})\n{p['text']}")
    return "\n\n".join(lines)


def retrieve(query: str) -> list[dict]:
    return retrieve_passages(query)


def draft_answer(query: str, passages: list[dict]) -> str:
    if not passages:
        return "NOT_GROUNDED"

    passages_block = _format_passages(passages)
    response = _client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1000,
        system=DRAFT_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Question: {query}\n\nRetrieved passages:\n{passages_block}",
            }
        ],
    )
    return response.content[0].text.strip()


def revise_answer(query: str, passages: list[dict], previous_draft: str, reviewer_feedback: str) -> str:
    passages_block = _format_passages(passages)
    response = _client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1000,
        system=REVISE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Retrieved passages:\n{passages_block}\n\n"
                    f"Previous draft:\n{previous_draft}\n\n"
                    f"Reviewer feedback:\n{reviewer_feedback}"
                ),
            }
        ],
    )
    return response.content[0].text.strip()
