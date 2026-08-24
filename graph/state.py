from typing import TypedDict


class QAState(TypedDict, total=False):
    query: str
    passages: list[dict]          # from Researcher's retrieval step
    draft: str                    # current draft answer
    reviewer_verdict: str         # "GROUNDED" | "NOT_GROUNDED"
    reviewer_feedback: str
    unsupported_claims: list[str]
    revision_count: int
    final_answer: str
    final_verdict: str            # what gets shown in the UI
