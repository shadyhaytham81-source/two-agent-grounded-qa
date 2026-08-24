"""
LangGraph orchestration for the two-agent pipeline.

This is a real graph with a conditional handoff, not a fixed A-then-B chain:

    retrieve -> draft -> review --(GROUNDED)--> finalize -> END
                            |
                    (NOT_GROUNDED and retries left)
                            v
                          revise -> review (loop back)

review is where control genuinely branches based on the Reviewer agent's
verdict — if there are no retries left it finalizes even on a NOT_GROUNDED
verdict, but labels that clearly in the UI rather than silently upgrading it.
"""
from langgraph.graph import StateGraph, START, END

import config
from agents import researcher, reviewer
from graph.state import QAState


def retrieve_node(state: QAState) -> QAState:
    passages = researcher.retrieve(state["query"])
    return {"passages": passages}


def draft_node(state: QAState) -> QAState:
    draft = researcher.draft_answer(state["query"], state["passages"])
    return {"draft": draft, "revision_count": state.get("revision_count", 0)}


def review_node(state: QAState) -> QAState:
    result = reviewer.review(state["query"], state["draft"], state["passages"])
    return {
        "reviewer_verdict": result["verdict"],
        "reviewer_feedback": result.get("feedback", ""),
        "unsupported_claims": result.get("unsupported_claims", []),
    }


def revise_node(state: QAState) -> QAState:
    revised = researcher.revise_answer(
        query=state["query"],
        passages=state["passages"],
        previous_draft=state["draft"],
        reviewer_feedback=state["reviewer_feedback"],
    )
    return {"draft": revised, "revision_count": state.get("revision_count", 0) + 1}


def finalize_node(state: QAState) -> QAState:
    draft = state["draft"]
    if draft.strip() == "NOT_GROUNDED":
        final_answer = researcher.REFUSAL_MESSAGE
    else:
        final_answer = draft

    if state.get("reviewer_verdict") == "GROUNDED":
        final_verdict = "✅ Grounded — approved by Reviewer"
    else:
        final_verdict = (
            "⚠️ Sent back for revision once; some claims may still be weakly "
            "supported. Treat with caution — see reviewer feedback."
        )

    return {"final_answer": final_answer, "final_verdict": final_verdict}


def route_after_review(state: QAState) -> str:
    if state["reviewer_verdict"] == "GROUNDED":
        return "finalize"
    if state.get("revision_count", 0) < config.MAX_REVIEWER_RETRIES:
        return "revise"
    return "finalize"  # retries exhausted — finalize with the caution label


def build_graph():
    graph = StateGraph(QAState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("draft", draft_node)
    graph.add_node("review", review_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {"revise": "revise", "finalize": "finalize"},
    )
    graph.add_edge("revise", "review")  # genuine handoff back to the Reviewer
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(query: str) -> QAState:
    graph = get_graph()
    return graph.invoke({"query": query, "revision_count": 0})
