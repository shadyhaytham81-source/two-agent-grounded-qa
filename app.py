"""
Streamlit chat interface for the two-agent grounded Q&A assistant.

Run with:
    streamlit run app.py
"""
import streamlit as st

import config
import llm
from graph.pipeline import run_pipeline

st.set_page_config(page_title=f"Grounded Q&A — {config.CORPUS_NAME}", page_icon="🔎")

st.title("🔎 Grounded Q&A Assistant")
st.caption(
    f"Researcher + Reviewer agents, orchestrated with LangGraph, answering "
    f"strictly from an ingested copy of *{config.CORPUS_NAME}*."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources, verdict}


def render_sources(sources: list[dict]):
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            page = f"p. {s['page']}" if s.get("page") is not None else "page unknown"
            st.markdown(f"**[{i}]** {s['title']} — {page} · relevance {s['score']}")


# --- render history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("verdict"):
                st.markdown(f"**Reviewer verdict:** {msg['verdict']}")
            if msg.get("sources"):
                render_sources(msg["sources"])

# --- handle new input ---
if question := st.chat_input(f"Ask something about {config.CORPUS_NAME}..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researcher is retrieving passages and drafting an answer..."):
            try:
                result = run_pipeline(question)
            except Exception as exc:  # noqa: BLE001
                if llm._is_rate_limit(exc):
                    st.warning(
                        "The free LLM tier is rate limited and is currently busy "
                        "(this also happens while the 100-question test suite is "
                        "running). Wait a few seconds and ask again."
                    )
                else:
                    st.error(f"Pipeline error: {exc}")
                st.stop()

        st.markdown(result["final_answer"])
        st.markdown(f"**Reviewer verdict:** {result['final_verdict']}")

        if result.get("reviewer_feedback") and result.get("reviewer_verdict") != "GROUNDED":
            st.info(f"Reviewer feedback: {result['reviewer_feedback']}")

        sources = result.get("passages", [])
        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["final_answer"],
            "verdict": result["final_verdict"],
            "sources": sources,
        }
    )

with st.sidebar:
    st.subheader("About")
    st.markdown(
        f"- **Corpus**: *{config.CORPUS_NAME}* (a local PDF, ingested into a "
        f"remote Qdrant collection — the PDF itself is never redistributed).\n"
        f"- **Agent 1 — Researcher**: retrieves passages from Qdrant and drafts "
        f"a cited, paraphrased answer from them alone.\n"
        f"- **Agent 2 — Reviewer**: checks every claim against those passages "
        f"and sends the draft back once if anything is unsupported or "
        f"over-quoted.\n"
        f"- Questions the book doesn't cover are refused rather than guessed at."
    )
    st.subheader("Configuration")
    st.markdown(
        f"- LLM: `{config.LLM_PROVIDER}` / `{config.LLM_MODEL}`\n"
        f"- Collection: `{config.QDRANT_COLLECTION_NAME}`\n"
        f"- Top-k passages: `{config.TOP_K_PASSAGES}`\n"
        f"- Min relevance: `{config.MIN_RELEVANCE_SCORE}`"
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
