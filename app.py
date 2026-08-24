"""
Streamlit chat interface for the two-agent grounded Q&A assistant.

Run with:
    streamlit run app.py
"""
import streamlit as st

import config
from graph.pipeline import run_pipeline

st.set_page_config(page_title="Grounded Q&A — LangChain & Qdrant docs", page_icon="🔎")

st.title("🔎 Grounded Q&A Assistant")
st.caption(
    "Researcher + Reviewer agents, orchestrated with LangGraph, answering "
    "strictly from the ingested LangChain and Qdrant documentation."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources, verdict}


def render_sources(sources: list[dict]):
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}]** `{s['source']}` — [{s['title']}]({s['url']}) "
                f"· relevance {s['score']}"
            )


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
if question := st.chat_input("Ask about LangChain or Qdrant..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researcher is retrieving passages and drafting an answer..."):
            try:
                result = run_pipeline(question)
            except Exception as exc:  # noqa: BLE001
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
        "- **Corpus**: the LangChain and Qdrant documentation, ingested into a "
        "remote Qdrant collection.\n"
        "- **Agent 1 — Researcher**: retrieves passages from Qdrant and drafts "
        "a cited answer from them alone.\n"
        "- **Agent 2 — Reviewer**: checks every claim against those passages "
        "and sends the draft back once if anything is unsupported.\n"
        "- Questions the documentation doesn't cover are refused rather than "
        "guessed at."
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
