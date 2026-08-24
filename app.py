"""
Streamlit chat interface for the two-agent grounded Q&A assistant.

Run with:
    streamlit run app.py
"""
import streamlit as st

import config
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
                st.error(f"Pipeline error: {exc}")
                st.stop()

        st.markdown(result["final_answer"])
        st.markdown(f"**Reviewer verdict:** {result['final_verdict']}")

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
        f"- **Corpus**: *{config.CORPUS_NAME}* (local PDF, ingested into a "
        f"remote Qdrant collection — never redistributed).\n"
        f"- **Agent 1 — Researcher**: retrieves passages from Qdrant and "
        f"drafts a cited, paraphrased answer.\n"
        f"- **Agent 2 — Reviewer**: checks each claim against the retrieved "
        f"passages and sends the draft back once if anything is unsupported "
        f"or over-quoted.\n"
        f"- Questions the book doesn't cover are refused rather than guessed."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
