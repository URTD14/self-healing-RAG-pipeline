"""
Self-Healing RAG Pipeline — Streamlit UI
"""

import tempfile
from pathlib import Path

import streamlit as st
from ingest import get_doc_count, ingest_files
from pipeline import graph, MAX_RETRIES

st.set_page_config(page_title="Self-Healing RAG", page_icon="", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — Document Ingestion
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Document Ingestion")
    st.caption("Upload PDF or TXT files to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Ingest Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            paths = []
            for f in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(f.name).suffix
                )
                tmp.write(f.read())
                tmp.close()
                paths.append(tmp.name)

            with st.spinner("Processing documents..."):
                num_chunks = ingest_files(paths)

            st.success(f"Ingested **{num_chunks}** chunks from **{len(uploaded_files)}** file(s).")

    st.divider()
    doc_count = get_doc_count()
    st.metric("Chunks in Vector Store", doc_count)

    st.divider()
    st.caption("Built with LangGraph + Groq + Chroma")

# ---------------------------------------------------------------------------
# Main — Query Interface
# ---------------------------------------------------------------------------

st.title("Self-Healing RAG Pipeline")
st.markdown(
    "Ask a question. The system retrieves, generates, critiques its own answer, "
    "and retries with reformulated queries if needed."
)

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is retrieval-augmented generation?",
    label_visibility="collapsed",
)

if question and st.button("Run Pipeline", type="primary"):
    if doc_count == 0:
        st.error("No documents in the vector store. Please ingest files first.")
    else:
        with st.spinner("Running self-healing RAG pipeline..."):
            state = graph.invoke({
                "question": question,
                "context": [],
                "answer": "",
                "critique": {},
                "retry_count": 0,
                "reformulated_query": "",
            })

        retries = state.get("retry_count", 0)
        verdict = state.get("critique", {}).get("verdict", "UNKNOWN")

        # ----- Pipeline summary -----
        cols = st.columns(4)
        cols[0].metric("Question", question[:50] + "...")
        cols[1].metric("Retries Used", retries)
        cols[2].metric("Final Verdict", verdict)
        cols[3].metric("Chunks Retrieved", len(state.get("context", [])))

        st.divider()

        # ----- Final Answer -----
        st.subheader("Final Answer")
        if retries >= MAX_RETRIES and verdict != "PASS":
            st.warning(
                "The system was unable to produce a satisfactory answer after "
                f"{MAX_RETRIES} attempts."
            )
        st.markdown(state.get("answer", ""))

        # ----- Critique -----
        critique = state.get("critique", {})
        if critique:
            st.divider()
            st.subheader("Critique Scores")
            c1, c2, c3 = st.columns(3)
            c1.metric("Faithfulness", f"{critique.get('faithfulness', 0):.2f}")
            c2.metric("Relevance", f"{critique.get('relevance', 0):.2f}")
            c3.metric("Completeness", f"{critique.get('completeness', 0):.2f}")

            feedback = critique.get("feedback", "")
            if feedback:
                st.info(f"**Feedback:** {feedback}")

        # ----- Retrieved Context -----
        st.divider()
        with st.expander("Retrieved Context"):
            for i, doc in enumerate(state.get("context", []), 1):
                source = doc.metadata.get("source", "unknown")
                st.markdown(f"**[{i}]** `{source}`")
                st.text(doc.page_content[:500])
                st.markdown("---")
