"""
Self-Healing RAG Pipeline using LangGraph.

Flow: Retrieve → Generate → Critique → (Pass: Return) / (Fail: Reformulate → Retry)
Max retries: 3
"""

import json
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from ingest import get_retriever

load_dotenv()


def _get_groq_key() -> str:
    """Get Groq API key: Streamlit secrets (cloud) or .env (local)."""
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except (ImportError, FileNotFoundError, KeyError):
        return os.getenv("GROQ_API_KEY", "")

MAX_RETRIES = 3

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=_get_groq_key(),
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str
    context: Annotated[list[Document], "retrieved documents"]
    answer: str
    critique: dict
    retry_count: int
    reformulated_query: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> RAGState:
    retriever = get_retriever(k=5)
    query = state.get("reformulated_query") or state["question"]
    docs = retriever.invoke(query)
    return {"context": docs}


def generate(state: RAGState) -> RAGState:
    context_text = "\n\n".join(doc.page_content for doc in state["context"])

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question based ONLY "
                "on the provided context. If the context does not contain enough "
                "information to answer, say 'I cannot answer this question based on "
                "the available context.' Be concise and accurate."
            )
        ),
        HumanMessage(
            content=(
                f"Context:\n{context_text}\n\n"
                f"Question: {state['reformulated_query'] or state['question']}\n\n"
                f"Answer:"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}


def critique(state: RAGState) -> RAGState:
    prompt = f"""You are a strict quality evaluator for a RAG system.

Question: {state['question']}
Answer: {state['answer']}
Retrieved context: {[doc.page_content for doc in state['context']]}

Evaluate the answer on these criteria and respond with ONLY valid JSON:

{{
    "faithfulness": <0.0 to 1.0 — is the answer grounded in the context?>,
    "relevance": <0.0 to 1.0 — does it answer the actual question?>,
    "completeness": <0.0 to 1.0 — is it thorough enough?>,
    "verdict": <"PASS" or "FAIL">,
    "feedback": <specific explanation of issues, or "Answer is good.">,
    "reformulated_query": <if FAIL, an improved search query to retry; if PASS, empty string>
}}

Verdict should be PASS if faithfulness >= 0.6 AND relevance >= 0.6 AND completeness >= 0.5.
Otherwise FAIL.

Respond with ONLY the JSON object, no markdown fences, no extra text."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        critique_result = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        critique_result = {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "verdict": "FAIL",
            "feedback": "Could not parse critique response.",
            "reformulated_query": state["question"],
        }

    return {"critique": critique_result}


def reformulate(state: RAGState) -> RAGState:
    return {
        "retry_count": state["retry_count"] + 1,
        "reformulated_query": state["critique"].get("reformulated_query", state["question"]),
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_critique(state: RAGState) -> Literal["end", "reformulate"]:
    verdict = state.get("critique", {}).get("verdict", "FAIL")
    retries = state.get("retry_count", 0)

    if verdict == "PASS" or retries >= MAX_RETRIES:
        return "end"
    return "reformulate"


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("critique", critique)
builder.add_node("reformulate", reformulate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "critique")
builder.add_conditional_edges(
    "critique",
    route_critique,
    {
        "end": END,
        "reformulate": "reformulate",
    },
)
builder.add_edge("reformulate", "retrieve")

graph = builder.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(question: str) -> dict:
    """Run the self-healing RAG pipeline and return the final state."""
    initial_state: RAGState = {
        "question": question,
        "context": [],
        "answer": "",
        "critique": {},
        "retry_count": 0,
        "reformulated_query": "",
    }
    return graph.invoke(initial_state)
