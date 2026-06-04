"""
Self-Healing RAG Pipeline — Streamlit UI
"""

import os

# Must be set before any protobuf imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tempfile
from pathlib import Path

import streamlit as st
from ingest import get_doc_count, ingest_files
from pipeline import graph, MAX_RETRIES

st.set_page_config(page_title="Self-Healing RAG", page_icon="", layout="wide")
