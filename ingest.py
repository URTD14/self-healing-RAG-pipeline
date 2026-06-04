"""
Document ingestion pipeline for the Self-Healing RAG System.
Loads PDFs and text files, splits them into chunks, embeds them,
and stores them in a Chroma vector database.
"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx", ".html"}


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="documents",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def load_documents(file_paths: list[str]) -> list[Document]:
    docs = []
    for path in file_paths:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif ext in (".txt", ".md"):
                loader = TextLoader(path)
                docs.extend(loader.load())
            elif ext == ".csv":
                loader = CSVLoader(path)
                docs.extend(loader.load())
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
                docs.extend(loader.load())
            elif ext == ".html":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": path},
                        ))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_files(file_paths: list[str]) -> int:
    """Ingest files into the vector store. Returns number of chunks added."""
    docs = load_documents(file_paths)
    if not docs:
        return 0

    chunks = split_documents(docs)
    if not chunks:
        return 0

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def get_retriever(k: int = 5):
    """Return a retriever from the vector store."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_doc_count() -> int:
    """Return the number of documents in the vector store."""
    vectorstore = get_vectorstore()
    return vectorstore._collection.count()
