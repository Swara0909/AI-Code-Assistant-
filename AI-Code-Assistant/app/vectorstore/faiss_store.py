"""
FAISS vector store — create, save, load, and search.
Uses the custom CodeBERT embeddings.

BUG FIX: Original loaded embeddings at module level (via _get_embeddings
called from top-level code paths) causing the 500 MB model to load on
every Streamlit hot-reload. Now uses a proper module-level lazy singleton
that is only initialised on the very first call to an embedding function.
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.models.embeddings import CodeBERTEmbeddings
from app.config.settings import FAISS_INDEX_DIR


# ── lazy singleton ────────────────────────────────────────────────────────────
_embeddings: Optional[CodeBERTEmbeddings] = None


def _get_embeddings() -> CodeBERTEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = CodeBERTEmbeddings()
    return _embeddings


# ── public API ────────────────────────────────────────────────────────────────

def build_vectorstore(documents: List[Document]) -> FAISS:
    """Build a new FAISS index from a list of LangChain Documents."""
    return FAISS.from_documents(documents, _get_embeddings())


def save_vectorstore(vectorstore: FAISS) -> None:
    """Persist FAISS index to disk."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))


def load_vectorstore() -> Optional[FAISS]:
    """Load FAISS index from disk. Returns None if not found."""
    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if not os.path.exists(index_file):
        return None
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def add_documents(vectorstore: FAISS, documents: List[Document]) -> FAISS:
    """Add new documents to an existing FAISS index."""
    vectorstore.add_documents(documents)
    return vectorstore


def get_retriever(vectorstore: FAISS, k: int = 4):
    """Return a LangChain retriever from the FAISS store."""
    return vectorstore.as_retriever(search_kwargs={"k": k})
