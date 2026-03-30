"""
ChatService — single entry-point for the UI layer.

BUG FIX: Original routing logic set has_knowledge_base as the gate for
ALL chat.  When no KB was loaded, the UI showed "Upload files first" and
blocked the user — making it a file-based QA system instead of a
conversational coding assistant.

Fixed routing:
  • Default (always available) → LLM-only chain.  User pastes code/question.
  • Opt-in RAG mode            → activated only when user explicitly indexes
                                  a snippet via ingest_text/ingest_file.

The assistant NOW works without any file upload, exactly like ChatGPT.
"""

from __future__ import annotations

from typing import Optional
from openai import APIConnectionError, AuthenticationError
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.chains.rag_chain import build_rag_chain, build_llm_only_chain
from app.vectorstore.faiss_store import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    add_documents,
    get_retriever,
)
from app.memory.memory_store import clear_memory, list_sessions
from app.utils.helpers import load_file, load_directory, load_text_snippet
from app.config.settings import RETRIEVER_K


class ChatService:
    """
    Stateful service shared across Streamlit re-runs via st.session_state.

    The vectorstore and FAISS index are OPTIONAL — the assistant works without
    them.  They are only populated when the user explicitly indexes a snippet.
    """

    def __init__(self) -> None:
        # Try loading a persisted FAISS index from disk; None = no KB yet
        self._vectorstore: Optional[FAISS] = load_vectorstore()
        self._chains: dict = {}  # session_id → chain

    # ── Knowledge-base management (opt-in) ───────────────────────────────────

    def _rebuild_chains(self) -> None:
        """Invalidate cached chains after the vectorstore changes."""
        self._chains.clear()

    def ingest_file(self, file_path: str) -> int:
        docs = load_file(file_path)
        self._upsert_docs(docs)
        return len(docs)

    def ingest_directory(self, dir_path: str) -> int:
        docs = load_directory(dir_path)
        self._upsert_docs(docs)
        return len(docs)

    def ingest_text(self, text: str, source: str = "user_snippet") -> int:
        """Index a raw code/text snippet into FAISS. Returns chunk count."""
        docs = load_text_snippet(text, source)
        self._upsert_docs(docs)
        return len(docs)

    def _upsert_docs(self, docs: list[Document]) -> None:
        if not docs:
            return
        if self._vectorstore is None:
            self._vectorstore = build_vectorstore(docs)
        else:
            self._vectorstore = add_documents(self._vectorstore, docs)
        save_vectorstore(self._vectorstore)
        self._rebuild_chains()

    def clear_knowledge_base(self) -> None:
        """Remove the FAISS index from memory (does not delete disk files)."""
        self._vectorstore = None
        self._rebuild_chains()

    @property
    def has_knowledge_base(self) -> bool:
        return self._vectorstore is not None

    # ── Chain access ──────────────────────────────────────────────────────────

    def _get_llm_chain(self, session_id: str):
        key = f"llm_{session_id}"
        if key not in self._chains:
            self._chains[key] = build_llm_only_chain(session_id)
        return self._chains[key]

    def _get_rag_chain(self, session_id: str):
        key = f"rag_{session_id}"
        if key not in self._chains:
            retriever = get_retriever(self._vectorstore, k=RETRIEVER_K)
            self._chains[key] = build_rag_chain(retriever, session_id)
        return self._chains[key]

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, question: str, session_id: str) -> dict:
        """
        Route the question to the right chain.

        BUG FIX: Original blocked chat entirely when no KB was loaded and
        showed "Upload files first".  Now the LLM-only chain is ALWAYS the
        default; RAG is only used when a KB exists.

        Returns:
            {
                "answer": str,
                "source_documents": list,
                "mode": "llm" | "rag",
            }
        """
        try:
            if self._vectorstore is not None:
                # Optional RAG mode — only when user has indexed something
                chain = self._get_rag_chain(session_id)
                result = chain.invoke({"question": question})
                return {
                    "answer": result.get("answer", ""),
                    "source_documents": result.get("source_documents", []),
                    "mode": "rag",
                }

            # Default LLM-only mode — always works, no upload required
            chain = self._get_llm_chain(session_id)
            result = chain.invoke({"question": question})
            return {
                "answer": result.get("text", ""),   # BUG FIX: key is "text" not "answer"
                "source_documents": [],
                "mode": "llm",
            }
        except AuthenticationError as exc:
            raise RuntimeError(
                "OpenRouter authentication failed (401). "
                "Check OPENROUTER_API_KEY in .env and ensure the key is active."
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                "OpenRouter connection failed. "
                "If your network uses custom SSL inspection, set OPENROUTER_VERIFY_SSL=false "
                "or configure OPENROUTER_CA_BUNDLE in .env."
            ) from exc

    # ── Session helpers ───────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        clear_memory(session_id)
        # Remove both possible chain variants
        self._chains.pop(f"llm_{session_id}", None)
        self._chains.pop(f"rag_{session_id}", None)

    @staticmethod
    def list_sessions() -> list[str]:
        return list_sessions()
