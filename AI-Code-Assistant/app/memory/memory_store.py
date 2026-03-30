"""
Persistent conversation memory backed by SQLite.

BUG FIX: The original code set output_key="answer" on ALL memory objects,
but LLMChain returns its result under key "text", not "answer".  When the
LLM-only chain was used (no knowledge base), LangChain raised:

    ValueError: One output key expected, got dict_keys(['text'])

Fix: expose two factory functions — one per chain type — with the
correct output_key for each.
"""

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_classic.memory import ConversationBufferWindowMemory

from app.config.settings import SQLITE_DB_PATH, MEMORY_WINDOW_K, SESSION_ID_DEFAULT


def _get_sql_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=f"sqlite:///{SQLITE_DB_PATH}",
    )


def get_memory_for_rag(session_id: str = SESSION_ID_DEFAULT) -> ConversationBufferWindowMemory:
    """
    Memory for ConversationalRetrievalChain.
    output_key='answer' matches the chain's return dict key.
    """
    history = _get_sql_history(session_id)
    return ConversationBufferWindowMemory(
        chat_memory=history,
        k=MEMORY_WINDOW_K,
        memory_key="chat_history",
        return_messages=True,   # ConversationalRetrievalChain needs message objects
        output_key="answer",    # chain returns {"answer": ..., "source_documents": ...}
    )


def get_memory_for_llm(session_id: str = SESSION_ID_DEFAULT) -> ConversationBufferWindowMemory:
    """
    Memory for LLMChain (direct, no retrieval).
    output_key='text' matches LLMChain's return dict key.

    BUG FIX: Original used output_key='answer' here which broke the
    LLM-only path with a KeyError / ValueError.
    """
    history = _get_sql_history(session_id)
    return ConversationBufferWindowMemory(
        chat_memory=history,
        k=MEMORY_WINDOW_K,
        memory_key="chat_history",
        return_messages=True,
        output_key="text",      # LLMChain returns {"text": ...}
    )


def clear_memory(session_id: str = SESSION_ID_DEFAULT) -> None:
    """Wipe the conversation history for session_id."""
    _get_sql_history(session_id).clear()


def list_sessions() -> list[str]:
    """Return all session IDs stored in SQLite."""
    import sqlite3, os
    if not os.path.exists(SQLITE_DB_PATH):
        return []
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.execute(
        "SELECT DISTINCT session_id FROM message_store ORDER BY session_id"
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows
