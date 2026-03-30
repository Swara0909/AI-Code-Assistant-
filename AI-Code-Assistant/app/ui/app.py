"""
Streamlit UI for the AI Code Assistant.

BUG FIX SUMMARY
────────────────
1. Removed the `if not service.has_knowledge_base: st.warning("Upload files first")`
   gate that blocked ALL chat when no KB was loaded.  The assistant now works
   immediately on launch — the user just types or pastes code.

2. Replaced the "you must upload files" flow with a proper chat-first UI:
   • Main panel: conversational chat with code/question input.
   • Sidebar (optional): "Index a snippet" expander for opt-in RAG mode.

3. Fixed chat message rendering to use st.chat_message() instead of raw markdown
   so user/assistant bubbles render correctly.

4. Added mode badge so users see whether the response came from LLM or RAG.
"""

from __future__ import annotations

import sys
import os
import uuid
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

from app.services.chat_service import ChatService
from app.config.settings import APP_TITLE, APP_ICON, SESSION_ID_DEFAULT


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state() -> None:
    if "service" not in st.session_state:
        with st.spinner("Initialising AI Code Assistant…"):
            st.session_state.service = ChatService()
    if "session_id" not in st.session_state:
        st.session_state.session_id = SESSION_ID_DEFAULT
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add chat history (example titles and messages)
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = [
            {"title": "Git Error Fix", "messages": [
                {"role": "user", "content": "How do I fix this git error?"},
                {"role": "assistant", "content": "Try running 'git fetch --all' and then 'git reset --hard origin/main'."}
            ]},
            {"title": "Cloned Project Not Visible", "messages": [
                {"role": "user", "content": "I cloned a project but can't see it in VS Code."},
                {"role": "assistant", "content": "Check if you opened the correct folder. Use 'File > Open Folder' in VS Code."}
            ]},
            {"title": "Best Udemy AI Courses", "messages": [
                {"role": "user", "content": "What are the best Udemy courses for AI?"},
                {"role": "assistant", "content": "Some top-rated courses are 'Python for Data Science and Machine Learning Bootcamp' and 'Deep Learning A-Z'."}
            ]},
            {"title": "Missing package installation", "messages": [
                {"role": "user", "content": "ModuleNotFoundError: No module named 'numpy'"},
                {"role": "assistant", "content": "Install it using 'pip install numpy' in your terminal."}
            ]},
            {"title": "AI Engineer Roadmap Guide", "messages": [
                {"role": "user", "content": "Can you give me a roadmap to become an AI engineer?"},
                {"role": "assistant", "content": "Start with Python, learn ML basics, then deep learning, NLP, and deployment. Practice with projects."}
            ]},
        ]
    if "selected_chat_index" not in st.session_state:
        st.session_state.selected_chat_index = 0


_init_state()
service: ChatService = st.session_state.service


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Paste code directly — no file upload needed.")
    st.markdown("---")

    # Chat history section
    st.markdown("#### Your chats")
    chat_titles = [chat["title"] for chat in st.session_state.chat_histories]
    selected = st.radio("Select a chat", chat_titles, index=st.session_state.selected_chat_index, key="chat_history_radio")
    selected_index = chat_titles.index(selected)
    if selected_index != st.session_state.selected_chat_index:
        st.session_state.selected_chat_index = selected_index
        # Load the selected chat's messages
        st.session_state.messages = st.session_state.chat_histories[selected_index]["messages"].copy()
        st.rerun()
    st.markdown("---")

    # Session controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑 Clear history", use_container_width=True):
            service.clear_session(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # ── Optional: index a snippet for RAG ─────────────────────────────────────
    with st.expander("🔍 Index snippet for deeper search (optional)", expanded=False):
        st.caption(
            "Paste a large codebase or file here to enable retrieval-augmented "
            "answers.  For most questions, just use the chat directly — this is optional."
        )
        snippet = st.text_area("Paste code / text to index", height=200, key="rag_snippet")
        snippet_label = st.text_input("Label (optional)", value="my_snippet")

        if st.button("Index snippet"):
            if not snippet.strip():
                st.warning("Nothing to index.")
            else:
                with st.spinner("Embedding with CodeBERT…"):
                    n = service.ingest_text(snippet.strip(), source=snippet_label)
                st.success(f"Indexed {n} chunk(s). RAG mode is now active.")

        if service.has_knowledge_base:
            st.info("📚 RAG mode active — answers grounded in indexed snippets.")
            if st.button("Clear knowledge base"):
                service.clear_knowledge_base()
                st.rerun()

    st.markdown("---")
    st.markdown(
        "**Tips**\n"
        "- Paste code and ask *'Explain this'* or *'Fix the bug'*\n"
        "- Ask *'How do I sort a list in Python?'* — no upload needed\n"
        "- Multi-turn: follow-up questions keep context"
    )
    st.caption(f"Session: `{st.session_state.session_id}`")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(
    "Ask coding questions, paste code to explain/debug/improve, "
    "or discuss any programming topic.  No file upload required."
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("mode"):
            badge = "📚 RAG" if msg["mode"] == "rag" else "🤖 LLM"
            st.caption(badge)

# Chat input
placeholder = (
    "Paste code here or ask a programming question…\n\n"
    "Examples:\n"
    "• 'Explain what this does: def fib(n): return n if n<2 else fib(n-1)+fib(n-2)'\n"
    "• 'What's wrong with: for i in range(len(lst)): lst.remove(lst[i])'\n"
    "• 'How do I implement a binary search tree in Python?'"
)

if prompt := st.chat_input("Ask a coding question or paste code…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = service.chat(prompt, st.session_state.session_id)
                answer = result["answer"]
                mode = result["mode"]
            except Exception as exc:
                answer = f"⚠️ Error: {exc}"
                mode = "error"

        st.markdown(answer)
        badge = "📚 RAG" if mode == "rag" else "🤖 LLM"
        st.caption(badge)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "mode": mode}
    )
