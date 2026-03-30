"""
Chain builders.

Two modes:
  ① Conversational Code Assistant (no KB) — pure LLM with rolling memory
  ② RAG mode (code snippet added to KB) — retrieval-augmented

BUG FIXES
─────────
1. Memory factory mismatch: original passed get_memory() (output_key="answer")
   to LLMChain which returns key "text" → ValueError on every LLM-only call.
   Fixed: use get_memory_for_rag / get_memory_for_llm.

2. LLM-only prompt had no system message guiding the model to act as a coding
   assistant.  Added a strong system prompt so the assistant never asks users
   to upload files.

3. RAG system prompt now explicitly tells the model to use its general coding
   knowledge when context is thin (avoids "I don't have information" non-answers).
"""

from __future__ import annotations

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.vectorstores import VectorStoreRetriever

from app.models.llm_model import get_llm
from app.memory.memory_store import get_memory_for_rag, get_memory_for_llm
from app.config.settings import RETRIEVER_K


# ── Shared prompts ────────────────────────────────────────────────────────────

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the conversation history and a follow-up message, rephrase the
follow-up as a fully self-contained standalone question so it can be
understood without the history.

Chat History:
{chat_history}

Follow-up: {question}

Standalone question:"""
)

# ─── LLM-only prompt (PRIMARY mode — no file upload needed) ──────────────────
# BUG FIX: original had a weak system prompt that didn't suppress file-upload
# behaviour.  This version explicitly forbids asking for file uploads and lists
# the assistant's core capabilities.
LLM_ONLY_SYSTEM = """\
You are an expert AI Code Assistant — think of yourself as ChatGPT specialised \
for software development. You help developers with ALL of the following tasks:

  • Explain code  — break down what a snippet does, line by line if needed.
  • Suggest improvements  — refactor, optimise, and follow best practices.
  • Detect & fix bugs  — identify issues and provide corrected code.
  • Answer programming questions  — architecture, algorithms, libraries, tools.
  • Write new code  — generate functions, classes, scripts on request.
  • Compare approaches  — pros/cons of different implementations.

RULES:
  - NEVER ask the user to upload a file.  They paste code directly — that is \
the entire interface.
  - Always format code inside markdown fences with the correct language tag, \
e.g. ```python ... ```.
  - Be concise but thorough.  Explain your reasoning when non-obvious.
  - If you are unsure about something, say so clearly instead of guessing.
"""

LLM_ONLY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(LLM_ONLY_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}"),
])

# ─── RAG prompt (when code snippet is indexed into FAISS) ────────────────────
RAG_SYSTEM_TEMPLATE = """\
You are an expert AI Code Assistant.

Use the retrieved code context below to answer the user's question.
If the answer is not fully in the context, supplement with your own coding \
knowledge but make it clear which parts come from the context and which from \
general knowledge.

Always format code with markdown fences (e.g. ```python ... ```).
NEVER ask the user to upload files — they have already provided the code above.

Context (retrieved snippets):
{context}
"""

RAG_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(RAG_SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
])


# ── Chain factories ───────────────────────────────────────────────────────────

def build_llm_only_chain(session_id: str) -> LLMChain:
    """
    Pure LLM chain — the PRIMARY mode.
    User pastes code/questions directly, no retrieval needed.
    Memory: ConversationBufferWindowMemory with output_key='text'.
    """
    llm = get_llm()
    memory = get_memory_for_llm(session_id)  # BUG FIX: was get_memory()

    return LLMChain(
        llm=llm,
        prompt=LLM_ONLY_PROMPT,
        memory=memory,
        verbose=False,
    )


def build_rag_chain(
    retriever: VectorStoreRetriever,
    session_id: str,
) -> ConversationalRetrievalChain:
    """
    RAG chain — activated when user explicitly indexes a code snippet.
    Memory: ConversationBufferWindowMemory with output_key='answer'.
    """
    llm = get_llm()
    memory = get_memory_for_rag(session_id)  # BUG FIX: was get_memory()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": RAG_CHAT_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
