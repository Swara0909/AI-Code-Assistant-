# AI Code Assistant

A conversational coding assistant built with LangChain, CodeBERT, FAISS, and DeepSeek-Coder via OpenRouter.

---

## Architecture

```
User types/pastes code or question
         │
         ▼
  ┌──────────────┐
  │  Streamlit   │  ← no file upload gate; chat works immediately
  │    UI        │
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────┐
  │             ChatService                  │
  │  ┌─────────────────┐  ┌───────────────┐ │
  │  │  LLM-only chain │  │  RAG chain    │ │
  │  │  (default)      │  │  (opt-in)     │ │
  │  └────────┬────────┘  └──────┬────────┘ │
  └───────────┼──────────────────┼──────────┘
              │                  │
              ▼                  ▼
      DeepSeek-Coder      CodeBERT → FAISS
      (via OpenRouter)    (retrieval)
              │
              ▼
   SQLite-backed memory
   (ConversationBufferWindowMemory)
```

### Modes

| Mode | When | How |
|------|------|-----|
| **LLM-only** | Always (default) | Question → DeepSeek-Coder → Answer |
| **RAG** | When user indexes a snippet | Question → CodeBERT embed → FAISS retrieve → DeepSeek-Coder → Answer |

---

## Bug Fixes Applied

### 1. File-upload gate removed (root cause of QA-system behaviour)
**File:** `app/ui/app.py`

Original code blocked ALL chat with `"Upload files first"` unless a file was ingested:
```python
# ORIGINAL (broken)
if not service.has_knowledge_base:
    st.warning("Upload files first")
```
Fixed: LLM-only chain is now the default — it always works. File indexing is optional.

---

### 2. `output_key` mismatch in memory (crashed LLM-only mode)
**File:** `app/memory/memory_store.py`

`LLMChain` returns `{"text": ...}` but memory was configured with `output_key="answer"`:
```python
# ORIGINAL (broken) — one memory factory for both chain types
memory = ConversationBufferWindowMemory(..., output_key="answer")
```
Fixed: two separate factories — `get_memory_for_rag()` (`output_key="answer"`) and `get_memory_for_llm()` (`output_key="text"`).

---

### 3. Wrong result key in `chat()` for LLM-only mode
**File:** `app/services/chat_service.py`

```python
# ORIGINAL (broken) — LLMChain returns "text", not "answer"
answer = result.get("answer", "")   # always empty string
```
Fixed:
```python
answer = result.get("text", "")     # correct key for LLMChain
```

---

### 4. CodeBERT loading fix
**File:** `app/models/embeddings.py`

- Added clear `RuntimeError` with actionable message if the model can't be downloaded.
- Added batched encoding (`batch_size=8`) to avoid OOM on large document sets.
- Lazy singleton in `faiss_store.py` prevents the 500 MB model from loading on every Streamlit hot-reload.

---

### 5. System prompt engineering
**File:** `app/chains/rag_chain.py`

Added an explicit system prompt that:
- Lists the assistant's capabilities (explain, debug, optimise, answer questions, write code).
- Explicitly forbids asking for file uploads.
- Instructs the model to always use markdown code fences.

---

## Setup

```bash
# 1. Clone / unzip the project
cd AI-Code-Assistant

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and paste your OpenRouter key

# 5. Run
streamlit run app/ui/app.py
```

---

## Usage

- **Just type a question** — no file upload needed:
  - `"Explain what this does: [paste code]"`
  - `"Fix the bug in: [paste code]"`
  - `"How do I implement a LRU cache in Python?"`

- **Optional RAG mode** — open the sidebar expander "Index snippet for deeper search", paste a large codebase, then ask questions about it.

---

## Why CodeBERT + FAISS?

CodeBERT (`microsoft/codebert-base`) is pre-trained on code and natural language pairs, making its 768-dimensional embeddings ideal for semantic code search. FAISS provides fast approximate nearest-neighbour search over these embeddings. Together they power the optional RAG mode for large codebase Q&A.

For most conversational use (explaining, debugging, writing code), the LLM-only path using DeepSeek-Coder is sufficient and faster.
