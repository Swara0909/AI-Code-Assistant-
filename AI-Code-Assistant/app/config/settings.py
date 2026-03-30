import os
from pathlib import Path
from dotenv import load_dotenv

# ── Base Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
SQLITE_DB_PATH = DATA_DIR / "chat.db"

# Ensure project-level .env is loaded regardless of launch directory.
load_dotenv(BASE_DIR / ".env")

# ── OpenRouter / DeepSeek ───────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
OPENROUTER_VERIFY_SSL = os.getenv("OPENROUTER_VERIFY_SSL", "true").strip().lower() in {
	"1",
	"true",
	"yes",
	"on",
}
OPENROUTER_CA_BUNDLE = os.getenv("OPENROUTER_CA_BUNDLE", "").strip()

# ── LLM Params ──────────────────────────────────────────────
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2048

# ── Embeddings (CodeBERT) ───────────────────────────────────
CODEBERT_MODEL = "microsoft/codebert-base"
EMBEDDING_DIM = 768
EMBEDDING_DEVICE = "cpu"

# ── FAISS / Retrieval ───────────────────────────────────────
RETRIEVER_K = 4
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ── Memory ──────────────────────────────────────────────────
# FIX: MEMORY_WINDOW_K controls rolling context window.
# output_key must be "text" for LLMChain and "answer" for ConversationalRetrievalChain.
MEMORY_WINDOW_K = 6
SESSION_ID_DEFAULT = "default_session"

# ── UI ──────────────────────────────────────────────────────
APP_TITLE = "AI Code Assistant"
APP_ICON = "🤖"
