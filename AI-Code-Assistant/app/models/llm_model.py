import httpx

from langchain_openai import ChatOpenAI
from app.config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    OPENROUTER_VERIFY_SSL,
    OPENROUTER_CA_BUNDLE,
)


def get_llm() -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "❌  OPENROUTER_API_KEY is missing.\n"
            "Set it in your .env file:\n"
            "  OPENROUTER_API_KEY=sk-or-..."
        )

    http_client = None
    if not OPENROUTER_VERIFY_SSL:
        http_client = httpx.Client(verify=False)
    elif OPENROUTER_CA_BUNDLE:
        http_client = httpx.Client(verify=OPENROUTER_CA_BUNDLE)

    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        http_client=http_client,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Code Assistant",
        },
    )
