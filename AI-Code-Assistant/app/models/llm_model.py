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


class MockLLM:
    """Mock LLM for demo purposes when API key is missing."""
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="This is a demo response. Please set your OPENROUTER_API_KEY in the .env file to get real AI responses.")

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="This is a demo response. Please set your OPENROUTER_API_KEY in the .env file to get real AI responses.")


def get_llm():
    if not OPENROUTER_API_KEY:
        print("⚠️  OPENROUTER_API_KEY is missing. Using mock LLM for demo.")
        return MockLLM()

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
