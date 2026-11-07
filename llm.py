# llm.py
"""
Unified LLM provider. Single source of truth for all models, clients,
and fallback ordering. Raises LLMUnavailable when every provider fails —
callers MUST handle it.

Public API:
    generate_text(prompt, system_instruction="", max_tokens=None) -> str
    LLMUnavailable                                            # exception
    MODEL_GEMINI, MODEL_GROQ, MODEL_OPENROUTER                # model strings
    _gemini_client, _groq_client, _openrouter_client          # clients
                                                              (for chat_agent.py)
"""

import os
import logging
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """All configured providers failed. Callers must catch this."""


# ---- Keys & models --------------------------------------------------------
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

MODEL_GEMINI     = os.getenv("MODEL_GEMINI", "gemini-3.1-flash-lite")
MODEL_GROQ       = os.getenv("MODEL_GROQ", "openai/gpt-oss-120b")
MODEL_OPENROUTER = os.getenv("MODEL_OPENROUTER", "meta-llama/llama-3.3-70b-instruct")

# NOTE: default order = free providers first, paid last.
DEFAULT_ORDER    = "gemini,groq,openrouter"
LLM_ORDER        = [p.strip().lower() for p in os.getenv("LLM_ORDER", DEFAULT_ORDER).split(",") if p.strip()]
DEFAULT_MAX_TOKENS = max(128, int(os.getenv("LLM_MAX_TOKENS", "1024")))


# ---- Clients (constructed only when keys present) -------------------------
_gemini_client = None
_groq_client = None
_openrouter_client = None

if GOOGLE_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as exc:
        logger.warning("Could not init Gemini client: %s", exc)

if GROQ_API_KEY:
    try:
        _groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY,
                              timeout=30, max_retries=0)
    except Exception as exc:
        logger.warning("Could not init Groq client: %s", exc)

if OPENROUTER_API_KEY:
    try:
        _openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                                    timeout=30, max_retries=0)
    except Exception as exc:
        logger.warning("Could not init OpenRouter client: %s", exc)


# ---- Per-provider calls ---------------------------------------------------
@retry(retry=retry_if_exception_type(ServerError),
       stop=stop_after_attempt(2),
       wait=wait_exponential(multiplier=1, min=2, max=5),
       reraise=True)
def _call_gemini(prompt: str, system: str, max_tokens: int) -> str:
    if _gemini_client is None:
        raise RuntimeError("Gemini not configured")
    resp = _gemini_client.models.generate_content(
        model=MODEL_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=max_tokens,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Empty Gemini response")
    return text


def _openai_compatible(client, model: str, prompt: str, system: str, max_tokens: int) -> str:
    if client is None:
        raise RuntimeError(f"{model} client not configured")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"Empty {model} response")
    return text


def _call_groq(prompt: str, system: str, max_tokens: int) -> str:
    return _openai_compatible(_groq_client, MODEL_GROQ, prompt, system, max_tokens)


def _call_openrouter(prompt: str, system: str, max_tokens: int) -> str:
    return _openai_compatible(_openrouter_client, MODEL_OPENROUTER, prompt, system, max_tokens)


# ---- Public router --------------------------------------------------------
_DISPATCH = {
    "gemini":     _call_gemini,
    "groq":       _call_groq,
    "openrouter": _call_openrouter,
}


def generate_text(
    prompt: str,
    system_instruction: str = "",
    max_tokens: Optional[int] = None,
) -> str:
    """
    Try each provider in LLM_ORDER. Returns first successful text.
    Raises LLMUnavailable if every provider fails.
    """
    tokens = max_tokens or DEFAULT_MAX_TOKENS
    errors = []
    for provider in LLM_ORDER:
        fn = _DISPATCH.get(provider)
        if fn is None:
            continue
        try:
            return fn(prompt, system_instruction, tokens)
        except Exception as exc:
            logger.warning("LLM provider %s failed: %s", provider, exc)
            errors.append(f"{provider}: {exc}")

    raise LLMUnavailable(
        "All LLM providers failed: " + ("; ".join(errors) if errors else "none configured")
    )


# ---- Backward-compat aliases (legacy code importing these names) -----------
# Chat agent imports these directly; keeping them here means one client per
# provider for the whole process.
gemini_client     = _gemini_client
groq_client       = _groq_client
openrouter_client = _openrouter_client