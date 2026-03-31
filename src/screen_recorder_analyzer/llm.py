"""
Multi-LLM abstraction layer for screen-recorder-analyzer.

Routes text-generation requests to OpenAI, Anthropic, or any LiteLLM-supported
provider based on the ``LLM_PROVIDER`` and ``LLM_MODEL`` environment variables.

Environment variables
---------------------
LLM_PROVIDER : str
    "openai" (default) | "anthropic" | "litellm"
LLM_MODEL : str
    Model name override.  Defaults per provider:
        openai    -> gpt-4o
        anthropic -> claude-sonnet-4-20250514
        litellm   -> gpt-4o  (passed through to litellm router)
OPENAI_API_KEY : str
    Required when LLM_PROVIDER is "openai".
ANTHROPIC_API_KEY : str
    Required when LLM_PROVIDER is "anthropic".
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Provider / model resolution
# ---------------------------------------------------------------------------

_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower().strip()

_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "litellm": "gpt-4o",
}

_MODEL = os.environ.get("LLM_MODEL", "").strip() or _DEFAULT_MODELS.get(_PROVIDER, "gpt-4o")


def get_provider() -> str:
    return _PROVIDER


def get_model() -> str:
    return _MODEL


# ---------------------------------------------------------------------------
# Core ask_llm helper
# ---------------------------------------------------------------------------

def ask_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.3,
    response_json: bool = False,
) -> str:
    """
    Send *prompt* (with optional *system* message) to the configured LLM
    provider and return the assistant's text response.

    Parameters
    ----------
    prompt : str
        The user message.
    system : str
        An optional system/instruction message.
    max_tokens : int
        Maximum tokens in the response.
    temperature : float
        Sampling temperature.
    response_json : bool
        When True, request a JSON-formatted response (supported by OpenAI and
        LiteLLM; for Anthropic, the instruction is placed in the system prompt).

    Returns
    -------
    str
        The model's text output.
    """
    if _PROVIDER == "openai":
        return _ask_openai(prompt, system, max_tokens, temperature, response_json)
    elif _PROVIDER == "anthropic":
        return _ask_anthropic(prompt, system, max_tokens, temperature, response_json)
    elif _PROVIDER == "litellm":
        return _ask_litellm(prompt, system, max_tokens, temperature, response_json)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {_PROVIDER!r}. Choose openai, anthropic, or litellm.")


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _ask_openai(prompt, system, max_tokens, temperature, response_json):
    import openai as _openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Export it or switch LLM_PROVIDER.")
    client = _openai.OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _ask_anthropic(prompt, system, max_tokens, temperature, response_json):
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    sys_text = system or ""
    if response_json:
        sys_text += "\n\nYou MUST respond with valid JSON only. No prose outside the JSON object."

    kwargs = dict(
        model=_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if sys_text.strip():
        kwargs["system"] = sys_text.strip()

    resp = client.messages.create(**kwargs)
    return resp.content[0].text


# ---------------------------------------------------------------------------
# LiteLLM backend (routes to any provider LiteLLM supports)
# ---------------------------------------------------------------------------

def _ask_litellm(prompt, system, max_tokens, temperature, response_json):
    import litellm

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = litellm.completion(**kwargs)
    return resp.choices[0].message.content
