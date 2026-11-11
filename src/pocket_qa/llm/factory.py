"""Build a chat client from a provider name. Offline does not come through here."""

from __future__ import annotations

from pocket_qa.llm.anthropic import AnthropicClient
from pocket_qa.llm.azure import AzureOpenAIClient
from pocket_qa.llm.base import LLMClient
from pocket_qa.llm.openai_compat import OpenAICompatibleClient


def build_llm(provider: str) -> LLMClient:
    if provider == "openai":
        return OpenAICompatibleClient()
    if provider == "azure":
        return AzureOpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "offline":
        raise RuntimeError("The offline provider does not call a model.")
    if provider in {"local-nli", "hybrid"}:
        raise RuntimeError(f"{provider} is invoked from the summary checker, not the chat factory.")
    raise ValueError(f"Unknown provider {provider!r}")
