"""Anthropic Messages API."""

from __future__ import annotations

import os

import httpx

from pocket_qa.llm.parsing import parse_json_object


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model if model is not None else os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Use provider=offline to run without secrets.")
        payload = {
            "model": self.model,
            "max_tokens": 512,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            response = client.post(f"{self.base_url}/v1/messages", json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        text = "".join(
            block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
        )
        return parse_json_object(text)
