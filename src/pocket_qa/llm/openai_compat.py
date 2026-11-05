"""OpenAI-compatible chat completions, including any proxy that speaks that API."""

from __future__ import annotations

import os

import httpx

from pocket_qa.llm.parsing import parse_json_object


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model if model is not None else os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self._transport = transport

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use provider=offline to run without secrets.")
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        content = body["choices"][0]["message"]["content"]
        return parse_json_object(content)
