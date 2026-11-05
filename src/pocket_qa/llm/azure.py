"""Azure OpenAI chat completions."""

from __future__ import annotations

import os

import httpx

from pocket_qa.llm.parsing import parse_json_object


class AzureOpenAIClient:
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = (
            endpoint if endpoint is not None else os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("AZURE_OPENAI_API_KEY", "")
        self.deployment = (
            deployment if deployment is not None else os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        )
        self.api_version = (
            api_version
            if api_version is not None
            else os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        )
        self.timeout = timeout
        self._transport = transport

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        if not self.api_key or not self.endpoint or not self.deployment:
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "and AZURE_OPENAI_DEPLOYMENT. Use provider=offline to run without secrets."
            )
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        payload = {
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"api-key": self.api_key}
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        content = body["choices"][0]["message"]["content"]
        return parse_json_object(content)
