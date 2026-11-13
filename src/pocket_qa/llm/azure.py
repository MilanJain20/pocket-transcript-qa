"""Azure OpenAI claim grounding through LangChain."""

from __future__ import annotations

import os
import ssl
from pathlib import Path

import httpx

from pocket_qa.llm.parsing import ModelOutputError, parse_json_object
from pocket_qa.paths import repo_root


class AzureOpenAIClient:
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        timeout: float = 45.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        try:
            from dotenv import dotenv_values
        except ImportError as exc:
            raise RuntimeError('Azure provider requires pip install -e ".[azure]".') from exc

        config = dotenv_values(repo_root() / ".env")

        def setting(name: str, default: str = "") -> str:
            return os.environ.get(name, config.get(name) or default)

        self.endpoint = (endpoint if endpoint is not None else setting("AZURE_OPENAI_ENDPOINT")).rstrip("/")
        self.api_key = api_key if api_key is not None else setting("AZURE_OPENAI_API_KEY")
        self.deployment = deployment if deployment is not None else setting("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = api_version if api_version is not None else setting("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.ca_bundle = setting("AZURE_OPENAI_CA_BUNDLE")
        self.timeout = timeout
        self._transport = transport

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        if not self.api_key or not self.endpoint or not self.deployment:
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "and AZURE_OPENAI_DEPLOYMENT. Use provider=offline to run without secrets."
            )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:
            raise RuntimeError('Azure provider requires pip install -e ".[azure]".') from exc

        verify: ssl.SSLContext | bool = True
        if self.ca_bundle:
            path = Path(self.ca_bundle).expanduser()
            if not path.is_absolute():
                path = repo_root() / path
            verify = ssl.create_default_context(cafile=str(path))
        with httpx.Client(timeout=self.timeout, transport=self._transport, verify=verify) as client:
            llm = AzureChatOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                azure_deployment=self.deployment,
                api_version=self.api_version,
                temperature=0,
                max_tokens=350,
                timeout=self.timeout,
                max_retries=0,
                http_client=client,
            )
            result = llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)],
                response_format={"type": "json_object"},
            )
        content = result.content
        if not isinstance(content, str):
            raise ModelOutputError("Model output was not text")
        return parse_json_object(content)
