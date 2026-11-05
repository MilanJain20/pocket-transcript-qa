"""HTTP client for the documented Pocket Public API.

This client is real: paths, query parameters, and auth match the public docs.
It does not run unless you pass an API key. This project ships no key, and
the benchmark never constructs this client.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from pocket_qa import __version__
from pocket_qa.pocket.errors import PocketAPIError, PocketAuthError, PocketNotFoundError
from pocket_qa.pocket.schemas import PocketListResponse, PocketRecording, PocketRecordingResponse

DOCUMENTED_BASE_URL = "https://public.heypocketai.com/api/v1"
USER_AGENT = f"pocket-transcript-qa/{__version__} (independent eval harness; not Pocket)"


class HttpPocketClient:
    """Calls https://public.heypocketai.com/api/v1 with a Bearer pk_ key.

    Not a mock. Requires credentials this repository does not contain.
    """

    is_mock = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("POCKET_API_KEY", "")
        self.base_url = (
            base_url if base_url is not None else os.environ.get("POCKET_API_BASE_URL", DOCUMENTED_BASE_URL)
        ).rstrip("/")
        self.timeout = timeout
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise PocketAuthError(
                "POCKET_API_KEY is not set. Use MockPocketClient, which is the default and does not call Pocket."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            response = client.request(method, url, headers=self._headers(), **kwargs)
        if response.status_code == 401:
            raise PocketAuthError("Pocket rejected the API key.")
        if response.status_code == 404:
            raise PocketNotFoundError(f"Pocket resource was not found: {path}")
        if response.status_code >= 400:
            message = response.text[:300]
            raise PocketAPIError(response.status_code, message)
        return response.json()

    def list_recordings(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        start_date: str | None = None,
        end_date: str | None = None,
        tag_ids: str | None = None,
    ) -> PocketListResponse:
        """GET /public/recordings. Dates are YYYY-MM-DD, interpreted as UTC by Pocket."""
        params: dict[str, Any] = {"page": page, "limit": limit}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if tag_ids:
            params["tag_ids"] = tag_ids
        body = self._request("GET", "/public/recordings", params=params)
        return PocketListResponse.model_validate(body)

    def get_recording(
        self,
        recording_id: str,
        *,
        include_transcript: bool = True,
        include_summarizations: bool = True,
        summarization_id: str | None = None,
    ) -> PocketRecording:
        """GET /public/recordings/{id}."""
        params: dict[str, Any] = {
            "include_transcript": include_transcript,
            "include_summarizations": include_summarizations,
        }
        if summarization_id:
            params["summarization_id"] = summarization_id
        body = self._request("GET", f"/public/recordings/{recording_id}", params=params)
        parsed = PocketRecordingResponse.model_validate(body)
        if parsed.data is None:
            raise PocketNotFoundError(f"Recording {recording_id} was not in the response.")
        return parsed.data

    def search(self, query: str, limit: int | None = None) -> dict[str, Any]:
        """POST /public/search.

        The public docs show a JSON body with `query`. A limit control is
        documented as default 8 and range 1..20, so `limit` is sent only when
        the caller sets it. The response schema was not fully expanded in the
        docs page this client was written against; the raw JSON is returned.
        """
        if not query.strip():
            raise ValueError("query is required")
        payload: dict[str, Any] = {"query": query}
        if limit is not None:
            if not 1 <= limit <= 20:
                raise ValueError("limit must be between 1 and 20")
            payload["limit"] = limit
        body = self._request("POST", "/public/search", json=payload)
        if isinstance(body, dict):
            return body
        raise PocketAPIError(200, "Search response was not a JSON object")

    def get_organization_recording(
        self,
        org_id: str,
        recording_id: str,
        *,
        include: str = "transcript,summary",
    ) -> PocketRecording:
        """GET /organization/{orgId}/recordings/{recordingId}.

        This org endpoint returns `summary.markdown` and a transcript array,
        which is a different shape from the personal recordings endpoint.
        """
        body = self._request(
            "GET",
            f"/organization/{org_id}/recordings/{recording_id}",
            params={"include": include},
        )
        return PocketRecording.model_validate(body)
