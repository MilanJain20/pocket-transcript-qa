"""Pull a JSON object out of a chat completion."""

from __future__ import annotations

import json
import re


class ModelOutputError(ValueError):
    """The model response did not contain a JSON object."""


def parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ModelOutputError("Model output was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("Model output JSON was not an object")
    return payload
