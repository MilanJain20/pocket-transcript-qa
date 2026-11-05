"""Chat-model interface used by the grounding check."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Return a JSON object for a system/user prompt. Implementations must not log secrets."""

    def complete_json(self, system: str, user: str) -> dict[str, object]:
        """Call the provider and parse one JSON object."""
        ...
