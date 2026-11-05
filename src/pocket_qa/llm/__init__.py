"""Pluggable model clients. Offline mode does not use this package."""

from pocket_qa.llm.factory import build_llm

__all__ = ["build_llm"]
