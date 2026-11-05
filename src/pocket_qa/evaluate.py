"""Public entry point used by the CLI, API, MCP server, and benchmark."""

from __future__ import annotations

from pocket_qa.models import EvaluateRequest, EvaluationReport
from pocket_qa.pipeline import get_graph


def evaluate(request: EvaluateRequest) -> EvaluationReport:
    result = get_graph().invoke({"request": request.model_dump()})
    return EvaluationReport.model_validate(result["report"])
