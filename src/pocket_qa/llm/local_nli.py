"""Optional local cross-encoder. Not installed by default and never required.

The default extra is `sentence-transformers`. The model id is a public
cross-encoder fine-tuned on NLI. The first call downloads weights, so the
benchmark and CI do not select this provider.
"""

from __future__ import annotations

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
_MODEL = None


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _model() -> object:
    global _MODEL
    if _MODEL is None:
        if not available():
            raise RuntimeError(
                "provider=local-nli requires the optional local-nli extra "
                "(sentence-transformers). The default path is provider=offline, which needs no download."
            )
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder(MODEL_ID)
    return _MODEL


def score_nli(premise: str, hypothesis: str) -> tuple[str, float]:
    """Return (label, score) using the cross-encoder's three-way NLI head.

    sentence-transformers NLI models emit logits ordered contradiction, entailment, neutral
    for `cross-encoder/nli-deberta-v3-xsmall`.
    """
    model = _model()
    logits = model.predict([(premise, hypothesis)])  # type: ignore[attr-defined]
    row = list(logits[0])
    if len(row) != 3:
        raise RuntimeError(f"Unexpected NLI output width: {len(row)}")
    # Stable softmax without numpy.
    peak = max(row)
    exps = [pow(2.718281828, value - peak) for value in row]
    total = sum(exps)
    probs = [value / total for value in exps]
    labels = ("contradicted", "entailed", "unsupported")
    index = max(range(3), key=lambda item: probs[item])
    return labels[index], float(probs[index])
