"""Optional local cross-encoder for semantic claim support.

Not installed by default and never required by CI. Install the extra with
`pip install -e ".[local-nli]"`. The model id is a public DeBERTa-v3 xsmall
cross-encoder fine-tuned on NLI. The first call downloads weights unless
they are already on disk. Inference runs on CPU. Transcripts are not sent
to a hosted model API.

The benchmark default and CI stay on `provider=offline`. Select this path
with `--provider local-nli` or `--provider hybrid`, or set
`POCKET_QA_PROVIDER`. Override the checkpoint with `POCKET_QA_NLI_MODEL`
(a Hugging Face id or a local directory).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"

# Public model card order for cross-encoder/nli-deberta-v3-xsmall, used only
# when the loaded config does not name its three labels.
_FALLBACK_LABELS = ("contradicted", "entailed", "unsupported")
_CANONICAL = {
    "contradiction": "contradicted",
    "contradicted": "contradicted",
    "entailment": "entailed",
    "entailed": "entailed",
    "neutral": "unsupported",
    "unsupported": "unsupported",
}

_MODEL = None
_LABELS: tuple[str, str, str] | None = None


@dataclass(frozen=True)
class NliDistribution:
    """Three-way probabilities. Keys are contradicted, entailed, unsupported."""

    probabilities: dict[str, float]

    def prob(self, label: str) -> float:
        return float(self.probabilities[label])

    @property
    def label(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)


def model_id() -> str:
    return os.environ.get("POCKET_QA_NLI_MODEL", MODEL_ID).strip() or MODEL_ID


def available() -> bool:
    """True when the optional sentence-transformers extra imports."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def weights_cached() -> bool:
    """True when the checkpoint is on disk. Does not download."""
    if not available():
        return False
    target = model_id()
    if Path(target).exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    for filename in ("config.json", "model.safetensors", "pytorch_model.bin"):
        cached = try_to_load_from_cache(target, filename)
        if isinstance(cached, str):
            return True
    return False


def _require_extra() -> None:
    if not available():
        raise RuntimeError(
            "provider local-nli requires the optional local-nli extra "
            "(pip install -e \".[local-nli]\"). The default path is provider=offline, "
            "which needs no download and no API key."
        )


def _label_names(model: object) -> tuple[str, str, str]:
    config = getattr(getattr(model, "model", None), "config", None)
    mapping = getattr(config, "id2label", None)
    if isinstance(mapping, dict) and mapping:
        names: list[str] = []
        for index in range(len(mapping)):
            raw = mapping.get(index, mapping.get(str(index)))
            if not isinstance(raw, str):
                names = []
                break
            canonical = _CANONICAL.get(raw.lower().strip())
            if canonical is None:
                names = []
                break
            names.append(canonical)
        if len(names) == 3 and len(set(names)) == 3:
            return (names[0], names[1], names[2])
    return _FALLBACK_LABELS


def load_model() -> object:
    """Load the cross-encoder on CPU. Downloads weights on first use."""
    global _MODEL, _LABELS
    if _MODEL is not None:
        return _MODEL
    _require_extra()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import CrossEncoder

    _MODEL = CrossEncoder(model_id(), device="cpu")
    _LABELS = _label_names(_MODEL)
    return _MODEL


def _as_rows(raw: object) -> list[list[float]]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        raise RuntimeError(f"Unexpected NLI output type: {type(raw).__name__}")
    # A single pair comes back as one vector, not a batch of vectors.
    if raw and isinstance(raw[0], (int, float)):
        raw = [raw]
    rows: list[list[float]] = []
    for item in raw:
        if hasattr(item, "tolist"):
            item = item.tolist()
        rows.append([float(value) for value in item])
    return rows


def _probabilities(row: list[float]) -> list[float]:
    """Softmax over logits. If the row is already a probability vector, keep it."""
    if len(row) != 3:
        raise RuntimeError(f"Unexpected NLI output width: {len(row)}")
    total = sum(row)
    if all(value >= -1e-6 for value in row) and abs(total - 1.0) < 1e-3:
        cleaned = [max(0.0, value) for value in row]
        scale = sum(cleaned) or 1.0
        return [value / scale for value in cleaned]
    peak = max(row)
    exps = [math.exp(value - peak) for value in row]
    scale = sum(exps)
    return [value / scale for value in exps]


def _predict(model: object, pairs: list[tuple[str, str]]) -> list[list[float]]:
    """Return one probability row per pair.

    Cross-encoder NLI heads emit three logits. Some sentence-transformers
    versions apply sigmoid unless an identity activation is passed. Softmax
    of sigmoid outputs is not a distribution over the three labels, so
    identity is requested when the installed predict() accepts it.
    """
    batch = [(premise, hypothesis) for premise, hypothesis in pairs]
    predict = model.predict  # type: ignore[attr-defined]
    try:
        # Identity then softmax. Some checkpoints set a sigmoid activation, which
        # is the wrong conversion for a three-way NLI head.
        raw = predict(
            batch,
            show_progress_bar=False,
            activation_fn=lambda logits: logits,
            apply_softmax=True,
        )
    except TypeError:
        raw = predict(batch, show_progress_bar=False)
    return [_probabilities(row) for row in _as_rows(raw)]


def score_pairs(pairs: list[tuple[str, str]]) -> list[NliDistribution]:
    """Score (premise, hypothesis) pairs. Empty input returns an empty list."""
    if not pairs:
        return []
    model = load_model()
    labels = _LABELS or _label_names(model)
    rows = _predict(model, pairs)
    if len(rows) != len(pairs):
        raise RuntimeError(f"NLI returned {len(rows)} rows for {len(pairs)} pairs.")
    scored: list[NliDistribution] = []
    for row in rows:
        probabilities = {label: row[index] for index, label in enumerate(labels)}
        scored.append(NliDistribution(probabilities=probabilities))
    return scored


def score_nli(premise: str, hypothesis: str) -> tuple[str, float]:
    """Return (argmax label, probability of that label) for one pair."""
    distribution = score_pairs([(premise, hypothesis)])[0]
    label = distribution.label
    return label, distribution.prob(label)
