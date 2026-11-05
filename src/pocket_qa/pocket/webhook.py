"""Verify Pocket webhook signatures and remember recent deliveries.

Signature rule, from the public docs: HMAC-SHA256 of `{timestamp}.{rawBody}`
with the webhook secret, hex-encoded, compared to `X-HeyPocket-Signature`.
`X-HeyPocket-Timestamp` is unix time in milliseconds.

Pocket's docs do not specify a replay window. This receiver rejects timestamps
more than five minutes from the local clock. Delivery is at-least-once, so a
repeated raw body returns the stored evaluation instead of running twice.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import OrderedDict

from pocket_qa.models import EvaluationReport

MAX_SKEW_MS = 5 * 60 * 1000
_CACHE: OrderedDict[str, EvaluationReport] = OrderedDict()
_CACHE_LIMIT = 100


class WebhookVerificationError(ValueError):
    """The delivery did not pass signature or freshness checks."""


def sign_payload(secret: str, timestamp_ms: str, body: bytes) -> str:
    message = timestamp_ms.encode("utf-8") + b"." + body
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_signature(
    secret: str | None,
    timestamp_ms: str | None,
    body: bytes,
    signature: str | None,
    *,
    allow_unsigned: bool = False,
    now_ms: int | None = None,
) -> None:
    if not secret:
        if allow_unsigned:
            return
        raise WebhookVerificationError(
            "POCKET_WEBHOOK_SECRET is not set. Refusing an unsigned webhook. "
            "Set the secret, or set POCKET_QA_ALLOW_UNSIGNED_WEBHOOKS=1 for a local demo."
        )
    if not timestamp_ms or not signature:
        raise WebhookVerificationError("Missing X-HeyPocket-Timestamp or X-HeyPocket-Signature.")
    if not timestamp_ms.isdigit():
        raise WebhookVerificationError("X-HeyPocket-Timestamp must be unix milliseconds.")
    current = int(time.time() * 1000) if now_ms is None else now_ms
    if abs(current - int(timestamp_ms)) > MAX_SKEW_MS:
        raise WebhookVerificationError("Webhook timestamp is outside the 5 minute window.")
    expected = sign_payload(secret, timestamp_ms, body)
    if not hmac.compare_digest(expected, signature.strip()):
        raise WebhookVerificationError("Webhook signature did not match.")


def body_key(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def remember(key: str, report: EvaluationReport) -> None:
    _CACHE[key] = report
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_LIMIT:
        _CACHE.popitem(last=False)


def recall(key: str) -> EvaluationReport | None:
    report = _CACHE.get(key)
    if report is not None:
        _CACHE.move_to_end(key)
    return report


def recent(limit: int = 20) -> list[EvaluationReport]:
    items = list(_CACHE.values())
    return list(reversed(items[-limit:]))


def clear_cache() -> None:
    _CACHE.clear()
