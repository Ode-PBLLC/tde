"""Shared retry/backoff helpers for LLM SDK calls.

These helpers centralise exponential backoff with jitter for rate-limited or
transient failures when interacting with OpenAI or Anthropic. They deliberately
avoid importing heavy SDK symbols at module import time so they can be shared by
the standalone MCP server processes.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")

_LOGGER = logging.getLogger("llm.retry")


def _default_should_retry(error: Exception) -> bool:
    """Heuristic to decide if an exception is retryable."""

    lower = str(error).lower()

    # Fast exit for common non-retryable cases.
    non_retry_tokens = (
        "context length",
        "maximum context length",
        "invalid api key",
        "invalid_request",
        "does not exist",
        "unsupported",
        "bad request",
    )
    if any(token in lower for token in non_retry_tokens):
        return False

    # HTTP-style status codes exposed by both SDKs.
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int) and status in {408, 409, 429, 500, 502, 503, 504}:
        return True

    # Anthropic overload surfaces as `overloaded_error`.
    if getattr(error, "type", "").lower() == "overloaded_error":
        return True

    retry_tokens = (
        "rate limit",
        "rate-limit",
        "too many requests",
        "try again",
        "temporarily unavailable",
        "timeout",
        "overloaded",
    )
    if any(token in lower for token in retry_tokens):
        return True

    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return True

    return False


async def call_llm_with_retries(
    call: Callable[[], Awaitable[T]] | Callable[[], T],
    *,
    is_async: bool = False,
    max_attempts: int = 3,
    base_delay: float = 0.4,
    max_delay: float = 2.0,
    jitter: float = 0.2,
    timeout: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    classify: Optional[Callable[[Exception], bool]] = None,
    provider: Optional[str] = None,
) -> T:
    """Execute an LLM call with exponential backoff retries.

    Args:
        call: Zero-argument callable returning either the result or an awaitable
            that resolves to the result.
        is_async: When True, `call` is expected to return an awaitable and will
            be awaited directly. Otherwise, it is executed in a background
            thread to avoid blocking the event loop.
        max_attempts: Maximum number of attempts (initial try + retries).
        base_delay: Initial delay in seconds used for backoff.
        max_delay: Upper bound on the sleep between attempts.
        jitter: Fractional jitter applied (+/-) to each delay.
        timeout: Optional wall-clock budget in seconds across all attempts.
        logger: Optional logger to emit retry events (defaults to module logger).
        classify: Optional predicate to decide if an exception is retryable.
        provider: Identifier used in log messages for context.

    Raises:
        The last exception from `call` if retries are exhausted or the error is
        deemed non-retryable.
    """

    attempt = 0
    start = time.monotonic()
    log = logger or _LOGGER
    classifier = classify or _default_should_retry
    label = provider or "llm"

    while True:
        attempt += 1
        try:
            if is_async:
                result = await call()  # type: ignore[arg-type]
            else:
                result = await asyncio.to_thread(call)
            return result
        except Exception as exc:  # pragma: no cover - defensive path
            if attempt >= max_attempts or not classifier(exc):
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                delay *= 1 + random.uniform(-jitter, jitter)
                delay = max(delay, 0.05)

            if timeout is not None:
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise
                delay = min(delay, max(0.05, remaining))

            log.warning(
                "Retrying %s call after %.2fs (attempt %s/%s): %s",
                label,
                delay,
                attempt,
                max_attempts,
                exc,
            )

            await asyncio.sleep(delay)


def call_llm_with_retries_sync(
    call: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.4,
    max_delay: float = 2.0,
    jitter: float = 0.2,
    timeout: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    classify: Optional[Callable[[Exception], bool]] = None,
    provider: Optional[str] = None,
) -> T:
    """Synchronous variant for contexts that cannot await.

    Intended for early-stage helpers that still run outside the async pipeline
    (e.g., query enrichment during request pre-processing).
    """

    attempt = 0
    start = time.monotonic()
    log = logger or _LOGGER
    classifier = classify or _default_should_retry
    label = provider or "llm"

    while True:
        attempt += 1
        try:
            return call()
        except Exception as exc:  # pragma: no cover - defensive path
            if attempt >= max_attempts or not classifier(exc):
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                delay *= 1 + random.uniform(-jitter, jitter)
                delay = max(delay, 0.05)

            if timeout is not None:
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise
                delay = min(delay, max(0.05, remaining))

            log.warning(
                "Retrying %s call after %.2fs (attempt %s/%s): %s",
                label,
                delay,
                attempt,
                max_attempts,
                exc,
            )

            time.sleep(delay)
