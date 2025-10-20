"""Helpers for working with MCP tool responses."""

from __future__ import annotations

import json
from typing import Any, Dict


def mcp_payload_from_result(result: Any) -> Dict[str, Any]:
    """Return a Python dict from a FastMCP tool result.

    FastMCP may return either structured JSON blocks or plain text. This helper
    prefers JSON blocks (which some servers emit directly) and falls back to
    parsing the first text block when no JSON payload is available.
    """

    content = getattr(result, "content", None)
    if not content:
        raise ValueError("empty MCP result content")

    for block in content:
        obj = getattr(block, "json", None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            return obj
        # Some FastMCP content types expose a ``json`` method rather than a
        # materialised dict. Treat callable values as "no json payload" so we
        # can fall back to the text representation instead of raising.
        if callable(obj):
            continue
        raise TypeError("JSON block must be a dict")

    text = getattr(content[0], "text", None)
    if not text:
        raise ValueError("tool returned neither json nor text")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("tool returned non-JSON text") from exc
