#!/usr/bin/env python3
"""Minimal multi-turn conversation smoke test for the Climate Policy Radar API.

This helper script exercises the session tracking logic by issuing two
sequential `/query/stream` calls:

1. Ask the assistant to remember a specific number.
2. Reuse the returned `session_id` to ask what number was just provided.

The script prints the `session_id` and short text snippets from both
responses so you can confirm that context is being preserved across turns.

Usage:
    python test_scripts/test_multiturn_minimal.py [BASE_URL] [NUMBER]

Defaults:
    BASE_URL = "http://localhost:8098"
    NUMBER = "42"
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional

import json
import requests

DEFAULT_BASE_URL = "http://localhost:8098"
DEFAULT_NUMBER = "42"
TEXT_MODULE_TYPES = {"text", "focus", "callout"}


def extract_text_snippets(modules: Iterable[dict], *, limit: int = 2) -> list[str]:
    """Return up to ``limit`` trimmed text snippets from response modules."""

    snippets: list[str] = []
    for module in modules:
        if module.get("type") not in TEXT_MODULE_TYPES:
            continue

        content = module.get("content")
        if isinstance(content, str) and content.strip():
            snippets.append(content.strip())

        texts = module.get("texts")
        if isinstance(texts, list):
            for text in texts:
                if isinstance(text, str) and text.strip():
                    snippets.append(text.strip())

        if len(snippets) >= limit:
            break

    return snippets[:limit]


def stream_request(
    base_url: str,
    query: str,
    *,
    conversation_id: Optional[str] = None,
) -> tuple[str, dict]:
    """Call `/query/stream`, returning the session_id and complete payload."""

    payload: dict[str, object] = {"query": query}
    if conversation_id:
        payload["conversation_id"] = conversation_id

    response = requests.post(
        f"{base_url.rstrip('/')}/query/stream",
        json=payload,
        timeout=120,
        stream=True,
    )
    response.raise_for_status()

    session_id: Optional[str] = conversation_id
    complete_payload: Optional[dict] = None

    try:
        for line_bytes in response.iter_lines(decode_unicode=True):
            if not line_bytes:
                continue
            if not line_bytes.startswith("data: "):
                continue
            try:
                event = json.loads(line_bytes[6:])
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            data = event.get("data", {})

            if event_type == "conversation_id" and isinstance(data, dict):
                conv_from_event = data.get("conversation_id")
                if isinstance(conv_from_event, str):
                    session_id = conv_from_event
            elif event_type == "complete":
                if isinstance(data, dict):
                    complete_payload = data
                break
            elif event_type == "error":
                message = data.get("message") if isinstance(data, dict) else None
                raise RuntimeError(f"Stream returned error event: {message}")
    finally:
        response.close()

    if not isinstance(session_id, str):
        raise RuntimeError("Stream did not return a conversation_id event; session tracking failed.")
    if not isinstance(complete_payload, dict):
        raise RuntimeError("Stream ended without a complete payload; check API logs.")

    return session_id, complete_payload


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    number = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NUMBER

    remember_query = f"Give me the three Brazilian states with the highest installed solar capacity. List them as A, B, and C, include approximate MW figures, and cite your sources."
    recall_query = "For the state you labeled B, explain the main drivers of its solar growth and point me to the cited data you used."

    print(f"Sending first turn without conversation_id -> {remember_query}")
    session_id, first_payload = stream_request(base_url, remember_query)

    first_snippets = extract_text_snippets(first_payload.get("modules", []))
    print(f"Session ID: {session_id}")
    if first_snippets:
        print("First response snippet(s):")
        for snippet in first_snippets:
            print(f"  - {snippet}")

    conversation_context = [
        {"role": "user", "content": remember_query},
    ]
    if first_snippets:
        conversation_context.append({"role": "assistant", "content": first_snippets[0]})

    print("\nContext that the follow-up turn will inherit:")
    for message in conversation_context:
        role = message["role"].capitalize()
        content = message["content"]
        print(f"  {role}: {content}")

    print("\nSending follow-up turn with conversation_id set")
    _, second_payload = stream_request(
        base_url,
        recall_query,
        conversation_id=session_id,
    )

    second_snippets = extract_text_snippets(second_payload.get("modules", []))
    print(f"Second response status: OK (conversation reused {session_id})")
    if second_snippets:
        print("Second response snippet(s):")
        for snippet in second_snippets:
            print(f"  - {snippet}")
    else:
        print("No text modules returned in second response; inspect raw output below.")

    print("\nFull second response payload (modules truncated to first entry for brevity):")
    trimmed_response = dict(second_payload)
    modules = second_payload.get("modules", [])
    trimmed_response["modules"] = modules[:1] if isinstance(modules, list) else modules
    print(trimmed_response)


if __name__ == "__main__":
    main()
