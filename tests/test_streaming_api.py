#!/usr/bin/env python3
"""Simple CLI to exercise the mcp_chat_v2 streaming interface."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _ensure_pythonpath() -> None:
    current = os.environ.get("PYTHONPATH")
    if current:
        if str(REPO_ROOT) not in current.split(":"):
            os.environ["PYTHONPATH"] = f"{REPO_ROOT}:{current}"
    else:
        os.environ["PYTHONPATH"] = str(REPO_ROOT)


async def stream_query_to_stdout(query: str) -> None:
    from mcp.mcp_chat_v2 import cleanup_v2_client, stream_query

    try:
        async for event in stream_query(query):
            print(f"data: {json.dumps(event, ensure_ascii=False)}", flush=True)
    finally:
        try:
            await cleanup_v2_client()
        except Exception as exc:  # pragma: no cover - cleanup best effort
            print(f"Warning: failed to cleanup MCP client: {exc}", file=sys.stderr)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream thinking and final response from mcp_chat_v2"
    )
    parser.add_argument("query", help="User query to send to the orchestrator")
    args = parser.parse_args()

    await stream_query_to_stdout(args.query)


if __name__ == "__main__":
    _ensure_pythonpath()
    asyncio.run(main())
