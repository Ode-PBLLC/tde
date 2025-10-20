#!/usr/bin/env python3
"""Run a query directly against the CPR MCP server."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure support_intent alias resolves regardless of import path
support_intent_module = importlib.import_module("mcp.servers_v2.support_intent")
sys.modules.setdefault("mcp.support_intent", support_intent_module)

from mcp.servers_v2.cpr_server_v2 import CPRServerV2


def _load_context(context_arg: str | None) -> dict:
    if not context_arg:
        return {}
    try:
        return json.loads(context_arg)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid context JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a run_query against the CPR server")
    parser.add_argument("query", help="User query to send to the CPR server")
    parser.add_argument(
        "--context",
        dest="context",
        default=None,
        help="Optional JSON object to include as the run_query context",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON response",
    )
    args = parser.parse_args()

    context = _load_context(args.context)

    server = CPRServerV2()
    response = server.handle_run_query(query=args.query, context=context)
    payload = response.model_dump(mode="json")

    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
