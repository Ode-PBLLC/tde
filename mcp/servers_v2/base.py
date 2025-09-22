"""Shared helpers for MCP servers implementing `run_query`.

The mixin keeps the FastMCP code tiny while guaranteeing that every
server returns a payload conforming to :mod:`mcp.contracts_v2`.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import sys
from pathlib import Path

from fastmcp import FastMCP

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mcp.contracts_v2 import RunQueryResponse  # type: ignore
else:
    from ..contracts_v2 import RunQueryResponse


class RunQueryMixin:
    """Mixin that serialises :class:`RunQueryResponse` into MCP output."""

    mcp: FastMCP

    def _register_run_query_tool(self) -> None:
        @self.mcp.tool()
        def run_query(query: str, context: dict) -> str:  # type: ignore[misc]
            response = self.handle_run_query(query=query, context=context)
            if not isinstance(response, RunQueryResponse):
                raise TypeError(
                    "handle_run_query must return RunQueryResponse (received "
                    f"{type(response)!r})"
                )
            return json.dumps(response.model_dump(mode="json"))

    # To be implemented by subclasses --------------------------------------------------
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        raise NotImplementedError
