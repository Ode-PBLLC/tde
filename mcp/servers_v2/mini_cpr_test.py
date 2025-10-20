# mini_bound_methods.py  (⚠️ do NOT use: from __future__ import annotations)
import json
import inspect, typing
from typing import Annotated, Dict, Any
from fastmcp import FastMCP

class CPRMini:
    def __init__(self):
        self.mcp = FastMCP("cpr-mini-bound")
        # Register BOUND methods (no decorator on the defs)
        self.mcp.tool(self.describe_capabilities)
        self.mcp.tool(self.query_support)

    def describe_capabilities(
        self,
        format: Annotated[str, "Output format: 'json' or 'text'"] = "json",
    ) -> str:
        """Describe the CPR KG dataset, provenance, and key tools."""
        payload = {"name": "cpr", "tools": ["describe_capabilities", "query_support"]}
        return json.dumps(payload) if format == "json" else str(payload)

    def query_support(
        self,
        query: Annotated[str, "User question to evaluate for CPR KG suitability"],
        context: Annotated[Dict[str, Any], "Optional orchestrator context"] = None,
    ) -> str:
        """Decide if the CPR knowledge graph should handle this query."""
        if context is None:
            context = {}
        return json.dumps({
            "server": "cpr",
            "query": query,
            "supported": True,
            "reasons": ["demo"],
        })

if __name__ == "__main__":
    srv = CPRMini()

    # Quick local introspection (helps debug missing params/docs)
    for fn in [srv.describe_capabilities, srv.query_support]:
        print(fn.__name__, inspect.signature(fn))
        print("  hints:", typing.get_type_hints(fn, include_extras=True))
        print("  doc  :", (fn.__doc__ or "").strip(), "\n")

    # Start the MCP server if you want to exercise it with your test harness
    # srv.mcp.run()
