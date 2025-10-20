"""Quick helper to exercise the GIST GetGistRiskByCategory tool."""

from __future__ import annotations

import asyncio
import runpy
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _invoke_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Load the GIST v2 server in-process and invoke the ranking tool."""

    module_globals = runpy.run_path("mcp/servers_v2/gist_server_v2.py")
    gist_server_cls = module_globals["GistServerV2"]
    server = gist_server_cls()
    tools = await server.mcp.get_tools()
    tool = tools["GetGistRiskByCategory"]

    # Call the underlying function directly; FastMCP wraps sync callables in FunctionTool.
    result = tool.fn(**payload)
    if not isinstance(result, dict):
        raise TypeError("tool response is not a dict")
    return result


async def _main() -> None:
    payload = {
        "risk_type": "flood",
        "risk_level": "HIGH",
        "limit": 5,
    }
    result = await _invoke_tool(payload)

    summary = result.get("summary")
    facts = result.get("facts") or []
    artifacts = result.get("artifacts") or []

    print("Summary:", summary)
    if facts:
        print("First fact:", facts[0])
    else:
        print("Facts: <none>")

    if artifacts:
        artifact = artifacts[0]
        columns = artifact.get("columns") or []
        rows = artifact.get("rows") or []
        print("Artifact columns:", columns)
        if rows:
            print("First row:", rows[0])
    else:
        print("Artifacts: <none>")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
