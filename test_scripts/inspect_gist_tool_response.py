"""Inspect the raw MCP content blocks for GetGistRiskByCategory."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _inspect_tool() -> None:
    params = StdioServerParameters(
        command="python",
        args=[str(ROOT / "mcp" / "servers_v2" / "gist_server_v2.py")],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep),
        },
    )

    async with stdio_client(params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "GetGistRiskByCategory",
                {
                    "risk_type": "flood",
                    "risk_level": "HIGH",
                    "limit": 5,
                },
            )

            print("Content blocks:")
            for index, block in enumerate(result.content, start=1):
                json_payload: Any = getattr(block, "json", None)
                text_payload: Any = getattr(block, "text", None)
                print(f"  Block {index} json: {json_payload!r}")
                print(f"  Block {index} text: {text_payload!r}")


def main() -> None:
    asyncio.run(_inspect_tool())


if __name__ == "__main__":
    main()
