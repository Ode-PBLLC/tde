"""Verify `mcp_payload_from_result` handles callable JSON blocks."""

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

from mcp.utils_mcp import mcp_payload_from_result


async def _main() -> None:
    params = StdioServerParameters(
        command="python",
        args=[str(ROOT / "mcp" / "servers_v2" / "gist_server_v2.py")],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), os.environ.get("PYTHONPATH", "")]
            ).strip(os.pathsep),
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

            payload: dict[str, Any] = mcp_payload_from_result(result)
            print("Payload keys:", sorted(payload.keys()))
            print("Summary:", payload.get("summary"))
            companies = (payload.get("details") or {}).get("companies") or []
            if companies:
                print("Top company:", companies[0])


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
