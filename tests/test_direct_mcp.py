#!/usr/bin/env python3
"""Test MCP servers directly to check responsiveness."""

import asyncio
import json
from mcp import MultiServerClient

async def test_servers():
    """Test basic MCP server calls."""
    servers = {
        "solar": {
            "command": ["python", "mcp/solar_facilities_server.py"],
            "tool": "GetSolarFacilitiesByCountry",
            "args": {"country": "Brazil", "limit": 5}
        },
        "gist": {
            "command": ["python", "mcp/gist_server.py"],
            "tool": "GetGistCompanies",
            "args": {"limit": 5}
        },
        "brazilian-admin": {
            "command": ["python", "mcp/brazilian_admin_server.py"],
            "tool": "GetTopBrazilianCitiesByPopulation",
            "args": {"top_n": 5}
        }
    }

    for server_name, config in servers.items():
        print(f"\nTesting {server_name} server...")
        try:
            client = MultiServerClient()
            await client.connect_to_server(server_name, config["command"])

            # Test the tool
            result = await client.call_tool(server_name, config["tool"], config["args"])
            print(f"✅ {server_name}: Success")
            print(f"   Result keys: {list(result.keys())[:5] if isinstance(result, dict) else type(result)}")

            await client.disconnect_from_server(server_name)
        except Exception as e:
            print(f"❌ {server_name}: Failed - {str(e)[:100]}")

    print("\nDone testing servers directly.")

if __name__ == "__main__":
    asyncio.run(test_servers())