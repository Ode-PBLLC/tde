"""Test LSE semantic search through actual MCP protocol.

This script tests whether semantic search works when the LSE server
is accessed through the MCP protocol (as it would be in production).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def test_lse_semantic_via_mcp():
    """Test LSE server semantic search through MCP protocol."""

    print("=" * 70)
    print("LSE SEMANTIC SEARCH TEST VIA MCP PROTOCOL")
    print("=" * 70)
    print()

    # Load .env file
    env_path = ROOT / ".env"
    if env_path.exists():
        print(f"✓ .env file exists at: {env_path}")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            print("✓ .env file loaded")
        except Exception as e:
            print(f"⚠ Could not load .env file: {e}")
    else:
        print(f"✗ .env file NOT found at: {env_path}")

    # Check OPENAI_API_KEY
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✓ OPENAI_API_KEY is set in parent process (length: {len(openai_key)})")
    else:
        print("✗ OPENAI_API_KEY is NOT set in parent process")

    print()
    print("-" * 70)
    print("Connecting to LSE server via MCP...")
    print("-" * 70)

    # Import MCP client components
    from mcp.mcp_chat_v2 import MultiServerClient

    servers_v2_dir = ROOT / "mcp" / "servers_v2"
    lse_server_path = servers_v2_dir / "lse_server_v2.py"

    if not lse_server_path.exists():
        print(f"✗ LSE server not found at: {lse_server_path}")
        return

    print(f"LSE server path: {lse_server_path}")

    async with MultiServerClient() as client:
        try:
            await client.connect_to_server("lse", str(lse_server_path))
            print("✓ Connected to LSE server via MCP")
        except Exception as e:
            print(f"✗ Failed to connect to LSE server: {e}")
            import traceback
            traceback.print_exc()
            return

        session = client.sessions.get("lse")
        if not session:
            print("✗ No session established")
            return

        print()
        print("-" * 70)
        print("Testing SearchLSEContent tool...")
        print("-" * 70)

        # Test multiple queries
        test_queries = [
            "climate adaptation",
            "greenhouse gas emissions",
            "renewable energy",
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("." * 60)

            try:
                # Call SearchLSEContent tool
                result = await session.call_tool(
                    "SearchLSEContent",
                    arguments={
                        "search_term": query,
                        "limit": 5,
                    }
                )

                # Parse the result
                if hasattr(result, 'content'):
                    content = result.content
                    if content and len(content) > 0:
                        text_content = content[0].text if hasattr(content[0], 'text') else str(content[0])

                        # Try to parse as JSON
                        try:
                            data = json.loads(text_content)

                            # Check what method was used
                            method = data.get("method", "unknown")
                            count = data.get("count", 0)

                            if method == "semantic":
                                print(f"✓ Using SEMANTIC search")
                                print(f"  Results: {count}")

                                # Show first result
                                results = data.get("results", [])
                                if results:
                                    first = results[0]
                                    print(f"  First result:")
                                    print(f"    Title: {first.get('title', 'Unknown')}")
                                    print(f"    Module: {first.get('module', 'Unknown')}")
                                    print(f"    Score: {first.get('score', 0):.4f}")
                            else:
                                print(f"✗ Using fallback catalog search (method: {method})")
                                print(f"  Results: {count}")
                                print(f"  → Semantic search is NOT working in MCP!")

                                # Show first result
                                results = data.get("results", [])
                                if results:
                                    first = results[0]
                                    print(f"  First result:")
                                    print(f"    Title: {first.get('title', 'Unknown')}")
                                    print(f"    Module: {first.get('module', 'Unknown')}")

                        except json.JSONDecodeError:
                            print(f"Response (raw): {text_content[:200]}...")
                    else:
                        print("✗ Empty content in result")
                else:
                    print(f"✗ Unexpected result format: {result}")

            except Exception as e:
                print(f"✗ Error calling SearchLSEContent: {e}")
                import traceback
                traceback.print_exc()

        print()
        print("-" * 70)
        print("Checking LSE server capabilities...")
        print("-" * 70)

        try:
            # Call describe_capabilities to see server state
            result = await session.call_tool("describe_capabilities", arguments={})

            if hasattr(result, 'content') and result.content:
                text_content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                try:
                    data = json.loads(text_content)
                    print(f"Dataset: {data.get('dataset', 'Unknown')}")
                    print(f"Modules: {len(data.get('modules', []))}")
                    print(f"Tools: {len(data.get('tools', []))}")

                    # Check if semantic search is mentioned in capabilities
                    capabilities_text = json.dumps(data, indent=2)
                    if 'semantic' in capabilities_text.lower():
                        print("✓ Semantic search mentioned in capabilities")
                    else:
                        print("⚠ Semantic search NOT mentioned in capabilities")

                except json.JSONDecodeError:
                    print(f"Response: {text_content[:300]}...")

        except Exception as e:
            print(f"Note: Could not call describe_capabilities: {e}")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()
    print("DIAGNOSIS:")
    print("----------")
    print("If the test shows 'Using fallback catalog search', then semantic")
    print("search is NOT working when the LSE server runs as an MCP subprocess.")
    print()
    print("Possible causes:")
    print("1. OPENAI_API_KEY not propagated to subprocess")
    print("2. OpenAI client initialization failing in subprocess")
    print("3. Semantic index file not being loaded in subprocess")
    print("4. _semantic_search() returning empty results")
    print()


if __name__ == "__main__":
    asyncio.run(test_lse_semantic_via_mcp())
