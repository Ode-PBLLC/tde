"""Test LSE combined semantic + token search implementation.

This script tests the new dual search strategy that combines:
1. Semantic search (embedding-based conceptual matching)
2. Token search (n-gram based literal keyword matching)
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


def test_direct_methods():
    """Test the search methods directly on the server instance."""

    print("=" * 70)
    print("DIRECT METHOD TESTS")
    print("=" * 70)
    print()

    # Load .env
    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            print("✓ .env file loaded")
        except Exception as e:
            print(f"⚠ Could not load .env file: {e}")

    # Initialize server
    from mcp.servers_v2.lse_server_v2 import LSEServerV2
    print("Initializing LSE server...")
    server = LSEServerV2()
    print(f"✓ Server initialized")
    print(f"  Catalog sheets: {len(server.catalog.sheets)}")
    print(f"  Semantic records: {len(server._semantic_records)}")
    print()

    # Test queries
    test_cases = [
        {
            "query": "climate adaptation",
            "description": "Conceptual query (should match semantically)"
        },
        {
            "query": "São Paulo emissions",
            "description": "Specific entity + keyword (should match both ways)"
        },
        {
            "query": "renewable energy targets",
            "description": "Mixed conceptual + literal"
        },
        {
            "query": "deforestation",
            "description": "Single keyword"
        },
    ]

    for test_case in test_cases:
        query = test_case["query"]
        description = test_case["description"]

        print("-" * 70)
        print(f"Query: '{query}'")
        print(f"Type: {description}")
        print("." * 70)

        # Test semantic search
        semantic_results = server._semantic_search(query, limit=5)
        print(f"\nSemantic search: {len(semantic_results)} results")
        if semantic_results:
            for i, r in enumerate(semantic_results[:3], 1):
                score = r.get('score', 0)
                print(f"  {i}. {r['title']} (score: {score:.4f})")

        # Test token search
        token_results = server._token_search(query, limit=5)
        print(f"\nToken search: {len(token_results)} results")
        if token_results:
            for i, r in enumerate(token_results[:3], 1):
                print(f"  {i}. {r['title']}")

        # Check overlap
        semantic_slugs = {r['slug'] for r in semantic_results}
        token_slugs = {r['slug'] for r in token_results}
        overlap = semantic_slugs & token_slugs

        print(f"\nOverlap: {len(overlap)} results in both")
        print(f"Semantic-only: {len(semantic_slugs - token_slugs)}")
        print(f"Token-only: {len(token_slugs - semantic_slugs)}")
        print()

    print()


async def test_via_mcp():
    """Test SearchLSEContent tool via MCP protocol."""

    print("=" * 70)
    print("MCP PROTOCOL TEST")
    print("=" * 70)
    print()

    # Load .env
    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except Exception:
            pass

    from mcp.mcp_chat_v2 import MultiServerClient

    servers_v2_dir = ROOT / "mcp" / "servers_v2"
    lse_server_path = servers_v2_dir / "lse_server_v2.py"

    async with MultiServerClient() as client:
        await client.connect_to_server("lse", str(lse_server_path))
        print("✓ Connected to LSE server via MCP")
        print()

        session = client.sessions.get("lse")

        # Test queries
        test_queries = [
            ("climate adaptation", "Should use semantic"),
            ("São Paulo", "Should use token for entity name"),
            ("greenhouse gas reduction targets", "Should use both"),
        ]

        for query, expectation in test_queries:
            print("-" * 70)
            print(f"Query: '{query}'")
            print(f"Expected: {expectation}")
            print("." * 70)

            try:
                result = await session.call_tool(
                    "SearchLSEContent",
                    arguments={"search_term": query, "limit": 10}
                )

                if hasattr(result, 'content') and result.content:
                    text_content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                    data = json.loads(text_content)

                    method = data.get("method", "unknown")
                    count = data.get("count", 0)
                    results = data.get("results", [])

                    print(f"\nMethod: {method}")
                    print(f"Results: {count}")

                    # Show method breakdown
                    if "semantic" in method and "token" in method:
                        print("✓ Using BOTH semantic and token search")
                    elif "semantic" in method:
                        print("✓ Using semantic search")
                    elif "token" in method:
                        print("✓ Using token search")
                    else:
                        print(f"⚠ Using fallback method: {method}")

                    # Show top results
                    if results:
                        print(f"\nTop 3 results:")
                        for i, r in enumerate(results[:3], 1):
                            title = r.get('title', 'Unknown')
                            module = r.get('module', 'Unknown')
                            score = r.get('score')
                            score_str = f" (score: {score:.4f})" if score is not None else ""
                            print(f"  {i}. [{module}] {title}{score_str}")

            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()

            print()

    print()


def main():
    """Run all tests."""

    print("\n" + "=" * 70)
    print("LSE COMBINED SEARCH TEST")
    print("=" * 70 + "\n")

    # Test 1: Direct method tests
    test_direct_methods()

    # Test 2: MCP protocol test
    asyncio.run(test_via_mcp())

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("- Semantic search uses embeddings for conceptual matching")
    print("- Token search uses n-grams for literal keyword matching")
    print("- SearchLSEContent combines both: semantic first, then token")
    print("- This provides best coverage for all query types")
    print()


if __name__ == "__main__":
    main()
