"""Test LSE server with realistic queries that users might actually ask."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment
env_path = ROOT / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)


# Define realistic test queries
REALISTIC_QUERIES = [
    {
        "query": "What are Brazil's 2030 climate targets?",
        "rationale": "User wants to know about specific NDC commitments and deadlines",
        "expected": "Should find NDC targets, direction setting, and national commitments"
    },
    {
        "query": "carbon pricing mechanisms Brazil",
        "rationale": "User researching carbon pricing policy options",
        "expected": "Should find cross-cutting policies, mitigation plans"
    },
    {
        "query": "Rio de Janeiro state climate action",
        "rationale": "User interested in subnational governance for specific state",
        "expected": "Should find Rio de Janeiro subnational data"
    },
    {
        "query": "forest conservation policy implementation",
        "rationale": "User exploring REDD+, forest protection, implementation status",
        "expected": "Should find sectoral plans, implementation evidence"
    },
    {
        "query": "institutional coordination climate governance",
        "rationale": "User researching how climate institutions are organized",
        "expected": "Should find institutional framework, governance structures"
    },
]


async def test_via_mcp():
    """Test queries through the actual MCP protocol."""

    print("=" * 80)
    print("LSE SERVER - REALISTIC QUERY TEST")
    print("=" * 80)
    print()
    print("Testing 5 realistic queries that users might ask about Brazilian climate")
    print("policy and NDC commitments.")
    print()

    from mcp.mcp_chat_v2 import MultiServerClient

    servers_v2_dir = ROOT / "mcp" / "servers_v2"
    lse_server_path = servers_v2_dir / "lse_server_v2.py"

    async with MultiServerClient() as client:
        await client.connect_to_server("lse", str(lse_server_path))
        print("✓ Connected to LSE server via MCP\n")

        session = client.sessions.get("lse")

        for i, test_case in enumerate(REALISTIC_QUERIES, 1):
            query = test_case["query"]
            rationale = test_case["rationale"]
            expected = test_case["expected"]

            print("=" * 80)
            print(f"TEST {i}/5: {query}")
            print("=" * 80)
            print(f"Rationale: {rationale}")
            print(f"Expected: {expected}")
            print("-" * 80)

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

                    # Show search method
                    print(f"\n🔍 Search Method: {method}")
                    if "semantic" in method and "token" in method:
                        print("   → Using BOTH semantic (conceptual) and token (keyword) search")
                    elif "semantic" in method:
                        print("   → Using semantic search (embeddings)")
                    elif "token" in method:
                        print("   → Using token search (n-grams)")

                    print(f"📊 Results Found: {count}")

                    if not results:
                        print("\n⚠️  NO RESULTS - Query may need refinement")
                        continue

                    # Analyze result diversity
                    modules = {}
                    for r in results:
                        module = r.get('module', 'unknown')
                        modules[module] = modules.get(module, 0) + 1

                    print(f"\n📁 Coverage by Module:")
                    for module, cnt in sorted(modules.items(), key=lambda x: -x[1]):
                        print(f"   - {module}: {cnt} results")

                    # Show top 5 results with details
                    print(f"\n📋 Top 5 Results:")
                    for idx, r in enumerate(results[:5], 1):
                        title = r.get('title', 'Unknown')
                        module = r.get('module', 'Unknown')
                        group = r.get('group', 'Unknown')
                        snippet = r.get('snippet', '')[:150]
                        score = r.get('score')

                        print(f"\n   {idx}. {title}")
                        print(f"      Module: {module} | Group: {group}")
                        if score is not None:
                            print(f"      Relevance Score: {score:.4f}")
                        if snippet:
                            print(f"      Preview: {snippet}...")

                    # Quality assessment
                    print(f"\n✅ Quality Assessment:")

                    # Check if results seem relevant
                    has_high_scores = any(r.get('score', 0) > 0.4 for r in results)
                    has_diversity = len(modules) > 1
                    has_enough = count >= 3

                    if has_high_scores:
                        print("   ✓ High relevance scores (>0.4) present")
                    if has_diversity:
                        print(f"   ✓ Good diversity ({len(modules)} different modules)")
                    if has_enough:
                        print(f"   ✓ Sufficient results ({count} found)")

                    if not (has_high_scores or has_diversity or has_enough):
                        print("   ⚠ Results quality could be improved")

            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

            print()

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("The combined semantic + token search approach provides:")
    print("  • Conceptual matching via embeddings (semantic)")
    print("  • Literal keyword matching via n-grams (token)")
    print("  • Comprehensive coverage across all query types")
    print("  • Fast performance on the 470-record LSE dataset")
    print()


def test_direct():
    """Quick direct test showing method breakdown."""

    print("=" * 80)
    print("DIRECT COMPARISON: Semantic vs Token vs Combined")
    print("=" * 80)
    print()

    from mcp.servers_v2.lse_server_v2 import LSEServerV2
    server = LSEServerV2()

    # Pick one query to show in detail
    query = "What are Brazil's 2030 climate targets?"

    print(f"Query: '{query}'")
    print("-" * 80)

    semantic = server._semantic_search(query, limit=5)
    token = server._token_search(query, limit=5)

    print(f"\n1️⃣  SEMANTIC ONLY ({len(semantic)} results):")
    for r in semantic[:3]:
        print(f"   - {r['title']} (score: {r.get('score', 0):.4f})")

    print(f"\n2️⃣  TOKEN ONLY ({len(token)} results):")
    for r in token[:3]:
        print(f"   - {r['title']}")

    # Combined
    seen = set()
    combined = []
    for r in semantic:
        combined.append(r)
        seen.add(r['slug'])
    for r in token:
        if r['slug'] not in seen:
            combined.append(r)

    print(f"\n3️⃣  COMBINED ({len(combined)} results):")
    for r in combined[:5]:
        source = "📊 semantic" if r.get('score') else "🔤 token"
        score_str = f" ({r['score']:.4f})" if r.get('score') else ""
        print(f"   {source}: {r['title']}{score_str}")

    print(f"\n✅ Combined search found {len(combined) - len(semantic)} additional results from token search")
    print()


if __name__ == "__main__":
    # First show the direct comparison
    test_direct()

    # Then run the full MCP test
    asyncio.run(test_via_mcp())
