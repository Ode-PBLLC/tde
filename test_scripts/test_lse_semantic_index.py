"""Debug script to diagnose LSE semantic search issues.

This script thoroughly tests the semantic search functionality and identifies
what could be preventing it from working.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_environment():
    """Check environment variables and dependencies."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    # Load .env file if it exists
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
        print(f"✓ OPENAI_API_KEY is set (length: {len(openai_key)})")
    else:
        print("✗ OPENAI_API_KEY is NOT set")
        print("  → Semantic index building will be skipped!")

    # Check OpenAI library
    try:
        from openai import OpenAI
        print("✓ openai library is installed")
    except ImportError as e:
        print(f"✗ openai library NOT installed: {e}")
        return False

    # Check numpy
    try:
        import numpy as np
        print(f"✓ numpy is installed (version: {np.__version__})")
    except ImportError as e:
        print(f"✗ numpy NOT installed: {e}")
        return False

    # Check sklearn
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("✓ sklearn is installed")
    except ImportError as e:
        print(f"✗ sklearn NOT installed: {e}")
        return False

    print()
    return True


def test_semantic_index_file():
    """Check if semantic index file exists."""
    print("=" * 60)
    print("SEMANTIC INDEX FILE CHECK")
    print("=" * 60)

    # Correct path: extras/ not static/extras/
    index_path = ROOT / "extras" / "lse_semantic_index.jsonl"

    if index_path.exists():
        print(f"✓ Semantic index file exists at: {index_path}")
        size = index_path.stat().st_size
        print(f"  File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

        # Count lines
        try:
            with index_path.open("r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            print(f"  Number of records: {lines}")

            # Check first record
            with index_path.open("r", encoding="utf-8") as f:
                first_line = f.readline()
                import json
                first_record = json.loads(first_line)
                embedding_dim = len(first_record.get("embedding", []))
                print(f"  Embedding dimension: {embedding_dim}")
        except Exception as e:
            print(f"  Warning: Could not analyze records: {e}")
    else:
        print(f"✗ Semantic index file NOT found at: {index_path}")
        print("  → Server will attempt to build it on initialization")

    print()


def test_server_initialization():
    """Test LSE server initialization and semantic search state."""
    print("=" * 60)
    print("SERVER INITIALIZATION CHECK")
    print("=" * 60)

    try:
        # Import and initialize server
        from mcp.servers_v2.lse_server_v2 import LSEServerV2

        print("Initializing LSEServerV2...")
        server = LSEServerV2()

        # Check OpenAI client
        if server._openai_client is not None:
            print("✓ OpenAI client is initialized")
        else:
            print("✗ OpenAI client is NOT initialized")
            print("  → Semantic search will not work!")
            print("  → Set OPENAI_API_KEY environment variable")

        # Check semantic records
        num_records = len(server._semantic_records)
        print(f"Semantic records loaded: {num_records}")
        if num_records > 0:
            print("✓ Semantic index has records")
        else:
            print("✗ Semantic index has NO records")
            print("  → Semantic search will always return empty results")

        # Check semantic matrix
        if server._semantic_matrix is not None:
            shape = server._semantic_matrix.shape
            print(f"✓ Semantic matrix exists: shape={shape}")
        else:
            print("✗ Semantic matrix is None")
            print("  → Semantic search will not work")

        # Check catalog
        num_sheets = len(server.catalog.sheets)
        print(f"Catalog sheets loaded: {num_sheets}")
        if num_sheets > 0:
            print("✓ LSE data is loaded")
        else:
            print("✗ LSE data is NOT loaded")

        print()
        return server

    except Exception as e:
        print(f"✗ Error initializing server: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_semantic_search(server):
    """Test semantic search functionality."""
    print("=" * 60)
    print("SEMANTIC SEARCH TEST")
    print("=" * 60)

    if server is None:
        print("Cannot test - server initialization failed")
        return

    # Test queries
    test_queries = [
        "climate adaptation",
        "greenhouse gas emissions",
        "renewable energy",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        # Direct semantic search
        results = server._semantic_search(query, limit=3)

        if results:
            print(f"✓ Semantic search returned {len(results)} results")
            for i, result in enumerate(results, 1):
                score = result.get("score", 0)
                title = result.get("title", "Unknown")
                module = result.get("module", "Unknown")
                snippet = result.get("snippet", "")[:100]
                print(f"  {i}. [{module}] {title}")
                print(f"     Score: {score:.4f}")
                print(f"     Snippet: {snippet}...")
        else:
            print("✗ Semantic search returned NO results")
            print("  This indicates the semantic index is not working")

    print()


def test_search_tool(server):
    """Test the SearchLSEContent tool."""
    print("=" * 60)
    print("SEARCH TOOL TEST")
    print("=" * 60)

    if server is None:
        print("Cannot test - server initialization failed")
        return

    # Find the SearchLSEContent tool
    search_tool = None
    try:
        tools = server.mcp.get_tools()
        for tool in tools:
            if tool.name == "SearchLSEContent":
                search_tool = tool
                break
    except Exception as e:
        print(f"Note: Could not enumerate tools: {e}")

    if search_tool:
        print(f"✓ Found SearchLSEContent tool")
    else:
        print("Note: Proceeding with direct method test")

    # Test search by simulating what the SearchLSEContent tool does
    query = "climate adaptation"
    print(f"\nSimulating SearchLSEContent tool with query: '{query}'")
    print("-" * 40)

    try:
        # This is exactly what the SearchLSEContent tool does
        # See lse_server_v2.py lines 1556-1573
        term = query

        # Step 1: Try semantic search
        semantic_hits = server._semantic_search(term, limit=5)

        if semantic_hits:
            print(f"✓ Tool would use SEMANTIC search")
            print(f"  Results count: {len(semantic_hits)}")
            print(f"  Method: semantic")

            # Show first result
            first = semantic_hits[0]
            print(f"\nFirst result:")
            print(f"  Title: {first.get('title', 'Unknown')}")
            print(f"  Module: {first.get('module', 'Unknown')}")
            print(f"  Score: {first.get('score', 0):.4f}")
            print(f"  Snippet: {first.get('snippet', '')[:100]}...")
        else:
            print(f"✗ Semantic search returned empty, falling back to catalog")
            # Step 2: Fallback to catalog search
            result = server.catalog.search(term, limit=5)
            print(f"  Results count: {result.get('count', 0)}")
            print(f"  Method: catalog (fallback)")

            # Show first result
            results = result.get("results", [])
            if results:
                first = results[0]
                print(f"\nFirst result:")
                print(f"  Title: {first.get('title', 'Unknown')}")
                print(f"  Module: {first.get('module', 'Unknown')}")

    except Exception as e:
        print(f"✗ Error during search simulation: {e}")
        import traceback
        traceback.print_exc()

    print()


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 60)
    print("LSE SEMANTIC SEARCH DIAGNOSTIC")
    print("=" * 60 + "\n")

    # Test 1: Environment
    env_ok = test_environment()

    # Test 2: Index file
    test_semantic_index_file()

    # Test 3: Server initialization
    server = test_server_initialization()

    # Test 4: Semantic search
    if server:
        test_semantic_search(server)
        test_search_tool(server)

    # Summary
    print("=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("\nFor semantic search to work, you need:")
    print("1. OPENAI_API_KEY environment variable set")
    print("2. Semantic index file built (or API key to build it)")
    print("3. OpenAI client successfully initialized")
    print("4. Non-empty semantic_records and semantic_matrix")
    print("\nIf semantic search is not working:")
    print("- Set OPENAI_API_KEY in your .env file")
    print("- Run: python test_scripts/build_lse_semantic_index.py")
    print("- Restart the server to reload the index")
    print()


if __name__ == "__main__":
    main()
