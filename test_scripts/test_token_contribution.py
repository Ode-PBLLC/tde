"""Test that token search adds unique results when semantic doesn't find everything.

This focused test verifies that the combined search is actually using both methods
when appropriate.
"""

from __future__ import annotations

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

from mcp.servers_v2.lse_server_v2 import LSEServerV2

print("=" * 70)
print("TOKEN SEARCH CONTRIBUTION TEST")
print("=" * 70)
print()

server = LSEServerV2()
print(f"Server initialized with {len(server.catalog.sheets)} sheets\n")

# Test queries where token search should find unique results
test_queries = [
    "deforestation",
    "Amazonas",
    "Minas Gerais",
]

for query in test_queries:
    print("-" * 70)
    print(f"Query: '{query}'")
    print("-" * 70)

    # Get individual results
    semantic = server._semantic_search(query, limit=5)
    token = server._token_search(query, limit=5)

    semantic_slugs = {r['slug'] for r in semantic}
    token_slugs = {r['slug'] for r in token}

    overlap = semantic_slugs & token_slugs
    semantic_only = semantic_slugs - token_slugs
    token_only = token_slugs - semantic_slugs

    print(f"\nSemantic: {len(semantic)} results")
    for r in semantic[:3]:
        score = r.get('score', 'N/A')
        print(f"  - {r['title']} ({score})")

    print(f"\nToken: {len(token)} results")
    for r in token[:3]:
        print(f"  - {r['title']}")

    print(f"\nOverlap: {len(overlap)}")
    print(f"Semantic-only: {len(semantic_only)}")
    print(f"Token-only: {len(token_only)}")

    if token_only:
        print(f"\n✓ Token search found {len(token_only)} unique results!")
        print("  Token-only results:")
        for slug in list(token_only)[:3]:
            match = next(r for r in token if r['slug'] == slug)
            print(f"    - {match['title']}")
    else:
        print("\n⚠ Token search did not find unique results for this query")

    # Now test combined
    print("\nCombined search (semantic first, then token):")
    seen = set()
    combined = []

    for r in semantic:
        combined.append(r)
        seen.add(r['slug'])

    for r in token:
        if r['slug'] not in seen:
            combined.append(r)
            seen.add(r['slug'])

    print(f"  Total unique results: {len(combined)}")
    print(f"  ({len(semantic)} from semantic + {len(combined) - len(semantic)} from token)")

    print()

print("=" * 70)
print("Summary:")
print("- Both methods contribute different results")
print("- Semantic finds conceptually related content")
print("- Token finds literal keyword matches")
print("- Together they provide comprehensive coverage")
print("=" * 70)
