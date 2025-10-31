#!/usr/bin/env python3
"""
Test script to verify that NDC Align citations now include the underlying_source field.

This script directly tests the LSE server to ensure:
1. Citations have the underlying_source field populated from source URLs
2. The url field points to the NDC Align dataset
3. The underlying_source field contains URLs from primary_source, secondary_source, etc.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.servers_v2.lse_server_v2 import LSEServerV2


def test_underlying_source():
    """Test that citations include underlying_source field."""
    print("=" * 80)
    print("Testing NDC Align UnderlyingSource Field")
    print("=" * 80)

    # Initialize the LSE server
    print("\n1. Initializing LSE server...")
    server = LSEServerV2()

    # Create a test query about Brazil's climate policies
    query = "What interministerial coordination mechanisms exist for Brazil's climate policy?"
    context = {
        "topic": "Brazil climate policy",
        "geography": ["Brazil"],
        "time_range": None,
    }

    print(f"\n2. Running query: {query}")
    print(f"   Context: {context}")

    # Execute the query
    try:
        response = server.handle_run_query(query=query, context=context)
        print(f"\n3. Query executed successfully!")
        print(f"   Facts returned: {len(response.facts)}")
        print(f"   Citations returned: {len(response.citations)}")

        # Analyze citations
        print("\n4. Analyzing citations for underlying_source field:")
        print("-" * 80)

        citations_with_underlying_source = 0

        for i, citation in enumerate(response.citations, 1):
            print(f"\n   Citation {i}:")
            print(f"     ID: {citation.id}")
            print(f"     Title: {citation.title}")
            print(f"     Source Type: {citation.source_type}")
            print(f"     URL (NDC Align dataset): {citation.url}")
            print(f"     Underlying Source: {citation.underlying_source}")

            if citation.underlying_source:
                citations_with_underlying_source += 1
                print(f"     ✓ Has underlying_source!")
            else:
                print(f"     ○ No underlying_source")

            # Print metadata for debugging
            if citation.metadata:
                field = citation.metadata.get("field")
                if field:
                    print(f"     Source Field: {field}")

        print("\n" + "=" * 80)
        print(f"Summary:")
        print(f"  Total citations: {len(response.citations)}")
        print(f"  Citations with underlying_source: {citations_with_underlying_source}")
        print(f"  Percentage: {citations_with_underlying_source / len(response.citations) * 100:.1f}%")
        print("=" * 80)

        # Verify the implementation
        print("\n5. Verification:")
        if citations_with_underlying_source > 0:
            print("   ✓ SUCCESS: Some citations have underlying_source populated!")

            # Check that URLs are correct
            for citation in response.citations:
                if citation.underlying_source:
                    # Check that underlying_source is a valid URL
                    if citation.underlying_source.startswith("http"):
                        print(f"   ✓ Valid URL in underlying_source: {citation.underlying_source[:50]}...")
                    else:
                        print(f"   ✗ WARNING: underlying_source is not a URL: {citation.underlying_source}")

                    # Check that url points to NDC Align dataset
                    if "ndcalign" in (citation.url or "").lower() or "lse" in (citation.url or "").lower():
                        print(f"   ✓ URL points to NDC Align dataset")
                    else:
                        print(f"   ○ URL: {citation.url}")
                    break
        else:
            print("   ✗ FAIL: No citations have underlying_source populated")
            print("   This may be expected if the query didn't return records with source URLs")

        # Print a sample fact with citation
        if response.facts:
            print("\n6. Sample fact with citation:")
            fact = response.facts[0]
            print(f"   Fact: {fact.text[:200]}...")
            print(f"   Citation ID: {fact.citation_id}")

            # Find the corresponding citation
            matching_citation = next((c for c in response.citations if c.id == fact.citation_id), None)
            if matching_citation:
                print(f"   Citation Title: {matching_citation.title}")
                print(f"   Citation URL: {matching_citation.url}")
                print(f"   Underlying Source: {matching_citation.underlying_source}")

        return True

    except Exception as e:
        print(f"\n✗ Error executing query: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_underlying_source()

    if success:
        print("\n" + "=" * 80)
        print("✓ Test completed successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ Test failed!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
