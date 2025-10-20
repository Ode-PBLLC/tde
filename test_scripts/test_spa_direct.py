#!/usr/bin/env python3
"""
Direct test of SPA server functions
"""
import os
import sys
import json
import time
from pathlib import Path

# Setup path and environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp'))

# Set environment variable for OpenAI
os.environ['OPENAI_API_KEY'] = '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'

# Now import the spa_server module
import spa_server

def test_direct_functions():
    """Test the internal functions directly"""
    print("\n" + "="*60)
    print("Testing SPA Server Direct Functions")
    print("="*60)

    # Test the search function directly
    print("\n1. Testing _search function directly:")
    try:
        results = spa_server._search("carbon sequestration amazon", 3)
        print(f"✓ Search returned {len(results)} results")
        if results:
            print(f"  Top result: {results[0]['file']} p.{results[0]['page']}")
            print(f"  Similarity: {results[0]['similarity']}")
            print(f"  Text preview: {results[0]['text'][:100]}...")
    except Exception as e:
        print(f"❌ Search error: {e}")

    # Test embed function
    print("\n2. Testing _embed_query function:")
    try:
        embedding = spa_server._embed_query("test query")
        print(f"✓ Embedding generated: {len(embedding)} dimensions")
        print(f"  First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Embedding error: {e}")

    return results if 'results' in locals() else []

def test_mcp_tools():
    """Test the MCP tool functions"""
    print("\n" + "="*60)
    print("Testing MCP Tool Functions")
    print("="*60)

    # Get the actual functions from the MCP tool decorators
    # The decorated functions are still callable directly

    print("\n1. Testing AmazonAssessmentListDocs:")
    try:
        # Call the function directly (it's still a regular function)
        docs = spa_server.AmazonAssessmentListDocs.__wrapped__(limit=5)
        print(f"✓ ListDocs returned {len(docs)} documents")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  {i}. {doc['file']} - page {doc['page']}")
    except AttributeError:
        # If __wrapped__ doesn't exist, try calling directly
        try:
            docs = spa_server.AmazonAssessmentListDocs(limit=5)
            print(f"✓ ListDocs returned {len(docs)} documents")
            for i, doc in enumerate(docs[:3], 1):
                print(f"  {i}. {doc['file']} - page {doc['page']}")
        except Exception as e:
            print(f"❌ ListDocs error: {e}")

    print("\n2. Testing AmazonAssessmentSearch:")
    try:
        # Try the __wrapped__ attribute first
        try:
            results = spa_server.AmazonAssessmentSearch.__wrapped__(
                query="carbon sequestration amazon biome",
                k=5
            )
        except AttributeError:
            results = spa_server.AmazonAssessmentSearch(
                query="carbon sequestration amazon biome",
                k=5
            )

        print(f"✓ Search returned {len(results)} results")
        if results:
            print(f"  Top result similarity: {results[0]['similarity']}")
            print(f"  Files found: {set(r['file'] for r in results)}")
    except Exception as e:
        print(f"❌ Search error: {e}")

    print("\n3. Testing AmazonAssessmentAsk (main test):")
    question = "How much carbon does the Amazon sequester every year?"
    print(f"   Question: {question}")

    try:
        start_time = time.time()

        # Try to call the ask function
        try:
            result = spa_server.AmazonAssessmentAsk.__wrapped__(
                query=question,
                k=6,
                max_tokens=400
            )
        except AttributeError:
            result = spa_server.AmazonAssessmentAsk(
                query=question,
                k=6,
                max_tokens=400
            )

        elapsed = time.time() - start_time

        print(f"\n✓ Response generated in {elapsed:.2f} seconds")
        print("\n📝 Answer:")
        print("-" * 40)
        print(result['answer'])
        print("-" * 40)

        print(f"\n📚 Citations: {len(result['citations'])} sources")
        for i, cite in enumerate(result['citations'][:3], 1):
            print(f"  {i}. {cite['file']} p.{cite['page']} (similarity: {cite['similarity']})")
            print(f"     Preview: {cite['preview'][:80]}...")

        # Check for numeric values in answer
        import re
        numbers = re.findall(r'\d+\.?\d*\s*(?:GtC|PgC|GtCO2|billion|million)', result['answer'])
        if numbers:
            print(f"\n✓ Found numeric values: {numbers}")

        # Save result
        output_dir = Path("static/cache/spa_query_examples")
        output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"carbon_sequestration_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump({
                'question': question,
                'result': result,
                'elapsed_seconds': elapsed,
                'timestamp': timestamp
            }, f, indent=2)

        print(f"\n💾 Saved to: {output_file}")

    except Exception as e:
        print(f"❌ Ask error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("\n" + "="*70)
    print("SPA Server Direct Testing")
    print("="*70)

    # Check environment
    api_key = os.environ.get('OPENAI_API_KEY')
    print(f"✓ OpenAI API key set: {len(api_key) if api_key else 0} chars")

    # Check index
    index_path = Path("data/spa_index")
    if index_path.exists():
        print(f"✓ Index found at: {index_path}")
        # Check what's in the index
        manifest = index_path / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            print(f"  Manifest has {len(data.get('items', []))} items")
    else:
        print(f"❌ Index not found at: {index_path}")
        return

    # Run tests
    print("\nRunning tests...")

    # Test direct functions first
    test_direct_functions()

    # Test MCP tools
    test_mcp_tools()

    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()