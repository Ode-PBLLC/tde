#!/usr/bin/env python3
"""
Live test of SPA server with proper FastMCP function access
"""
import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp'))

# Set OpenAI key
os.environ['OPENAI_API_KEY'] = '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'

import spa_server

def run_tests():
    """Run all SPA server tests"""
    print("\n" + "="*70)
    print("SPA Server Live Test Suite")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Verify setup
    print("\n✓ OpenAI API key loaded")
    print(f"✓ Index found at: data/spa_index")

    # Test 1: List Documents
    print("\n" + "="*60)
    print("Test 1: AmazonAssessmentListDocs")
    print("="*60)

    try:
        docs = spa_server.AmazonAssessmentListDocs.fn(limit=10)
        print(f"✓ Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  {i}. {doc['file']} - page {doc['page']}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: Search
    print("\n" + "="*60)
    print("Test 2: AmazonAssessmentSearch")
    print("="*60)

    search_queries = [
        ("carbon sequestration amazon biome", 5),
        ("deforestation rates", 3),
        ("GtC per year", 4)
    ]

    for query, k in search_queries:
        print(f"\n🔍 Query: '{query}' (k={k})")
        try:
            start = time.time()
            results = spa_server.AmazonAssessmentSearch.fn(query=query, k=k)
            elapsed = time.time() - start

            print(f"✓ Found {len(results)} results in {elapsed:.2f}s")
            if results:
                top = results[0]
                print(f"  Top hit: {top['file']} p.{top['page']}")
                print(f"  Similarity: {top['similarity']}")
                print(f"  Preview: {top['text'][:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Test 3: Ask (Main Test)
    print("\n" + "="*60)
    print("Test 3: AmazonAssessmentAsk - Carbon Sequestration")
    print("="*60)

    question = "How much carbon does the Amazon sequester every year?"
    print(f"\n❓ Question: {question}")
    print(f"   Parameters: k=6, max_tokens=400")

    try:
        start = time.time()
        result = spa_server.AmazonAssessmentAsk.fn(
            query=question,
            k=6,
            max_tokens=400
        )
        elapsed = time.time() - start

        print(f"\n⏱️ Response time: {elapsed:.2f}s")

        # Display answer
        answer = result['answer']
        citations = result['citations']

        print(f"\n📝 Answer ({len(answer)} characters):")
        print("-" * 50)
        print(answer)
        print("-" * 50)

        # Check for numeric values
        numbers = re.findall(r'\d+\.?\d*\s*(?:GtC|PgC|GtCO₂|billion|million|Gt C)', answer)
        if numbers:
            print(f"\n✅ Found numeric values with units: {numbers}")
        else:
            print("\n⚠️ No specific numeric values found in answer")

        # Display citations
        print(f"\n📚 Citations: {len(citations)} sources")
        for i, cite in enumerate(citations[:4], 1):
            print(f"\n  Citation {i}:")
            print(f"    File: {cite['file']}")
            print(f"    Page: {cite['page']}")
            print(f"    Similarity: {cite['similarity']}")
            preview = cite['preview'][:100] + '...' if len(cite['preview']) > 100 else cite['preview']
            print(f"    Preview: {preview}")

        # Check for inline citations
        inline_cites = re.findall(r'\([^)]*p\.\s*\d+\)', answer)
        if inline_cites:
            print(f"\n✅ Found {len(inline_cites)} inline citations")
            print(f"   Examples: {inline_cites[:3]}")

        # Save result
        cache_dir = Path("static/cache/spa_query_examples")
        cache_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = cache_dir / f"carbon_sequestration_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'question': question,
                'answer': answer,
                'citations': citations,
                'response_time': elapsed,
                'numeric_values_found': numbers,
                'inline_citations_found': inline_cites
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error in Ask: {e}")
        import traceback
        traceback.print_exc()

    # Additional test questions
    print("\n" + "="*60)
    print("Test 4: Additional Questions")
    print("="*60)

    additional_questions = [
        "What percentage of the Amazon has been deforested?",
        "What are the main drivers of deforestation in the Amazon?"
    ]

    for q in additional_questions:
        print(f"\n❓ Question: {q}")
        try:
            start = time.time()
            result = spa_server.AmazonAssessmentAsk.fn(query=q, k=4, max_tokens=300)
            elapsed = time.time() - start

            answer = result['answer']
            print(f"✓ Answer generated in {elapsed:.2f}s ({len(answer)} chars)")
            print(f"  Citations: {len(result['citations'])} sources")

            # Show first 200 chars of answer
            preview = answer[:200] + '...' if len(answer) > 200 else answer
            print(f"  Preview: {preview}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*70)
    print("✅ All tests completed!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    run_tests()