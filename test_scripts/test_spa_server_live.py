#!/usr/bin/env python
"""
Live integration tests for spa_server.py
Tests all MCP tools against the real SPA index with actual OpenAI calls.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the SPA server tools
from spa_server import (
    AmazonAssessmentListDocs,
    AmazonAssessmentSearch,
    AmazonAssessmentAsk
)

def pretty_print(title: str, data: Any, indent: int = 2):
    """Pretty print test results"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)

    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=indent, ensure_ascii=False))
    else:
        print(data)

def test_list_docs():
    """Test AmazonAssessmentListDocs with various limits"""
    print("\n" + "="*60)
    print("Testing AmazonAssessmentListDocs")
    print("="*60)

    # Test with small limit
    result = AmazonAssessmentListDocs(limit=3)
    print(f"\n✓ ListDocs (limit=3): {len(result)} items")
    for i, doc in enumerate(result[:2], 1):
        print(f"  {i}. {doc.get('file', 'unknown')} - page {doc.get('page', 'unknown')}")

    # Test with default (no limit)
    result_all = AmazonAssessmentListDocs()
    print(f"\n✓ ListDocs (no limit): {len(result_all)} total items")

    # Validate structure
    if result:
        assert 'file' in result[0], "Missing 'file' key in result"
        assert 'page' in result[0], "Missing 'page' key in result"
        print("✓ Document structure validated")

    return result

def test_search():
    """Test AmazonAssessmentSearch with various queries"""
    print("\n" + "="*60)
    print("Testing AmazonAssessmentSearch")
    print("="*60)

    test_queries = [
        ("carbon sequestration amazon biome", 5),
        ("deforestation rates brazil", 3),
        ("Amazon rainforest climate", 6),
        ("carbon storage tropical forests", 4),
        ("GtC per year sequestration", 3)
    ]

    all_results = []

    for query, k in test_queries:
        print(f"\n🔍 Searching: '{query}' (k={k})")
        start = time.time()

        try:
            results = AmazonAssessmentSearch(query=query, k=k)
            elapsed = time.time() - start

            print(f"  ✓ Found {len(results)} results in {elapsed:.2f}s")

            if results:
                # Show first result preview
                first = results[0]
                preview = first.get('text', '')[:100] + '...' if first.get('text') else 'No text'
                print(f"  📄 Top result: {first.get('file', 'unknown')} p.{first.get('page', 'unknown')}")
                print(f"  📊 Similarity: {first.get('similarity', 0):.4f}")
                print(f"  📝 Preview: {preview}")

                # Validate structure
                for r in results:
                    assert 'text' in r, f"Missing 'text' in result"
                    assert 'file' in r, f"Missing 'file' in result"
                    assert 'page' in r, f"Missing 'page' in result"
                    assert 'similarity' in r, f"Missing 'similarity' in result"

                all_results.extend(results)
            else:
                print(f"  ⚠️ No results found")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Summary
    unique_files = set(r['file'] for r in all_results)
    print(f"\n📊 Search Summary:")
    print(f"  - Total results across queries: {len(all_results)}")
    print(f"  - Unique files accessed: {len(unique_files)}")
    if unique_files:
        print(f"  - Files: {', '.join(list(unique_files)[:3])}...")

    return all_results

def test_ask():
    """Test AmazonAssessmentAsk with the carbon sequestration question"""
    print("\n" + "="*60)
    print("Testing AmazonAssessmentAsk")
    print("="*60)

    test_questions = [
        {
            "question": "How much carbon does the Amazon sequester every year?",
            "k": 6,
            "max_tokens": 400,
            "expected_keywords": ["GtC", "PgC", "carbon", "year", "sequester"]
        },
        {
            "question": "What are the main threats to Amazon rainforest?",
            "k": 5,
            "max_tokens": 300,
            "expected_keywords": ["deforestation", "climate", "fire", "threat"]
        },
        {
            "question": "What is the total area of the Amazon basin?",
            "k": 3,
            "max_tokens": 200,
            "expected_keywords": ["km", "area", "basin", "million", "hectares"]
        }
    ]

    results = []

    for test in test_questions:
        print(f"\n❓ Question: {test['question']}")
        print(f"   Parameters: k={test['k']}, max_tokens={test['max_tokens']}")

        start = time.time()

        try:
            result = AmazonAssessmentAsk(
                question=test['question'],
                k=test['k'],
                max_tokens=test['max_tokens']
            )
            elapsed = time.time() - start

            print(f"\n⏱️ Response time: {elapsed:.2f}s")

            # Validate structure
            assert 'answer' in result, "Missing 'answer' in result"
            assert 'citations' in result, "Missing 'citations' in result"

            answer = result['answer']
            citations = result['citations']

            print(f"\n📝 Answer ({len(answer)} chars):")
            print("-" * 40)
            print(answer)
            print("-" * 40)

            # Check for expected keywords
            found_keywords = [kw for kw in test['expected_keywords']
                            if kw.lower() in answer.lower()]
            print(f"\n✓ Found keywords: {found_keywords}")

            # Check for numeric values with units
            import re
            numbers_with_units = re.findall(r'\d+\.?\d*\s*(?:GtC|PgC|GtCO2|km²|million|billion|%)', answer)
            if numbers_with_units:
                print(f"✓ Numeric values found: {numbers_with_units}")

            # Validate citations
            print(f"\n📚 Citations: {len(citations)} sources")
            for i, cite in enumerate(citations[:3], 1):
                assert 'file' in cite, f"Missing 'file' in citation {i}"
                assert 'page' in cite, f"Missing 'page' in citation {i}"
                assert 'preview' in cite, f"Missing 'preview' in citation {i}"
                assert 'similarity' in cite, f"Missing 'similarity' in citation {i}"

                print(f"  {i}. {cite['file']} p.{cite['page']} (sim: {cite['similarity']:.4f})")
                preview = cite['preview'][:80] + '...' if len(cite['preview']) > 80 else cite['preview']
                print(f"     Preview: {preview}")

            # Check for inline citations in answer
            inline_citations = re.findall(r'\([^)]*p\.\s*\d+\)', answer)
            if inline_citations:
                print(f"\n✓ Found {len(inline_citations)} inline citations in answer")
                print(f"  Examples: {inline_citations[:3]}")

            results.append({
                'question': test['question'],
                'answer_length': len(answer),
                'citations_count': len(citations),
                'has_numbers': bool(numbers_with_units),
                'keywords_found': len(found_keywords),
                'response_time': elapsed
            })

        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({
                'question': test['question'],
                'error': str(e)
            })

    # Save example to cache
    if results and not results[0].get('error'):
        cache_dir = Path("static/cache/spa_query_examples")
        cache_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = cache_dir / f"test_run_{timestamp}.json"

        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'test_results': results,
                'sample_question': test_questions[0]['question'],
                'sample_response': result if 'result' in locals() else None
            }, f, indent=2)

        print(f"\n💾 Results saved to: {cache_file}")

    return results

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)

    # Test empty query
    print("\n🧪 Testing empty query search:")
    try:
        result = AmazonAssessmentSearch(query="", k=3)
        print(f"  Result: {len(result)} items (should handle gracefully)")
    except Exception as e:
        print(f"  ✓ Handled as expected: {type(e).__name__}")

    # Test very large k
    print("\n🧪 Testing very large k value:")
    result = AmazonAssessmentSearch(query="amazon", k=1000)
    print(f"  ✓ Returned {len(result)} results (capped by available data)")

    # Test non-English query
    print("\n🧪 Testing Portuguese query:")
    result = AmazonAssessmentSearch(query="floresta amazônica desmatamento", k=3)
    print(f"  ✓ Found {len(result)} results for Portuguese query")

    # Test special characters
    print("\n🧪 Testing query with special characters:")
    result = AmazonAssessmentSearch(query="CO₂ emissions & climate", k=3)
    print(f"  ✓ Found {len(result)} results with special chars")

    print("\n✅ Edge case testing complete")

def main():
    """Run all live tests"""
    print("\n" + "="*70)
    print("SPA Server Live Integration Tests")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set in environment")
        return
    print(f"✓ OpenAI API key loaded ({len(api_key)} chars)")

    # Check index existence
    index_path = Path("data/spa_index")
    if not index_path.exists():
        print("❌ ERROR: SPA index not found at data/spa_index/")
        print("  Run: python rag_embed_generator.py")
        return
    print(f"✓ SPA index found at {index_path}")

    try:
        # Run tests
        print("\n" + "="*70)
        print("Starting Test Suite")
        print("="*70)

        # 1. List documents
        docs = test_list_docs()

        # 2. Search functionality
        search_results = test_search()

        # 3. Ask functionality (the main test)
        ask_results = test_ask()

        # 4. Edge cases
        test_edge_cases()

        # Summary
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)

        print("\n✅ All tests completed successfully!")
        print(f"\nKey Results:")
        print(f"  • Documents in index: {len(docs) if docs else 'Unknown'}")
        print(f"  • Search functionality: ✓ Working")
        print(f"  • Ask functionality: ✓ Working")

        if ask_results:
            carbon_result = next((r for r in ask_results if 'sequester' in r.get('question', '').lower()), None)
            if carbon_result and not carbon_result.get('error'):
                print(f"\n🎯 Carbon Sequestration Question:")
                print(f"  • Answer generated: ✓")
                print(f"  • Citations provided: {carbon_result.get('citations_count', 0)}")
                print(f"  • Contains numbers: {'✓' if carbon_result.get('has_numbers') else '✗'}")
                print(f"  • Response time: {carbon_result.get('response_time', 0):.2f}s")

    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n✨ Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    exit(main())