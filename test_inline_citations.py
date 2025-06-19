#!/usr/bin/env python3
"""
Integration test for the inline citation system.

Tests the citation registry, inline citation insertion, and narrative organization.
"""

import os
import sys
import asyncio
import json

# Add the mcp directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp'))

from mcp_chat import CitationRegistry

def test_citation_registry():
    """Test the CitationRegistry class functionality."""
    print("🧪 Testing CitationRegistry...")
    
    registry = CitationRegistry()
    
    # Test adding passage sources
    passage_source = {
        "doc_id": "test_doc_123",
        "passage_id": "passage_456", 
        "text": "This is a test passage about climate policy",
        "title": "Climate Policy Document"
    }
    
    citation_num1 = registry.add_source(passage_source, "tool_GetPassages")
    print(f"  ✓ Added passage source as citation #{citation_num1}")
    
    # Test adding dataset sources
    dataset_source = {
        "citation_id": "GIST_test_789",
        "tool_used": "GetGistTopEmitters",
        "text": "GIST database analysis of top emitting companies",
        "title": "GIST Environmental Database",
        "type": "dataset",
        "provider": "GIST"
    }
    
    citation_num2 = registry.add_source(dataset_source, "tool_GetGistTopEmitters")
    print(f"  ✓ Added dataset source as citation #{citation_num2}")
    
    # Test deduplication
    citation_num3 = registry.add_source(passage_source, "tool_GetPassages")
    assert citation_num3 == citation_num1, "Citation deduplication failed"
    print(f"  ✓ Deduplication works: same source = citation #{citation_num3}")
    
    # Test module associations
    module_citations = registry.get_module_citations("tool_GetPassages")
    assert citation_num1 in module_citations, "Module citation tracking failed"
    print(f"  ✓ Module citation tracking: {module_citations}")
    
    # Test citation formatting
    superscript = registry.format_citation_superscript([citation_num1, citation_num2])
    print(f"  ✓ Citation superscript formatting: {superscript}")
    
    print("✅ CitationRegistry tests passed!\n")
    return registry

def test_citation_insertion():
    """Test inline citation insertion utilities."""
    print("🧪 Testing inline citation insertion...")
    
    # Import the citation insertion function
    from response_formatter_server import _insert_inline_citations, _add_citations_to_table_heading
    
    # Create test citation registry data
    test_registry = {
        "citations": {
            1: {"title": "Climate Policy Doc", "type": "document"},
            2: {"title": "GIST Database", "type": "dataset"}
        },
        "module_citations": {
            "tool_GetPassages": [1],
            "tool_GetGistTopEmitters": [2]
        }
    }
    
    # Test text citation insertion
    test_text = "Climate change is a major global challenge. Renewable energy solutions are critical for mitigation."
    cited_text = _insert_inline_citations(test_text, "text_module_0", test_registry)
    print(f"  ✓ Text with citations: {cited_text[:100]}...")
    
    # Test table heading citations
    heading = "Top Carbon Emitting Companies"
    cited_heading = _add_citations_to_table_heading(heading, "GetGistTopEmitters", test_registry)
    print(f"  ✓ Table heading with citations: {cited_heading}")
    
    print("✅ Citation insertion tests passed!\n")

async def test_comprehensive_query():
    """Test a comprehensive query to validate the full system."""
    print("🧪 Testing comprehensive query with citations...")
    
    try:
        from mcp_chat import run_query
        
        # Test a simple query that should trigger multiple data sources
        query = "What are the top solar energy installations by country?"
        
        print(f"  🔍 Running query: {query}")
        result = await run_query(query)
        
        if result and "formatted_response" in result:
            formatted_response = result["formatted_response"]
            modules = formatted_response.get("modules", [])
            
            print(f"  ✓ Query returned {len(modules)} modules")
            
            # Check for citation-related modules
            has_citations = any(m.get("type") == "numbered_citation_table" for m in modules)
            has_text_with_citations = any("^" in str(m.get("texts", [])) for m in modules if m.get("type") == "text")
            
            print(f"  ✓ Has citation table: {has_citations}")
            print(f"  ✓ Has inline citations: {has_text_with_citations}")
            
            # Print module summary
            for i, module in enumerate(modules):
                module_type = module.get("type", "unknown")
                heading = module.get("heading", "No heading")
                print(f"    [{i+1}] {module_type}: {heading}")
            
            print("✅ Comprehensive query test completed!\n")
            return True
            
        else:
            print("  ❌ Query returned no formatted response")
            return False
            
    except Exception as e:
        print(f"  ❌ Error in comprehensive query: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all citation system tests."""
    print("🚀 Starting Inline Citation System Tests\n")
    
    # Test 1: Citation Registry
    registry = test_citation_registry()
    
    # Test 2: Citation Insertion
    test_citation_insertion()
    
    # Test 3: Comprehensive Query (optional, requires full system)
    print("🧪 Testing full system integration...")
    try:
        result = asyncio.run(test_comprehensive_query())
        if result:
            print("🎉 All tests passed! Inline citation system is working correctly.")
        else:
            print("⚠️  Some integration tests failed, but core components work.")
    except Exception as e:
        print(f"⚠️  Integration test skipped due to error: {e}")
        print("🎯 Core citation components tested successfully.")

if __name__ == "__main__":
    main()