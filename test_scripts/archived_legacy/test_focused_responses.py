#!/usr/bin/env python3
"""
Test script to verify the focused response improvements.
Tests that responses are more focused and include proper units.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path to import mcp modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.mcp_chat_redo import MCPChatSystem, initialize_mcp_client

async def test_ndc_query():
    """Test that NDC queries get focused policy responses without facility counts."""
    print("\n" + "="*60)
    print("TEST 1: NDC Query - Should focus on policy, not infrastructure")
    print("="*60)
    
    # Initialize the chat system
    chat_system = MCPChatSystem()
    
    # Test query about Brazil's NDC commitments
    query = "What are Brazil's NDC commitments related to energy?"
    print(f"Query: {query}")
    print("-"*40)
    
    try:
        # Process the query
        result = await chat_system.process_query(query)
        
        # Check the response
        if 'modules' in result:
            print("\nModules in response:")
            for i, module in enumerate(result['modules']):
                module_type = module.get('type', 'unknown')
                heading = module.get('heading', 'No heading')
                print(f"  {i+1}. [{module_type}] {heading}")
                
                # Check if first text module starts with direct answer
                if module_type == 'text' and i == 0:
                    texts = module.get('texts', [])
                    if texts:
                        first_text = texts[0][:200] if texts[0] else ""
                        print(f"     First text: {first_text}...")
                        
                        # Check for problematic content
                        if 'facilities' in first_text.lower():
                            print("     ⚠️ WARNING: Contains facility data in policy query!")
                        if 'units' not in first_text and any(char.isdigit() for char in first_text):
                            print("     ⚠️ WARNING: May have numbers without units!")
                
                # Check for maps that shouldn't be there
                if module_type == 'map':
                    print(f"     ⚠️ WARNING: Map included for NDC policy query!")
        
        # Check metadata
        metadata = result.get('metadata', {})
        print(f"\nMetadata:")
        print(f"  Servers queried: {metadata.get('servers_queried', 'N/A')}")
        print(f"  Facts collected: {metadata.get('facts_collected', 'N/A')}")
        print(f"  Has maps: {metadata.get('has_maps', False)}")
        print(f"  Has charts: {metadata.get('has_charts', False)}")
        print(f"  Has tables: {metadata.get('has_tables', False)}")
        
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()

async def test_map_query():
    """Test that geographic queries properly include maps."""
    print("\n" + "="*60)
    print("TEST 2: Geographic Query - Should include relevant map")
    print("="*60)
    
    # Initialize the chat system
    chat_system = MCPChatSystem()
    
    # Test query about solar facility locations
    query = "Where are the largest solar facilities in Brazil?"
    print(f"Query: {query}")
    print("-"*40)
    
    try:
        # Process the query
        result = await chat_system.process_query(query)
        
        # Check the response
        if 'modules' in result:
            has_map = False
            print("\nModules in response:")
            for i, module in enumerate(result['modules']):
                module_type = module.get('type', 'unknown')
                heading = module.get('heading', 'No heading')
                print(f"  {i+1}. [{module_type}] {heading}")
                
                if module_type == 'map':
                    has_map = True
                    print(f"     ✓ Map correctly included for geographic query")
            
            if not has_map:
                print("     ⚠️ WARNING: No map for geographic 'where' query!")
        
    except Exception as e:
        print(f"Error processing query: {e}")

async def test_comparison_query():
    """Test that comparison queries include appropriate tables."""
    print("\n" + "="*60)
    print("TEST 3: Comparison Query - Should include comparison table")
    print("="*60)
    
    # Initialize the chat system
    chat_system = MCPChatSystem()
    
    # Test comparison query
    query = "Compare renewable energy targets for Brazil, India, and China"
    print(f"Query: {query}")
    print("-"*40)
    
    try:
        # Process the query
        result = await chat_system.process_query(query)
        
        # Check the response
        if 'modules' in result:
            has_table = False
            print("\nModules in response:")
            for i, module in enumerate(result['modules']):
                module_type = module.get('type', 'unknown')
                heading = module.get('heading', 'No heading')
                print(f"  {i+1}. [{module_type}] {heading}")
                
                if module_type == 'table':
                    has_table = True
                    print(f"     ✓ Table correctly included for comparison query")
                    
                    # Check if table has proper units in column names
                    if 'data' in module and 'columns' in module['data']:
                        columns = module['data']['columns']
                        print(f"     Columns: {columns}")
                        if '%' not in str(columns) and 'percent' not in str(columns).lower():
                            print(f"     ⚠️ WARNING: Table may lack units in column names")
            
            if not has_table:
                print("     ⚠️ WARNING: No table for comparison query!")
        
    except Exception as e:
        print(f"Error processing query: {e}")

async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING FOCUSED RESPONSE IMPROVEMENTS")
    print("="*60)
    
    # Initialize the global MCP client
    print("\nInitializing MCP client...")
    await initialize_mcp_client()
    
    # Run tests
    await test_ndc_query()
    await test_map_query()
    await test_comparison_query()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
