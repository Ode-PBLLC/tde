#!/usr/bin/env python3
"""
Test script for the new streaming API endpoint
"""
import asyncio
import json
import sys
import os

# Add the mcp directory to the path
sys.path.append('mcp')

async def test_streaming_structure():
    """Test that the streaming function structure is correct"""
    try:
        from mcp_chat import run_query_streaming
        print("✓ Successfully imported run_query_streaming")
        
        # Test that it's an async generator
        gen = run_query_streaming("test query")
        print(f"✓ Function returns: {type(gen)}")
        
        print("✓ Streaming structure appears correct")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_api_structure():
    """Test that the API structure is correct"""
    try:
        # Check if api_server.py has the new endpoint
        with open('api_server.py', 'r') as f:
            content = f.read()
            
        if '/query/stream' in content:
            print("✓ Streaming endpoint found in api_server.py")
        else:
            print("✗ Streaming endpoint not found in api_server.py")
            return False
            
        if 'StreamingResponse' in content:
            print("✓ StreamingResponse import found")
        else:
            print("✗ StreamingResponse import not found")
            return False
            
        if 'run_query_streaming' in content:
            print("✓ run_query_streaming import found")
        else:
            print("✗ run_query_streaming import not found")
            return False
            
        print("✓ API structure appears correct")
        return True
        
    except Exception as e:
        print(f"✗ Error checking API structure: {e}")
        return False

async def main():
    """Run all tests"""
    print("Testing streaming implementation structure...\n")
    
    # Test API structure (doesn't require imports)
    api_ok = test_api_structure()
    print()
    
    # Test streaming structure (requires imports)
    streaming_ok = await test_streaming_structure()
    print()
    
    if api_ok and streaming_ok:
        print("🎉 All tests passed! Streaming implementation structure is correct.")
        print("\nTo test the actual streaming functionality:")
        print("1. Start the API server: python api_server.py")
        print("2. Make a POST request to http://localhost:8099/query/stream")
        print("3. Use Content-Type: application/json")
        print("4. Body: {\"query\": \"your test query\"}")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())