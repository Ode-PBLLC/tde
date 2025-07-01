#!/usr/bin/env python3
"""
Generate cached responses for featured queries.

This script reads featured queries from static/featured_queries.json
and generates cached API responses for each query.
"""

import json
import os
import sys
from datetime import datetime
import asyncio
import aiohttp
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
API_BASE_URL = "http://localhost:8098"  # Update this if your API runs on a different port
FEATURED_QUERIES_PATH = Path("static/featured_queries.json")
CACHE_DIR = Path("static/cache")
TIMEOUT = 300  # 5 minutes timeout for each query


async def fetch_query_response(session: aiohttp.ClientSession, query: str) -> dict:
    """Fetch response from the API for a given query."""
    url = f"{API_BASE_URL}/query"
    payload = {
        "query": query,
        "include_thinking": False
    }
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"API returned status {response.status}: {error_text}")
    except asyncio.TimeoutError:
        raise Exception(f"Query timed out after {TIMEOUT} seconds")
    except Exception as e:
        raise Exception(f"Error fetching query response: {str(e)}")


async def generate_cache_for_query(session: aiohttp.ClientSession, query_data: dict) -> tuple[str, bool, str]:
    """Generate cache for a single query."""
    query_id = query_data["id"]
    query_text = query_data["query"]
    
    print(f"\nProcessing: {query_data['title']}")
    print(f"Query ID: {query_id}")
    
    try:
        # Fetch the response
        print("  Fetching response from API...")
        response = await fetch_query_response(session, query_text)
        
        # Prepare cached data
        cached_data = {
            "cached_at": datetime.now().isoformat(),
            "query_id": query_id,
            "query": query_text,
            "response": response
        }
        
        # Save to cache file
        cache_file = CACHE_DIR / f"{query_id}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cached_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Cached successfully to {cache_file}")
        return query_id, True, "Success"
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Error: {error_msg}")
        return query_id, False, error_msg


async def main():
    """Main function to generate all caches."""
    print("Featured Query Cache Generator")
    print("=" * 50)
    
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load featured queries
    if not FEATURED_QUERIES_PATH.exists():
        print(f"Error: Featured queries file not found at {FEATURED_QUERIES_PATH}")
        sys.exit(1)
    
    with open(FEATURED_QUERIES_PATH, 'r', encoding='utf-8') as f:
        featured_data = json.load(f)
    
    queries = featured_data.get("featured_queries", [])
    
    if not queries:
        print("No featured queries found.")
        return
    
    print(f"Found {len(queries)} featured queries to cache")
    
    # Check if API is running
    print(f"\nChecking API availability at {API_BASE_URL}...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print("✓ API is running")
                else:
                    print(f"✗ API returned status {response.status}")
                    print("Make sure the API server is running: python api_server.py")
                    sys.exit(1)
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print("Make sure the API server is running: python api_server.py")
        sys.exit(1)
    
    # Process queries
    results = []
    async with aiohttp.ClientSession() as session:
        # Process queries sequentially to avoid overwhelming the API
        for query_data in queries:
            result = await generate_cache_for_query(session, query_data)
            results.append(result)
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"  Total queries: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed queries:")
        for query_id, success, error in results:
            if not success:
                print(f"  - {query_id}: {error}")
    
    # Update metadata in featured queries
    if successful > 0:
        featured_data["metadata"]["cache_generated_at"] = datetime.now().isoformat()
        featured_data["metadata"]["cached_queries"] = successful
        
        with open(FEATURED_QUERIES_PATH, 'w', encoding='utf-8') as f:
            json.dump(featured_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Updated featured_queries.json with cache metadata")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())