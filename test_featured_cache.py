#!/usr/bin/env python3
"""
Test script for featured queries caching system.
"""

import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8099"

def test_featured_queries():
    """Test the featured queries endpoints."""
    
    print("Testing Featured Queries Cache System")
    print("=" * 50)
    
    # Test 1: Get featured queries without cache
    print("\n1. Testing /featured-queries (without cache)")
    try:
        response = requests.get(f"{API_BASE_URL}/featured-queries")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Found {len(data['featured_queries'])} featured queries")
            print(f"   ✓ Categories: {', '.join(data['metadata'].get('categories', []))}")
        else:
            print(f"   ✗ Error: Status {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Get featured queries with cache
    print("\n2. Testing /featured-queries?include_cached=true")
    try:
        response = requests.get(f"{API_BASE_URL}/featured-queries?include_cached=true")
        if response.status_code == 200:
            data = response.json()
            cached_count = sum(1 for q in data['featured_queries'] if 'cached_response' in q)
            print(f"   ✓ Found {cached_count}/{len(data['featured_queries'])} queries with cached responses")
            
            # Show cache status for each query
            for query in data['featured_queries']:
                if 'cached_response' in query:
                    cached_at = query.get('cached_at', 'Unknown')
                    print(f"   ✓ {query['id']}: Cached at {cached_at}")
                else:
                    print(f"   ✗ {query['id']}: No cache available")
        else:
            print(f"   ✗ Error: Status {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Get specific cached query
    print("\n3. Testing specific cached query endpoint")
    # Get the first query ID
    try:
        response = requests.get(f"{API_BASE_URL}/featured-queries")
        if response.status_code == 200:
            data = response.json()
            if data['featured_queries']:
                first_query_id = data['featured_queries'][0]['id']
                
                # Try to get its cached response
                cache_response = requests.get(f"{API_BASE_URL}/featured-queries/{first_query_id}/cached")
                if cache_response.status_code == 200:
                    cached_data = cache_response.json()
                    print(f"   ✓ Successfully retrieved cache for: {first_query_id}")
                    print(f"   ✓ Cached at: {cached_data.get('cached_at', 'Unknown')}")
                    print(f"   ✓ Response has {len(cached_data.get('response', {}).get('modules', []))} modules")
                elif cache_response.status_code == 404:
                    print(f"   ℹ No cache found for: {first_query_id} (run generate_featured_cache.py first)")
                else:
                    print(f"   ✗ Error retrieving cache: Status {cache_response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("To generate caches, run: python scripts/generate_featured_cache.py")


if __name__ == "__main__":
    test_featured_cache()