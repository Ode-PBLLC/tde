#!/usr/bin/env python3
"""
Simple API test to verify citation improvements
"""

import requests
import json

def test_simple_query():
    """Test a very simple query to see citation behavior"""
    
    query = "mitigation"  # One word query that we know exists
    
    print(f"Testing simple query: '{query}'")
    print("Making API call...")
    
    try:
        response = requests.post(
            "http://localhost:8098/query",
            json={"query": query, "include_thinking": False},
            timeout=120  # 2 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API call successful")
            
            # Save full response for inspection
            with open('last_api_response.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("✓ Full response saved to last_api_response.json")
            
            modules = result.get("modules", [])
            print(f"✓ Found {len(modules)} modules")
            
            for i, module in enumerate(modules):
                module_type = module.get("type", "unknown")
                heading = module.get("heading", "no heading")
                print(f"  Module {i+1}: {module_type} - {heading}")
                
                if module_type == "table" and "source" in heading.lower():
                    print(f"    🎯 SOURCES FOUND!")
                    columns = module.get("columns", [])
                    rows = module.get("rows", [])
                    print(f"    Columns: {columns}")
                    print(f"    Number of sources: {len(rows)}")
                    
                    if rows:
                        print(f"    First source: {rows[0]}")
                        return True
            
            print("  ❌ No sources table found")
            return False
            
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced Citations - Simple Query")
    print("=" * 50)
    
    success = test_simple_query()
    
    if success:
        print("\n🎉 SUCCESS! Citations are working!")
    else:
        print("\n❌ Citations not found in response")
        print("Check last_api_response.json for details")