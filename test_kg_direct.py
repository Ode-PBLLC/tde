#!/usr/bin/env python3
"""
Test the KG server directly to show concepts and relationships data
"""

import asyncio
import aiohttp
import json

async def test_kg_server_directly():
    """Test the KG visualization server directly for concepts/relationships"""
    
    # This is the KG server, not the main API server
    url = "http://localhost:8100/api/kg/query-subgraph"
    query = "Solar energy in Brazil"
    
    payload = {
        "query": query,
        "depth": 2,
        "max_nodes": 50,
        "include_datasets": True,
        "include_passages": False
    }
    
    print(f"🎯 Testing KG Server Directly")
    print(f"URL: {url}")
    print(f"Query: '{query}'")
    print("=" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}")
                    text = await response.text()
                    print(f"Response: {text}")
                    return
                
                data = await response.json()
                
                print(f"✅ KG Server Response:")
                print(f"Total nodes: {len(data.get('nodes', []))}")
                print(f"Total edges: {len(data.get('edges', []))}")
                
                # THIS IS WHERE THE CONCEPTS ARE
                concepts = data.get('concepts', [])
                print(f"\n📝 CONCEPTS ({len(concepts)}):")
                for i, concept in enumerate(concepts[:10], 1):
                    relevance = "★" if concept.get('is_query_relevant') else " "
                    print(f"{i:2d}. {relevance} {concept['label']}")
                
                # THIS IS WHERE THE RELATIONSHIPS ARE  
                relationships = data.get('relationships', [])
                print(f"\n🔗 RELATIONSHIPS ({len(relationships)}):")
                for i, rel in enumerate(relationships[:10], 1):
                    print(f"{i:2d}. {rel['formatted']}")
                
                print(f"\n💾 Full JSON structure:")
                print(f"Keys: {list(data.keys())}")
                
                return data
                
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the KG visualization server is running on http://localhost:8100")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def compare_apis():
    """Show the difference between the two APIs"""
    
    print(f"\n📊 API COMPARISON")
    print("=" * 60)
    
    print(f"""
🏢 MAIN API SERVER (port 8098)
├── Endpoint: POST /query/stream  
├── Purpose: Climate policy analysis
├── Response: Modules (text, maps, tables, citations)
└── KG Integration: Only links in metadata

🕸️  KG VISUALIZATION SERVER (port 8100)  
├── Endpoint: POST /api/kg/query-subgraph
├── Purpose: Knowledge graph analysis
├── Response: Concepts + relationships + graph data
└── Usage: Pure graph visualization + structured data

FOR CONCEPTS & RELATIONSHIPS DATA:
→ Use the KG server (port 8100), not the main API server (port 8098)
""")

if __name__ == "__main__":
    print("🚀 Testing KG Server for Concepts & Relationships")
    print("🎯 This is the server that has the data you're looking for!")
    print()
    
    asyncio.run(test_kg_server_directly())
    asyncio.run(compare_apis())