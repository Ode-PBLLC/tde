#!/usr/bin/env python3
"""
Test the pure API approach - graph visualization with concepts/relationships data only in API
"""

import asyncio
import aiohttp
import json

async def test_pure_query_api():
    """Test the clean API that returns concepts and relationships as data only"""
    
    url = "http://localhost:8100/api/kg/query-subgraph"
    query = "Climate policy effectiveness in Brazil"
    
    payload = {
        "query": query,
        "depth": 2,
        "max_nodes": 50,
        "include_datasets": True,
        "include_passages": False
    }
    
    print(f"🎯 Testing Pure API Approach")
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
                
                # Show API structure
                print(f"✅ API Response Structure:")
                print(f"├── nodes: {len(data.get('nodes', []))} items")
                print(f"├── edges: {len(data.get('edges', []))} items") 
                print(f"├── concepts: {len(data.get('concepts', []))} items")
                print(f"├── relationships: {len(data.get('relationships', []))} items")
                print(f"├── query_concepts: {len(data.get('query_concepts', []))} items")
                print(f"└── total_found: {data.get('total_found', 0)}")
                
                # Show concepts data
                concepts = data.get('concepts', [])
                print(f"\n📝 CONCEPTS DATA (for your dev team):")
                print("-" * 50)
                if concepts:
                    print("JSON structure:")
                    print(json.dumps(concepts[:3], indent=2))  # Show first 3 as example
                    
                    print(f"\nQuery-relevant concepts:")
                    relevant = [c for c in concepts if c.get('is_query_relevant')]
                    for concept in relevant:
                        print(f"  ★ {concept['label']} (ID: {concept['id']})")
                else:
                    print("No concepts found")
                
                # Show relationships data  
                relationships = data.get('relationships', [])
                print(f"\n🔗 RELATIONSHIPS DATA (for your dev team):")
                print("-" * 50)
                if relationships:
                    print("JSON structure:")
                    print(json.dumps(relationships[:2], indent=2))  # Show first 2 as example
                    
                    print(f"\nFormatted relationships:")
                    for i, rel in enumerate(relationships[:5], 1):
                        print(f"  {i}. {rel['formatted']}")
                    
                    if len(relationships) > 5:
                        print(f"     ... and {len(relationships) - 5} more")
                else:
                    print("No relationships found")
                
                print(f"\n🎨 Visualization:")
                print(f"├── Pure graph at: http://localhost:8100")
                print(f"├── No UI overlays or input boxes")
                print(f"└── All data available via API for your team to display elsewhere")
                
                return data
                
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the KG visualization server is running on http://localhost:8100")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def show_api_documentation():
    """Show the API documentation for the dev team"""
    
    print(f"\n📚 API DOCUMENTATION FOR DEV TEAM")
    print("=" * 60)
    
    print(f"""
🔥 Pure Graph + API Data Approach

VISUALIZATION:
• Pure D3.js graph at http://localhost:8100
• No input boxes, no overlays, no UI clutter
• Just the interactive, zoomable knowledge graph

API ENDPOINTS:
• POST /api/kg/query-subgraph
• POST /api/kg/subgraph  
• Both return identical data structure

RESPONSE FORMAT:
{{
  "nodes": [...],           // Graph nodes for visualization
  "edges": [...],           // Graph edges for visualization  
  "concepts": [             // Clean concept list for your UI
    {{
      "id": "Q123",
      "label": "Climate Policy", 
      "is_query_relevant": true
    }}
  ],
  "relationships": [        // Clean relationships for your UI
    {{
      "source_label": "Climate Policy",
      "target_label": "Carbon Pricing",
      "relationship_type": "RELATED_TO", 
      "formatted": "Climate Policy -> Carbon Pricing (RELATED_TO)"
    }}
  ],
  "query_concepts": [...],  // IDs of query-relevant concepts
  "total_found": 25
}}

INTEGRATION:
1. Embed the pure graph: <iframe src="http://localhost:8100"></iframe>
2. Fetch API data: POST to /api/kg/query-subgraph
3. Display concepts/relationships in your own UI components
4. Perfect separation of concerns!

EXAMPLE USAGE:
• Graph shows visual relationships
• Your UI shows structured lists
• Users get both visual + textual understanding
""")

if __name__ == "__main__":
    print("🚀 Pure API + Graph Testing")
    print("🎯 Clean separation: Visual graph + API data")
    print()
    
    asyncio.run(test_pure_query_api())
    asyncio.run(show_api_documentation())