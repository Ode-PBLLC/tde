#!/usr/bin/env python3
"""
Test the enhanced KG visualization with concepts and relationships lists
"""

import asyncio
import aiohttp
import json

async def test_query_subgraph():
    """Test the query subgraph endpoint for concepts and relationships"""
    
    url = "http://localhost:8100/api/kg/query-subgraph"
    query = "Climate policy effectiveness"
    
    payload = {
        "query": query,
        "depth": 2,
        "max_nodes": 50,
        "include_datasets": True,
        "include_passages": False
    }
    
    print(f"🔍 Testing query subgraph: '{query}'")
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
                
                # Display results
                print(f"✅ Query processed successfully!")
                print(f"📊 Total nodes: {data.get('total_found', 0)}")
                print(f"🎯 Query concepts: {', '.join(data.get('query_concept_labels', []))}")
                
                # Display concepts list
                concepts = data.get('concepts', [])
                print(f"\n📝 CONCEPTS ({len(concepts)}):")
                print("-" * 40)
                for i, concept in enumerate(concepts[:10], 1):  # Show first 10
                    indicator = "★" if concept.get('is_query_relevant') else " "
                    print(f"{i:2d}. {indicator} {concept['label']}")
                
                if len(concepts) > 10:
                    print(f"    ... and {len(concepts) - 10} more")
                
                # Display relationships list
                relationships = data.get('relationships', [])
                print(f"\n🔗 RELATIONSHIPS ({len(relationships)}):")
                print("-" * 40)
                for i, rel in enumerate(relationships[:10], 1):  # Show first 10
                    print(f"{i:2d}. {rel['formatted']}")
                
                if len(relationships) > 10:
                    print(f"    ... and {len(relationships) - 10} more")
                
                print(f"\n💡 Visit http://localhost:8100?query={query.replace(' ', '%20')} to see the visualization!")
                
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the KG visualization server is running on http://localhost:8100")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def test_regular_subgraph():
    """Test regular subgraph endpoint"""
    
    url = "http://localhost:8100/api/kg/subgraph"
    
    # First get a concept ID
    concepts_url = "http://localhost:8100/api/kg/top_concepts?limit=5"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get a concept to test with
            async with session.get(concepts_url) as response:
                if response.status != 200:
                    print("❌ Could not get top concepts")
                    return
                
                concepts_data = await response.json()
                if not concepts_data.get('concepts'):
                    print("❌ No concepts found")
                    return
                
                test_concept = concepts_data['concepts'][0]
                concept_id = test_concept['id']
                concept_label = test_concept['label']
                
                print(f"\n🔍 Testing regular subgraph for: '{concept_label}'")
                print("=" * 60)
                
                # Test regular subgraph
                payload = {
                    "node_id": concept_id,
                    "depth": 1,
                    "max_nodes": 30
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        print(f"❌ Error: HTTP {response.status}")
                        return
                    
                    data = await response.json()
                    
                    concepts = data.get('concepts', [])
                    relationships = data.get('relationships', [])
                    
                    print(f"✅ Regular subgraph processed!")
                    print(f"📊 Total nodes: {len(data.get('nodes', []))}")
                    print(f"📝 Concepts: {len(concepts)}")
                    print(f"🔗 Relationships: {len(relationships)}")
                    
                    if concepts:
                        print(f"\nSample concepts: {', '.join([c['label'] for c in concepts[:5]])}")
                    if relationships:
                        print(f"Sample relationships:")
                        for rel in relationships[:3]:
                            print(f"  - {rel['formatted']}")
                
    except Exception as e:
        print(f"❌ Error testing regular subgraph: {e}")

if __name__ == "__main__":
    print("🚀 Testing Enhanced KG Visualization")
    print("🔧 Requirements:")
    print("   - KG visualization server running on http://localhost:8100")
    print()
    
    asyncio.run(test_query_subgraph())
    asyncio.run(test_regular_subgraph())