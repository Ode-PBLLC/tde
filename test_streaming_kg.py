#!/usr/bin/env python3
"""
Test script for streaming API with KG visualization integration
"""

import asyncio
import aiohttp
import json

async def test_streaming_with_kg():
    """Test the streaming endpoint and show KG visualization links"""
    
    url = "http://localhost:8098/query/stream"
    query = "Climate policy effectiveness in Brazil"
    
    payload = {
        "query": query
    }
    
    print(f"🔄 Testing streaming query: '{query}'")
    print("=" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}")
                    return
                
                print("📡 Streaming events:")
                print("-" * 40)
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith('data: '):
                        event_data = line_str[6:]  # Remove 'data: ' prefix
                        
                        try:
                            event = json.loads(event_data)
                            event_type = event.get("type", "unknown")
                            
                            if event_type == "thinking":
                                message = event["data"]["message"]
                                print(f"🤔 {message}")
                                
                            elif event_type == "thinking_complete":
                                message = event["data"]["message"]
                                print(f"✅ {message}")
                                
                            elif event_type == "complete":
                                print("\n🎉 Query completed!")
                                
                                # Extract and display KG visualization info
                                metadata = event["data"]["metadata"]
                                modules_count = metadata.get("modules_count", 0)
                                
                                print(f"📊 Response contains {modules_count} modules")
                                
                                # Show KG visualization links
                                kg_viz_url = metadata.get("kg_visualization_url")
                                kg_query_url = metadata.get("kg_query_url")
                                
                                if kg_viz_url:
                                    print(f"\n🔗 Knowledge Graph Visualization:")
                                    print(f"   General KG Explorer: {kg_viz_url}")
                                    
                                if kg_query_url:
                                    print(f"   Query-Specific View: {kg_query_url}")
                                    
                                print(f"\n💡 Visit the URLs above to explore the knowledge graph!")
                                print(f"   The query-specific view will automatically analyze:")
                                print(f"   '{query}'")
                                
                                return
                                
                            elif event_type == "error":
                                error_msg = event["data"]["message"]
                                print(f"❌ Error: {error_msg}")
                                return
                                
                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON
                            
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the API server is running on http://localhost:8098")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def test_both_endpoints():
    """Test both regular and streaming endpoints"""
    
    print("🧪 Testing KG Visualization Integration")
    print("=" * 50)
    
    # Test regular endpoint first
    print("\n1️⃣ Testing regular /query endpoint:")
    print("-" * 30)
    
    try:
        async with aiohttp.ClientSession() as session:
            url = "http://localhost:8098/query"
            payload = {
                "query": "Renewable energy policy in Brazil",
                "include_thinking": False
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    metadata = data.get("metadata", {})
                    
                    print(f"✅ Regular endpoint works!")
                    print(f"📊 Modules: {metadata.get('modules_count', 0)}")
                    
                    kg_viz_url = metadata.get("kg_visualization_url")
                    kg_query_url = metadata.get("kg_query_url")
                    
                    if kg_viz_url:
                        print(f"🔗 KG Visualization: {kg_viz_url}")
                    if kg_query_url:
                        print(f"🔗 Query-Specific: {kg_query_url}")
                else:
                    print(f"❌ Regular endpoint failed: HTTP {response.status}")
                    
    except Exception as e:
        print(f"❌ Regular endpoint error: {e}")
    
    # Test streaming endpoint
    print("\n2️⃣ Testing streaming /query/stream endpoint:")
    print("-" * 40)
    
    await test_streaming_with_kg()

if __name__ == "__main__":
    print("🚀 Starting KG Visualization Integration Test")
    print("🔧 Requirements:")
    print("   - API server running on http://localhost:8098")
    print("   - KG visualization server running on http://localhost:8100")
    print()
    
    asyncio.run(test_both_endpoints())