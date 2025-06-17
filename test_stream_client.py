#!/usr/bin/env python3
"""
Test client for the streaming API endpoint
"""
import asyncio
import aiohttp
import json
import sys

async def test_streaming_endpoint():
    """Test the streaming endpoint with a simple query"""
    url = "http://localhost:8099/query/stream"
    test_query = {"query": "What is solar energy?"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                json=test_query,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}")
                    text = await response.text()
                    print(f"Response: {text}")
                    return False
                
                print("✅ Connected to streaming endpoint")
                print("📡 Receiving streaming events:")
                print("-" * 50)
                
                event_count = 0
                async for line in response.content:
                    line = line.decode().strip()
                    
                    if line.startswith("data: "):
                        event_count += 1
                        data = line[6:]  # Remove "data: " prefix
                        
                        try:
                            event = json.loads(data)
                            event_type = event.get("type", "unknown")
                            
                            print(f"Event {event_count}: {event_type}")
                            
                            if event_type == "thinking":
                                text = event.get("data", {}).get("text", "")
                                print(f"  💭 Thinking: {text[:100]}...")
                            elif event_type == "tool_call":
                                tool = event.get("data", {}).get("tool", "")
                                print(f"  🔧 Tool Call: {tool}")
                            elif event_type == "tool_result":
                                tool = event.get("data", {}).get("tool", "")
                                print(f"  ✅ Tool Result: {tool}")
                            elif event_type == "complete":
                                modules = event.get("data", {}).get("modules", [])
                                print(f"  🎯 Complete: {len(modules)} modules")
                                break
                            elif event_type == "error":
                                error_msg = event.get("data", {}).get("message", "")
                                print(f"  ❌ Error: {error_msg}")
                                break
                            
                        except json.JSONDecodeError:
                            print(f"  ⚠️ Invalid JSON: {data[:50]}...")
                
                print("-" * 50)
                print(f"✅ Streaming test completed! Received {event_count} events")
                return True
                
    except aiohttp.ClientConnectorError:
        print("❌ Could not connect to API server")
        print("💡 Make sure the server is running: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def main():
    print("🧪 Testing streaming API endpoint...")
    print("🔗 Connecting to http://localhost:8099/query/stream")
    print()
    
    success = await test_streaming_endpoint()
    
    if success:
        print("\n🎉 Streaming test passed!")
    else:
        print("\n💥 Streaming test failed!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")