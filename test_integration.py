#!/usr/bin/env python3
"""
Simple integration test for multi-server MCP setup
"""
import asyncio
import sys
import os
sys.path.append('mcp')

from mcp_chat import MultiServerClient

async def test_servers():
    """Test connection to both servers and basic tool calls"""
    print("Testing multi-server MCP integration...")
    
    async with MultiServerClient() as client:
        print("1. Connecting to servers...")
        try:
            await client.connect_to_server("kg", "mcp/cpr_kg_server.py")
            print("   ✓ Connected to KG server")
        except Exception as e:
            print(f"   ✗ Failed to connect to KG server: {e}")
            return
            
        try:
            await client.connect_to_server("solar", "mcp/solar_facilities_server.py")
            print("   ✓ Connected to Solar server")
        except Exception as e:
            print(f"   ✗ Failed to connect to Solar server: {e}")
            return
        
        print("\n2. Testing tool discovery...")
        try:
            all_tools = await client.get_all_available_tools()
            for server_name, tools in all_tools.items():
                print(f"   {server_name} server: {len(tools)} tools")
                # Show first few tool names
                tool_names = [t['name'] for t in tools[:3]]
                print(f"     Sample tools: {', '.join(tool_names)}...")
        except Exception as e:
            print(f"   ✗ Failed to get tools: {e}")
            return
            
        print("\n3. Testing KG server tools...")
        try:
            # Test getting available datasets
            result = await client.call_tool("GetAvailableDatasets", {}, "kg")
            print(f"   ✓ GetAvailableDatasets returned {len(result.content)} items")
        except Exception as e:
            print(f"   ✗ GetAvailableDatasets failed: {e}")
            
        print("\n4. Testing Solar server tools...")
        try:
            # Test getting solar capacity by country
            result = await client.call_tool("GetSolarCapacityByCountry", {}, "solar")
            print(f"   ✓ GetSolarCapacityByCountry returned data")
        except Exception as e:
            print(f"   ✗ GetSolarCapacityByCountry failed: {e}")
            
        try:
            # Test getting solar facilities for Brazil
            result = await client.call_tool("GetSolarFacilitiesByCountry", {"country": "Brazil", "limit": 5}, "solar")
            print(f"   ✓ GetSolarFacilitiesByCountry for Brazil returned data")
        except Exception as e:
            print(f"   ✗ GetSolarFacilitiesByCountry failed: {e}")

    print("\n✓ Integration test completed!")

if __name__ == "__main__":
    # Set working directory to repo root
    os.chdir("/mnt/o/Ode/Github/tde")
    asyncio.run(test_servers())