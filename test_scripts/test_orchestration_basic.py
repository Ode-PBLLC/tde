#!/usr/bin/env python3
"""
Basic test of the MCP orchestration system.
Tests the core components without requiring actual MCP server connections.
"""
import sys
import os

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_root, 'mcp'))

async def test_basic_orchestration():
    """Test basic orchestration functionality."""
    print("🧪 Testing MCP Orchestration System")
    
    try:
        # Test imports
        from mcp_chat import MultiServerClient, run_query_structured, run_query_streaming, run_query
        print("✅ All imports successful")
        
        # Test MultiServerClient initialization
        client = MultiServerClient()
        print("✅ MultiServerClient initialized")
        
        # Test method availability
        required_methods = ['process_query', 'process_query_streaming', 'get_all_available_tools', 'call_tool']
        for method in required_methods:
            if hasattr(client, method):
                print(f"✅ Method {method} available")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        # Test citation system
        from mcp_chat import CitationRegistry, load_citation_mapping, create_citation_info_for_tool
        print("✅ Citation system components available")
        
        # Test citation registry
        registry = CitationRegistry()
        test_citation = {
            "source_name": "Test Source",
            "provider": "Test Provider",
            "spatial_coverage": "Global",
            "temporal_coverage": "Current"
        }
        citation_id = registry.add_citation(test_citation)
        retrieved = registry.get_citation(citation_id)
        if retrieved == test_citation:
            print("✅ Citation registry working")
        else:
            print("❌ Citation registry test failed")
            return False
        
        # Test citation info creation
        citation_info = create_citation_info_for_tool("TestTool", {"country": "Test"})
        if "source_name" in citation_info:
            print("✅ Citation info creation working")
        else:
            print("❌ Citation info creation failed")
            return False
        
        # Test configuration loading functions
        from mcp_chat import load_server_configuration, load_citation_mapping
        print("✅ Configuration loading functions available")
        
        print("\n🎉 All basic orchestration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_basic_orchestration())
    exit(0 if success else 1)