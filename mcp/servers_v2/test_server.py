#!/usr/bin/env python3
"""Fixed test server with proper parameter descriptions."""

import json
from typing import Annotated, Dict, Any, Optional
from pydantic import Field
from fastmcp import FastMCP

# Create FastMCP instance
mcp = FastMCP("fixed-test-server")

class FixedTestServer:
    """Test server with proper parameter descriptions."""
    
    def __init__(self):
        self.mcp = mcp
        self._register_tools()
    
    def _register_tools(self):
        """Register tools with proper Field descriptions."""
        
        # Use Field for parameter descriptions
        @self.mcp.tool
        def test_with_field(
            text: Annotated[str, Field(description="Input text to process")],
            limit: Annotated[int, Field(description="Maximum results to return", ge=1, le=100)] = 10
        ) -> str:
            """Test tool with Field descriptions."""
            return self.handle_simple(text, limit)
        
        # Alternative: Direct Field usage (without Annotated)
        @self.mcp.tool
        def test_direct_field(
            query: str = Field(description="User question to evaluate"),
            context: Optional[Dict[str, Any]] = Field(None, description="Optional context dictionary"),
        ) -> str:
            """Test tool with direct Field usage."""
            return self.handle_context(query, context)
        
        # Test with complex Field constraints
        @self.mcp.tool
        def test_complex_params(
            name: Annotated[str, Field(description="Name of the person", min_length=1, max_length=50)],
            age: Annotated[int, Field(description="Age in years", ge=0, le=150)],
            email: Annotated[Optional[str], Field(None, description="Optional email address", pattern=r'^[^@]+@[^@]+\.[^@]+$')],
            tags: Annotated[list[str], Field(description="List of tags", max_items=10)] = []
        ) -> str:
            """Test tool with complex parameter constraints."""
            result = {
                "name": name,
                "age": age,
                "email": email,
                "tags": tags,
                "message": f"Hello {name}, age {age}!"
            }
            return json.dumps(result, ensure_ascii=False)
    
    # Instance methods that do the actual work
    def handle_simple(self, text: str, limit: int = 10) -> str:
        """Handle simple test."""
        result = {
            "input_text": text,
            "limit": limit,
            "result": f"Processed '{text}' with limit {limit}"
        }
        return json.dumps(result, ensure_ascii=False)
    
    def handle_context(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Handle context test."""
        result = {
            "query": query,
            "context": context or {},
            "result": f"Processed query: {query}"
        }
        return json.dumps(result, ensure_ascii=False)


def create_server() -> FastMCP:
    """Create and return the test server."""
    server = FixedTestServer()
    return server.mcp


if __name__ == "__main__":
    # Test parameter detection before running
    import asyncio
    
    server_instance = create_server()
    
    async def debug_tools():
        tools_result = await server_instance.get_tools()
        print(f"=== Fixed Server Tools ===")
        
        if isinstance(tools_result, dict):
            print(f"Found {len(tools_result)} tools:")
            
            for tool_name, tool in tools_result.items():
                print(f"\n--- Tool: {tool_name} ---")
                print(f"Description: {tool.description}")
                
                # Check parameters in detail
                if 'properties' in tool.parameters:
                    props = tool.parameters['properties']
                    print("Parameters:")
                    for param_name, param_def in props.items():
                        print(f"  {param_name}:")
                        print(f"    Type: {param_def.get('type', 'unknown')}")
                        print(f"    Description: {param_def.get('description', 'MISSING!')}")
                        if 'default' in param_def:
                            print(f"    Default: {param_def['default']}")
                        # Check for validation constraints
                        for constraint in ['minimum', 'maximum', 'minLength', 'maxLength', 'pattern']:
                            if constraint in param_def:
                                print(f"    {constraint}: {param_def[constraint]}")
                
                print(f"Required params: {tool.parameters.get('required', [])}")
    
    # Run the async debug function
    asyncio.run(debug_tools())
    
    print("\n=== Starting fixed server ===")
    server_instance.run()