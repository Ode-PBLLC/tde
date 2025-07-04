#!/usr/bin/env python3
"""
Count actual MCP tools and estimate their schema sizes.
"""

import re
import os
from typing import Dict, List, Any

def count_tools_in_file(filepath: str) -> Dict[str, Any]:
    """Count tools in a specific MCP server file."""
    
    if not os.path.exists(filepath):
        return {'error': f'File not found: {filepath}'}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for @mcp.tool() decorators
    tool_pattern = r'@mcp\.tool\(\)\s*\ndef\s+(\w+)\('
    tool_matches = re.findall(tool_pattern, content, re.MULTILINE)
    
    # Extract function signatures and docstrings for schema estimation
    tools_with_details = []
    for tool_name in tool_matches:
        # Find the function definition and docstring
        func_pattern = rf'@mcp\.tool\(\)\s*\ndef\s+{tool_name}\((.*?)\).*?:\s*"""(.*?)"""'
        func_match = re.search(func_pattern, content, re.DOTALL)
        
        if func_match:
            params = func_match.group(1)
            docstring = func_match.group(2)
            
            # Estimate schema size based on parameters and docstring
            estimated_schema_size = len(params) + len(docstring) + 200  # Base overhead
            
            tools_with_details.append({
                'name': tool_name,
                'params': params,
                'docstring_length': len(docstring),
                'estimated_schema_chars': estimated_schema_size
            })
    
    total_schema_chars = sum(tool['estimated_schema_chars'] for tool in tools_with_details)
    
    return {
        'tool_count': len(tool_matches),
        'tools': tools_with_details,
        'total_schema_chars': total_schema_chars,
        'file_size': len(content)
    }

def main():
    """Analyze all MCP servers."""
    
    servers = [
        {'name': 'Knowledge Graph', 'path': 'mcp/cpr_kg_server.py'},
        {'name': 'Solar Facilities', 'path': 'mcp/solar_facilities_server.py'},
        {'name': 'GIST Environmental', 'path': 'mcp/gist_server.py'},
        {'name': 'LSE Policy', 'path': 'mcp/lse_server.py'},
        {'name': 'Response Formatter', 'path': 'mcp/response_formatter_server.py'}
    ]
    
    print("🔧 MCP Tool Analysis")
    print("=" * 50)
    
    total_tools = 0
    total_schema_size = 0
    
    for server in servers:
        print(f"\n📋 {server['name']}:")
        analysis = count_tools_in_file(server['path'])
        
        if 'error' in analysis:
            print(f"   ❌ {analysis['error']}")
            continue
        
        tool_count = analysis['tool_count']
        schema_size = analysis['total_schema_chars']
        
        print(f"   Tools: {tool_count}")
        print(f"   Schema Size: {schema_size:,} chars ({schema_size//4:,} tokens)")
        
        if tool_count > 0:
            print("   Tool Names:")
            for tool in analysis['tools'][:5]:  # Show first 5
                print(f"     • {tool['name']} ({tool['estimated_schema_chars']:,} chars)")
            if len(analysis['tools']) > 5:
                print(f"     ... and {len(analysis['tools']) - 5} more tools")
        
        total_tools += tool_count
        total_schema_size += schema_size
    
    print(f"\n" + "=" * 50)
    print("📊 TOTALS:")
    print(f"   Total Tools: {total_tools}")
    print(f"   Total Schema Size: {total_schema_size:,} chars ({total_schema_size//4:,} tokens)")
    
    # Estimate prompt components
    system_prompt_size = 13567  # From previous analysis
    message_growth = 10500  # From previous analysis
    
    total_prompt_size = system_prompt_size + total_schema_size + message_growth
    
    print(f"\n🎯 PROMPT BREAKDOWN:")
    print(f"   System Prompt: {system_prompt_size:,} chars ({system_prompt_size//4:,} tokens) - {(system_prompt_size/total_prompt_size)*100:.1f}%")
    print(f"   Tool Schemas: {total_schema_size:,} chars ({total_schema_size//4:,} tokens) - {(total_schema_size/total_prompt_size)*100:.1f}%")
    print(f"   Message Growth: {message_growth:,} chars ({message_growth//4:,} tokens) - {(message_growth/total_prompt_size)*100:.1f}%")
    print(f"   TOTAL: {total_prompt_size:,} chars ({total_prompt_size//4:,} tokens)")
    
    # Rate limit analysis
    rate_limit_tokens = 100000
    current_tokens = total_prompt_size // 4
    
    print(f"\n🚨 RATE LIMIT ANALYSIS:")
    print(f"   Current Usage: {current_tokens:,} tokens")
    print(f"   Rate Limit: {rate_limit_tokens:,} tokens")
    print(f"   Usage: {(current_tokens/rate_limit_tokens)*100:.1f}%")
    
    if current_tokens > rate_limit_tokens * 0.8:
        print("   ⚠️  APPROACHING RATE LIMIT!")
    
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    components = [
        ("Tool Schemas", total_schema_size, "Reduce tool descriptions, combine similar tools"),
        ("System Prompt", system_prompt_size, "Shorten verbose instructions, remove repetition"),
        ("Message Growth", message_growth, "Truncate tool results, summarize instead of full content")
    ]
    
    components.sort(key=lambda x: x[1], reverse=True)
    
    for i, (component, size, recommendation) in enumerate(components, 1):
        savings_potential = size * 0.3  # Assume 30% reduction possible
        print(f"   {i}. {component}: {size:,} chars")
        print(f"      → {recommendation}")
        print(f"      → Potential savings: {savings_potential:,.0f} chars ({savings_potential//4:,.0f} tokens)")

if __name__ == "__main__":
    main()