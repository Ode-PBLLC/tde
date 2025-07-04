#!/usr/bin/env python3
"""
Analyze prompt size components in the MCP chat system.

This script measures the relative sizes of different prompt components
to identify optimization targets for reducing prompt length.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

# Add mcp directory to path
sys.path.append('mcp')

# Constants for token estimation (rough approximation: 4 chars ≈ 1 token)
CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    if isinstance(text, str):
        return len(text) // CHARS_PER_TOKEN
    else:
        return text // CHARS_PER_TOKEN

def analyze_system_prompt() -> Dict[str, Any]:
    """Analyze the system prompt from mcp_chat.py"""
    
    # Read the mcp_chat.py file
    with open('mcp/mcp_chat.py', 'r') as f:
        content = f.read()
    
    # Extract the main system prompt (lines ~973-1136)
    # Look for the large system prompt in process_query function
    system_prompt_match = re.search(r'system_prompt = """(.*?)"""', content, re.DOTALL)
    
    if system_prompt_match:
        system_prompt = system_prompt_match.group(1)
        
        # Break down the system prompt into logical sections
        sections = {}
        
        # Define section patterns to look for
        section_patterns = [
            ("Core Task", r"Core Task:(.*?)(?=Available Data Sources|Tool Usage|$)"),
            ("Available Data Sources", r"Available Data Sources:(.*?)(?=Tool Usage|Cross-Reference|$)"),
            ("Tool Usage Guidelines", r"Tool Usage Guidelines:(.*?)(?=Cross-Reference|Enhanced Data|$)"),
            ("Cross-Reference Strategy", r"Cross-Reference Strategy:(.*?)(?=Enhanced Data|Multi-Table|$)"),
            ("Enhanced Data Discovery", r"Enhanced Data Discovery & Multi-Table Strategy:(.*?)(?=Multi-Table Response|Visualization|$)"),
            ("Multi-Table Response Patterns", r"Multi-Table Response Patterns.*?:(.*?)(?=Implementation Guidelines|Visualization|$)"),
            ("Visualization Guidelines", r"Visualization and Chart Generation Guidelines:(.*?)(?=Output Format|$)"),
            ("Output Format", r"Output Format:(.*?)$")
        ]
        
        for section_name, pattern in section_patterns:
            match = re.search(pattern, system_prompt, re.DOTALL | re.IGNORECASE)
            if match:
                section_content = match.group(1).strip()
                sections[section_name] = {
                    'content': section_content,
                    'chars': len(section_content),
                    'tokens': estimate_tokens(section_content)
                }
        
        return {
            'total_chars': len(system_prompt),
            'total_tokens': estimate_tokens(system_prompt),
            'sections': sections
        }
    
    return {'error': 'Could not find system prompt'}

def analyze_mcp_servers() -> Dict[str, Any]:
    """Analyze tool schemas from MCP servers"""
    
    server_configs = [
        {'name': 'kg_server', 'path': 'mcp/cpr_kg_server.py'},
        {'name': 'solar_server', 'path': 'mcp/solar_facilities_server.py'}, 
        {'name': 'gist_server', 'path': 'mcp/gist_server.py'},
        {'name': 'lse_server', 'path': 'mcp/lse_server.py'},
        {'name': 'formatter_server', 'path': 'mcp/response_formatter_server.py'}
    ]
    
    servers_analysis = {}
    total_tools = 0
    total_schema_size = 0
    
    for server in server_configs:
        try:
            with open(server['path'], 'r') as f:
                content = f.read()
            
            # Look for MCP tool definitions - different patterns in different servers
            tool_patterns = [
                r'@server\.call_tool\s*\ndef\s+(\w+)',  # @server.call_tool decorator
                r'@server\.call_tool\s*\nasync\s+def\s+(\w+)',  # async version
                r'tools\.append\(\s*Tool\(\s*name="([^"]+)"',  # Tool() constructor
                r'"name":\s*"([^"]+)".*?"description"',  # Direct tool dict
            ]
            
            tool_names = set()
            for pattern in tool_patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                tool_names.update(matches)
            
            # Also extract actual tool schemas for size estimation
            # Look for Tool constructors and input_schema definitions
            schema_patterns = [
                r'Tool\([^)]*input_schema=\{[^}]+\}[^)]*\)',  # Tool constructor with schema
                r'input_schema\s*=\s*\{[^}]+\}',  # Standalone input_schema
                r'"input_schema":\s*\{[^}]+\}',  # JSON style schema
            ]
            
            estimated_schema_chars = 0
            for pattern in schema_patterns:
                schema_matches = re.findall(pattern, content, re.DOTALL)
                for schema in schema_matches:
                    estimated_schema_chars += len(schema)
            
            # If we didn't find schemas, estimate based on typical MCP tool schema size
            if estimated_schema_chars == 0 and tool_names:
                # Estimate ~200 chars per tool schema (conservative)
                estimated_schema_chars = len(tool_names) * 200
            
            servers_analysis[server['name']] = {
                'tool_count': len(tool_names),
                'tool_names': list(tool_names)[:5],  # Show first 5 for debugging
                'schema_chars': estimated_schema_chars,
                'schema_tokens': estimate_tokens(str(estimated_schema_chars)),
                'file_size': len(content)
            }
            
            total_tools += len(tool_names)
            total_schema_size += estimated_schema_chars
            
        except Exception as e:
            servers_analysis[server['name']] = {'error': str(e)}
    
    return {
        'servers': servers_analysis,
        'total_tools': total_tools,
        'total_schema_chars': total_schema_size,
        'total_schema_tokens': estimate_tokens(str(total_schema_size))
    }

def analyze_message_patterns() -> Dict[str, Any]:
    """Analyze typical message accumulation patterns"""
    
    # Simulate a typical query flow
    typical_flow = {
        'initial_query': 'Show me solar facilities and capacity data across Brazil, India, South Africa, and Vietnam. Include maps of major installations.',
        'estimated_tool_calls': 5,  # Based on typical featured queries
        'estimated_results_per_call': 2000  # Average tool result size
    }
    
    # Calculate message growth
    base_message_size = len(typical_flow['initial_query'])
    
    # Each tool call adds:
    # 1. Assistant message requesting tool
    # 2. Tool result message with response
    message_growth = 0
    
    for i in range(typical_flow['estimated_tool_calls']):
        # Assistant tool request (~100 chars)
        message_growth += 100
        # Tool result (~2000 chars average)
        message_growth += typical_flow['estimated_results_per_call']
    
    return {
        'base_query_chars': base_message_size,
        'estimated_tool_calls': typical_flow['estimated_tool_calls'],
        'message_growth_chars': message_growth,
        'message_growth_tokens': estimate_tokens(str(message_growth)),
        'total_conversation_chars': base_message_size + message_growth
    }

def analyze_featured_queries() -> Dict[str, Any]:
    """Analyze the featured queries that are causing issues"""
    
    try:
        with open('static/featured_queries.json', 'r') as f:
            data = json.load(f)
        
        queries = data.get('featured_queries', [])
        
        query_analysis = {}
        total_query_chars = 0
        
        for query in queries:
            query_text = query.get('query', '')
            query_analysis[query.get('id', 'unknown')] = {
                'title': query.get('title', ''),
                'chars': len(query_text),
                'tokens': estimate_tokens(query_text),
                'has_map_keywords': any(keyword in query_text.lower() for keyword in ['map', 'geographic', 'facilities', 'installations'])
            }
            total_query_chars += len(query_text)
        
        return {
            'total_queries': len(queries),
            'average_query_chars': total_query_chars // len(queries) if queries else 0,
            'queries': query_analysis
        }
        
    except Exception as e:
        return {'error': str(e)}

def main():
    """Main analysis function"""
    
    print("🔍 MCP Chat Prompt Size Analysis")
    print("=" * 50)
    
    # Analyze system prompt
    print("\n1. System Prompt Analysis:")
    system_analysis = analyze_system_prompt()
    if 'error' not in system_analysis:
        print(f"   Total: {system_analysis['total_chars']:,} chars ({system_analysis['total_tokens']:,} tokens)")
        
        # Show top sections by size
        sections = sorted(system_analysis['sections'].items(), 
                         key=lambda x: x[1]['chars'], reverse=True)
        
        print("   Top sections by size:")
        for section_name, section_data in sections[:5]:
            print(f"     • {section_name[:40]}: {section_data['chars']:,} chars ({section_data['tokens']:,} tokens)")
    else:
        print(f"   Error: {system_analysis['error']}")
    
    # Analyze MCP servers and tools
    print("\n2. MCP Tool Schema Analysis:")
    mcp_analysis = analyze_mcp_servers()
    print(f"   Total Tools: {mcp_analysis['total_tools']}")
    print(f"   Total Schema Size: {mcp_analysis['total_schema_chars']:,} chars ({mcp_analysis['total_schema_tokens']:,} tokens)")
    
    print("   By server:")
    for server_name, server_data in mcp_analysis['servers'].items():
        if 'error' not in server_data:
            print(f"     • {server_name}: {server_data['tool_count']} tools, {server_data['schema_chars']:,} chars")
        else:
            print(f"     • {server_name}: Error - {server_data['error']}")
    
    # Analyze message patterns
    print("\n3. Message Accumulation Analysis:")
    message_analysis = analyze_message_patterns()
    print(f"   Base Query: {message_analysis['base_query_chars']:,} chars")
    print(f"   Tool Calls: {message_analysis['estimated_tool_calls']}")
    print(f"   Message Growth: {message_analysis['message_growth_chars']:,} chars ({message_analysis['message_growth_tokens']:,} tokens)")
    print(f"   Total Conversation: {message_analysis['total_conversation_chars']:,} chars")
    
    # Analyze featured queries
    print("\n4. Featured Query Analysis:")
    query_analysis = analyze_featured_queries()
    if 'error' not in query_analysis:
        print(f"   Total Queries: {query_analysis['total_queries']}")
        print(f"   Average Query Length: {query_analysis['average_query_chars']:,} chars")
        
        # Show longest queries
        longest_queries = sorted(query_analysis['queries'].items(), 
                               key=lambda x: x[1]['chars'], reverse=True)
        
        print("   Longest queries:")
        for query_id, query_data in longest_queries[:3]:
            print(f"     • {query_data['title'][:40]}: {query_data['chars']:,} chars")
    else:
        print(f"   Error: {query_analysis['error']}")
    
    # Summary and recommendations
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    
    if 'error' not in system_analysis and 'error' not in mcp_analysis:
        total_base_prompt = (system_analysis['total_chars'] + 
                           mcp_analysis['total_schema_chars'])
        
        print(f"Base Prompt Size: {total_base_prompt:,} chars ({estimate_tokens(str(total_base_prompt)):,} tokens)")
        print(f"  • System Prompt: {system_analysis['total_chars']:,} chars ({(system_analysis['total_chars']/total_base_prompt)*100:.1f}%)")
        print(f"  • Tool Schemas: {mcp_analysis['total_schema_chars']:,} chars ({(mcp_analysis['total_schema_chars']/total_base_prompt)*100:.1f}%)")
        
        with_conversation = total_base_prompt + message_analysis['message_growth_chars']
        print(f"With Conversation: {with_conversation:,} chars ({estimate_tokens(str(with_conversation)):,} tokens)")
        
        # Rate limit context (from CLAUDE.md)
        rate_limit_tokens = 100000  # From the error message
        current_estimated = estimate_tokens(str(with_conversation))
        
        print(f"\n🚨 Rate Limit Analysis:")
        print(f"   Estimated tokens: {current_estimated:,}")
        print(f"   Rate limit: {rate_limit_tokens:,} tokens")
        print(f"   Usage: {(current_estimated/rate_limit_tokens)*100:.1f}%")
        
        if current_estimated > rate_limit_tokens * 0.8:
            print("   ⚠️  APPROACHING RATE LIMIT!")
        
        print(f"\n🎯 Optimization Targets:")
        components = [
            ("Tool Schemas", mcp_analysis['total_schema_chars']),
            ("System Prompt", system_analysis['total_chars']),
            ("Message Accumulation", message_analysis['message_growth_chars'])
        ]
        
        components.sort(key=lambda x: x[1], reverse=True)
        
        for i, (component, size) in enumerate(components, 1):
            print(f"   {i}. {component}: {size:,} chars ({(size/with_conversation)*100:.1f}% of total)")

if __name__ == "__main__":
    main()