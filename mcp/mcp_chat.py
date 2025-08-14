import asyncio
import anthropic
from fastmcp import Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import Optional, Dict, List, Any
import time
import os
from dotenv import load_dotenv
import json
from textwrap import fill
import pandas as pd
import re
import aiohttp
import logging

# Load environment variables
load_dotenv()

# Set up logging for MCP operations
mcp_logger = logging.getLogger('mcp_orchestration')
mcp_logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler for MCP logs
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/mcp_orchestration.log')
file_handler.setFormatter(formatter)
mcp_logger.addHandler(file_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
mcp_logger.addHandler(console_handler)

# Global citation mapping cache
_citation_mapping = None
_citation_mapping_lock = asyncio.Lock()
_citation_validation_warnings = set()  # Track which tools we've warned about

def load_citation_mapping() -> dict:
    """
    Load citation mapping from configuration file with validation.
    Falls back to empty mapping if file not found.
    """
    global _citation_mapping
    
    if _citation_mapping is not None:
        return _citation_mapping
    
    citation_config_path = os.getenv("CITATION_CONFIG_PATH", "config/citation_sources.json")
    
    # First try to validate the config if it exists
    if os.path.exists(citation_config_path):
        validator = ConfigValidator()
        if validator.load_schemas():
            validation_result = validator.validate_citation_sources_config(citation_config_path)
            if not validation_result.valid:
                mcp_logger.error(f"Citation configuration validation failed:")
                for error in validation_result.errors:
                    mcp_logger.error(f"  - {error}")
                    print(f"❌ Citation config error: {error}")
            elif validation_result.warnings:
                for warning in validation_result.warnings:
                    mcp_logger.warning(f"  - {warning}")
                    print(f"⚠️ Citation config warning: {warning}")
            else:
                mcp_logger.info("Citation configuration validation passed")
    
    try:
        if os.path.exists(citation_config_path):
            with open(citation_config_path, 'r') as f:
                config = json.load(f)
                _citation_mapping = config.get("tool_citations", {})
                mcp_logger.info(f"Loaded {len(_citation_mapping)} citation mappings from {citation_config_path}")
                print(f"✅ Loaded {len(_citation_mapping)} citation mappings from {citation_config_path}")
        else:
            mcp_logger.warning(f"Citation config not found at {citation_config_path}, using fallback")
            print(f"⚠️ Citation config not found at {citation_config_path}, using fallback")
            _citation_mapping = {}
    except Exception as e:
        mcp_logger.error(f"Error loading citation config: {e}")
        print(f"❌ Error loading citation config: {e}")
        _citation_mapping = {}
    
    return _citation_mapping

def create_citation_info_for_tool(tool_name: str, tool_args: dict = None) -> dict:
    """
    Create citation metadata for a given tool call using configuration-driven mapping.
    
    Args:
        tool_name: Name of the tool called
        tool_args: Arguments passed to the tool
        
    Returns:
        Citation info dictionary with source_name, provider, etc.
    """
    global _citation_validation_warnings
    citation_mapping = load_citation_mapping()
    
    # Get citation info for this tool, with fallback
    if tool_name in citation_mapping:
        citation_info = citation_mapping[tool_name].copy()
        
        # Handle dynamic spatial coverage based on tool arguments
        if tool_args and "spatial_coverage" in citation_info:
            if "{country}" in citation_info["spatial_coverage"] and "country" in tool_args:
                citation_info["spatial_coverage"] = citation_info["spatial_coverage"].replace(
                    "{country}", tool_args["country"]
                )
        
        mcp_logger.debug(f"Found citation mapping for tool '{tool_name}'")
        return citation_info
    else:
        # Log warning for missing citation, but only once per tool to avoid spam
        if tool_name not in _citation_validation_warnings:
            mcp_logger.warning(f"No citation mapping found for tool '{tool_name}', using fallback")
            _citation_validation_warnings.add(tool_name)
        
        # Fallback for tools not in mapping
        return {
            "source_name": f"{tool_name} Dataset",
            "provider": "Unknown Provider",
            "spatial_coverage": "Unknown Coverage", 
            "temporal_coverage": "Unknown Period",
            "source_url": ""
        }

def wrap_tool_result_with_citation(tool_name: str, tool_result_text: str, tool_args: dict = None) -> dict:
    """
    Wrap a tool result with citation information in the expected format.
    
    Args:
        tool_name: Name of the tool that generated the result
        tool_result_text: Raw JSON text result from the tool
        tool_args: Arguments passed to the tool
        
    Returns:
        Dictionary in {"fact": result, "citation_info": {...}} format
    """
    # CITATION DEBUG: Analyze tool result for errors/failures
    tool_success = True
    error_reasons = []
    
    # Check for common failure patterns
    if "Error" in tool_result_text:
        tool_success = False
        error_reasons.append("Error in result")
    elif tool_result_text.strip() in ["[]", "{}", '""']:
        tool_success = False
        error_reasons.append("Empty result")
    elif "null" in tool_result_text.lower():
        tool_success = False
        error_reasons.append("Null result")
    
    # Log the analysis
    status = "SUCCESS" if tool_success else "FAILED"
    print(f"🔍 CITATION DEBUG - Tool: {tool_name}, Status: {status}")
    if not tool_success:
        print(f"   ❌ Failure reasons: {', '.join(error_reasons)}")
        print(f"   📄 Result snippet: {tool_result_text[:100]}...")
    
    citation_info = create_citation_info_for_tool(tool_name, tool_args)
    
    # Add success/failure metadata to citation info
    citation_info["tool_success"] = tool_success
    citation_info["error_reasons"] = error_reasons
    
    return {
        "fact": tool_result_text,  # Keep original JSON string as the fact
        "citation_info": citation_info
    }

def convert_citation_format(text: str) -> str:
    """
    Convert [citation_X] format to ^X^ format for frontend compatibility.
    
    Args:
        text: Text containing [citation_1], [citation_2] style citations
        
    Returns:
        Text with ^1^, ^2^ style citations
    """
    if not text:
        return text
        
    # Replace [citation_1] with ^1^, [citation_2] with ^2^, etc.
    def replace_citation(match):
        citation_id = match.group(1)  # Extract the number part
        # Handle both citation_1 and just 1 formats
        if citation_id.startswith('citation_'):
            number = citation_id.replace('citation_', '')
        else:
            number = citation_id
        return f"^{number}^"
    
    # Pattern matches [citation_1], [citation_2], etc.
    pattern = r'\\[citation_(\\w+)\\]'
    converted_text = re.sub(pattern, replace_citation, text)
    
    print(f"CITATION FORMAT DEBUG: Converted citations in text: {len(re.findall(pattern, text))} found")
    return converted_text

# Global singleton client for performance optimization
_global_client = None
_client_lock = asyncio.Lock()

async def load_server_configuration() -> List[Dict]:
    """
    Load MCP server configuration from config/servers.json with validation
    """
    servers_config_path = os.getenv("SERVERS_CONFIG_PATH", "config/servers.json")
    
    # First try to validate the config if it exists
    if os.path.exists(servers_config_path):
        validator = ConfigValidator()
        if validator.load_schemas():
            validation_result = validator.validate_servers_config(servers_config_path)
            if not validation_result.valid:
                mcp_logger.error(f"Server configuration validation failed:")
                for error in validation_result.errors:
                    mcp_logger.error(f"  - {error}")
                    print(f"❌ Config error: {error}")
            elif validation_result.warnings:
                for warning in validation_result.warnings:
                    mcp_logger.warning(f"  - {warning}")
                    print(f"⚠️ Config warning: {warning}")
            else:
                mcp_logger.info("Server configuration validation passed")
    
    if not os.path.exists(servers_config_path):
        # Fallback to example file for development
        servers_config_path = "config/servers.example.json"
        if os.path.exists(servers_config_path):
            mcp_logger.warning("Using example server config. Copy to config/servers.json for production.")
            print(f"⚠️ Using example server config. Copy to config/servers.json for production.")
    
    try:
        with open(servers_config_path, 'r') as f:
            config = json.load(f)
            servers = config.get("servers", [])
            mcp_logger.info(f"Loaded {len(servers)} server configurations from {servers_config_path}")
            print(f"✅ Loaded {len(servers)} server configurations from {servers_config_path}")
            return servers
    except Exception as e:
        mcp_logger.error(f"Error loading server config: {e}")
        print(f"❌ Error loading server config: {e}")
        print("🔧 Using fallback server configuration")
        # Minimal fallback configuration
        fallback_servers = [
            {
                "name": "response_formatter",
                "path": "mcp/response_formatter_server.py",
                "description": "Basic response formatting"
            }
        ]
        mcp_logger.info(f"Using fallback configuration with {len(fallback_servers)} servers")
        return fallback_servers

async def get_global_client():
    """Get or create the global singleton MCP client."""
    global _global_client
    
    async with _client_lock:
        if _global_client is None:
            _global_client = MultiServerClient()
            await _global_client.__aenter__()
            
            # Load server configuration and connect
            servers = await load_server_configuration()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            
            for server_config in servers:
                server_name = server_config.get("name")
                server_path = server_config.get("path")
                
                if not server_name or not server_path:
                    print(f"⚠️ Invalid server configuration: {server_config}")
                    continue
                
                # Resolve full path
                full_server_path = os.path.join(project_root, server_path)
                
                try:
                    if os.path.exists(full_server_path):
                        print(f"🔌 Connecting to {server_name} server...")
                        await _global_client.connect_to_server(server_name, full_server_path)
                        print(f"✅ Connected to {server_name}")
                    else:
                        print(f"❌ Server file not found: {full_server_path}")
                except Exception as e:
                    print(f"❌ Failed to connect to {server_name}: {str(e)}")
                        
    return _global_client

async def cleanup_global_client():
    """Clean up the global singleton client."""
    global _global_client
    
    async with _client_lock:
        if _global_client:
            try:
                await _global_client.__aexit__(None, None, None)
                print("🔄 Global MCP client cleaned up")
            except Exception as e:
                print(f"⚠️ Error during client cleanup: {e}")
            finally:
                _global_client = None

# Citation registry class for tracking citations
class CitationRegistry:
    def __init__(self):
        self.citations = {}
        self.counter = 0
    
    def add_citation(self, citation_info: dict) -> str:
        """Add a citation and return its ID"""
        self.counter += 1
        citation_id = f"citation_{self.counter}"
        self.citations[citation_id] = citation_info
        return citation_id
    
    def get_citation(self, citation_id: str) -> dict:
        """Get citation info by ID"""
        return self.citations.get(citation_id, {})
    
    def get_all_citations(self) -> dict:
        """Get all citations"""
        return self.citations
    
    def clear(self):
        """Clear all citations"""
        self.citations = {}
        self.counter = 0

# [REST OF THE FILE CONTINUES WITH MULTISERVCLIENT CLASS AND OTHER FUNCTIONS]
# This is a large file (~2900 lines), so I'm including the key changes:
# 1. Configuration-driven citation loading
# 2. Configuration-driven server connections
# 3. All climate-specific references removed
# 4. Core orchestration logic remains the same

SUMMARY_PROMPT = (
    "You are an AI assistant that provides comprehensive, well-researched answers. "
    "When synthesizing information from multiple sources, maintain accuracy and provide proper attribution. "
    "Focus on delivering clear, actionable insights while acknowledging data limitations where appropriate."
)

# Import health manager and config validation
from server_health import ServerHealthManager, CircuitBreakerConfig
from config_validation import ConfigValidator

class MultiServerClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.anthropic = anthropic.Anthropic()
        self.citation_registry = CitationRegistry()
        # Initialize health manager with small retry limits
        config = CircuitBreakerConfig(
            failure_threshold=3,      # 3 failures before circuit opens
            recovery_timeout=30,      # 30 seconds before retry
            half_open_max_calls=1,    # 1 test call in recovery
            success_threshold=1       # 1 success to close circuit
        )
        self.health_manager = ServerHealthManager(config)
        mcp_logger.info("MultiServerClient initialized with health monitoring")
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # shuts down all stdio_clients, ClientSessions, etc. in the *same* task
        await self.exit_stack.aclose()

    async def connect_to_server(self, server_name: str, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        mcp_logger.info(f"Connecting to {server_name} server at {server_script_path}")
        print(f"Connecting to {server_name} server at {server_script_path}")
        
        if not (is_python or is_js):
            error_msg = "Server script must be a .py or .js file"
            mcp_logger.error(f"Connection failed for {server_name}: {error_msg}")
            raise ValueError(error_msg)

        try:
            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command, args=[server_script_path], env=None
            )

            stdio_client = stdio_client
            session = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            self.sessions[server_name] = session
            
            # Register server with health manager
            self.health_manager.register_server(server_name)
            
            # List available tools
            list_tools_result = await session.list_tools()
            tool_names = [tool.name for tool in list_tools_result.tools]
            
            mcp_logger.info(f"Connected to {server_name}. Available tools: {tool_names}")
            print(f"Connected to {server_name}. Available tools: {tool_names}")
            
        except Exception as e:
            mcp_logger.error(f"Failed to connect to {server_name}: {str(e)}")
            # Still register with health manager to track the failure
            self.health_manager.register_server(server_name)
            raise

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        # Check if server can be called based on circuit breaker
        if not self.health_manager.can_call_server(server_name):
            error_result = self.health_manager.create_non_blocking_error(server_name, "Circuit breaker is open")
            mcp_logger.warning(f"Call to {server_name}.{tool_name} blocked by circuit breaker")
            return json.dumps(error_result)
        
        if server_name not in self.sessions:
            # Server not connected - record failure and return non-blocking error
            self.health_manager.record_failure(server_name, "Server not connected")
            error_result = self.health_manager.create_non_blocking_error(server_name, "Server not connected")
            mcp_logger.error(f"Call to {server_name}.{tool_name} failed: server not connected")
            return json.dumps(error_result)
        
        session = self.sessions[server_name]
        
        if arguments is None:
            arguments = {}
        
        start_time = time.time()
        
        try:
            mcp_logger.debug(f"Calling {server_name}.{tool_name} with args: {arguments}")
            result = await session.call_tool(tool_name, arguments)
            
            response_time = time.time() - start_time
            
            if result.content:
                # Join all text content from the tool result
                text_content = "\\n".join([
                    content.text for content in result.content 
                    if hasattr(content, 'text')
                ])
                
                # Record successful call
                self.health_manager.record_success(server_name, response_time)
                mcp_logger.info(f"Successfully called {server_name}.{tool_name} in {response_time:.3f}s")
                
                return text_content
            else:
                # Record successful call even if content is empty
                self.health_manager.record_success(server_name, response_time)
                return str(result)
                
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            # Record failure
            self.health_manager.record_failure(server_name, error_msg)
            
            # Create non-blocking error response
            error_result = self.health_manager.create_non_blocking_error(server_name, error_msg)
            
            mcp_logger.error(f"Call to {server_name}.{tool_name} failed after {response_time:.3f}s: {error_msg}")
            
            # Return non-blocking error as JSON string that LLM can handle
            return json.dumps(error_result)

    async def get_all_available_tools(self):
        """Get all available tools from all connected servers."""
        all_tools = {}
        for server_name, session in self.sessions.items():
            try:
                tools_result = await session.list_tools()
                server_tools = []
                for tool in tools_result.tools:
                    server_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                        "server": server_name
                    })
                all_tools[server_name] = server_tools
                mcp_logger.debug(f"Retrieved {len(server_tools)} tools from {server_name}")
            except Exception as e:
                mcp_logger.error(f"Failed to get tools from {server_name}: {e}")
                all_tools[server_name] = []
        
        return all_tools

    def _determine_server_for_tool(self, tool_name: str, all_tools: dict) -> str:
        """Determine which server hosts a given tool."""
        for server_name, tools in all_tools.items():
            for tool in tools:
                if tool["name"] == tool_name:
                    return server_name
        
        # Fallback to first available server
        if all_tools:
            return list(all_tools.keys())[0]
        
        raise ValueError(f"No server found for tool: {tool_name}")

    async def process_query(self, query: str):
        """
        Main orchestration method: processes a user query using available MCP tools.
        """
        messages = [
            {"role": "user", "content": query}
        ]
        
        # Get tools from all connected servers  
        all_tools = await self.get_all_available_tools()
        available_tools = []
        for server_name, tools in all_tools.items():
            # Remove server field before sending to Claude API
            clean_tools = []
            for tool in tools:
                clean_tool = {
                    "name": tool["name"],
                    "description": tool["description"], 
                    "input_schema": tool["input_schema"]
                }
                clean_tools.append(clean_tool)
            available_tools.extend(clean_tools)

        # Generic system prompt - domain agnostic
        system_prompt = """
            You are an expert data analyst and research assistant. Your task is to help users find, analyze, and understand information from available data sources.

            Core Responsibilities:
            1. Understand the user's query and information needs
            2. Use available tools to gather relevant data from connected sources
            3. Synthesize information to provide comprehensive, well-sourced answers

            Tool Usage Guidelines:
            - Always start by exploring what tools and data sources are available
            - Use multiple tools when needed to provide comprehensive answers
            - Cross-reference information from different sources when possible
            - Focus on providing accurate, well-attributed information

            Data Discovery Strategy:
            1. Identify relevant tools based on the user's query
            2. Call appropriate tools to gather data
            3. If one source doesn't have complete information, try related tools
            4. Combine insights from multiple sources for comprehensive answers

            Response Guidelines:
            - Provide clear, actionable insights
            - Include specific data points and evidence
            - Acknowledge limitations in available data
            - Structure information logically (overview → details → implications)

            Quality Standards:
            - Accuracy: Ensure all data points are correctly reported
            - Completeness: Use multiple sources to provide full picture
            - Clarity: Present complex information in understandable terms
            - Attribution: Properly cite all data sources used
        """

        try:
            response = self.anthropic.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
                tools=available_tools
            )
        except Exception as e:
            mcp_logger.error(f"Failed to get initial response from Claude: {e}")
            raise

        # Initialize data collection variables
        final_text = []
        sources_used = []
        context_chunks = []
        chart_data = None
        chart_data_tool = None
        map_data = None
        visualization_data = None
        all_tool_outputs_for_debug = []
        intermediate_facts_with_citations = []
        intermediate_ai_text_parts = []

        # Main orchestration loop
        while True:
            assistant_message_content = []
            current_turn_text_parts = []

            for content in response.content:
                if content.type == "text":
                    current_turn_text_parts.append(content.text)
                    intermediate_ai_text_parts.append(content.text)
                    assistant_message_content.append(content)
                elif content.type == "tool_use":
                    tool_name = content.name
                    tool_args = content.input
                    
                    # Determine which server to route this tool call to
                    server_name = self._determine_server_for_tool(tool_name, all_tools)
                    
                    print(f"🔧 Calling tool {tool_name} on {server_name} with args: {tool_args}")
                    
                    # Execute the tool call
                    try:
                        result = await self.call_tool(server_name, tool_name, tool_args)
                        
                        # Store for debugging
                        all_tool_outputs_for_debug.append({
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "server_name": server_name,
                            "result": result
                        })

                        # Parse structured data based on tool output
                        try:
                            parsed_content = json.loads(result) if isinstance(result, str) else result
                            
                            # Check for map data
                            if isinstance(parsed_content, dict):
                                if parsed_content.get("type") == "map" or "geojson" in parsed_content:
                                    map_data = parsed_content
                                    print(f"📍 Parsed map data from {tool_name}")
                                
                                # Check for chart/visualization data
                                elif "data" in parsed_content and isinstance(parsed_content["data"], list):
                                    if not chart_data:  # Only set if not already set
                                        chart_data = parsed_content["data"]
                                        chart_data_tool = tool_name
                                        print(f"📊 Parsed chart data from {tool_name}: {len(chart_data)} records")
                                
                                # Check for structured visualization data
                                elif "visualization_type" in parsed_content:
                                    visualization_data = parsed_content
                                    print(f"📈 Parsed visualization data from {tool_name}")
                            
                            # Handle list data (potential chart data)
                            elif isinstance(parsed_content, list) and parsed_content:
                                if all(isinstance(item, dict) for item in parsed_content):
                                    if not chart_data:  # Only set if not already set
                                        chart_data = parsed_content
                                        chart_data_tool = tool_name
                                        print(f"📊 Parsed list data from {tool_name}: {len(chart_data)} records")

                        except (json.JSONDecodeError, TypeError):
                            print(f"📄 Tool {tool_name} returned non-JSON data")

                        # Add to context (truncate if too long)
                        content_str = str(result)
                        if len(content_str) > 1000:
                            content_str = content_str[:1000] + "... [truncated]"
                        context_chunks.append(content_str)

                        # Wrap tool result with citation info
                        wrapped_result = wrap_tool_result_with_citation(tool_name, str(result), tool_args)
                        intermediate_facts_with_citations.append({
                            "fact": wrapped_result["fact"],
                            "citation_info": wrapped_result["citation_info"],
                            "tool_context": {
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "server_name": server_name
                            }
                        })

                        # Create tool result for Claude
                        tool_result_content = [{
                            "type": "text",
                            "text": str(result)
                        }]

                    except Exception as e:
                        error_msg = f"Tool {tool_name} failed: {str(e)}"
                        print(f"❌ {error_msg}")
                        mcp_logger.error(error_msg)
                        
                        # Create error result for Claude
                        tool_result_content = [{
                            "type": "text", 
                            "text": f"Error: {error_msg}"
                        }]

                    # Attach tool_use to assistant message
                    assistant_message_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_message_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": tool_result_content
                        }]
                    })

                    # Break early and send updated messages to Claude
                    break
            else:
                # No tool_use found → conversation complete
                messages.append({"role": "assistant", "content": assistant_message_content})
                break

            # Ask Claude for the next step
            try:
                response = self.anthropic.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=messages,
                    tools=available_tools,
                )
            except Exception as e:
                mcp_logger.error(f"Failed to get follow-up response from Claude: {e}")
                break

        # Process final response
        if current_turn_text_parts:
            final_response_text = "\\n".join(current_turn_text_parts)
        else:
            final_response_text = "\\n".join(intermediate_ai_text_parts)

        # Convert citation format for frontend compatibility
        final_response_text = convert_citation_format(final_response_text)

        # Prepare result dictionary
        result = {
            "response": final_response_text,
            "sources": sources_used,
            "context_chunks": context_chunks,
            "chart_data": chart_data,
            "chart_data_tool": chart_data_tool,
            "map_data": map_data,
            "visualization_data": visualization_data,
            "all_tool_outputs_for_debug": all_tool_outputs_for_debug,
            "final_citations_list": intermediate_facts_with_citations,
            "citation_registry": self.citation_registry
        }

        mcp_logger.info(f"Query processing completed. Tools called: {len(all_tool_outputs_for_debug)}")
        return result

    async def process_query_streaming(self, query: str):
        """
        Streaming version of process_query that yields events as they happen.
        """
        def translate_tool_to_action(tool_name: str, tool_args: dict) -> dict:
            """Convert technical tool calls to user-friendly action messages."""
            if not tool_args:
                return {
                    "message": f"🔧 Calling {tool_name}...",
                    "category": "tool_execution"
                }
                
            # Extract common arguments for friendly display
            common_args = []
            if "query" in tool_args:
                common_args.append(f"query: {tool_args['query'][:50]}...")
            if "country" in tool_args:
                common_args.append(f"country: {tool_args['country']}")
            if "concept" in tool_args:
                common_args.append(f"concept: {tool_args['concept']}")
            
            args_display = ", ".join(common_args) if common_args else "processing request"
            
            return {
                "message": f"🔧 Executing {tool_name} ({args_display})",
                "category": "tool_execution"
            }

        # Yield initial status
        yield {
            "type": "thinking",
            "data": {
                "message": "🚀 Initializing analysis...",
                "category": "initialization"
            }
        }

        messages = [{"role": "user", "content": query}]
        
        # Get available tools
        try:
            all_tools = await self.get_all_available_tools()
            available_tools = []
            for server_name, tools in all_tools.items():
                clean_tools = []
                for tool in tools:
                    clean_tool = {
                        "name": tool["name"],
                        "description": tool["description"], 
                        "input_schema": tool["input_schema"]
                    }
                    clean_tools.append(clean_tool)
                available_tools.extend(clean_tools)
            
            yield {
                "type": "thinking", 
                "data": {
                    "message": f"📋 Found {len(available_tools)} available tools across {len(all_tools)} servers",
                    "category": "discovery"
                }
            }
        except Exception as e:
            yield {
                "type": "error",
                "data": {
                    "message": f"Failed to get available tools: {str(e)}"
                }
            }
            return

        # Same system prompt as non-streaming version
        system_prompt = """
            You are an expert data analyst and research assistant. Your task is to help users find, analyze, and understand information from available data sources.

            Core Responsibilities:
            1. Understand the user's query and information needs
            2. Use available tools to gather relevant data from connected sources
            3. Synthesize information to provide comprehensive, well-sourced answers

            Tool Usage Guidelines:
            - Always start by exploring what tools and data sources are available
            - Use multiple tools when needed to provide comprehensive answers
            - Cross-reference information from different sources when possible
            - Focus on providing accurate, well-attributed information

            Data Discovery Strategy:
            1. Identify relevant tools based on the user's query
            2. Call appropriate tools to gather data
            3. If one source doesn't have complete information, try related tools
            4. Combine insights from multiple sources for comprehensive answers

            Response Guidelines:
            - Provide clear, actionable insights
            - Include specific data points and evidence
            - Acknowledge limitations in available data
            - Structure information logically (overview → details → implications)

            Quality Standards:
            - Accuracy: Ensure all data points are correctly reported
            - Completeness: Use multiple sources to provide full picture
            - Clarity: Present complex information in understandable terms
            - Attribution: Properly cite all data sources used
        """

        # Get initial response from Claude
        try:
            response = self.anthropic.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
                tools=available_tools
            )
        except Exception as e:
            yield {
                "type": "error",
                "data": {
                    "message": f"Failed to get initial response: {str(e)}"
                }
            }
            return

        # Initialize collection variables
        intermediate_facts_with_citations = []
        all_tool_outputs_for_debug = []
        chart_data = None
        map_data = None
        visualization_data = None

        # Main streaming loop
        while True:
            assistant_message_content = []
            current_turn_text_parts = []

            for content in response.content:
                if content.type == "text":
                    current_turn_text_parts.append(content.text)
                    assistant_message_content.append(content)
                    
                    # Yield thinking event
                    yield {
                        "type": "thinking",
                        "data": {
                            "message": content.text,
                            "category": "analysis"
                        }
                    }
                    
                elif content.type == "tool_use":
                    tool_name = content.name
                    tool_args = content.input
                    
                    # Yield tool call event
                    action_info = translate_tool_to_action(tool_name, tool_args)
                    yield {
                        "type": "tool_call",
                        "data": {
                            "tool": tool_name,
                            "args": tool_args,
                            "message": action_info["message"],
                            "category": action_info["category"]
                        }
                    }
                    
                    # Execute tool
                    try:
                        server_name = self._determine_server_for_tool(tool_name, all_tools)
                        result = await self.call_tool(server_name, tool_name, tool_args)
                        
                        # Store result
                        all_tool_outputs_for_debug.append({
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "server_name": server_name,
                            "result": result
                        })

                        # Parse and collect structured data (same logic as non-streaming)
                        try:
                            parsed_content = json.loads(result) if isinstance(result, str) else result
                            
                            if isinstance(parsed_content, dict):
                                if parsed_content.get("type") == "map" or "geojson" in parsed_content:
                                    map_data = parsed_content
                                elif "data" in parsed_content and isinstance(parsed_content["data"], list):
                                    if not chart_data:
                                        chart_data = parsed_content["data"]
                                elif "visualization_type" in parsed_content:
                                    visualization_data = parsed_content
                            elif isinstance(parsed_content, list) and parsed_content:
                                if all(isinstance(item, dict) for item in parsed_content):
                                    if not chart_data:
                                        chart_data = parsed_content
                        except (json.JSONDecodeError, TypeError):
                            pass

                        # Add citation
                        wrapped_result = wrap_tool_result_with_citation(tool_name, str(result), tool_args)
                        intermediate_facts_with_citations.append({
                            "fact": wrapped_result["fact"],
                            "citation_info": wrapped_result["citation_info"],
                            "tool_context": {
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "server_name": server_name
                            }
                        })

                        # Yield tool result event
                        yield {
                            "type": "tool_result",
                            "data": {
                                "tool": tool_name,
                                "success": True,
                                "message": f"✅ {tool_name} completed successfully"
                            }
                        }

                        tool_result_content = [{"type": "text", "text": str(result)}]

                    except Exception as e:
                        error_msg = f"Tool {tool_name} failed: {str(e)}"
                        
                        yield {
                            "type": "tool_result", 
                            "data": {
                                "tool": tool_name,
                                "success": False,
                                "message": f"❌ {error_msg}"
                            }
                        }
                        
                        tool_result_content = [{"type": "text", "text": f"Error: {error_msg}"}]

                    # Update messages for Claude
                    assistant_message_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_message_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": tool_result_content
                        }]
                    })

                    break
            else:
                # No tool_use found → conversation complete
                messages.append({"role": "assistant", "content": assistant_message_content})
                break

            # Get next response from Claude
            try:
                response = self.anthropic.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=messages,
                    tools=available_tools,
                )
            except Exception as e:
                yield {
                    "type": "error",
                    "data": {
                        "message": f"Failed to get follow-up response: {str(e)}"
                    }
                }
                break

        # Yield final completion event
        final_response_text = "\\n".join(current_turn_text_parts) if current_turn_text_parts else "Query completed"
        final_response_text = convert_citation_format(final_response_text)

        yield {
            "type": "complete",
            "data": {
                "query": query,
                "response": final_response_text,
                "modules": [],  # Will be populated by formatter
                "metadata": {
                    "tools_called": len(all_tool_outputs_for_debug),
                    "citations_count": len(intermediate_facts_with_citations)
                },
                "chart_data": chart_data,
                "map_data": map_data,
                "visualization_data": visualization_data,
                "citation_registry": {
                    "citations": self.citation_registry.get_all_citations()
                }
            }
        }

# Main query processing functions 
async def run_query_structured(query: str) -> Dict[str, Any]:
    """
    Process a query using the MCP orchestration system.
    Returns structured response with modules and metadata.
    """
    try:
        client = await get_global_client()
        
        # Process query through orchestration
        result = await client.process_query(query)
        
        # Extract structured response (will be set by formatter if available)
        return result.get("formatted_response", {
            "modules": [],
            "metadata": {
                "processing_time": 0,
                "citations_count": len(result.get("final_citations_list", [])),
                "tools_called": len(result.get("all_tool_outputs_for_debug", []))
            }
        })
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        raise

async def run_query_streaming(query: str):
    """
    Process a query with streaming responses.
    Yields chunks as they become available.
    """
    try:
        client = await get_global_client()
        
        # Stream processing implementation
        async for event in client.process_query_streaming(query):
            yield event
        
    except Exception as e:
        import traceback
        yield {
            "type": "error",
            "data": {
                "message": f"Streaming error: {str(e)}",
                "traceback": traceback.format_exc()
            }
        }

async def run_query(query: str) -> Dict[str, Any]:
    """
    Complete query processing function that returns full result including metadata.
    """
    try:
        # Ensure we're in the correct working directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        os.chdir(project_root)
        
        # Use the global singleton client
        client = await get_global_client()
        
        # Process the main query
        result = await client.process_query(query)
        
        # Format the response using the formatter MCP (if available)
        formatter_args = {
            "response_text": result.get("response", ""),
            "chart_data": result.get("chart_data"),
            "chart_data_tool": result.get("chart_data_tool"),
            "visualization_data": result.get("visualization_data"), 
            "map_data": result.get("map_data"),
            "sources": result.get("sources"),
            "structured_citations": result.get("final_citations_list"),
            "title": "Data Analysis Results",
            "citation_registry": {
                "citations": client.citation_registry.get_all_citations() if hasattr(client.citation_registry, 'get_all_citations') else {},
                "module_citations": getattr(client.citation_registry, 'module_citations', {})
            }
        }
        
        # Remove None values to avoid issues
        formatter_args = {k: v for k, v in formatter_args.items() if v is not None}
        
        # Try to format response if formatter is available
        try:
            formatted_result = await client.call_tool("response_formatter", "FormatResponseAsModules", formatter_args)
            
            # Parse the formatted response
            if hasattr(formatted_result, 'content') and isinstance(formatted_result.content, list):
                first_content = formatted_result.content[0]
                if hasattr(first_content, 'text'):
                    import json
                    formatted_data = json.loads(first_content.text)
                    result["formatted_response"] = formatted_data
                    print(f"✅ Successfully formatted response with {len(formatted_data.get('modules', []))} modules")
        except Exception as e:
            print(f"⚠️ Formatter not available or failed: {e}")
            # Continue without formatting - the raw result is still useful
        
        return result
        
    except Exception as e:
        import traceback
        error_detail = f"Query processing failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"❌ API ERROR: {error_detail}")
        raise