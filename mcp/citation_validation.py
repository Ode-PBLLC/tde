"""
ODE MCP Generic - Citation Validation System
Validates citation mappings for all MCP tools to ensure proper attribution.
"""
import os
import json
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from mcp import ClientSession

# Configure logging for citation validation
citation_logger = logging.getLogger('citation_validation')

class CitationValidationResult:
    """Result of citation validation process"""
    
    def __init__(self):
        self.valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.missing_citations: Set[str] = set()
        self.orphaned_citations: Set[str] = set()
        self.tools_validated: Set[str] = set()
        self.servers_validated: Set[str] = set()

    def add_error(self, message: str):
        """Add an error to the validation result"""
        self.valid = False
        self.errors.append(message)
        citation_logger.error(f"Citation validation error: {message}")

    def add_warning(self, message: str):
        """Add a warning to the validation result"""
        self.warnings.append(message)
        citation_logger.warning(f"Citation validation warning: {message}")

    def summary(self) -> str:
        """Generate a summary of the validation results"""
        if self.valid:
            return f"✅ Citation validation passed: {len(self.tools_validated)} tools validated across {len(self.servers_validated)} servers"
        else:
            return f"❌ Citation validation failed: {len(self.errors)} errors, {len(self.warnings)} warnings"

class CitationValidator:
    """Validates citation mappings against available MCP tools"""
    
    def __init__(self, citation_config_path: str = "config/citation_sources.json"):
        self.citation_config_path = citation_config_path
        self.citation_mapping: Dict[str, Dict] = {}
        
    def load_citation_mapping(self) -> bool:
        """Load citation mapping from configuration file"""
        try:
            if not os.path.exists(self.citation_config_path):
                citation_logger.error(f"Citation config file not found: {self.citation_config_path}")
                return False
                
            with open(self.citation_config_path, 'r') as f:
                config = json.load(f)
                self.citation_mapping = config.get("tool_citations", {})
                citation_logger.info(f"Loaded {len(self.citation_mapping)} citation mappings from {self.citation_config_path}")
                return True
                
        except Exception as e:
            citation_logger.error(f"Failed to load citation config: {e}")
            return False

    def get_all_server_tools(self, sessions: Dict[str, ClientSession]) -> Dict[str, List[str]]:
        """Get all available tools from connected MCP servers"""
        server_tools = {}
        
        for server_name, session in sessions.items():
            try:
                # Get tools list from the session
                tools_result = session._tools if hasattr(session, '_tools') else []
                if hasattr(session, 'list_tools'):
                    # If we can call list_tools, do that instead
                    import asyncio
                    if asyncio.iscoroutinefunction(session.list_tools):
                        # We can't call async functions from here, so we'll need to handle this differently
                        tools_result = getattr(session, '_cached_tools', [])
                    else:
                        tools_result = session.list_tools()
                
                if hasattr(tools_result, 'tools'):
                    tool_names = [tool.name for tool in tools_result.tools]
                else:
                    # Fallback: try to extract from session attributes
                    tool_names = []
                    
                server_tools[server_name] = tool_names
                citation_logger.info(f"Found {len(tool_names)} tools in server '{server_name}': {tool_names}")
                
            except Exception as e:
                citation_logger.error(f"Failed to get tools from server '{server_name}': {e}")
                server_tools[server_name] = []
                
        return server_tools

    async def get_all_server_tools_async(self, sessions: Dict[str, ClientSession]) -> Dict[str, List[str]]:
        """Async version of get_all_server_tools"""
        server_tools = {}
        
        for server_name, session in sessions.items():
            try:
                # Call list_tools async
                tools_result = await session.list_tools()
                tool_names = [tool.name for tool in tools_result.tools]
                server_tools[server_name] = tool_names
                citation_logger.info(f"Found {len(tool_names)} tools in server '{server_name}': {tool_names}")
                
            except Exception as e:
                citation_logger.error(f"Failed to get tools from server '{server_name}': {e}")
                server_tools[server_name] = []
                
        return server_tools

    def validate_citations_against_tools(self, server_tools: Dict[str, List[str]]) -> CitationValidationResult:
        """Validate that all tools have corresponding citation mappings"""
        result = CitationValidationResult()
        
        if not self.citation_mapping:
            result.add_error("No citation mappings loaded")
            return result
        
        # Collect all available tools from all servers
        all_tools = set()
        for server_name, tools in server_tools.items():
            all_tools.update(tools)
            result.servers_validated.add(server_name)
        
        citation_logger.info(f"Validating citations for {len(all_tools)} tools across {len(server_tools)} servers")
        
        # Check for missing citations
        for tool_name in all_tools:
            if tool_name in self.citation_mapping:
                result.tools_validated.add(tool_name)
                # Validate citation structure
                citation_info = self.citation_mapping[tool_name]
                if not self._validate_citation_structure(tool_name, citation_info, result):
                    continue
            else:
                result.missing_citations.add(tool_name)
                result.add_warning(f"Tool '{tool_name}' has no citation mapping - will use fallback")
        
        # Check for orphaned citations (citations for non-existent tools)
        for citation_tool in self.citation_mapping.keys():
            if citation_tool not in all_tools:
                result.orphaned_citations.add(citation_tool)
                result.add_warning(f"Citation exists for non-existent tool '{citation_tool}'")
        
        # Summary logging
        citation_logger.info(f"Citation validation summary: {len(result.tools_validated)} valid, "
                           f"{len(result.missing_citations)} missing, {len(result.orphaned_citations)} orphaned")
        
        return result

    def _validate_citation_structure(self, tool_name: str, citation_info: Dict, result: CitationValidationResult) -> bool:
        """Validate that a citation has all required fields"""
        required_fields = ["source_name", "provider", "spatial_coverage", "temporal_coverage", "source_url"]
        
        if not isinstance(citation_info, dict):
            result.add_error(f"Citation for '{tool_name}' is not a dictionary")
            return False
        
        for field in required_fields:
            if field not in citation_info:
                result.add_error(f"Citation for '{tool_name}' missing required field '{field}'")
                return False
            
            if not isinstance(citation_info[field], str):
                result.add_error(f"Citation for '{tool_name}' field '{field}' must be a string")
                return False
        
        # Check for empty required fields (except source_url which can be empty)
        for field in required_fields:
            if field != "source_url" and not citation_info[field].strip():
                result.add_warning(f"Citation for '{tool_name}' has empty field '{field}'")
        
        return True

    def generate_missing_citation_template(self, missing_tools: Set[str]) -> Dict[str, Dict]:
        """Generate a template for missing citation mappings"""
        template = {}
        
        for tool_name in sorted(missing_tools):
            template[tool_name] = {
                "source_name": f"TODO: Add source name for {tool_name}",
                "provider": "TODO: Add provider organization",
                "spatial_coverage": "TODO: Add geographic coverage",
                "temporal_coverage": "TODO: Add temporal coverage", 
                "source_url": "TODO: Add source URL or leave empty"
            }
        
        return template

def validate_citation_coverage_sync(citation_config_path: str = "config/citation_sources.json") -> CitationValidationResult:
    """
    Synchronous validation for use during startup.
    This validates citation file structure but cannot validate against live servers.
    """
    validator = CitationValidator(citation_config_path)
    result = CitationValidationResult()
    
    if not validator.load_citation_mapping():
        result.add_error("Failed to load citation mapping")
        return result
    
    # Validate citation file structure
    for tool_name, citation_info in validator.citation_mapping.items():
        if validator._validate_citation_structure(tool_name, citation_info, result):
            result.tools_validated.add(tool_name)
    
    citation_logger.info(f"Citation file structure validation: {len(result.tools_validated)} tools with valid citations")
    return result

async def validate_citation_coverage_async(sessions: Dict[str, ClientSession], 
                                         citation_config_path: str = "config/citation_sources.json") -> CitationValidationResult:
    """
    Full async validation that checks citations against live server tools.
    Use this for comprehensive validation when servers are running.
    """
    validator = CitationValidator(citation_config_path)
    
    if not validator.load_citation_mapping():
        result = CitationValidationResult()
        result.add_error("Failed to load citation mapping")
        return result
    
    # Get all tools from connected servers
    server_tools = await validator.get_all_server_tools_async(sessions)
    
    # Validate citations against actual tools
    return validator.validate_citations_against_tools(server_tools)

def setup_citation_logging():
    """Set up logging for citation validation"""
    citation_logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler for citation logs
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler('logs/citation_validation.log')
    file_handler.setFormatter(formatter)
    citation_logger.addHandler(file_handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    citation_logger.addHandler(console_handler)

# Initialize logging when module is imported
setup_citation_logging()