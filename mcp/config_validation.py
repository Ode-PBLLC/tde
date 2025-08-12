"""
ODE MCP Generic - Configuration Validation
Validates JSON configuration files against schemas to ensure correct setup.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from jsonschema import validate, ValidationError, Draft7Validator
from jsonschema.exceptions import SchemaError

# Configure logging for config validation
config_logger = logging.getLogger('config_validation')

class ConfigValidationResult:
    """Result of configuration validation"""
    
    def __init__(self, config_name: str):
        self.config_name = config_name
        self.valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.schema_path: Optional[str] = None
        self.config_path: Optional[str] = None

    def add_error(self, message: str):
        """Add an error to the validation result"""
        self.valid = False
        self.errors.append(message)
        config_logger.error(f"Config validation error for {self.config_name}: {message}")

    def add_warning(self, message: str):
        """Add a warning to the validation result"""
        self.warnings.append(message)
        config_logger.warning(f"Config validation warning for {self.config_name}: {message}")

    def summary(self) -> str:
        """Generate a summary of the validation results"""
        if self.valid:
            warning_text = f" ({len(self.warnings)} warnings)" if self.warnings else ""
            return f"✅ {self.config_name} validation passed{warning_text}"
        else:
            return f"❌ {self.config_name} validation failed: {len(self.errors)} errors, {len(self.warnings)} warnings"

class ConfigValidator:
    """Validates configuration files against JSON schemas"""
    
    def __init__(self, schemas_dir: str = "config/schemas"):
        self.schemas_dir = schemas_dir
        self.schemas: Dict[str, Dict] = {}
        
    def load_schemas(self) -> bool:
        """Load all JSON schemas from the schemas directory"""
        if not os.path.exists(self.schemas_dir):
            config_logger.error(f"Schemas directory not found: {self.schemas_dir}")
            return False
        
        schema_files = {
            "servers": "servers.schema.json",
            "citation_sources": "citation_sources.schema.json", 
            "featured_queries": "featured_queries.schema.json"
        }
        
        success = True
        for schema_name, schema_file in schema_files.items():
            schema_path = os.path.join(self.schemas_dir, schema_file)
            
            try:
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schema = json.load(f)
                        # Validate the schema itself
                        Draft7Validator.check_schema(schema)
                        self.schemas[schema_name] = schema
                        config_logger.info(f"Loaded schema for {schema_name}")
                else:
                    config_logger.warning(f"Schema file not found: {schema_path}")
                    success = False
                    
            except (json.JSONDecodeError, SchemaError) as e:
                config_logger.error(f"Invalid schema file {schema_path}: {e}")
                success = False
            except Exception as e:
                config_logger.error(f"Failed to load schema {schema_path}: {e}")
                success = False
        
        return success

    def validate_servers_config(self, config_path: str = "config/servers.json") -> ConfigValidationResult:
        """Validate servers configuration file"""
        result = ConfigValidationResult("servers.json")
        result.config_path = config_path
        
        if "servers" not in self.schemas:
            result.add_error("Servers schema not loaded")
            return result
        
        # Load and validate configuration
        config_data = self._load_config_file(config_path, result)
        if config_data is None:
            return result
        
        # Validate against schema
        self._validate_against_schema(config_data, self.schemas["servers"], result)
        
        # Additional domain-specific validations
        if result.valid and "servers" in config_data:
            self._validate_servers_specific(config_data["servers"], result)
        
        return result

    def validate_citation_sources_config(self, config_path: str = "config/citation_sources.json") -> ConfigValidationResult:
        """Validate citation sources configuration file"""
        result = ConfigValidationResult("citation_sources.json")
        result.config_path = config_path
        
        if "citation_sources" not in self.schemas:
            result.add_error("Citation sources schema not loaded")
            return result
        
        # Load and validate configuration
        config_data = self._load_config_file(config_path, result)
        if config_data is None:
            return result
        
        # Validate against schema
        self._validate_against_schema(config_data, self.schemas["citation_sources"], result)
        
        # Additional domain-specific validations
        if result.valid and "tool_citations" in config_data:
            self._validate_citations_specific(config_data["tool_citations"], result)
        
        return result

    def validate_featured_queries_config(self, config_path: str = "config/featured_queries.json") -> ConfigValidationResult:
        """Validate featured queries configuration file"""
        result = ConfigValidationResult("featured_queries.json")
        result.config_path = config_path
        
        if "featured_queries" not in self.schemas:
            result.add_error("Featured queries schema not loaded")
            return result
        
        # Load and validate configuration
        config_data = self._load_config_file(config_path, result)
        if config_data is None:
            return result
        
        # Validate against schema
        self._validate_against_schema(config_data, self.schemas["featured_queries"], result)
        
        # Additional domain-specific validations
        if result.valid and "queries" in config_data:
            self._validate_queries_specific(config_data["queries"], result)
        
        return result

    def validate_all_configs(self) -> List[ConfigValidationResult]:
        """Validate all configuration files"""
        results = []
        
        # Validate servers config
        servers_result = self.validate_servers_config()
        results.append(servers_result)
        
        # Validate citation sources config
        citation_result = self.validate_citation_sources_config()
        results.append(citation_result)
        
        # Validate featured queries config (optional)
        if os.path.exists("config/featured_queries.json"):
            queries_result = self.validate_featured_queries_config()
            results.append(queries_result)
        else:
            config_logger.info("Featured queries config not found - skipping (optional)")
        
        return results

    def _load_config_file(self, config_path: str, result: ConfigValidationResult) -> Optional[Dict]:
        """Load configuration file with error handling"""
        if not os.path.exists(config_path):
            result.add_error(f"Configuration file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in configuration file: {e}")
            return None
        except Exception as e:
            result.add_error(f"Failed to load configuration file: {e}")
            return None

    def _validate_against_schema(self, config_data: Dict, schema: Dict, result: ConfigValidationResult):
        """Validate configuration data against schema"""
        try:
            validate(instance=config_data, schema=schema)
            config_logger.info(f"Schema validation passed for {result.config_name}")
        except ValidationError as e:
            # Make the error message more user-friendly
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            result.add_error(f"Schema validation failed at '{error_path}': {e.message}")

    def _validate_servers_specific(self, servers: List[Dict], result: ConfigValidationResult):
        """Additional validation specific to servers configuration"""
        server_names = set()
        
        for i, server in enumerate(servers):
            # Check for duplicate server names
            server_name = server.get("name", f"server_{i}")
            if server_name in server_names:
                result.add_error(f"Duplicate server name '{server_name}' found")
            else:
                server_names.add(server_name)
            
            # Check if server file exists (warning, not error)
            server_path = server.get("path", "")
            if server_path and not os.path.exists(server_path):
                result.add_warning(f"Server file not found: {server_path}")

    def _validate_citations_specific(self, citations: Dict, result: ConfigValidationResult):
        """Additional validation specific to citations configuration"""
        # Check for potentially problematic URLs
        for tool_name, citation_info in citations.items():
            source_url = citation_info.get("source_url", "")
            
            # Check for placeholder URLs
            if "TODO" in source_url or "example.com" in source_url:
                result.add_warning(f"Citation for '{tool_name}' has placeholder URL")
            
            # Check for empty required fields
            for field in ["source_name", "provider", "spatial_coverage", "temporal_coverage"]:
                value = citation_info.get(field, "")
                if "TODO" in value or not value.strip():
                    result.add_warning(f"Citation for '{tool_name}' has placeholder/empty field '{field}'")

    def _validate_queries_specific(self, queries: List[Dict], result: ConfigValidationResult):
        """Additional validation specific to queries configuration"""
        query_ids = set()
        
        for query in queries:
            # Check for duplicate query IDs
            query_id = query.get("id", "")
            if query_id in query_ids:
                result.add_error(f"Duplicate query ID '{query_id}' found")
            else:
                query_ids.add(query_id)
            
            # Check query length (very long queries might cause issues)
            query_text = query.get("query", "")
            if len(query_text) > 500:
                result.add_warning(f"Query '{query_id}' is very long ({len(query_text)} characters)")

def validate_config_file_with_schema(config_path: str, schema_path: str) -> ConfigValidationResult:
    """
    Validate a single configuration file against a specific schema.
    Utility function for one-off validations.
    """
    config_name = os.path.basename(config_path)
    result = ConfigValidationResult(config_name)
    result.config_path = config_path
    result.schema_path = schema_path
    
    # Load schema
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
            Draft7Validator.check_schema(schema)  # Validate schema itself
    except Exception as e:
        result.add_error(f"Failed to load schema {schema_path}: {e}")
        return result
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        result.add_error(f"Failed to load config {config_path}: {e}")
        return result
    
    # Validate
    try:
        validate(instance=config_data, schema=schema)
        config_logger.info(f"Validation passed for {config_name}")
    except ValidationError as e:
        error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
        result.add_error(f"Validation failed at '{error_path}': {e.message}")
    
    return result

def setup_config_logging():
    """Set up logging for configuration validation"""
    config_logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler for config logs
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler('logs/config_validation.log')
    file_handler.setFormatter(formatter)
    config_logger.addHandler(file_handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    config_logger.addHandler(console_handler)

# Initialize logging when module is imported
setup_config_logging()