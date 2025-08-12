#!/usr/bin/env python3
"""
ODE MCP Generic - Configuration Validation CLI Tool
Validates all configuration files against their JSON schemas.
"""
import os
import sys
import argparse
from typing import List

# Add the mcp directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mcp'))

from config_validation import ConfigValidator, ConfigValidationResult

def print_validation_result(result: ConfigValidationResult, verbose: bool = False):
    """Print validation result in a readable format"""
    print(f"\n📋 {result.config_name}")
    print("=" * (len(result.config_name) + 4))
    
    if result.config_path:
        print(f"📁 Config file: {result.config_path}")
    if result.schema_path:
        print(f"📋 Schema file: {result.schema_path}")
    
    if result.valid:
        print(f"✅ {result.summary()}")
    else:
        print(f"❌ {result.summary()}")
        
        if result.errors:
            print(f"\n🚨 Errors ({len(result.errors)}):")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")
    
    if result.warnings and (verbose or not result.valid):
        print(f"\n⚠️ Warnings ({len(result.warnings)}):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. {warning}")

def validate_all_configs(verbose: bool = False) -> bool:
    """Validate all configuration files"""
    print("🔍 ODE MCP Generic - Configuration Validation")
    print("=" * 50)
    
    validator = ConfigValidator()
    
    # Load schemas first
    print("📚 Loading validation schemas...")
    if not validator.load_schemas():
        print("❌ Failed to load validation schemas")
        return False
    
    print("✅ Schemas loaded successfully")
    
    # Validate all configs
    results = validator.validate_all_configs()
    
    all_valid = True
    total_errors = 0
    total_warnings = 0
    
    for result in results:
        print_validation_result(result, verbose)
        if not result.valid:
            all_valid = False
        total_errors += len(result.errors)
        total_warnings += len(result.warnings)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    print("=" * 50)
    
    if all_valid:
        print("🎉 All configuration files are valid!")
    else:
        print("❌ Some configuration files have errors")
    
    print(f"📋 Files validated: {len(results)}")
    print(f"🚨 Total errors: {total_errors}")
    print(f"⚠️ Total warnings: {total_warnings}")
    
    if total_warnings > 0:
        print("\n💡 Tip: Run with --verbose to see all warnings")
    
    return all_valid

def validate_single_file(config_path: str, schema_path: str = None, verbose: bool = False) -> bool:
    """Validate a single configuration file"""
    from config_validation import validate_config_file_with_schema
    
    print(f"🔍 Validating {config_path}")
    print("=" * 50)
    
    if schema_path:
        # Use specific schema
        result = validate_config_file_with_schema(config_path, schema_path)
    else:
        # Auto-detect schema based on filename
        validator = ConfigValidator()
        if not validator.load_schemas():
            print("❌ Failed to load validation schemas")
            return False
        
        filename = os.path.basename(config_path)
        if "servers" in filename:
            result = validator.validate_servers_config(config_path)
        elif "citation" in filename:
            result = validator.validate_citation_sources_config(config_path)
        elif "queries" in filename:
            result = validator.validate_featured_queries_config(config_path)
        else:
            print(f"❌ Cannot auto-detect schema for file: {filename}")
            print("Please specify --schema-path for this file")
            return False
    
    print_validation_result(result, verbose)
    
    return result.valid

def main():
    parser = argparse.ArgumentParser(
        description="Validate ODE MCP Generic configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all configuration files
  python scripts/validate_config.py
  
  # Validate with verbose output
  python scripts/validate_config.py --verbose
  
  # Validate a specific file
  python scripts/validate_config.py --file config/servers.json
  
  # Validate with custom schema
  python scripts/validate_config.py --file myconfig.json --schema config/schemas/servers.schema.json
        """
    )
    
    parser.add_argument(
        "--file", "-f",
        help="Validate a specific configuration file"
    )
    
    parser.add_argument(
        "--schema", "-s",
        help="Schema file to use for validation (for single file validation)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including all warnings"
    )
    
    args = parser.parse_args()
    
    # Change to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    try:
        if args.file:
            # Validate single file
            success = validate_single_file(args.file, args.schema, args.verbose)
        else:
            # Validate all files
            success = validate_all_configs(args.verbose)
        
        if success:
            print("\n✅ Validation completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Validation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()