#!/usr/bin/env python3
"""
Test script for configuration validation system.
Tests JSON schema validation and configuration loading.
"""
import os
import sys
import json
import tempfile
from typing import Dict, Any

# Add the mcp directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mcp'))

from config_validation import ConfigValidator, validate_config_file_with_schema

def create_test_config(config_type: str, valid: bool = True) -> Dict[str, Any]:
    """Create test configuration data"""
    
    if config_type == "servers":
        if valid:
            return {
                "servers": [
                    {
                        "name": "test_server",
                        "path": "mcp/test_server.py",
                        "description": "Test server for validation"
                    },
                    {
                        "name": "another_server",
                        "path": "mcp/another_server.js",
                        "description": "Another test server",
                        "enabled": True,
                        "required": False
                    }
                ]
            }
        else:
            return {
                "servers": [
                    {
                        "name": "",  # Invalid: empty name
                        "path": "invalid_extension.txt",  # Invalid: wrong extension
                        # Missing description
                    },
                    {
                        "name": "duplicate_name",
                        "path": "mcp/server1.py",
                        "description": "First server"
                    },
                    {
                        "name": "duplicate_name",  # Invalid: duplicate name
                        "path": "mcp/server2.py",
                        "description": "Second server"
                    }
                ]
            }
    
    elif config_type == "citation_sources":
        if valid:
            return {
                "tool_citations": {
                    "TestTool1": {
                        "source_name": "Test Data Source",
                        "provider": "Test Provider Organization",
                        "spatial_coverage": "Global",
                        "temporal_coverage": "2020-2024",
                        "source_url": "https://example.com/data"
                    },
                    "TestTool2": {
                        "source_name": "Another Data Source",
                        "provider": "Another Provider",
                        "spatial_coverage": "United States",
                        "temporal_coverage": "Real-time",
                        "source_url": ""  # Empty URL is allowed
                    }
                }
            }
        else:
            return {
                "tool_citations": {
                    "InvalidTool1": {
                        "source_name": "Missing fields example"
                        # Missing required fields
                    },
                    "InvalidTool2": {
                        "source_name": "",  # Empty required field
                        "provider": "Provider",
                        "spatial_coverage": "Coverage",
                        "temporal_coverage": "Time",
                        "source_url": "not-a-url"  # Invalid URL format
                    }
                }
            }
    
    elif config_type == "featured_queries":
        if valid:
            return {
                "queries": [
                    {
                        "id": "test_query_1",
                        "title": "Test Query 1",
                        "description": "A test query for validation",
                        "query": "Show me test data",
                        "category": "testing",
                        "difficulty": "beginner",
                        "tags": ["test", "validation"]
                    },
                    {
                        "id": "test_query_2",
                        "title": "Advanced Test Query",
                        "description": "An advanced test query",
                        "query": "Perform complex analysis on test data",
                        "category": "advanced",
                        "difficulty": "advanced"
                    }
                ]
            }
        else:
            return {
                "queries": [
                    {
                        "id": "",  # Invalid: empty ID
                        "title": "Invalid Query",
                        # Missing description, query, category
                        "difficulty": "invalid_difficulty"  # Invalid enum value
                    },
                    {
                        "id": "duplicate_id",
                        "title": "First Query",
                        "description": "First query description",
                        "query": "First query text",
                        "category": "test"
                    },
                    {
                        "id": "duplicate_id",  # Invalid: duplicate ID
                        "title": "Second Query",
                        "description": "Second query description", 
                        "query": "Second query text",
                        "category": "test"
                    }
                ]
            }
    
    return {}

def test_schema_loading():
    """Test schema loading functionality"""
    print("🧪 Testing Schema Loading")
    print("-" * 50)
    
    validator = ConfigValidator()
    success = validator.load_schemas()
    
    if success:
        print(f"✅ Successfully loaded {len(validator.schemas)} schemas")
        for schema_name in validator.schemas.keys():
            print(f"   - {schema_name}")
    else:
        print("❌ Failed to load schemas")
    
    # Test with non-existent schema directory
    invalid_validator = ConfigValidator("non/existent/path")
    success = invalid_validator.load_schemas()
    
    if not success:
        print("✅ Correctly handled missing schema directory")
    else:
        print("❌ Should have failed with missing schema directory")
        return False
    
    return success

def test_valid_configurations():
    """Test validation of valid configurations"""
    print("\n🧪 Testing Valid Configuration Validation")
    print("-" * 50)
    
    validator = ConfigValidator()
    if not validator.load_schemas():
        print("❌ Failed to load schemas")
        return False
    
    config_types = ["servers", "citation_sources", "featured_queries"]
    all_passed = True
    
    for config_type in config_types:
        print(f"\n📋 Testing valid {config_type} config...")
        
        # Create valid test config
        test_config = create_test_config(config_type, valid=True)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f, indent=2)
            temp_path = f.name
        
        try:
            # Validate based on type
            if config_type == "servers":
                result = validator.validate_servers_config(temp_path)
            elif config_type == "citation_sources":
                result = validator.validate_citation_sources_config(temp_path)
            elif config_type == "featured_queries":
                result = validator.validate_featured_queries_config(temp_path)
            
            if result.valid:
                print(f"   ✅ Valid {config_type} config passed validation")
            else:
                print(f"   ❌ Valid {config_type} config failed validation:")
                for error in result.errors:
                    print(f"      - {error}")
                all_passed = False
            
            if result.warnings:
                print(f"   ⚠️ Warnings for {config_type}:")
                for warning in result.warnings:
                    print(f"      - {warning}")
        
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    return all_passed

def test_invalid_configurations():
    """Test validation of invalid configurations"""
    print("\n🧪 Testing Invalid Configuration Validation")
    print("-" * 50)
    
    validator = ConfigValidator()
    if not validator.load_schemas():
        print("❌ Failed to load schemas")
        return False
    
    config_types = ["servers", "citation_sources", "featured_queries"]
    all_passed = True
    
    for config_type in config_types:
        print(f"\n📋 Testing invalid {config_type} config...")
        
        # Create invalid test config
        test_config = create_test_config(config_type, valid=False)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f, indent=2)
            temp_path = f.name
        
        try:
            # Validate based on type
            if config_type == "servers":
                result = validator.validate_servers_config(temp_path)
            elif config_type == "citation_sources":
                result = validator.validate_citation_sources_config(temp_path)
            elif config_type == "featured_queries":
                result = validator.validate_featured_queries_config(temp_path)
            
            if not result.valid:
                print(f"   ✅ Invalid {config_type} config correctly failed validation")
                print(f"      Found {len(result.errors)} errors as expected")
                for error in result.errors[:2]:  # Show first 2 errors
                    print(f"      - {error}")
                if len(result.errors) > 2:
                    print(f"      ... and {len(result.errors) - 2} more errors")
            else:
                print(f"   ❌ Invalid {config_type} config incorrectly passed validation")
                all_passed = False
        
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    return all_passed

def test_malformed_json():
    """Test handling of malformed JSON files"""
    print("\n🧪 Testing Malformed JSON Handling")
    print("-" * 50)
    
    validator = ConfigValidator()
    if not validator.load_schemas():
        print("❌ Failed to load schemas")
        return False
    
    # Create malformed JSON file
    malformed_json = '{"servers": [{"name": "test", "path": "test.py", "description": "test"'  # Missing closing braces
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(malformed_json)
        temp_path = f.name
    
    try:
        result = validator.validate_servers_config(temp_path)
        
        if not result.valid and any("JSON" in error for error in result.errors):
            print("✅ Malformed JSON correctly detected")
            return True
        else:
            print("❌ Malformed JSON not properly detected")
            return False
    
    finally:
        os.unlink(temp_path)

def test_missing_config_files():
    """Test handling of missing configuration files"""
    print("\n🧪 Testing Missing Config File Handling")
    print("-" * 50)
    
    validator = ConfigValidator()
    if not validator.load_schemas():
        print("❌ Failed to load schemas")
        return False
    
    # Test with non-existent file
    result = validator.validate_servers_config("non/existent/config.json")
    
    if not result.valid and any("not found" in error for error in result.errors):
        print("✅ Missing config file correctly detected")
        return True
    else:
        print("❌ Missing config file not properly detected")
        return False

def run_all_config_tests():
    """Run all configuration validation tests"""
    print("🚀 Running Configuration Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Schema Loading", test_schema_loading),
        ("Valid Configurations", test_valid_configurations),
        ("Invalid Configurations", test_invalid_configurations),
        ("Malformed JSON", test_malformed_json),
        ("Missing Config Files", test_missing_config_files)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n💥 Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Configuration Validation Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All configuration validation tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False

def main():
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    success = run_all_config_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()