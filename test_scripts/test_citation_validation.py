#!/usr/bin/env python3
"""
Test script for citation validation system.
Tests citation mapping coverage and validation functions.
"""
import os
import sys
import json
import asyncio
from typing import Dict, List

# Add the mcp directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mcp'))

from citation_validation import CitationValidator, validate_citation_coverage_sync

def test_citation_file_structure():
    """Test citation configuration file structure validation"""
    print("🧪 Testing Citation File Structure Validation")
    print("-" * 50)
    
    # Test with actual config file
    citation_config_path = "config/citation_sources.json"
    
    if not os.path.exists(citation_config_path):
        print("⚠️ Citation config file not found - creating test file")
        
        # Create a test citation file
        test_citations = {
            "tool_citations": {
                "TestTool1": {
                    "source_name": "Test Dataset 1",
                    "provider": "Test Provider",
                    "spatial_coverage": "Global",
                    "temporal_coverage": "2024",
                    "source_url": "https://example.com"
                },
                "TestTool2": {
                    "source_name": "Test Dataset 2",
                    "provider": "Another Provider",
                    "spatial_coverage": "United States",
                    "temporal_coverage": "2020-2024",
                    "source_url": ""
                }
            }
        }
        
        os.makedirs("config", exist_ok=True)
        with open(citation_config_path, 'w') as f:
            json.dump(test_citations, f, indent=2)
        print("✅ Created test citation config file")
    
    # Run validation
    result = validate_citation_coverage_sync(citation_config_path)
    
    print(f"📊 Validation Results:")
    print(f"   Valid: {result.valid}")
    print(f"   Tools with valid citations: {len(result.tools_validated)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    if result.errors:
        print("\n🚨 Errors found:")
        for error in result.errors:
            print(f"   - {error}")
    
    if result.warnings:
        print("\n⚠️ Warnings found:")
        for warning in result.warnings:
            print(f"   - {warning}")
    
    print(f"\n✅ Citation structure test completed")
    return result.valid

def test_citation_mapping_loading():
    """Test citation mapping loading and fallback behavior"""
    print("\n🧪 Testing Citation Mapping Loading")
    print("-" * 50)
    
    validator = CitationValidator()
    
    # Test loading existing config
    success = validator.load_citation_mapping()
    print(f"📁 Citation mapping loaded: {success}")
    print(f"📊 Number of citation mappings: {len(validator.citation_mapping)}")
    
    # Test with tools that exist vs don't exist
    test_tools = ["TestTool1", "NonExistentTool", "TestTool2"]
    
    for tool_name in test_tools:
        if tool_name in validator.citation_mapping:
            print(f"✅ Citation found for '{tool_name}'")
        else:
            print(f"⚠️ No citation mapping for '{tool_name}' - will use fallback")
    
    return success

def test_malformed_citation_config():
    """Test handling of malformed citation configuration"""
    print("\n🧪 Testing Malformed Citation Config Handling")
    print("-" * 50)
    
    # Create a malformed config file
    malformed_config_path = "config/test_malformed_citations.json"
    
    malformed_citations = {
        "tool_citations": {
            "MalformedTool1": {
                "source_name": "Missing required fields"
                # Missing provider, spatial_coverage, temporal_coverage, source_url
            },
            "MalformedTool2": "This should be an object, not a string"
        }
    }
    
    with open(malformed_config_path, 'w') as f:
        json.dump(malformed_citations, f, indent=2)
    
    # Test validation
    result = validate_citation_coverage_sync(malformed_config_path)
    
    print(f"📊 Malformed Config Results:")
    print(f"   Valid: {result.valid}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.errors:
        print("\n🚨 Expected errors found:")
        for error in result.errors:
            print(f"   - {error}")
    
    # Clean up
    if os.path.exists(malformed_config_path):
        os.remove(malformed_config_path)
        print("🧹 Cleaned up test file")
    
    return not result.valid  # Should be invalid

def test_missing_citation_template():
    """Test missing citation template generation"""
    print("\n🧪 Testing Missing Citation Template Generation")
    print("-" * 50)
    
    validator = CitationValidator()
    validator.load_citation_mapping()
    
    # Simulate missing tools
    missing_tools = {"MissingTool1", "MissingTool2", "AnotherMissingTool"}
    
    template = validator.generate_missing_citation_template(missing_tools)
    
    print(f"📝 Generated template for {len(missing_tools)} missing tools:")
    print(json.dumps(template, indent=2))
    
    # Verify template structure
    all_valid = True
    for tool_name, citation_info in template.items():
        if not isinstance(citation_info, dict):
            print(f"❌ Template for {tool_name} is not a dictionary")
            all_valid = False
            continue
        
        required_fields = ["source_name", "provider", "spatial_coverage", "temporal_coverage", "source_url"]
        for field in required_fields:
            if field not in citation_info:
                print(f"❌ Template for {tool_name} missing field: {field}")
                all_valid = False
    
    if all_valid:
        print("✅ Template generation test passed")
    else:
        print("❌ Template generation test failed")
    
    return all_valid

def run_all_citation_tests():
    """Run all citation validation tests"""
    print("🚀 Running Citation Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Citation File Structure", test_citation_file_structure),
        ("Citation Mapping Loading", test_citation_mapping_loading),
        ("Malformed Config Handling", test_malformed_citation_config),
        ("Missing Citation Template", test_missing_citation_template)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n💥 Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Citation Validation Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All citation validation tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False

def main():
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    success = run_all_citation_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()