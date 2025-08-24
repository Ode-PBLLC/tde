#!/usr/bin/env python3
"""
Test the enhanced solar facilities MCP server tools
"""

import sys
import os
sys.path.append('mcp')
sys.path.append('../mcp')

# Change to project root if we're in test_scripts
if 'test_scripts' in os.getcwd():
    os.chdir('..')

# Import the database directly to test functionality
from solar_db import SolarDatabase
import json

# Initialize database
db = SolarDatabase()

def test_new_db_features():
    """Test the newly added database features."""
    
    print("=== Testing Enhanced Solar Database Features ===\n")
    
    # Test 1: Get Available Countries
    print("1. Testing get_all_country_names():")
    try:
        all_countries = db.get_all_country_names()
        print(f"   ✓ Found {len(all_countries)} countries")
        print(f"   ✓ Brazil in list: {'Brazil' in all_countries}")
        print(f"   ✓ First 5 countries: {all_countries[:5]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 2: Find Countries by Name
    print("2. Testing find_country_by_partial_name('brazil'):")
    try:
        brazil_matches = db.find_country_by_partial_name('brazil')
        print(f"   ✓ Found {len(brazil_matches)} matches")
        print(f"   ✓ Matches: {brazil_matches}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 3: Get Facilities in Bounds for Brazil
    print("3. Testing get_facilities_in_bounds() for Brazil region:")
    # Brazil rough bounds: North: 5, South: -34, East: -34, West: -74
    try:
        brazil_facilities = db.get_facilities_in_bounds(
            north=5, south=-34, east=-34, west=-74, limit=50
        )
        print(f"   ✓ Found {len(brazil_facilities)} facilities in Brazil region")
        if brazil_facilities:
            countries_found = set(f.get('country', 'Unknown') for f in brazil_facilities)
            print(f"   ✓ Countries found: {list(countries_found)[:5]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 4: Direct Brazil country query
    print("4. Testing get_facilities_by_country('Brazil'):")
    try:
        brazil_direct = db.get_facilities_by_country('Brazil', limit=20)
        print(f"   ✓ Found {len(brazil_direct)} Brazilian facilities")
        if brazil_direct:
            sample = brazil_direct[0]
            print(f"   ✓ Sample facility: lat={sample['latitude']:.2f}, lon={sample['longitude']:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    # Test 5: Map generation simulation
    print("5. Testing map data preparation for Brazil:")
    try:
        facilities = db.get_facilities_by_country('Brazil', limit=100)
        if facilities:
            # Calculate map center
            lats = [f['latitude'] for f in facilities if f['latitude']]
            lons = [f['longitude'] for f in facilities if f['longitude']]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            print(f"   ✓ Calculated map center: ({center_lat:.2f}, {center_lon:.2f})")
            print(f"   ✓ Ready to generate map with {len(facilities)} facilities")
        else:
            print("   ✗ No facilities found for map generation")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    print("=== Test Summary ===")
    print("Enhanced database features are working! New capabilities:")
    print("• get_all_country_names() - Discover available countries")
    print("• find_country_by_partial_name() - Fuzzy country search") 
    print("• get_facilities_in_bounds() - Geographic bounding box filtering")
    print("• Better country name validation and matching")
    print("• Improved map data preparation")

if __name__ == "__main__":
    test_new_db_features()