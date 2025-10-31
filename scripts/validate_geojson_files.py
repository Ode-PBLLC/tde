#!/usr/bin/env python3
"""
Validate and clean up GeoJSON files in static/maps directory.

This script checks that GeoJSON files contain the correct data type based on their filename.
For example, files starting with "extreme_heat_" should contain heat zone data, not solar assets.
"""

import json
import os
import sys
from pathlib import Path


def validate_geojson_file(filepath: Path) -> tuple[bool, str]:
    """
    Validate that a GeoJSON file contains the correct data type.

    Returns:
        (is_valid, error_message) tuple
    """
    filename = filepath.name

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if data.get('type') != 'FeatureCollection':
            return False, f"Not a FeatureCollection"

        features = data.get('features', [])
        if not features:
            return False, f"No features found"

        # Check first feature properties to determine data type
        first_feature = features[0]
        props = first_feature.get('properties', {})
        layer = props.get('layer', '')

        # Validate extreme heat files
        if filename.startswith('extreme_heat_'):
            if layer != 'heat_zone':
                return False, f"Expected layer='heat_zone', got layer='{layer}'"
            if 'quintile' not in props:
                return False, f"Missing 'quintile' property for heat zone"
            return True, ""

        # Validate solar facility files
        if filename.startswith('solar_'):
            if layer != 'solar_facility':
                return False, f"Expected layer='solar_facility', got layer='{layer}'"
            if 'capacity_mw' not in props:
                return False, f"Missing 'capacity_mw' property for solar facility"
            return True, ""

        # Validate deforestation files
        if filename.startswith('deforestation_'):
            if layer not in ['deforestation', 'deforestation_polygon']:
                return False, f"Expected layer='deforestation*', got layer='{layer}'"
            return True, ""

        # Unknown file type - skip validation
        return True, ""

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def main():
    """Validate all GeoJSON files in static/maps directory."""

    # Determine static maps directory
    project_root = Path(__file__).resolve().parents[1]
    static_maps_dir = project_root / "static" / "maps"

    if not static_maps_dir.exists():
        print(f"Directory not found: {static_maps_dir}")
        print("Creating directory...")
        static_maps_dir.mkdir(parents=True, exist_ok=True)
        return 0

    geojson_files = list(static_maps_dir.glob("*.geojson"))

    if not geojson_files:
        print(f"No GeoJSON files found in {static_maps_dir}")
        return 0

    print(f"Validating {len(geojson_files)} GeoJSON files in {static_maps_dir}\n")

    corrupted_files = []

    for filepath in geojson_files:
        is_valid, error_msg = validate_geojson_file(filepath)

        if is_valid:
            print(f"✓ {filepath.name}")
        else:
            print(f"✗ {filepath.name}: {error_msg}")
            corrupted_files.append((filepath, error_msg))

    if corrupted_files:
        print(f"\n\nFound {len(corrupted_files)} corrupted files:")
        for filepath, error_msg in corrupted_files:
            print(f"  - {filepath.name}: {error_msg}")

        print("\nTo delete corrupted files, run:")
        print(f"  python3 {__file__} --delete")

        if '--delete' in sys.argv:
            print("\nDeleting corrupted files...")
            for filepath, _ in corrupted_files:
                print(f"  Deleting {filepath.name}")
                filepath.unlink()
            print("Done!")
            return 0
        else:
            return 1
    else:
        print(f"\n✓ All {len(geojson_files)} files are valid!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
