#!/usr/bin/env python3
"""Test script to verify deforestation data loading after S3 sync."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Looking for deforestation data in:")

# Check all candidate paths
candidates = [
    PROJECT_ROOT / "data/deforestation/deforestation.parquet",
    PROJECT_ROOT / "data/deforestation/deforestation_old.parquet",
    PROJECT_ROOT / "data/brazil_deforestation.geojson",
]

for path in candidates:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    print(f"  {'✓' if exists else '✗'} {path} ({size:,} bytes)" if exists else f"  ✗ {path} (not found)")

print("\nAttempting to load DeforestationPolygonProvider...")
try:
    from mcp.geospatial_datasets.deforestation import DeforestationPolygonProvider
    provider = DeforestationPolygonProvider()
    print(f"✓ Successfully loaded: {provider.dataset_name}")
    overview = provider.dataset_overview()
    print(f"  Polygon count: {overview.get('polygon_count')}")
    print(f"  Total area: {overview.get('total_area_hectares')} hectares")
except Exception as e:
    print(f"✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()