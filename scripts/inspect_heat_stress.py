#!/usr/bin/env python3
"""
Inspect heat-stress preprocessed datasets and print columns + samples.
Looks in: data/heat_stress/preprocessed and data/heat_stress/preprocessed/geojsons
"""

from pathlib import Path


def main():
    try:
        import geopandas as gpd  # noqa
    except Exception as e:
        print('✗ geopandas not available:', e)
        return

    base = Path(__file__).resolve().parent.parent
    root = base / 'data' / 'heat_stress' / 'preprocessed'
    geodir = root / 'geojsons'

    if not root.exists():
        print('✗ Heat-stress directory not found:', root)
        return

    # Prefer simplified geojsons
    files = list(geodir.glob('*.geojson')) if geodir.exists() else []
    if not files:
        files = list(root.glob('*.geojson')) + list(root.glob('*.gpkg'))

    if not files:
        print('✗ No heat-stress files found under', root)
        return

    f = files[0]
    print('✓ Inspecting', f)
    gdf = gpd.read_file(f)
    print('CRS:', gdf.crs)
    print('Columns:', list(gdf.columns))
    if 'quintile' in gdf.columns:
        print('Unique quintiles:', sorted(gdf['quintile'].dropna().unique().tolist()))
    print('Head(5):')
    print(gdf.head(5))


if __name__ == '__main__':
    main()

