#!/usr/bin/env python3
"""
Inspect deforestation dataset and print columns + sample rows.
Looks for:
- data/deforestation/deforestation.parquet
- data/brazil_deforestation.geojson

Prints:
- CRS, columns, head(5)
- Presence of 'year' column (for time series)
"""

import os
from pathlib import Path


def main():
    import geopandas as gpd  # noqa

    base = Path(__file__).resolve().parent.parent
    parquet_path = base / 'data' / 'deforestation' / 'deforestation.parquet'
    geojson_path = base / 'data' / 'brazil_deforestation.geojson'

    gdf = None
    src = None
    if parquet_path.exists():
        src = parquet_path
        gdf = gpd.read_parquet(parquet_path)
    elif geojson_path.exists():
        src = geojson_path
        gdf = gpd.read_file(geojson_path)
    else:
        print('✗ No deforestation dataset found at', parquet_path, 'or', geojson_path)
        return

    print('✓ Loaded deforestation from:', src)
    print('CRS:', gdf.crs)
    print('Columns:', list(gdf.columns))
    print('\nHead(5):')
    print(gdf.head(5))
    print('\nHas year column?:', 'year' in gdf.columns)


if __name__ == '__main__':
    main()

