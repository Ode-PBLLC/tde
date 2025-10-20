#!/usr/bin/env python3
"""
Inspect Brazilian municipalities datasets (CSV + GeoJSON/shapefile) and print columns + samples.
Looks in data/brazilian_munis/ for:
- br.csv (CSV attributes)
- BR_Municipios_2024_simplified.geojson OR municipality_shapes/BR_Municipios_2024.shp
"""

from pathlib import Path


def main():
    try:
        import pandas as pd
        import geopandas as gpd
    except Exception as e:
        print('✗ Required libraries missing:', e)
        return

    base = Path(__file__).resolve().parent.parent
    csv_path = base / 'data' / 'brazilian_munis' / 'br.csv'
    gj_path = base / 'data' / 'brazilian_munis' / 'BR_Municipios_2024_simplified.geojson'
    shp_path = base / 'data' / 'brazilian_munis' / 'municipality_shapes' / 'BR_Municipios_2024.shp'

    if csv_path.exists():
        print('✓ CSV:', csv_path)
        try:
            df = pd.read_csv(csv_path)
            print('CSV Columns:', list(df.columns))
            print('CSV Head(5):')
            print(df.head(5))
        except Exception as e:
            print('  (error reading CSV):', e)
    else:
        print('✗ CSV not found at', csv_path)

    geom_file = gj_path if gj_path.exists() else (shp_path if shp_path.exists() else None)
    if geom_file:
        print('✓ Geometry file:', geom_file)
        try:
            gdf = gpd.read_file(geom_file)
            print('CRS:', gdf.crs)
            print('Geometry Columns:', list(gdf.columns))
            print('Head(3):')
            print(gdf.head(3))
        except Exception as e:
            print('  (error reading geometry):', e)
    else:
        print('✗ No geometry file found at', gj_path, 'or', shp_path)


if __name__ == '__main__':
    main()

