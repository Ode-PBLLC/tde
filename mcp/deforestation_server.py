#!/usr/bin/env python3
"""
Deforestation Data Server

Serves polygon data from Brazil deforestation GeoJSON.
Uses FastMCP for consistent server architecture.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP
import hashlib
from datetime import datetime

# Initialize FastMCP server
mcp = FastMCP("deforestation-server")

# Try to import geospatial libraries
try:
    import geopandas as gpd
    from shapely.geometry import shape, Point, box
    import numpy as np
    import pandas as pd
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("Warning: GeoPandas not available. Install with: pip install geopandas shapely")

# Load deforestation polygons - automatically detect available datasets
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "deforestation"
PARQUET_PATH = DATA_DIR / "deforestation.parquet"
GEOJSON_PATH = BASE_DIR / "data" / "brazil_deforestation.geojson"
FALLBACK_DATASETS = [
    # Skip the large 520MB parquet file for now as it might be causing timeout
    # ("parquet", PARQUET_PATH),
    ("parquet", DATA_DIR / "deforestation_old.parquet"),
    ("geojson", GEOJSON_PATH),
]

# Initialize empty GeoDataFrame
deforestation_gdf = None

def _load_deforestation_data():
    """Attempt to load the deforestation dataset from known locations."""
    for fmt, path in FALLBACK_DATASETS:
        try:
            if not path.exists():
                continue

            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            print(f"Found {fmt} file at {path} ({file_size_mb:.1f} MB)")

            if fmt == "parquet":
                print(f"Loading deforestation data from parquet: {path}")
                import time
                start = time.time()
                gdf = gpd.read_parquet(path)
                elapsed = time.time() - start
                print(f"Loaded {len(gdf)} polygons in {elapsed:.1f} seconds")
            else:
                print(f"Loading deforestation data from GeoJSON: {path}")
                gdf = gpd.read_file(path)
            return gdf, path
        except Exception as exc:
            print(f"Failed to load deforestation data from {path}: {exc}")
            import traceback
            traceback.print_exc()
            continue
    return None, None


if GEOSPATIAL_AVAILABLE:
    try:
        loaded_gdf, source_path = _load_deforestation_data()
        if loaded_gdf is not None:
            deforestation_gdf = loaded_gdf
            print(f"Loaded {len(deforestation_gdf)} deforestation polygons from {source_path}")
            # Ensure CRS is set (parquet should preserve it)
            if deforestation_gdf.crs is None:
                deforestation_gdf.set_crs(epsg=4326, inplace=True)
                print("Set CRS to EPSG:4326")
            elif str(deforestation_gdf.crs).lower() not in {"epsg:4326", "ogc:crs84"}:
                original_crs = deforestation_gdf.crs
                deforestation_gdf = deforestation_gdf.to_crs('EPSG:4326')
                print(f"Reprojected dataset from {original_crs} to EPSG:4326")
            print(f"Available columns: {list(deforestation_gdf.columns)}")
            # Calculate area if not present
            if 'area_hectares' not in deforestation_gdf.columns:
                print("Calculating area in hectares...")
                try:
                    deforestation_projected = deforestation_gdf.to_crs('EPSG:5880')  # Brazil Polyconic
                    deforestation_gdf['area_hectares'] = (
                        deforestation_projected.geometry.area / 10000
                    )
                    print(
                        f"Calculated areas ranging from {deforestation_gdf['area_hectares'].min():.2f} "
                        f"to {deforestation_gdf['area_hectares'].max():.2f} hectares"
                    )
                except Exception as area_exc:
                    print(f"Warning: failed to compute area_hectares: {area_exc}")
                    deforestation_gdf['area_hectares'] = 0.0
            # Provide temporal context if missing
            if 'year' not in deforestation_gdf.columns:
                deforestation_gdf['year'] = None
        else:
            print(
                "Warning: No deforestation data found in expected locations. "
                "Proceeding with empty dataset."
            )
            deforestation_gdf = gpd.GeoDataFrame(columns=['geometry', 'area_hectares', 'year'])
            deforestation_gdf.set_crs(epsg=4326, inplace=True)
    except Exception as e:
        print(f"Error loading deforestation data: {e}")
        import traceback
        traceback.print_exc()
        deforestation_gdf = gpd.GeoDataFrame(columns=['geometry', 'area_hectares', 'year'])
        deforestation_gdf.set_crs(epsg=4326, inplace=True)


@mcp.tool()
def GetDeforestationAreas(
    min_area_hectares: float = 0,
    max_area_hectares: Optional[float] = None,
    limit: int = 0
) -> Dict[str, Any]:
    """
    Get deforestation polygons in Brazil matching criteria.
    
    Args:
        min_area_hectares: Minimum area in hectares
        max_area_hectares: Maximum area in hectares (optional)
        limit: Maximum number of polygons to return
        
    Returns:
        Dictionary with deforestation areas and statistics
    """
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    
    if deforestation_gdf is None or deforestation_gdf.empty:
        return {"error": "No deforestation data loaded"}
    
    # Apply a safety cap to avoid overwhelming the transport/UI with very large payloads
    # 0 or negative means "use default cap"; values above the cap are clamped.
    DEFAULT_MAX = 1000

    # Filter by area
    filtered = deforestation_gdf.copy()
    
    if min_area_hectares > 0:
        filtered = filtered[filtered['area_hectares'] >= min_area_hectares]
    
    if max_area_hectares is not None:
        filtered = filtered[filtered['area_hectares'] <= max_area_hectares]
    
    # Sort by area (largest first); enforce sane default cap if limit not provided
    filtered = filtered.sort_values('area_hectares', ascending=False)
    try:
        lim = int(limit or 0)
    except Exception:
        lim = 0
    if lim <= 0:
        lim = DEFAULT_MAX
    elif lim > DEFAULT_MAX:
        lim = DEFAULT_MAX
    filtered = filtered.head(lim)
    
    # Convert to JSON-serializable format
    areas = []
    for idx, row in filtered.iterrows():
        area_dict = {
            "id": f"deforest_{idx}",
            "geometry": row.geometry.__geo_interface__ if hasattr(row.geometry, '__geo_interface__') else {"type": "Polygon", "coordinates": []},
            "area_hectares": float(row['area_hectares'])
        }
        
        # Add other properties if they exist
        for col in filtered.columns:
            if col not in ['geometry', 'area_hectares']:
                try:
                    value = row[col]
                    # Convert numpy types to Python types
                    if hasattr(value, 'item'):
                        value = value.item()
                    # Convert NaN to None
                    elif pd.isna(value):
                        value = None
                    area_dict[col] = value
                except:
                    pass  # Skip problematic columns
        
        areas.append(area_dict)
    
    return {
        "deforestation_areas": areas,
        "total_count": len(areas),
        "total_area_hectares": float(filtered['area_hectares'].sum()),
        "average_area_hectares": float(filtered['area_hectares'].mean()) if len(filtered) > 0 else 0,
        "country": "Brazil",
        "filter_applied": {
            "min_area_hectares": min_area_hectares,
            "max_area_hectares": max_area_hectares
        }
    }


@mcp.tool()
def GetDeforestationInBounds(
    north: float, 
    south: float,
    east: float,
    west: float,
    limit: int = 500
) -> Dict[str, Any]:
    """
    Get deforestation areas within geographic bounds.
    
    Args:
        north: Northern latitude boundary
        south: Southern latitude boundary
        east: Eastern longitude boundary
        west: Western longitude boundary
        limit: Maximum number of results
        
    Returns:
        Deforestation areas within the specified bounds
    """
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    
    if deforestation_gdf is None or deforestation_gdf.empty:
        return {"error": "No deforestation data loaded"}
    
    # Create bounding box
    bbox = box(west, south, east, north)
    
    # Find intersecting polygons
    intersects = deforestation_gdf[deforestation_gdf.geometry.intersects(bbox)]
    
    # Sort by area and limit
    intersects = intersects.sort_values('area_hectares', ascending=False).head(limit)
    
    # Convert to JSON format
    areas = []
    for idx, row in intersects.iterrows():
        area_dict = {
            "id": f"deforest_{idx}",
            "geometry": row.geometry.__geo_interface__ if hasattr(row.geometry, '__geo_interface__') else {"type": "Polygon", "coordinates": []},
            "area_hectares": float(row.get('area_hectares', 0))
        }
        
        # Add other properties
        for col in intersects.columns:
            if col not in ['geometry', 'area_hectares']:
                try:
                    value = row[col]
                    if hasattr(value, 'item'):
                        value = value.item()
                    elif pd.isna(value):
                        value = None
                    area_dict[col] = value
                except:
                    pass  # Skip problematic columns
        
        areas.append(area_dict)
    
    return {
        "deforestation_areas": areas,
        "total_count": len(areas),
        "total_area_hectares": float(intersects['area_hectares'].sum()) if len(intersects) > 0 else 0,
        "bounds": {
            "north": north,
            "south": south,
            "east": east,
            "west": west
        },
        "country": "Brazil"
    }


@mcp.tool()
def GetDeforestationStatistics() -> Dict[str, Any]:
    """
    Get summary statistics about deforestation in Brazil.
    
    Returns:
        Statistical summary of deforestation data
    """
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    
    if deforestation_gdf is None or deforestation_gdf.empty:
        return {"error": "No deforestation data loaded"}
    
    areas = deforestation_gdf['area_hectares']
    
    stats = {
        "country": "Brazil",
        "total_polygons": len(deforestation_gdf),
        "total_area_hectares": float(areas.sum()),
        "total_area_km2": float(areas.sum() / 100),  # Convert hectares to km²
        "average_area_hectares": float(areas.mean()),
        "median_area_hectares": float(areas.median()),
        "min_area_hectares": float(areas.min()),
        "max_area_hectares": float(areas.max()),
        "std_dev_hectares": float(areas.std()),
        "area_percentiles": {
            "10th": float(areas.quantile(0.10)),
            "25th": float(areas.quantile(0.25)),
            "50th": float(areas.quantile(0.50)),
            "75th": float(areas.quantile(0.75)),
            "90th": float(areas.quantile(0.90)),
            "95th": float(areas.quantile(0.95)),
            "99th": float(areas.quantile(0.99))
        }
    }
    
    # Add year statistics if available
    if 'year' in deforestation_gdf.columns:
        year_stats = deforestation_gdf.groupby('year')['area_hectares'].agg(['sum', 'count'])
        stats['by_year'] = {
            int(year): {
                "area_hectares": float(row['sum']),
                "polygon_count": int(row['count'])
            }
            for year, row in year_stats.iterrows()
        }
    
    return stats


@mcp.tool()
def GetDeforestationWithMap(
    min_area_hectares: float = 0,
    limit: int = 1000
) -> Dict[str, Any]:
    """
    Get deforestation areas and generate a GeoJSON map file.
    Saves to static/maps/ directory for visualization.
    
    Args:
        min_area_hectares: Minimum area filter
        limit: Maximum number of polygons
        
    Returns:
        Deforestation data with map URL
    """
    # First get the deforestation areas
    result = GetDeforestationAreas(min_area_hectares, limit=limit)
    
    if "error" in result:
        return result
    
    areas = result["deforestation_areas"]
    
    # Generate GeoJSON for map
    features = []
    for area in areas:
        feature = {
            "type": "Feature",
            "geometry": area['geometry'],
            "properties": {
                "id": area['id'],
                "area_hectares": area['area_hectares'],
                # Styling for map display
                "fill": "#8B4513",  # Brown for deforestation
                "fill-opacity": 0.4,
                "stroke": "#654321",  # Darker brown border
                "stroke-width": 2,
                "stroke-opacity": 0.8,
                "title": f"Deforestation: {area['area_hectares']:.1f} hectares"
            }
        }
        
        # Add other properties
        for key, value in area.items():
            if key not in ['geometry', 'id', 'area_hectares']:
                feature['properties'][key] = value
        
        features.append(feature)
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated": datetime.now().isoformat(),
            "source": "Brazil deforestation data",
            "total_features": len(features),
            "total_area_hectares": result["total_area_hectares"],
            "filters": {
                "min_area_hectares": min_area_hectares
            }
        }
    }
    
    # Generate unique filename
    content_str = json.dumps(features, sort_keys=True)
    data_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
    filename = f"deforestation_brazil_{data_hash}.geojson"
    
    # Save to static/maps/
    static_maps_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "static", "maps"
    )
    os.makedirs(static_maps_dir, exist_ok=True)
    
    filepath = os.path.join(static_maps_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(geojson, f)
    
    # Add map info to result
    result["geojson_url"] = f"/static/maps/{filename}"
    result["geojson_filename"] = filename
    result["type"] = "deforestation_map"
    result["summary"] = {
        "description": f"Deforestation map of Brazil with {len(features)} areas",
        "total_area_hectares": result["total_area_hectares"],
        "styling": "Brown polygons with 40% opacity"
    }
    
    print(f"Generated deforestation map: {filename}")

    return result


@mcp.tool()
def DescribeServer() -> Dict[str, Any]:
    """Describe deforestation dataset, tools, and live stats."""
    try:
        if not GEOSPATIAL_AVAILABLE:
            return {"error": "GeoPandas not installed"}
        total = int(len(deforestation_gdf)) if deforestation_gdf is not None else 0
        total_area_km2 = 0.0
        if deforestation_gdf is not None and not deforestation_gdf.empty and 'area_hectares' in deforestation_gdf.columns:
            total_area_km2 = float(deforestation_gdf['area_hectares'].sum() / 100.0)
        tools = [
            "GetDeforestationAreas",
            "GetDeforestationInBounds",
            "GetDeforestationStatistics",
            "GetDeforestationWithMap"
        ]
        # Derive last_updated from source file timestamps
        last_updated = None
        try:
            from datetime import datetime as _dt
            paths = [p for p in [PARQUET_PATH, GEOJSON_PATH] if os.path.exists(p)]
            if paths:
                last_updated = _dt.fromtimestamp(max(os.path.getmtime(p) for p in paths)).isoformat()
        except Exception:
            pass
        return {
            "name": "Deforestation Server",
            "description": "Brazil deforestation polygons from processed satellite imagery",
            "version": "1.0",
            "dataset": "Brazil deforestation polygons",
            "metrics": {
                "total_polygons": total,
                "total_area_km2": total_area_km2
            },
            "coverage": {
                "country": "Brazil"
            },
            "tools": tools,
            "examples": [
                "Get deforestation areas over 10 hectares",
                "Intersect solar with deforestation in a region"
            ],
            "last_updated": last_updated
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
