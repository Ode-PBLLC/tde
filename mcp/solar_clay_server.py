import pandas as pd
import geopandas as gpd
from fastmcp import FastMCP
from typing import List, Optional, Dict, Any
import os
import json
import hashlib
from shapely import wkt
from shapely.geometry import shape, mapping
import numpy as np

mcp = FastMCP("solar-clay-server")

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Data paths
SOLAR_CHIP_DATA_PATH = os.path.join(project_root, "data", "top_potential_solar.parquet")
BRAZILIAN_STATES_PATH = os.path.join(project_root, "data", "brazilian_admin", "brazilian_states.geojson")

# Initialize data
solar_gdf = None
brazilian_states_gdf = None
metadata = {}

try:
    # Load solar data
    if os.path.exists(SOLAR_CHIP_DATA_PATH):
        # Read with pandas first (no geo metadata in file)
        df = pd.read_parquet(SOLAR_CHIP_DATA_PATH)

        # Convert WKB geometry to shapely objects
        if 'geometry' in df.columns:
            from shapely import wkb
            # Handle both WKB bytes and WKT strings
            if isinstance(df['geometry'].iloc[0], bytes):
                df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if x else None)
            elif isinstance(df['geometry'].iloc[0], str):
                df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x) if x else None)

        solar_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

        print(f"Loaded solar data with {len(solar_gdf)} geometries")

        # Perform spatial join with Brazilian states if states file exists
        if os.path.exists(BRAZILIAN_STATES_PATH):
            brazilian_states_gdf = gpd.read_file(BRAZILIAN_STATES_PATH)
            print(f"Loaded Brazilian states data with {len(brazilian_states_gdf)} states")

            # Spatial join to assign state to each geometry
            print("Performing spatial join to assign states to geometries...")
            solar_gdf = gpd.sjoin(
                solar_gdf,
                brazilian_states_gdf[['NM_UF', 'geometry']],
                how='left',
                predicate='intersects'
            )
            # Rename NM_UF to state for consistency
            if 'NM_UF' in solar_gdf.columns:
                solar_gdf = solar_gdf.rename(columns={'NM_UF': 'state'})
            # Drop the index_right column from spatial join
            if 'index_right' in solar_gdf.columns:
                solar_gdf = solar_gdf.drop(columns=['index_right'])

            print(f"Assigned states to {solar_gdf['state'].notna().sum()} geometries")

        metadata = {
            "Name": "Solar Geometries Server",
            "Description": "Top potential sites where solar farms could be installed based on Clay embeddings and PVGIS data",
            "Version": "1.0.0",
            "Author": "Ode Partners",
            "Dataset": "Top potential Solar Non-Intersecting Geometries",
            "total_geometries": len(solar_gdf),
            "columns": list(solar_gdf.columns),
            "has_yield_data": 'specific_yield_kwh_per_kwp_yr' in solar_gdf.columns
        }
    else:
        print(f"Warning: Solar data file not found at {SOLAR_CHIP_DATA_PATH}")
        metadata = {"error": f"Solar data file not found at {SOLAR_CHIP_DATA_PATH}"}

        # Still try to load Brazilian states for reference
        if os.path.exists(BRAZILIAN_STATES_PATH):
            brazilian_states_gdf = gpd.read_file(BRAZILIAN_STATES_PATH)
            print(f"Loaded Brazilian states data with {len(brazilian_states_gdf)} states")

except Exception as e:
    print(f"Error loading data: {e}")
    metadata = {"error": f"Data loading failed: {str(e)}"}


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        # Convert array to list and recursively process elements
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def generate_color_scale(value: float, min_val: float, max_val: float, colormap: str = "viridis") -> str:
    """Generate color from value using a colormap."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if pd.isna(value) or min_val == max_val:
        return "#808080"  # Gray for null or constant values

    # Normalize value to 0-1 range
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))  # Clamp to 0-1

    # Get color from matplotlib colormap
    cmap = plt.get_cmap(colormap)
    rgba = cmap(normalized)

    # Convert to hex
    return mcolors.to_hex(rgba)


def generate_and_save_geojson(
    gdf: gpd.GeoDataFrame,
    identifier: str = "solar",
    color_column: Optional[str] = None,
    extra_metadata: Dict = None
) -> Dict[str, Any]:
    """
    Generate GeoJSON file from GeoDataFrame and save to static/maps/.
    Returns URL, filename, and statistics about the data.
    """
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Calculate color scale if color column specified
    color_scale = None
    if color_column and color_column in gdf.columns:
        min_val = gdf[color_column].min()
        max_val = gdf[color_column].max()
        color_scale = (min_val, max_val)

    # Statistics tracking
    total_area = 0
    states = set()
    state_counts = {}
    geometry_types = set()

    # Track geographic bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    for idx, row in gdf.iterrows():
        # Track statistics
        if 'state' in row:
            state = row['state']
            states.add(state)
            state_counts[state] = state_counts.get(state, 0) + 1

        # Get geometry
        geom = row.geometry
        if geom is None:
            continue

        geometry_types.add(geom.geom_type)

        # Calculate area in hectares (assumes CRS is in degrees, rough approximation)
        try:
            area_ha = geom.area * 111 * 111 * 100  # Rough conversion
            total_area += area_ha
        except:
            area_ha = None

        # Determine color
        fill_color = "#3388ff"  # Default blue
        if color_scale and color_column in row.index and pd.notna(row[color_column]):
            fill_color = generate_color_scale(row[color_column], color_scale[0], color_scale[1])

        # Build properties
        properties = {
            "id": str(idx),
            "fill_color": fill_color,
            "fill_opacity": 0.7,
            "stroke_color": "#000000",
            "stroke_weight": 1,
            "stroke_opacity": 0.8
        }

        # Add all non-geometry columns as properties
        for col in gdf.columns:
            if col != 'geometry':
                val = row[col]
                # Use recursive converter for all types (handles nested numpy arrays)
                properties[col] = convert_to_json_serializable(val)

        if area_ha:
            properties['area_ha'] = round(area_ha, 2)

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": properties
        }
        geojson["features"].append(feature)

    # Calculate statistics
    stats = {
        "total_geometries": len(geojson["features"]),
        "total_area_ha": round(total_area, 1) if total_area > 0 else None,
        "states": sorted(list(states)) if states else None,
        "state_counts": state_counts if state_counts else None,
        "geometry_types": sorted(list(geometry_types)),
        "bounds": {
            "west": float(bounds[0]),
            "south": float(bounds[1]),
            "east": float(bounds[2]),
            "north": float(bounds[3])
        }
    }

    # Add color scale info if used
    if color_scale:
        stats["color_scale"] = {
            "column": color_column,
            "min": float(color_scale[0]),
            "max": float(color_scale[1]),
            "colormap": "viridis"
        }

    # Save GeoJSON file
    try:
        static_maps_dir = os.path.join(project_root, "static", "maps")
        os.makedirs(static_maps_dir, exist_ok=True)

        # Create unique filename
        identifier_str = identifier.lower().replace(" ", "_").replace(",", "")[:50]
        data_hash = hashlib.md5(f"{identifier_str}_{len(geojson['features'])}".encode()).hexdigest()[:8]
        filename = f"solar_geometries_{identifier_str}_{data_hash}.geojson"

        geojson_path = os.path.join(static_maps_dir, filename)

        # Add metadata to GeoJSON
        geojson["metadata"] = {**stats}
        if extra_metadata:
            geojson["metadata"].update(extra_metadata)

        with open(geojson_path, 'w') as f:
            json.dump(geojson, f)

        print(f"Generated GeoJSON file: {filename} with {len(geojson['features'])} features")

        return {
            "success": True,
            "geojson_url": f"/static/maps/{filename}",
            "geojson_filename": filename,
            "file_size_kb": round(os.path.getsize(geojson_path) / 1024, 1),
            "stats": stats
        }

    except Exception as e:
        print(f"Error generating GeoJSON file: {e}")
        return {
            "success": False,
            "error": str(e),
            "stats": stats
        }


@mcp.tool()
def GetSolarGeometriesMap(
    color_by: str = "specific_yield_kwh_per_kwp_yr",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Generate map of potential solar geometries colored by a specified column.

    Args:
        color_by: Column name to use for coloring polygons (default: specific_yield_kwh_per_kwp_yr)
        limit: Maximum number of geometries to show (default: all)

    Returns:
        Map data with GeoJSON URL and statistics
    """
    if solar_gdf is None:
        return {"error": "Solar data not available"}

    try:
        # Validate color column
        if color_by not in solar_gdf.columns:
            return {
                "error": f"Column '{color_by}' not found in data",
                "available_columns": list(solar_gdf.columns)
            }

        # Apply limit if specified
        gdf_subset = solar_gdf.head(limit) if limit else solar_gdf

        # Generate GeoJSON
        geojson_result = generate_and_save_geojson(
            gdf_subset,
            identifier=f"all_{color_by}",
            color_column=color_by,
            extra_metadata={
                "data_source": "Solar Top potential",
                "color_by": color_by,
                "limit": limit
            }
        )

        if not geojson_result["success"]:
            return {"error": f"Failed to generate GeoJSON: {geojson_result.get('error')}"}

        stats = geojson_result["stats"]

        # Create description
        description = f"Map of {stats['total_geometries']} Solar geometries colored by {color_by}"
        if stats.get('color_scale'):
            cs = stats['color_scale']
            description += f" (range: {cs['min']:.2f} - {cs['max']:.2f})"

        return {
            "type": "map_data",
            "summary": {
                "description": description,
                "total_geometries": stats['total_geometries'],
                "total_area_ha": stats.get('total_area_ha'),
                "states": stats.get('states'),
                "color_scale": stats.get('color_scale'),
                "geometry_types": stats['geometry_types'],
                "bounds": stats['bounds']
            },
            "geojson_url": geojson_result["geojson_url"],
            "geojson_filename": geojson_result["geojson_filename"],
            "file_size_kb": geojson_result["file_size_kb"],
            "metadata": {
                "data_source": "Solar Top potential",
                "color_by": color_by
            }
        }

    except Exception as e:
        return {"error": f"Map generation failed: {str(e)}"}


@mcp.tool()
def GetSolarGeometriesByState(
    state: str,
    color_by: str = "specific_yield_kwh_per_kwp_yr"
) -> Dict[str, Any]:
    """Generate map of potential solar geometries for a specific Brazilian state.

    Args:
        state: Brazilian state name (e.g., "São Paulo", "Mato Grosso")
        color_by: Column name to use for coloring polygons (default: specific_yield_kwh_per_kwp_yr)

    Returns:
        Map data with GeoJSON URL and statistics for the specified state
    """
    if solar_gdf is None:
        return {"error": "Solar data not available"}

    try:
        # Check if state column exists (should have been added via spatial join)
        if 'state' not in solar_gdf.columns:
            return {
                "error": "State column not found in solar data (spatial join may have failed)",
                "available_columns": list(solar_gdf.columns)
            }

        # Filter by state (case-insensitive)
        state_gdf = solar_gdf[solar_gdf['state'].str.lower() == state.lower()]

        if len(state_gdf) == 0:
            return {
                "error": f"No geometries found for state '{state}'",
                "available_states": sorted(solar_gdf['state'].unique().tolist())
            }

        # Validate color column
        if color_by not in state_gdf.columns:
            return {
                "error": f"Column '{color_by}' not found in data",
                "available_columns": list(state_gdf.columns)
            }

        # Generate GeoJSON
        geojson_result = generate_and_save_geojson(
            state_gdf,
            identifier=f"{state}_{color_by}",
            color_column=color_by,
            extra_metadata={
                "data_source": "Solar Top potential",
                "state_filter": state,
                "color_by": color_by
            }
        )

        if not geojson_result["success"]:
            return {"error": f"Failed to generate GeoJSON: {geojson_result.get('error')}"}

        stats = geojson_result["stats"]

        # Create description
        description = f"Map of {stats['total_geometries']} potential solar geometries in {state} colored by {color_by}"
        if stats.get('color_scale'):
            cs = stats['color_scale']
            description += f" (range: {cs['min']:.2f} - {cs['max']:.2f})"

        return {
            "type": "map_data",
            "summary": {
                "description": description,
                "state": state,
                "total_geometries": stats['total_geometries'],
                "total_area_ha": stats.get('total_area_ha'),
                "color_scale": stats.get('color_scale'),
                "geometry_types": stats['geometry_types'],
                "bounds": stats['bounds']
            },
            "geojson_url": geojson_result["geojson_url"],
            "geojson_filename": geojson_result["geojson_filename"],
            "file_size_kb": geojson_result["file_size_kb"],
            "metadata": {
                "data_source": "Solar Top potential",
                "state_filter": state,
                "color_by": color_by
            }
        }

    except Exception as e:
        return {"error": f"Map generation failed: {str(e)}"}


@mcp.tool()
def GetGeometriesByState() -> Dict[str, Any]:
    """Get count of potential solar geometries by Brazilian state.

    Returns:
        Statistics showing number of geometries per state
    """
    if solar_gdf is None:
        return {"error": "Solar data not available"}

    try:
        # Check if state column exists (should have been added via spatial join at startup)
        if 'state' not in solar_gdf.columns:
            return {
                "error": "State column not found (spatial join may have failed at startup)",
                "available_columns": list(solar_gdf.columns)
            }

        # Use existing state column
        state_counts = solar_gdf['state'].value_counts().to_dict()

        # Calculate total area per state if possible
        state_stats = []
        for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
            stat = {
                "state": state,
                "geometry_count": int(count)
            }

            # Add area if calculable
            if 'state' in solar_gdf.columns:
                state_gdf = solar_gdf[solar_gdf['state'] == state]
                total_area = state_gdf.geometry.area.sum() * 111 * 111 * 100  # Rough ha conversion
                stat["total_area_ha"] = round(total_area, 1)

            state_stats.append(stat)

        return {
            "total_geometries": len(solar_gdf),
            "total_states": len(state_counts),
            "state_statistics": state_stats,
            "top_5_states": state_stats[:5],
            "data_available": True
        }

    except Exception as e:
        return {"error": f"Failed to calculate state statistics: {str(e)}"}


@mcp.tool()
def GetSolarGeometriesInBounds(
    north: float,
    south: float,
    east: float,
    west: float,
    color_by: str = "specific_yield_kwh_per_kwp_yr"
) -> Dict[str, Any]:
    """Get potential solar geometries within geographic bounds.

    Args:
        north: Northern latitude bound
        south: Southern latitude bound
        east: Eastern longitude bound
        west: Western longitude bound
        color_by: Column name to use for coloring polygons

    Returns:
        Map data with GeoJSON URL for geometries in bounds
    """
    if solar_gdf is None:
        return {"error": "Solar data not available"}

    try:
        from shapely.geometry import box

        # Create bounding box
        bbox = box(west, south, east, north)

        # Filter geometries that intersect the bounding box
        in_bounds = solar_gdf[solar_gdf.geometry.intersects(bbox)]

        if len(in_bounds) == 0:
            return {
                "error": f"No geometries found in bounds: N{north} S{south} E{east} W{west}"
            }

        # Generate GeoJSON
        geojson_result = generate_and_save_geojson(
            in_bounds,
            identifier=f"bounds_n{north:.1f}_s{south:.1f}",
            color_column=color_by,
            extra_metadata={
                "data_source": "Solar Top potential",
                "bounds": {"north": north, "south": south, "east": east, "west": west},
                "color_by": color_by
            }
        )

        if not geojson_result["success"]:
            return {"error": f"Failed to generate GeoJSON: {geojson_result.get('error')}"}

        stats = geojson_result["stats"]

        return {
            "bounds": {"north": north, "south": south, "east": east, "west": west},
            "geometries_found": stats['total_geometries'],
            "total_area_ha": stats.get('total_area_ha'),
            "states": stats.get('states'),
            "color_scale": stats.get('color_scale'),
            "geojson_url": geojson_result["geojson_url"],
            "geojson_filename": geojson_result["geojson_filename"],
            "data_available": True
        }

    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}


@mcp.tool()
def GetSolarDataSummary() -> Dict[str, Any]:
    """Get summary statistics about the potential solar geometries dataset.

    Returns:
        Dataset metadata and statistics
    """
    if solar_gdf is None:
        return {"error": "Solar data not available"}

    try:
        summary = {
            "total_geometries": len(solar_gdf),
            "columns": list(solar_gdf.columns),
            "geometry_types": solar_gdf.geometry.geom_type.unique().tolist(),
            "bounds": {
                "west": float(solar_gdf.total_bounds[0]),
                "south": float(solar_gdf.total_bounds[1]),
                "east": float(solar_gdf.total_bounds[2]),
                "north": float(solar_gdf.total_bounds[3])
            }
        }

        # Add statistics for specific_yield_kwh_per_kwp_yr if available
        if 'specific_yield_kwh_per_kwp_yr' in solar_gdf.columns:
            yield_stats = solar_gdf['specific_yield_kwh_per_kwp_yr'].describe()
            summary["yield_statistics"] = {
                "min": float(yield_stats['min']),
                "max": float(yield_stats['max']),
                "mean": float(yield_stats['mean']),
                "median": float(yield_stats['50%']),
                "std": float(yield_stats['std'])
            }

        # Add state information if available
        if 'state' in solar_gdf.columns:
            summary["states"] = sorted(solar_gdf['state'].unique().tolist())
            summary["total_states"] = len(summary["states"])

        return summary

    except Exception as e:
        return {"error": f"Failed to generate summary: {str(e)}"}


@mcp.tool()
def GetBrazilianStates() -> Dict[str, Any]:
    """Get list of Brazilian states from the administrative boundaries dataset.

    Returns:
        List of all Brazilian states
    """
    if brazilian_states_gdf is None:
        return {"error": "Brazilian states data not available"}

    try:
        states = sorted(brazilian_states_gdf['NM_UF'].tolist())

        return {
            "total_states": len(states),
            "states": states,
            "data_available": True
        }

    except Exception as e:
        return {"error": f"Failed to retrieve states: {str(e)}"}


@mcp.tool()
def DescribeServer() -> Dict[str, Any]:
    """Describe this server, its dataset, and available tools."""
    tools = [
        "GetSolarGeometriesMap - Generate map of all geometries colored by a column",
        "GetSolarGeometriesByState - Generate map filtered to a specific state",
        "GetGeometriesByState - Get count of geometries per state",
        "GetSolarGeometriesInBounds - Get geometries within geographic bounds",
        "GetSolarDataSummary - Get dataset statistics and metadata",
        "GetBrazilianStates - Get list of Brazilian states"
    ]

    return {
        "name": metadata.get("Name", "Solar Geometries Server"),
        "description": metadata.get("Description", "Supply chain geometry visualization"),
        "version": metadata.get("Version"),
        "dataset": metadata.get("Dataset"),
        "metrics": {
            "total_geometries": metadata.get("total_geometries"),
            "total_states": metadata.get("total_states")
        },
        "tools": tools,
        "examples": [
            "Show map of potential solar geometries colored by solar yield",
            "Get geometry counts by Brazilian state",
            "Map potential solar geometries in São Paulo"
        ],
        "data_available": solar_gdf is not None
    }


if __name__ == "__main__":
    mcp.run()
