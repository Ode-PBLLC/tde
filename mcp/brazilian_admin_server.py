#!/usr/bin/env python3
"""
Brazilian Municipalities Server

Serves municipality data from Brazilian census and shapefiles.
Uses FastMCP for consistent server architecture.
Provides administrative boundaries and demographics for spatial analysis.
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from fastmcp import FastMCP
import hashlib
from datetime import datetime
import difflib
import unicodedata

# Initialize FastMCP server
mcp = FastMCP("brazilian-admin-server")

# Try to import geospatial libraries
try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, box
    from shapely.strtree import STRtree
    import numpy as np
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("Warning: GeoPandas not available. Install with: pip install geopandas shapely")

# Data paths
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "brazilian_admin" / "br.csv"
# Prefer simplified GeoJSON over the heavyweight shapefile if available
SIMPLIFIED_GEOJSON_PATH = BASE_DIR / "data" / "brazilian_admin" / "BR_Municipios_2024_simplified.geojson"
SHAPEFILE_PATH = BASE_DIR / "data" / "brazilian_admin" / "municipality_shapes" / "BR_Municipios_2024.shp"
STATEFILE_PATH = BASE_DIR / "data" / "brazilian_admin" / "brazilian_states.geojson"

# Global data storage
municipalities_gdf = None
municipalities_tree = None
states_gdf = None
states_tree = None
csv_data = None

# Fast mode removed: always load polygon geometries (GeoJSON/shapefile)

def load_municipality_data():
    """Load and merge municipality CSV and shapefile data."""
    global municipalities_gdf, municipalities_tree, csv_data
    
    if not GEOSPATIAL_AVAILABLE:
        print("GeoPandas not available - server functionality limited")
        return False
    
    try:
        # Load CSV data first (fast)
        print(f"Loading CSV data from {CSV_PATH}")
        csv_data = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(csv_data)} municipalities from CSV")
        
        # Choose geometry source: prefer simplified GeoJSON, then shapefile
        if (SIMPLIFIED_GEOJSON_PATH.exists() or SHAPEFILE_PATH.exists()):
            if SIMPLIFIED_GEOJSON_PATH.exists():
                print(f"Loading simplified GeoJSON from {SIMPLIFIED_GEOJSON_PATH}")
                shape_gdf = gpd.read_file(SIMPLIFIED_GEOJSON_PATH)
            else:
                print(f"Loading shapefile from {SHAPEFILE_PATH}")
                shape_gdf = gpd.read_file(SHAPEFILE_PATH)
            print(f"Loaded {len(shape_gdf)} municipality boundaries")
            
            # Ensure CRS is set to WGS84
            if shape_gdf.crs is None:
                shape_gdf.set_crs(epsg=4326, inplace=True)
            elif shape_gdf.crs.to_string() != 'EPSG:4326':
                shape_gdf = shape_gdf.to_crs('EPSG:4326')
            
            # Create a mapping from city names to CSV data
            # Normalize names for matching (handle encoding issues)
            csv_data['city_normalized'] = csv_data['city'].str.lower().str.strip()
            
            # Use actual, known IBGE columns from the dataset
            # - Municipality code: CD_MUN
            # - Municipality name: NM_MUN
            # - State name: NM_UF
            # - State code: SIGLA_UF
            if 'CD_MUN' in shape_gdf.columns:
                shape_gdf['muni_code'] = shape_gdf['CD_MUN'].astype(str)
            elif 'CD_GEOCMU' in shape_gdf.columns:
                shape_gdf['muni_code'] = shape_gdf['CD_GEOCMU'].astype(str)
            else:
                shape_gdf['muni_code'] = shape_gdf.index.astype(str)

            if 'NM_MUN' in shape_gdf.columns:
                shape_gdf['muni_name'] = shape_gdf['NM_MUN'].astype(str)
            else:
                # Fallbacks if different naming appears
                for col in ['NM_MUNICIP', 'NOME', 'NAME', 'municipio']:
                    if col in shape_gdf.columns:
                        shape_gdf['muni_name'] = shape_gdf[col].astype(str)
                        break
                else:
                    print("Warning: No municipality name column found in shapes")
                    shape_gdf['muni_name'] = 'Unknown'

            # State attributes
            if 'NM_UF' in shape_gdf.columns:
                shape_gdf['state_name'] = shape_gdf['NM_UF'].astype(str)
            else:
                shape_gdf['state_name'] = 'Unknown'
            if 'SIGLA_UF' in shape_gdf.columns:
                shape_gdf['state_code'] = shape_gdf['SIGLA_UF'].astype(str)
            else:
                shape_gdf['state_code'] = None

            shape_gdf['name_normalized'] = shape_gdf['muni_name'].str.lower().str.strip()
            
            # Try to match with CSV data
            # First attempt: exact name match
            merged_gdf = shape_gdf.merge(
                csv_data,
                left_on='name_normalized',
                right_on='city_normalized',
                how='left'
            )
            
            # Calculate or reuse area for each municipality
            if 'AREA_KM2' in shape_gdf.columns:
                # Use provided area if present
                shape_gdf['area_km2'] = pd.to_numeric(shape_gdf['AREA_KM2'], errors='coerce')
            else:
                # Project to equal area for accurate calculation
                shape_proj = shape_gdf.to_crs('EPSG:5880')  # Brazil Polyconic
                shape_gdf['area_km2'] = shape_proj.geometry.area / 1_000_000  # Convert to km²

            # Centroids calculated in projected CRS then converted to WGS84 for lat/lon
            centroids_proj = shape_gdf.to_crs('EPSG:5880').geometry.centroid
            centroids_ll = gpd.GeoSeries(centroids_proj, crs='EPSG:5880').to_crs('EPSG:4326')
            
            # Create final GeoDataFrame with all necessary fields
            municipalities_gdf = gpd.GeoDataFrame({
                'muni_id': 'muni_' + shape_gdf['muni_code'],
                'muni_code': shape_gdf['muni_code'],
                'name': shape_gdf['muni_name'],
                'state': shape_gdf['state_name'],
                'state_code': shape_gdf['state_code'],
                'geometry': shape_gdf['geometry'],
                'area_km2': shape_gdf['area_km2'],
                'latitude': centroids_ll.y.values,
                'longitude': centroids_ll.x.values,
            }, crs=shape_gdf.crs)
            
            # Add CSV data where available
            for idx, row in municipalities_gdf.iterrows():
                name_norm = row['name'].lower().strip()
                csv_match = csv_data[csv_data['city_normalized'] == name_norm]
                if not csv_match.empty:
                    csv_row = csv_match.iloc[0]
                    municipalities_gdf.at[idx, 'population'] = csv_row.get('population', 0)
                    municipalities_gdf.at[idx, 'population_proper'] = csv_row.get('population_proper', 0)
                    municipalities_gdf.at[idx, 'capital'] = csv_row.get('capital', '')
                    # Prefer CSV lat/lng points if present
                    if pd.notna(csv_row.get('lat')) and pd.notna(csv_row.get('lng')):
                        municipalities_gdf.at[idx, 'latitude'] = csv_row.get('lat')
                        municipalities_gdf.at[idx, 'longitude'] = csv_row.get('lng')
                else:
                    municipalities_gdf.at[idx, 'population'] = 0
                    municipalities_gdf.at[idx, 'population_proper'] = 0
                    municipalities_gdf.at[idx, 'capital'] = ''
            
            # Ensure numeric types
            municipalities_gdf['population'] = pd.to_numeric(municipalities_gdf['population'], errors='coerce').fillna(0).astype(int)
            municipalities_gdf['population_proper'] = pd.to_numeric(municipalities_gdf['population_proper'], errors='coerce').fillna(0).astype(int)
            municipalities_gdf['area_km2'] = pd.to_numeric(municipalities_gdf['area_km2'], errors='coerce').fillna(0)
            
            # Build spatial index for fast queries
            print("Building spatial index...")
            municipalities_tree = STRtree(municipalities_gdf.geometry.values)
            
            print(f"Successfully loaded {len(municipalities_gdf)} municipalities with boundaries")
            print(f"States represented: {municipalities_gdf['state'].nunique()}")
            print(f"Total population: {municipalities_gdf['population'].sum():,}")
            
            return True
            
        else:
            print(f"Geometry source not found: {SIMPLIFIED_GEOJSON_PATH} or {SHAPEFILE_PATH}")
            return False
            
    except Exception as e:
        print(f"Error loading municipality data: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_state_data() -> bool:
    """Load Brazilian states GeoJSON and compute basic attributes (area, centroid)."""
    global states_gdf, states_tree

    if not GEOSPATIAL_AVAILABLE:
        print("GeoPandas not available - cannot load states")
        return False

    if not STATEFILE_PATH.exists():
        print(f"State GeoJSON not found at {STATEFILE_PATH}")
        return False

    try:
        print(f"Loading states GeoJSON from {STATEFILE_PATH}")
        gdf = gpd.read_file(STATEFILE_PATH)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_string() != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')

        # Determine state name and code columns
        name_col = None
        for col in ['NM_UF', 'NOME', 'NAME', 'state', 'UF_NAME']:
            if col in gdf.columns:
                name_col = col
                break
        if name_col is None:
            name_col = 'name'
            gdf[name_col] = 'Unknown'

        code_col = None
        for col in ['SIGLA_UF', 'UF', 'STATE_ABBR', 'abbrev']:
            if col in gdf.columns:
                code_col = col
                break

        # Compute area in km² using an equal-area CRS
        gdf_proj = gdf.to_crs('EPSG:5880')  # Brazil Polyconic
        gdf['area_km2'] = gdf_proj.geometry.area / 1_000_000

        # Centroids from projected CRS, then convert to WGS84
        centroids_proj = gdf_proj.geometry.centroid
        centroids_ll = gpd.GeoSeries(centroids_proj, crs='EPSG:5880').to_crs('EPSG:4326')

        # Build final GeoDataFrame
        states_gdf = gpd.GeoDataFrame({
            'state_id': (
                'state_' + gdf[code_col].astype(str)
                if code_col is not None else 'state_' + gdf.index.astype(str)
            ),
            'state_code': (gdf[code_col].astype(str) if code_col is not None else None),
            'name': gdf[name_col].astype(str),
            'geometry': gdf.geometry,
            'area_km2': gdf['area_km2'],
            'latitude': centroids_ll.y.values,
            'longitude': centroids_ll.x.values,
        }, crs=gdf.crs)

        # If codes are missing in the states file, infer from municipalities mapping
        try:
            if ('state_code' in states_gdf.columns) and (states_gdf['state_code'].isna().all() or (states_gdf['state_code'] == None).all()):  # noqa: E711
                # Build mapping from municipalities if loaded
                from pandas import Series
                mapping = None
                if 'municipalities_gdf' in globals() and municipalities_gdf is not None and not municipalities_gdf.empty:
                    tmp = municipalities_gdf[['state', 'state_code']].dropna().drop_duplicates()
                    if not tmp.empty:
                        mapping = Series(tmp['state_code'].values, index=tmp['state']).to_dict()
                if mapping:
                    states_gdf['state_code'] = states_gdf['name'].map(mapping)
        except Exception:
            pass

        # Spatial index for potential downstream use
        print("Building state spatial index...")
        states_tree = STRtree(states_gdf.geometry.values)

        print(f"Successfully loaded {len(states_gdf)} states with boundaries")
        print(f"Total state area (km²): {states_gdf['area_km2'].sum():,.0f}")
        return True
    except Exception as e:
        print(f"Error loading states: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load data on startup
mun_loaded = load_municipality_data()
st_loaded = load_state_data()

def _municipalities_dataset_metadata_impl() -> Dict[str, Any]:
    """Internal helper to compute municipalities metadata."""
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    if municipalities_gdf is None or municipalities_gdf.empty:
        return {"error": "Municipality geometries not loaded"}
    total = int(len(municipalities_gdf))
    states = int(municipalities_gdf['state'].nunique() if 'state' in municipalities_gdf.columns else 0)
    total_pop = int(municipalities_gdf['population'].sum() if 'population' in municipalities_gdf.columns else 0)
    total_area = float(municipalities_gdf['area_km2'].sum() if 'area_km2' in municipalities_gdf.columns else 0.0)
    src = "simplified_geojson" if SIMPLIFIED_GEOJSON_PATH.exists() else ("shapefile" if SHAPEFILE_PATH.exists() else "unknown")
    return {
        "Name": "Brazilian Municipalities",
        "Description": "Administrative boundaries and demographics for Brazil's municipalities",
        "Version": "2024",
        "geometry_source": src,
        "total_municipalities": total,
        "states": states,
        "total_population": total_pop,
        "total_area_km2": total_area
    }

def _normalize(text: Any) -> str:
    """Lowercase, strip accents/diacritics, and trim whitespace for robust matching."""
    if text is None:
        return ''
    s = str(text).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c))

def _fuzzy_select_by_series(df, series_name: str, query: str, max_results: int = 5, cutoff: float = 0.75):
    """Return subset of df where the named series fuzzily matches query (accent-insensitive)."""
    ser = df[series_name].astype(str)
    values = ser.fillna('').map(_normalize)
    q = _normalize(query)
    # Unique candidates to rank
    unique_vals = list(dict.fromkeys(values.tolist()))
    candidates = difflib.get_close_matches(q, unique_vals, n=max_results, cutoff=cutoff)
    if not candidates:
        return df.iloc[0:0]
    return df[values.isin(candidates)]

def _admin_dataset_metadata_impl() -> Dict[str, Any]:
    """Combined metadata for Brazilian administrative boundaries (states, municipalities)."""
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    muni_total = int(len(municipalities_gdf)) if municipalities_gdf is not None else 0
    state_total = int(len(states_gdf)) if states_gdf is not None else 0
    muni_area = float(municipalities_gdf['area_km2'].sum()) if municipalities_gdf is not None and 'area_km2' in municipalities_gdf.columns else 0.0
    state_area = float(states_gdf['area_km2'].sum()) if states_gdf is not None and 'area_km2' in states_gdf.columns else 0.0
    return {
        "Name": "Brazilian Administrative Boundaries",
        "Description": "States and municipalities for Brazil (geometries, areas)",
        "Version": "2024",
        "boundary_types": [t for t, n in [("state", state_total), ("municipality", muni_total)] if n > 0],
        "counts": {
            "states": state_total,
            "municipalities": muni_total
        },
        "total_area_km2": {
            "states": state_area,
            "municipalities": muni_area
        }
    }

@mcp.tool()
def GetMunicipalitiesDatasetMetadata() -> Dict[str, Any]:
    """Return dynamic metadata for Brazilian municipalities dataset."""
    try:
        return _municipalities_dataset_metadata_impl()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def DescribeServer() -> Dict[str, Any]:
    """Describe municipalities dataset, tools, and live stats."""
    try:
        meta = _municipalities_dataset_metadata_impl()
        tools = [
            "GetMunicipalitiesByFilter",
            "GetMunicipalityBoundaries",
            "GetMunicipalitiesInBounds",
            "FindMunicipalitiesNearPoint",
            "GetMunicipalityStatistics",
            "GetTopBrazilianCitiesByPopulation",
            "GetPopulationByState",
            "GetMunicipalitiesDatasetMetadata",
            "GetAdminDatasetMetadata",
            "GetBoundaryCounts",
            "GetStateBoundaries",
            "ListStateNames",
            "SearchMunicipalities"
        ]
        # Derive last_updated from data files
        last_updated = None
        try:
            from datetime import datetime as _dt
            paths = []
            for p in [SIMPLIFIED_GEOJSON_PATH, SHAPEFILE_PATH, CSV_PATH]:
                try:
                    if p and os.path.exists(p):
                        paths.append(str(p))
                except Exception:
                    pass
            if paths:
                last_updated = _dt.fromtimestamp(max(os.path.getmtime(p) for p in paths)).isoformat()
        except Exception:
            pass
        admin_meta = _admin_dataset_metadata_impl()
        return {
            "name": "Brazilian Admin Boundaries Server",
            "description": "Administrative boundaries for Brazil (states, municipalities)",
            "version": meta.get("Version", "2024") if isinstance(meta, dict) else "2024",
            "dataset": "IBGE GeoJSON boundaries (states + municipalities) + CSV demographics (municipalities only)",
            "metrics": {
                "total_municipalities": meta.get("total_municipalities") if isinstance(meta, dict) else None,
                "states": admin_meta.get("counts", {}).get("states") if isinstance(admin_meta, dict) else None,
                "total_population": meta.get("total_population") if isinstance(meta, dict) else None
            },
            "coverage": {
                "geometry_source": meta.get("geometry_source") if isinstance(meta, dict) else None
            },
            "tools": tools,
            "examples": [
                "Which municipalities intersect recent deforestation?",
                "Top 20 cities by population",
                "Get polygons for specific states"
            ],
            "last_updated": last_updated
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def GetAdminDatasetMetadata() -> Dict[str, Any]:
    """Return metadata for Brazilian administrative boundaries (states, municipalities)."""
    try:
        return _admin_dataset_metadata_impl()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def GetBoundaryCounts() -> Dict[str, Any]:
    """Return counts and total area by boundary type (state, municipality)."""
    meta = _admin_dataset_metadata_impl()
    if 'error' in meta:
        return meta
    return {
        "boundary_types": meta.get("boundary_types"),
        "counts": meta.get("counts"),
        "total_area_km2": meta.get("total_area_km2")
    }

@mcp.tool()
def GetStateBoundaries(
    state_names: List[str],
    include_metadata: bool = True
) -> Dict[str, Any]:
    """Get full polygon boundaries for specific Brazilian states.

    Args:
        state_names: List of state names or codes to retrieve (case-insensitive). Supports partial matches.
        include_metadata: Include area and centroid metadata.

    Returns:
        Dictionary with state polygons and metadata.
    """
    if not GEOSPATIAL_AVAILABLE or states_gdf is None or states_gdf.empty:
        return {"error": "State data not available"}

    if not state_names:
        return {"error": "No state names provided"}

    results = []
    not_found = []

    # Prepare matching columns
    name_series = states_gdf['name'].astype(str)
    code_series = states_gdf['state_code'].astype(str) if 'state_code' in states_gdf.columns and states_gdf['state_code'].notna().any() else None
    norm_name_series = name_series.map(_normalize)
    norm_code_series = code_series.map(_normalize) if code_series is not None else None

    for raw in state_names:
        q = str(raw).strip()
        q_norm = _normalize(q)

        # Exact match by normalized name or code
        matches = states_gdf[norm_name_series == q_norm]
        if code_series is not None:
            matches = pd.concat([
                matches,
                states_gdf[norm_code_series == q_norm]
            ]).drop_duplicates()

        # Partial substring match fallback (accent-insensitive)
        if matches.empty:
            contains = states_gdf[norm_name_series.str.contains(q_norm, na=False)]
            if code_series is not None:
                contains = pd.concat([
                    contains,
                    states_gdf[norm_code_series.str.contains(q_norm, na=False)]
                ]).drop_duplicates()
            matches = contains

        # Fuzzy fallback for misspellings
        if matches.empty:
            fuzzy = _fuzzy_select_by_series(states_gdf, 'name', q, max_results=5, cutoff=0.75)
            if code_series is not None and fuzzy.empty:
                tmp = states_gdf.rename(columns={'state_code': 'name'}) if 'state_code' in states_gdf.columns else states_gdf.assign(name='')
                fuzzy = _fuzzy_select_by_series(tmp, 'name', q, max_results=5, cutoff=0.75)
            matches = fuzzy

        if matches.empty:
            not_found.append(q)
            continue

        # Return one entry per match (states are unique; but allow multiple if partial)
        for _, match in matches.iterrows():
            st = {
                "id": match['state_id'],
                "name": match['name'],
                "state_code": match.get('state_code', None),
                "geometry": match.geometry.__geo_interface__ if hasattr(match.geometry, '__geo_interface__') else None
            }
            if include_metadata:
                st.update({
                    "area_km2": float(match.get('area_km2', 0.0)),
                    "latitude": float(match.get('latitude', 0.0)),
                    "longitude": float(match.get('longitude', 0.0))
                })
            results.append(st)

    meta = {
        "requested": len(state_names),
        "found": len(results),
        "missing": len(not_found)
    }
    if include_metadata and results:
        meta.update({
            "total_area_km2": float(sum(r.get('area_km2', 0.0) for r in results))
        })

    return {
        "states": results,
        "total_count": len(results),
        "not_found": not_found,
        "metadata": meta
    }

@mcp.tool()
def ListStateNames() -> Dict[str, Any]:
    """Return list of Brazilian state names and codes for tool selection."""
    if not GEOSPATIAL_AVAILABLE or states_gdf is None or states_gdf.empty:
        return {"error": "State data not available"}
    rows = []
    for _, r in states_gdf.iterrows():
        rows.append({
            "name": str(r.get('name')),
            "state_code": (str(r.get('state_code')) if r.get('state_code') is not None else None),
        })
    return {"states": rows, "total_count": len(rows)}

@mcp.tool()
def SearchMunicipalities(query: str, state: Optional[str] = None, max_results: int = 20) -> Dict[str, Any]:
    """Fuzzy search for municipality names, optionally scoped by state.

    Args:
        query: municipality name (may be misspelled; accent-insensitive)
        state: optional state name or code to scope search
        max_results: limit results
    """
    if not GEOSPATIAL_AVAILABLE or municipalities_gdf is None or municipalities_gdf.empty:
        return {"error": "Municipality data not available"}
    df = municipalities_gdf
    if state:
        s = _normalize(state)
        df = df[(df['state'].map(_normalize) == s) | (df['state_code'].map(_normalize) == s)]
    # Fuzzy select on 'name'
    subset = _fuzzy_select_by_series(df, 'name', query, max_results=max_results, cutoff=0.65)
    out = []
    for _, r in subset.head(max_results).iterrows():
        out.append({
            "name": r['name'],
            "state": r['state'],
            "state_code": r.get('state_code'),
            "population": int(r.get('population') or 0),
            "area_km2": float(r.get('area_km2') or 0.0)
        })
    return {
        "municipalities": out,
        "total_count": len(out),
        "query": query,
        "state_scope": state
    }

@mcp.tool()
def GetTopBrazilianCitiesByPopulation(top_n: int = 10) -> Dict[str, Any]:
    """Return a chart spec for top N Brazilian cities by population (from CSV)."""
    try:
        import pandas as pd  # noqa
    except Exception:
        return {"error": "pandas not available"}
    global csv_data
    if csv_data is None or csv_data.empty:
        return {"error": "CSV data not loaded"}
    df = csv_data.copy()
    if 'population' not in df.columns or 'city' not in df.columns:
        return {"error": "CSV missing required columns 'city' and 'population'"}
    # Clean
    df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
    df = df.sort_values('population', ascending=False).head(max(1, int(top_n)))
    data = [{"city": r['city'], "population": int(r['population'])} for _, r in df.iterrows()]
    return {
        "visualization_type": "comparison",
        "data": data,
        "chart_config": {
            "x_axis": "city",
            "y_axis": "population",
            "title": f"Top {max(1, int(top_n))} Brazilian Cities by Population",
            "chart_type": "bar"
        }
    }

@mcp.tool()
def GetPopulationByState() -> Dict[str, Any]:
    """Return a chart spec for total population by Brazilian state."""
    try:
        import pandas as pd  # noqa
    except Exception:
        return {"error": "pandas not available"}
    global municipalities_gdf, csv_data
    # Prefer merged GeoDataFrame with state attribute; fallback to CSV
    if municipalities_gdf is not None and not municipalities_gdf.empty and 'state' in municipalities_gdf.columns and 'population' in municipalities_gdf.columns:
        df = municipalities_gdf[['state', 'population']].copy()
        df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
        agg = df.groupby('state', dropna=False)['population'].sum().reset_index()
    elif csv_data is not None and not csv_data.empty and 'admin_name' in csv_data.columns and 'population' in csv_data.columns:
        df = csv_data[['admin_name', 'population']].copy()
        df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
        agg = df.groupby('admin_name', dropna=False)['population'].sum().reset_index().rename(columns={'admin_name': 'state'})
    else:
        return {"error": "No state/population columns available"}
    agg = agg.sort_values('population', ascending=False)
    data = [{"state": str(r['state']), "population": int(r['population'])} for _, r in agg.iterrows()]
    return {
        "visualization_type": "comparison",
        "data": data,
        "chart_config": {
            "x_axis": "state",
            "y_axis": "population",
            "title": "Total Population by Brazilian State",
            "chart_type": "bar"
        }
    }

@mcp.tool()
def GetMunicipalityAreaHistogram(bins: int = 10) -> Dict[str, Any]:
    """Return a histogram-like bar chart of municipality areas (km²)."""
    try:
        import pandas as pd  # noqa
        import numpy as np  # noqa
    except Exception:
        return {"error": "pandas/numpy not available"}
    global municipalities_gdf
    if municipalities_gdf is None or municipalities_gdf.empty or 'area_km2' not in municipalities_gdf.columns:
        return {"error": "Municipality areas not available"}
    s = pd.to_numeric(municipalities_gdf['area_km2'], errors='coerce').dropna()
    if s.empty:
        return {"error": "No area data"}
    bins = max(2, int(bins))
    counts, edges = np.histogram(s.values, bins=bins)
    labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(edges)-1)]
    data = [{"bin": labels[i], "count": int(counts[i])} for i in range(len(counts))]
    return {
        "visualization_type": "comparison",
        "data": data,
        "chart_config": {
            "x_axis": "bin",
            "y_axis": "count",
            "title": "Distribution of Municipality Areas (km²)",
            "chart_type": "bar"
        }
    }

@mcp.tool()
def GetMunicipalitiesByFilter(
    state: Optional[str] = None,
    min_population: Optional[int] = None,
    max_population: Optional[int] = None,
    capital_only: bool = False,
    limit: int = 6000
) -> Dict[str, Any]:
    """
    Get Brazilian municipalities matching filter criteria.
    
    Args:
        state: State name or abbreviation (e.g., "São Paulo" or "SP")
        min_population: Minimum population threshold
        max_population: Maximum population threshold
        capital_only: Return only capital cities
        limit: Maximum number of results
        
    Returns:
        Dictionary with municipalities and statistics
    """
    if not GEOSPATIAL_AVAILABLE or municipalities_gdf is None:
        return {"error": "Municipality data not available"}
    
    # Start with all municipalities
    filtered = municipalities_gdf.copy()
    
    # Filter by state
    if state:
        state_upper = state.upper()
        filtered = filtered[
            (filtered['state'].str.upper() == state_upper) |
            (filtered['state'].str.contains(state, case=False, na=False))
        ]
    
    # Filter by population
    if min_population is not None:
        filtered = filtered[filtered['population'] >= min_population]
    
    if max_population is not None:
        filtered = filtered[filtered['population'] <= max_population]
    
    # Filter capitals
    if capital_only:
        filtered = filtered[filtered['capital'].isin(['admin', 'primary', 'minor'])]
    
    # Sort by population and limit
    filtered = filtered.sort_values('population', ascending=False).head(limit)
    
    # Convert to output format
    municipalities = []
    for idx, row in filtered.iterrows():
        muni_dict = {
            "id": row['muni_id'],
            "name": row['name'],
            "state": row['state'],
            "population": int(row['population']),
            "population_proper": int(row.get('population_proper', 0)),
            "area_km2": float(row.get('area_km2', 0)),
            "capital": row.get('capital', ''),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "geometry": row.geometry.__geo_interface__ if hasattr(row.geometry, '__geo_interface__') else None
        }
        municipalities.append(muni_dict)
    
    # Calculate statistics
    total_pop = filtered['population'].sum()
    avg_pop = filtered['population'].mean()
    total_area = filtered['area_km2'].sum()
    
    return {
        "municipalities": municipalities,
        "total_count": len(municipalities),
        "metadata": {
            "filter_applied": {
                "state": state,
                "min_population": min_population,
                "max_population": max_population,
                "capital_only": capital_only
            },
            "total_population": int(total_pop),
            "average_population": int(avg_pop) if not pd.isna(avg_pop) else 0,
            "total_area_km2": float(total_area)
        },
        "summary": f"Found {len(municipalities)} municipalities matching criteria"
    }


@mcp.tool()
def GetMunicipalityBoundaries(
    municipality_names: List[str],
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Get full polygon boundaries for specific municipalities.
    
    Args:
        municipality_names: List of municipality names to retrieve
        include_metadata: Include population and area data
        
    Returns:
        Dictionary with municipality polygons and metadata
    """
    if not GEOSPATIAL_AVAILABLE or municipalities_gdf is None:
        return {"error": "Municipality data not available"}
    
    if not municipality_names:
        return {"error": "No municipality names provided"}
    
    # Find matching municipalities (accent-insensitive; supports fuzzy fallback)
    municipalities = []
    not_found = []
    name_series = municipalities_gdf['name'].astype(str)
    norm_name_series = name_series.map(_normalize)

    for name in municipality_names:
        q = str(name).strip()
        q_norm = _normalize(q)
        # Exact normalized match
        matches = municipalities_gdf[norm_name_series == q_norm]
        # Partial substring match
        if matches.empty:
            matches = municipalities_gdf[norm_name_series.str.contains(q_norm, na=False)]
        # Fuzzy match
        if matches.empty:
            matches = _fuzzy_select_by_series(municipalities_gdf, 'name', q, max_results=5, cutoff=0.7)
        
        if not matches.empty:
            # Take the first match (highest population if multiple)
            match = matches.sort_values('population', ascending=False).iloc[0]
            
            muni_dict = {
                "id": match['muni_id'],
                "name": match['name'],
                "state": match['state'],
                "geometry": match.geometry.__geo_interface__ if hasattr(match.geometry, '__geo_interface__') else None
            }
            
            if include_metadata:
                muni_dict.update({
                    "population": int(match['population']),
                    "population_proper": int(match.get('population_proper', 0)),
                    "area_km2": float(match.get('area_km2', 0)),
                    "capital": match.get('capital', ''),
                    "latitude": float(match['latitude']),
                    "longitude": float(match['longitude'])
                })
            
            municipalities.append(muni_dict)
        else:
            not_found.append(name)
    
    result = {
        "municipalities": municipalities,
        "total_count": len(municipalities),
        "not_found": not_found,
        "metadata": {
            "requested": len(municipality_names),
            "found": len(municipalities),
            "missing": len(not_found)
        }
    }
    
    if municipalities and include_metadata:
        total_pop = sum(m['population'] for m in municipalities)
        total_area = sum(m['area_km2'] for m in municipalities)
        result["metadata"]["total_population"] = total_pop
        result["metadata"]["total_area_km2"] = total_area
    
    return result


@mcp.tool()
def GetMunicipalitiesInBounds(
    north: float,
    south: float, 
    east: float,
    west: float,
    min_area_km2: Optional[float] = None,
    limit: int = 6000
) -> Dict[str, Any]:
    """
    Get municipalities within a geographic bounding box.
    
    Args:
        north: Northern latitude boundary
        south: Southern latitude boundary
        east: Eastern longitude boundary
        west: Western longitude boundary
        min_area_km2: Minimum municipality area filter
        limit: Maximum number of results
        
    Returns:
        Dictionary with municipalities in the bounds
    """
    if not GEOSPATIAL_AVAILABLE or municipalities_gdf is None:
        return {"error": "Municipality data not available"}
    
    # Create bounding box
    bbox = box(west, south, east, north)
    
    # Find intersecting municipalities
    if municipalities_tree:
        # Use spatial index for fast query
        possible_matches_idx = list(municipalities_tree.query(bbox))
        possible_matches = municipalities_gdf.iloc[possible_matches_idx]
        intersects = possible_matches[possible_matches.geometry.intersects(bbox)]
    else:
        # Fallback to direct intersection
        intersects = municipalities_gdf[municipalities_gdf.geometry.intersects(bbox)]
    
    # Filter by minimum area
    if min_area_km2 is not None:
        intersects = intersects[intersects['area_km2'] >= min_area_km2]
    
    # Sort by population and limit
    intersects = intersects.sort_values('population', ascending=False).head(limit)
    
    # Convert to output format
    municipalities = []
    for idx, row in intersects.iterrows():
        muni_dict = {
            "id": row['muni_id'],
            "name": row['name'],
            "state": row['state'],
            "population": int(row['population']),
            "area_km2": float(row.get('area_km2', 0)),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "geometry": row.geometry.__geo_interface__ if hasattr(row.geometry, '__geo_interface__') else None
        }
        municipalities.append(muni_dict)
    
    return {
        "municipalities": municipalities,
        "total_count": len(municipalities),
        "bounds": {
            "north": north,
            "south": south,
            "east": east,
            "west": west
        },
        "metadata": {
            "total_population": int(intersects['population'].sum()),
            "total_area_km2": float(intersects['area_km2'].sum()),
            "states": list(intersects['state'].unique())
        },
        "summary": f"Found {len(municipalities)} municipalities in bounding box"
    }


@mcp.tool()
def GetMunicipalityStatistics(
    group_by: str = "state"
) -> Dict[str, Any]:
    """
    Get aggregate statistics about Brazilian municipalities.
    
    Args:
        group_by: Grouping field ('state', 'capital', or 'all')
        
    Returns:
        Statistical summary of municipality data
    """
    if not GEOSPATIAL_AVAILABLE or municipalities_gdf is None:
        return {"error": "Municipality data not available"}
    
    if group_by == "state":
        # Group by state
        grouped = municipalities_gdf.groupby('state').agg({
            'muni_id': 'count',
            'population': ['sum', 'mean', 'median'],
            'area_km2': 'sum'
        }).round(2)
        
        stats = {}
        for state in grouped.index:
            stats[state] = {
                'municipality_count': int(grouped.loc[state, ('muni_id', 'count')]),
                'total_population': int(grouped.loc[state, ('population', 'sum')]),
                'avg_population': int(grouped.loc[state, ('population', 'mean')]),
                'median_population': int(grouped.loc[state, ('population', 'median')]),
                'total_area_km2': float(grouped.loc[state, ('area_km2', 'sum')])
            }
        
        return {
            "statistics": stats,
            "group_by": "state",
            "total_municipalities": len(municipalities_gdf),
            "total_states": len(stats)
        }
    
    elif group_by == "capital":
        # Group by capital status
        capital_types = ['primary', 'admin', 'minor', '']
        stats = {}
        
        for cap_type in capital_types:
            if cap_type == '':
                subset = municipalities_gdf[municipalities_gdf['capital'] == '']
                label = 'non_capital'
            else:
                subset = municipalities_gdf[municipalities_gdf['capital'] == cap_type]
                label = f'capital_{cap_type}'
            
            if not subset.empty:
                stats[label] = {
                    'count': len(subset),
                    'total_population': int(subset['population'].sum()),
                    'avg_population': int(subset['population'].mean()),
                    'total_area_km2': float(subset['area_km2'].sum())
                }
        
        return {
            "statistics": stats,
            "group_by": "capital",
            "total_municipalities": len(municipalities_gdf)
        }
    
    else:  # all
        # Overall statistics
        pop_data = municipalities_gdf['population']
        area_data = municipalities_gdf['area_km2']
        
        return {
            "statistics": {
                "total_municipalities": len(municipalities_gdf),
                "total_population": int(pop_data.sum()),
                "population": {
                    "mean": int(pop_data.mean()),
                    "median": int(pop_data.median()),
                    "min": int(pop_data.min()),
                    "max": int(pop_data.max()),
                    "std": int(pop_data.std())
                },
                "area_km2": {
                    "total": float(area_data.sum()),
                    "mean": float(area_data.mean()),
                    "median": float(area_data.median()),
                    "min": float(area_data.min()),
                    "max": float(area_data.max())
                },
                "states": list(municipalities_gdf['state'].unique()),
                "state_count": municipalities_gdf['state'].nunique(),
                "capitals_count": len(municipalities_gdf[municipalities_gdf['capital'] != ''])
            },
            "group_by": "all"
        }


@mcp.tool()
def FindMunicipalitiesNearPoint(
    latitude: float,
    longitude: float,
    radius_km: float = 50.0,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Find municipalities within a radius of a geographic point.
    
    Args:
        latitude: Point latitude
        longitude: Point longitude
        radius_km: Search radius in kilometers
        limit: Maximum number of results
        
    Returns:
        Dictionary with nearby municipalities sorted by distance
    """
    if not GEOSPATIAL_AVAILABLE or municipalities_gdf is None:
        return {"error": "Municipality data not available"}
    
    # Create point and buffer
    point = Point(longitude, latitude)
    
    # Project to meter-based CRS for accurate distance
    gdf_proj = municipalities_gdf.to_crs('EPSG:3857')
    point_proj = gpd.GeoSeries([point], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
    
    # Buffer in meters
    buffer_m = radius_km * 1000
    buffer_proj = point_proj.buffer(buffer_m)
    
    # Find municipalities within buffer
    if municipalities_tree:
        # Use centroids for initial filtering with tree
        centroids_proj = gdf_proj.geometry.centroid
        possible_idx = []
        for idx, centroid in enumerate(centroids_proj):
            if centroid.distance(point_proj) <= buffer_m:
                possible_idx.append(idx)
        nearby = municipalities_gdf.iloc[possible_idx].copy()
    else:
        # Direct distance calculation
        nearby = municipalities_gdf.copy()
        nearby_proj = nearby.to_crs('EPSG:3857')
        distances = nearby_proj.geometry.centroid.distance(point_proj) / 1000  # Convert to km
        nearby = nearby[distances <= radius_km].copy()
        nearby['distance_km'] = distances[distances <= radius_km]
    
    # Calculate exact distances for results
    if not nearby.empty and 'distance_km' not in nearby.columns:
        nearby_proj = nearby.to_crs('EPSG:3857')
        nearby['distance_km'] = nearby_proj.geometry.centroid.distance(point_proj) / 1000
    
    # Sort by distance and limit
    if not nearby.empty:
        nearby = nearby.sort_values('distance_km').head(limit)
    
    # Convert to output format
    municipalities = []
    for idx, row in nearby.iterrows():
        muni_dict = {
            "id": row['muni_id'],
            "name": row['name'],
            "state": row['state'],
            "population": int(row['population']),
            "area_km2": float(row.get('area_km2', 0)),
            "distance_km": round(float(row['distance_km']), 2),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "geometry": row.geometry.__geo_interface__ if hasattr(row.geometry, '__geo_interface__') else None
        }
        municipalities.append(muni_dict)
    
    return {
        "municipalities": municipalities,
        "total_count": len(municipalities),
        "search_point": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km
        },
        "metadata": {
            "nearest": municipalities[0]['name'] if municipalities else None,
            "farthest": municipalities[-1]['name'] if municipalities else None,
            "total_population_in_radius": sum(m['population'] for m in municipalities)
        },
        "summary": f"Found {len(municipalities)} municipalities within {radius_km}km"
    }


if __name__ == "__main__":
    mcp.run()
