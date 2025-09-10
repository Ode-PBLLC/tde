#!/usr/bin/env python3
"""
Geospatial Correlation Server

Pure correlation engine for spatial analysis between different data types.
Uses FastMCP for consistent server architecture.
Performs point-in-polygon, proximity, and intersection analysis.
"""

import json
import os
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastmcp import FastMCP
import uuid
from pathlib import Path

# Initialize FastMCP server
mcp = FastMCP("geospatial-server")

# Try to import geospatial libraries
try:
    import geopandas as gpd
    from shapely.geometry import Point, shape, Polygon
    import pandas as pd
    import numpy as np
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("Warning: GeoPandas not available. Install with: pip install geopandas shapely")

# Load style configuration for consistent legend/labels/colors
STYLE_CONFIG = {
    "labels": {},
    "defaults": {
        "point": {"color": "#4CAF50", "stroke_color": "#FFFFFF", "stroke_width": 1},
        "polygon": {"fill": "#9E9E9E", "fill_opacity": 0.25, "stroke": "#666666", "stroke_width": 1}
    }
}

def _load_style_config():
    try:
        base_dir = Path(__file__).resolve().parents[1]
        cfg_path = base_dir / "config" / "style.json"
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    STYLE_CONFIG.update(data)
                    print("[geospatial] Loaded style config")
    except Exception as e:
        print(f"[geospatial] Failed to load style config: {e}")

_load_style_config()

# Global static deforestation index (read-only, shared across sessions)
STATIC_DEFOR = {
    'loaded': False,
    'geoms_3857': None,
    'ids': None,
    'props': None,
    'tree': None
}

# Session-scoped storage (cleared between queries and isolated per session)
SESSIONS: dict = {}

def _load_static_deforestation_index() -> bool:
    """Load deforestation polygons and build a global STRtree index (EPSG:3857)."""
    if not GEOSPATIAL_AVAILABLE or STATIC_DEFOR['loaded']:
        return STATIC_DEFOR['loaded']
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        pq_path = os.path.join(base_dir, 'data', 'deforestation', 'deforestation.parquet')
        gj_path = os.path.join(base_dir, 'data', 'brazil_deforestation.geojson')
        if os.path.exists(pq_path):
            gdf = gpd.read_parquet(pq_path)
        elif os.path.exists(gj_path):
            gdf = gpd.read_file(gj_path)
        else:
            print("[geospatial] Static deforestation data not found; static index disabled")
            return False
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        gdf_3857 = gdf.to_crs('EPSG:3857')
        ids, props, geoms = [], [], []
        for idx, row in gdf_3857.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            geoms.append(geom)
            ids.append(f"deforest_{idx}")
            row_props = {k: (None if (pd.isna(v) if hasattr(pd, 'isna') else False) else v) for k, v in row.items() if k != 'geometry'}
            try:
                props.append(json.dumps(row_props))
            except Exception:
                props.append(json.dumps({}))
        if not geoms:
            print("[geospatial] No deforestation geometries for static index")
            return False
        from shapely.strtree import STRtree
        tree = STRtree(geoms)
        STATIC_DEFOR.update({'geoms_3857': geoms, 'ids': ids, 'props': props, 'tree': tree, 'loaded': True})
        print(f"[geospatial] Static deforestation index loaded with {len(geoms)} polygons")
        return True
    except Exception as e:
        print(f"[geospatial] Failed to load static deforestation index: {e}")
        return False

def _get_session_store(session_id: str):
    """Get or create per-session storage for entities and correlations."""
    if session_id not in SESSIONS:
        if GEOSPATIAL_AVAILABLE:
            gdf = gpd.GeoDataFrame(columns=['entity_id', 'entity_type', 'geometry', 'properties'])
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        else:
            gdf = None
        SESSIONS[session_id] = {
            'entities_gdf': gdf,
            'last_correlations': []
        }
    return SESSIONS[session_id]


@mcp.tool()
def RegisterEntities(entity_type: str, entities: List[Dict], session_id: str = "_default") -> Dict[str, Any]:
    """
    Register geographic entities for correlation.
    Accepts points (lat/lon) or polygons (GeoJSON geometry).
    
    Args:
        entity_type: Type of entity (e.g., 'solar_facility', 'deforestation_area')
        entities: List of entities with geometry data
        
    Returns:
        Registration summary
    """
    store = _get_session_store(session_id)
    entities_gdf = store['entities_gdf']
    
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    
    if not entities:
        return {"error": "No entities provided"}
    
    new_entities = []
    
    for i, entity in enumerate(entities):
        try:
            # Create geometry from either lat/lon or GeoJSON
            if 'latitude' in entity and 'longitude' in entity:
                # Point geometry from coordinates
                geom = Point(float(entity['longitude']), float(entity['latitude']))
                entity_id = entity.get('id', f"{entity_type}_point_{len(entities_gdf)}_{i}")
            elif 'geometry' in entity:
                # Polygon/Point from GeoJSON
                geom = shape(entity['geometry'])
                entity_id = entity.get('id', f"{entity_type}_geom_{len(entities_gdf)}_{i}")
            else:
                continue  # Skip entities without geometry
            
            # Store entity with metadata
            properties_dict = {k: v for k, v in entity.items() 
                             if k not in ['geometry', 'latitude', 'longitude', 'id']}
            
            new_entities.append({
                'entity_id': entity_id,
                'entity_type': entity_type,
                'geometry': geom,
                'properties': json.dumps(properties_dict)
            })
        except Exception as e:
            print(f"Error processing entity {i}: {e}")
            continue
    
    if new_entities:
        # Add to global GeoDataFrame
        new_gdf = gpd.GeoDataFrame(new_entities, crs="EPSG:4326")
        entities_gdf = pd.concat([entities_gdf, new_gdf], ignore_index=True)
        
        print(f"Registered {len(new_entities)} {entity_type} entities")
        
        result = {
            "registered": len(new_entities),
            "entity_type": entity_type,
            "total_entities": len(entities_gdf),
            "entity_types": list(entities_gdf['entity_type'].unique()),
            "session_id": session_id
        }
        store['entities_gdf'] = entities_gdf
        return result
    
    return {"error": "No valid entities to register"}


@mcp.tool()
def FindSpatialCorrelations(
    entity_type1: str,
    entity_type2: str,
    method: str = "within",
    distance_km: float = 5.0,
    session_id: str = "_default"
) -> Dict[str, Any]:
    """
    Find spatial relationships between two entity types.
    
    Args:
        entity_type1: First entity type (e.g., 'solar_facility')
        entity_type2: Second entity type (e.g., 'deforestation_area')
        method: Correlation method - 'within', 'intersects', or 'proximity'
        distance_km: Distance threshold for proximity method
        
    Returns:
        List of correlated entity pairs with relationship details
    """
    global entities_gdf, last_correlations
    
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    
    store = _get_session_store(session_id)
    entities_gdf = store['entities_gdf']
    
    if entities_gdf is None or entities_gdf.empty:
        return {"error": "No entities registered"}
    
    # Get entities by type (dynamic points)
    entities1 = entities_gdf[entities_gdf['entity_type'] == entity_type1].copy()
    # Use static deforestation for entity_type2 when requested
    use_static_defor = (entity_type2 == 'deforestation_area') and _load_static_deforestation_index()
    entities2 = entities_gdf[entities_gdf['entity_type'] == entity_type2].copy() if not use_static_defor else None
    
    if entities1.empty:
        return {"error": f"No entities of type '{entity_type1}' registered"}
    if not use_static_defor and entities2.empty:
        return {"error": f"No entities of type '{entity_type2}' registered"}
    
    count2 = (len(entities2) if entities2 is not None else (len(STATIC_DEFOR['geoms_3857']) if use_static_defor and STATIC_DEFOR['loaded'] else 0))
    print(f"Correlating {len(entities1)} {entity_type1} with {count2} {entity_type2} using method '{method}'")
    
    correlations = []
    
    if method == "within":
        try:
            if use_static_defor:
                from shapely.prepared import prep
                prepared = [prep(g) for g in STATIC_DEFOR['geoms_3857']]
                pts = entities1.to_crs('EPSG:3857')
                for idx1, ent1 in pts.iterrows():
                    p = ent1.geometry
                    cand_idxs = STATIC_DEFOR['tree'].query(p)
                    if not cand_idxs:
                        continue
                    try:
                        props1 = json.loads(entities1.loc[idx1, 'properties'])
                    except Exception:
                        props1 = {}
                    for ci in cand_idxs:
                        try:
                            if prepared[ci].contains(p):
                                correlations.append({
                                    "entity1_id": entities1.loc[idx1, 'entity_id'],
                                    "entity1_type": entity_type1,
                                    "entity1_properties": props1,
                                    "entity2_id": STATIC_DEFOR['ids'][ci],
                                    "entity2_type": 'deforestation_area',
                                    "entity2_properties": json.loads(STATIC_DEFOR['props'][ci]) if STATIC_DEFOR['props'][ci] else {},
                                    "relationship": "within",
                                    "confidence": 1.0
                                })
                        except Exception:
                            continue
            else:
                left = entities1.copy().set_geometry('geometry')
                right = entities2.copy().set_geometry('geometry')
                joined = gpd.sjoin(left, right, predicate='within', how='inner', lsuffix='1', rsuffix='2')
                for _, row in joined.iterrows():
                    try:
                        props1 = json.loads(row['properties1'])
                        props2 = json.loads(row['properties2'])
                    except Exception:
                        props1, props2 = {}, {}
                    correlations.append({
                        "entity1_id": row['entity_id1'],
                        "entity1_type": entity_type1,
                        "entity1_properties": props1,
                        "entity2_id": row['entity_id2'],
                        "entity2_type": entity_type2,
                        "entity2_properties": props2,
                        "relationship": "within",
                        "confidence": 1.0
                    })
        except Exception as e:
            print(f"Error in within correlation: {e}")
            return {"error": f"Within correlation failed: {str(e)}"}
    
    elif method == "intersects":
        # Handle intersects for both dynamic and static polygon sources
        try:
            if use_static_defor:
                pts = entities1.to_crs('EPSG:3857')
                for idx1, ent1 in pts.iterrows():
                    p = ent1.geometry
                    cand_idxs = STATIC_DEFOR['tree'].query(p)
                    if len(cand_idxs) == 0:
                        continue
                    try:
                        props1 = json.loads(entities1.loc[idx1, 'properties'])
                    except Exception:
                        props1 = {}
                    for ci in cand_idxs:
                        try:
                            poly = STATIC_DEFOR['geoms_3857'][ci]
                            if p.intersects(poly):
                                correlations.append({
                                    "entity1_id": entities1.loc[idx1, 'entity_id'],
                                    "entity1_type": entity_type1,
                                    "entity1_properties": props1,
                                    "entity2_id": STATIC_DEFOR['ids'][ci],
                                    "entity2_type": 'deforestation_area',
                                    "entity2_properties": json.loads(STATIC_DEFOR['props'][ci]) if STATIC_DEFOR['props'][ci] else {},
                                    "relationship": "intersects"
                                })
                        except Exception as e:
                            print(f"Error checking static intersects: {e}")
                            continue
            else:
                for idx1, ent1 in entities1.iterrows():
                    for idx2, ent2 in entities2.iterrows():
                        try:
                            if ent1.geometry.intersects(ent2.geometry):
                                props1 = json.loads(ent1['properties'])
                                props2 = json.loads(ent2['properties'])
                                correlation = {
                                    "entity1_id": ent1['entity_id'],
                                    "entity1_type": entity_type1,
                                    "entity1_properties": props1,
                                    "entity2_id": ent2['entity_id'],
                                    "entity2_type": entity_type2,
                                    "entity2_properties": props2,
                                    "relationship": "intersects"
                                }
                                correlations.append(correlation)
                        except Exception as e:
                            print(f"Error checking intersection: {e}")
                            continue
        except Exception as e:
            print(f"Error in intersects correlation: {e}")
            return {"error": f"Intersects correlation failed: {str(e)}"}
    
    elif method == "proximity":
        # Point-first proximity using static index when available
        try:
            distance_m = float(distance_km) * 1000.0
        except Exception:
            distance_m = 1000.0
        try:
            if use_static_defor:
                e1 = entities1.to_crs('EPSG:3857')
                for idx1, ent1 in e1.iterrows():
                    pt = ent1.geometry
                    cand_idxs = STATIC_DEFOR['tree'].query(pt.buffer(distance_m))
                    if len(cand_idxs) == 0:
                        continue
                    try:
                        props1 = json.loads(entities1.loc[idx1, 'properties'])
                    except Exception:
                        props1 = {}
                    for ci in cand_idxs:
                        try:
                            poly = STATIC_DEFOR['geoms_3857'][ci]
                            if pt.distance(poly) <= distance_m:
                                correlations.append({
                                    "entity1_id": entities1.loc[idx1, 'entity_id'],
                                    "entity1_type": entity_type1,
                                    "entity1_properties": props1,
                                    "entity2_id": STATIC_DEFOR['ids'][ci],
                                    "entity2_type": 'deforestation_area',
                                    "entity2_properties": json.loads(STATIC_DEFOR['props'][ci]) if STATIC_DEFOR['props'][ci] else {},
                                    "relationship": "proximity",
                                    "distance_km": float(distance_km)
                                })
                        except Exception:
                            continue
            else:
                # Fallback: buffer+sjoin
                e1 = entities1.to_crs('EPSG:3857')
                e2 = entities2.to_crs('EPSG:3857')
                e2_buf = e2.copy()
                e2_buf['geometry'] = e2_buf.geometry.buffer(distance_m)
                joined = gpd.sjoin(e1, e2_buf[['geometry','entity_id','properties']], predicate='within', how='inner', lsuffix='1', rsuffix='2')
                for _, row in joined.iterrows():
                    try:
                        props1 = json.loads(entities1.loc[row.name, 'properties'])
                    except Exception:
                        props1 = {}
                    ent2_id = row.get('entity_id2') if 'entity_id2' in row else None
                    props2 = json.loads(row.get('properties2', '{}')) if 'properties2' in row else {}
                    correlations.append({
                        "entity1_id": entities1.loc[row.name, 'entity_id'] if row.name in entities1.index else row.get('entity_id1'),
                        "entity1_type": entity_type1,
                        "entity1_properties": props1,
                        "entity2_id": ent2_id,
                        "entity2_type": entity_type2,
                        "entity2_properties": props2,
                        "relationship": "proximity",
                        "distance_km": float(distance_km)
                    })
        except Exception as e:
            print(f"Error in proximity calculation: {e}")
            return {"error": f"Proximity calculation failed: {str(e)}"}
    
    else:
        return {"error": f"Unknown method: {method}. Use 'within', 'intersects', or 'proximity'"}
    
    # Store correlations for map generation
    store['last_correlations'] = correlations
    
    # Mark correlated entities in the GeoDataFrame
    correlated_ids = set()
    for corr in correlations:
        correlated_ids.add(corr['entity1_id'])
        correlated_ids.add(corr['entity2_id'])
    
    entities_gdf['correlated'] = entities_gdf['entity_id'].isin(correlated_ids)
    store['entities_gdf'] = entities_gdf
    
    print(f"Found {len(correlations)} correlations")
    
    # Compute uniqueness metrics
    try:
        unique_facilities = len({c['entity1_id'] for c in correlations})
        unique_polygons = len({c['entity2_id'] for c in correlations})
    except Exception:
        unique_facilities = len(correlations)
        unique_polygons = 0

    return {
        "correlations": correlations,
        "total_correlations": len(correlations),  # pairs count (backward-compatible)
        "pairs_count": len(correlations),
        "unique_facilities": unique_facilities,
        "unique_polygons": unique_polygons,
        "method": method,
        "parameters": {
            "distance_km": distance_km if method == "proximity" else None
        },
        "entity_counts": {
            entity_type1: len(entities1),
            entity_type2: (len(entities2) if entities2 is not None else (len(STATIC_DEFOR['geoms_3857']) if use_static_defor and STATIC_DEFOR['loaded'] else 0))
        },
        "correlation_rate": round(len(correlations) / max(len(entities1), 1) * 100, 1),
        "map_generation": "Use GenerateCorrelationMap to visualize results",
        "session_id": session_id
    }


@mcp.tool()
def GenerateCorrelationMap(
    correlation_type: str = "spatial_correlation",
    show_uncorrelated: bool = True,
    session_id: str = "_default"
) -> Dict[str, Any]:
    """
    Generate a multi-layer GeoJSON map showing correlation results.
    Saves to static/maps/ for frontend display.
    
    Args:
        correlation_type: Type description for the map
        show_uncorrelated: Whether to include uncorrelated entities
        
    Returns:
        Map generation result with GeoJSON URL
    """
    store = _get_session_store(session_id)
    entities_gdf = store['entities_gdf']
    last_correlations = store['last_correlations']
    
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    
    if entities_gdf is None or entities_gdf.empty:
        return {"error": "No entities registered for mapping"}
    
    features = []
    
    # Get correlated entity IDs
    correlated_pairs = {}
    for corr in last_correlations:
        entity1_id = corr['entity1_id']
        entity2_id = corr['entity2_id']
        
        if entity1_id not in correlated_pairs:
            correlated_pairs[entity1_id] = []
        correlated_pairs[entity1_id].append(entity2_id)
        
        if entity2_id not in correlated_pairs:
            correlated_pairs[entity2_id] = []
        correlated_pairs[entity2_id].append(entity1_id)
    
    # Process each entity
    for idx, entity in entities_gdf.iterrows():
        is_correlated = entity['entity_id'] in correlated_pairs
        
        if not show_uncorrelated and not is_correlated:
            continue
        
        try:
            properties = json.loads(entity['properties'])
        except:
            properties = {}
        
        # Add correlation status
        properties['layer'] = entity['entity_type']
        properties['entity_id'] = entity['entity_id']
        properties['correlated'] = is_correlated
        
        # Style based on entity type and correlation status
        if entity['entity_type'] == 'solar_facility':
            # Ensure capacity_mw is numeric for frontend sizing expressions
            raw_cap = properties.get('capacity_mw')
            cap_num: float
            cap_missing = False
            try:
                if raw_cap is None or (isinstance(raw_cap, str) and raw_cap.strip() == ""):
                    cap_missing = True
                    cap_num = 0.0
                else:
                    cap_num = float(raw_cap)
                    # Handle NaN/inf
                    if cap_num != cap_num or cap_num is None:  # NaN check
                        cap_missing = True
                        cap_num = 0.0
            except Exception:
                cap_missing = True
                cap_num = 0.0
            # Set numeric field used by styles and a human-readable label
            properties['capacity_mw'] = cap_num
            properties['capacity_label'] = "missing" if cap_missing else f"{cap_num:g} MW"

            # Point markers for solar facilities
            desired_color = '#FFD700' if is_correlated else '#FFA500'  # Gold if correlated, orange if not
            properties['marker-color'] = desired_color
            properties['marker_color'] = desired_color
            properties['color'] = desired_color
            properties['marker-size'] = 'medium' if is_correlated else 'small'
            properties['marker-symbol'] = 'circle'

            # Legend label must use consistent category expected by frontend
            label_cfg = (STYLE_CONFIG.get('labels', {}).get('solar_facility') or {})
            legend_label = label_cfg.get('legend_label', 'Solar Facilities')
            # Preserve original country for popup context
            original_country = properties.get('country')
            properties['facility_country'] = original_country
            properties['country'] = legend_label
            
            # Add descriptive title
            name = properties.get('name', 'Solar Facility')
            # Use the human-readable label; fall back to numeric
            capacity = properties.get('capacity_label') or (
                f"{properties.get('capacity_mw', 0)} MW"
            )
            status = 'IN deforestation' if is_correlated else 'Clear area'
            # Include original country in title for user clarity
            if original_country and isinstance(original_country, str):
                properties['title'] = f"{name} • {original_country} • capacity: {capacity} • {status}"
            else:
                properties['title'] = f"{name} • capacity: {capacity} • {status}"
            
        elif entity['entity_type'] == 'deforestation_area':
            # Polygon styling for deforestation
            has_solar = is_correlated
            properties['fill'] = '#8B4513'  # Brown
            properties['fill-opacity'] = 0.4 if has_solar else 0.25
            properties['stroke'] = '#654321'  # Darker brown
            properties['stroke-width'] = 2 if has_solar else 1
            properties['stroke-opacity'] = 0.8
            
            # Add descriptive title
            area = properties.get('area_hectares', 0)
            status = 'Contains solar' if has_solar else 'No solar'
            properties['title'] = f"Deforestation: {area:.1f} ha - {status}"
            
        elif entity['entity_type'] == 'water_stressed_asset':
            # Different styling for GIST assets
            properties['marker-color'] = '#4169E1' if is_correlated else '#87CEEB'  # Royal blue if correlated
            properties['marker-size'] = 'medium'
            properties['marker-symbol'] = 'water'
            properties['title'] = f"Water-stressed asset - {'At risk' if is_correlated else 'Safe'}"
        
        else:
            # Generic styling for other types
            properties['marker-color'] = '#808080'
            properties['marker-size'] = 'small'
        
        # Ensure legend-related properties for points
        if entity['entity_type'] == 'solar_facility':
            label_cfg = (STYLE_CONFIG.get('labels', {}).get('solar_facility') or {})
            if 'title' not in properties:
                properties['title'] = label_cfg.get('popup_title', 'Solar Facility')
        # Create feature
        feature = {
            "type": "Feature",
            "geometry": entity.geometry.__geo_interface__ if hasattr(entity.geometry, '__geo_interface__') else {"type": "Point", "coordinates": []},
            "properties": properties
        }
        features.append(feature)
    
    # Add correlated deforestation polygons from static index (only correlated ones)
    if GEOSPATIAL_AVAILABLE and STATIC_DEFOR['loaded'] and last_correlations:
        needed = {c['entity2_id'] for c in last_correlations if c.get('entity2_type') == 'deforestation_area' and c.get('entity2_id')}
        if needed:
            id_to_idx = {pid: i for i, pid in enumerate(STATIC_DEFOR['ids'])}
            for pid in needed:
                ci = id_to_idx.get(pid)
                if ci is None:
                    continue
                try:
                    geom_3857 = STATIC_DEFOR['geoms_3857'][ci]
                    # Reproject to EPSG:4326 for GeoJSON
                    g = gpd.GeoSeries([geom_3857], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
                    label_cfg = (STYLE_CONFIG.get('labels', {}).get('deforestation_area') or {})
                    legend_label = label_cfg.get('legend_label', 'Deforestation')
                    poly_style = label_cfg.get('polygon', {})
                    feature = {
                        "type": "Feature",
                        "geometry": g.__geo_interface__,
                        "properties": {
                            "layer": "deforestation_area",
                            # Set 'country' to legend label to ensure frontend legend groups correctly
                            "country": legend_label,
                            "entity_id": pid,
                            "correlated": True,
                            "fill": poly_style.get('fill', STYLE_CONFIG['defaults']['polygon']['fill']),
                            "fill-opacity": poly_style.get('fill_opacity', STYLE_CONFIG['defaults']['polygon']['fill_opacity']),
                            "stroke": poly_style.get('stroke', STYLE_CONFIG['defaults']['polygon']['stroke']),
                            "stroke-width": poly_style.get('stroke_width', STYLE_CONFIG['defaults']['polygon']['stroke_width']),
                            "stroke-opacity": 0.8,
                            "title": label_cfg.get('popup_title', 'Deforestation'),
                            "name": label_cfg.get('popup_title', 'Deforestation')
                        }
                    }
                    features.append(feature)
                except Exception as e:
                    print(f"Error adding correlated polygon feature: {e}")

    # Add correlation highlight layers
    for corr in last_correlations:
        if corr['relationship'] == 'within':
            # Create a highlight for point-in-polygon correlations
            entity1 = entities_gdf[entities_gdf['entity_id'] == corr['entity1_id']]
            if not entity1.empty:
                entity1 = entity1.iloc[0]
                if entity1.geometry.geom_type == 'Point':
                    # Create highlight circle around the point
                    highlight_geom = entity1.geometry.buffer(0.003)  # ~300m radius
                    
                    highlight_feature = {
                        "type": "Feature",
                        "geometry": highlight_geom.__geo_interface__ if hasattr(highlight_geom, '__geo_interface__') else {"type": "Polygon", "coordinates": []},
                        "properties": {
                            "layer": "correlation_highlight",
                            "fill": "#FF0000",
                            "fill-opacity": 0.3,
                            "stroke": "#FF0000",
                            "stroke-width": 3,
                            "stroke-opacity": 0.7,
                            "title": f"Correlation: {corr['entity1_type']} within {corr['entity2_type']}"
                        }
                    }
                    features.append(highlight_feature)
    
    # Compute bounds from features for better map framing
    min_lat = 90.0
    max_lat = -90.0
    min_lon = 180.0
    max_lon = -180.0
    def _update_bounds(geom):
        nonlocal min_lat, max_lat, min_lon, max_lon
        try:
            if geom and isinstance(geom, dict):
                gtype = geom.get('type')
                coords = geom.get('coordinates')
                def _iter_coords(c):
                    if isinstance(c, (list, tuple)):
                        for e in c:
                            yield from _iter_coords(e)
                    else:
                        return
                if gtype == 'Point' and isinstance(coords, (list, tuple)) and len(coords) == 2:
                    lon, lat = float(coords[0]), float(coords[1])
                    min_lat = min(min_lat, lat); max_lat = max(max_lat, lat)
                    min_lon = min(min_lon, lon); max_lon = max(max_lon, lon)
                else:
                    for pair in _iter_coords(coords):
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            lon, lat = pair
                            min_lat = min(min_lat, float(lat)); max_lat = max(max_lat, float(lat))
                            min_lon = min(min_lon, float(lon)); max_lon = max(max_lon, float(lon))
        except Exception:
            pass

    for f in features:
        _update_bounds(f.get('geometry'))
    if min_lat > max_lat or min_lon > max_lon:
        # Fallback bounds
        bounds = {"north": 50, "south": -50, "east": 180, "west": -180}
        center = [0, 0]
    else:
        bounds = {"north": max_lat, "south": min_lat, "east": max_lon, "west": min_lon}
        center = [(min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0]

    # Collect countries/labels from features for legend support
    countries = set()
    for f in features:
        props = f.get('properties', {}) or {}
        c = props.get('country')
        if isinstance(c, str) and c.strip():
            countries.add(c.strip())
    # Ensure deforestation appears in legend if any polygons are present
    if any((f.get('properties', {}) or {}).get('layer') == 'deforestation_area' for f in features):
        countries.add('Deforestation')

    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated": datetime.now().isoformat(),
            "correlation_type": correlation_type,
            "total_features": len(features),
            "correlations": len(last_correlations),
            "entity_types": list(entities_gdf['entity_type'].unique()),
            "styling": {
                "solar_correlated": "Gold markers - facilities in deforestation",
                "solar_clear": "Orange markers - facilities in clear areas",
                "deforestation_with_solar": "Brown polygons 40% opacity - contains solar",
                "deforestation_without": "Brown polygons 25% opacity - no solar",
                "correlation_highlights": "Red circles 30% opacity - correlation points"
            },
            "bounds": bounds,
            "center": center,
            # Keep exact casing for frontend legend matching
            "countries": sorted({str(c) for c in countries})
        }
    }
    
    # Generate unique filename
    content_str = json.dumps(features, sort_keys=True)
    data_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
    filename = f"correlation_{correlation_type.replace(' ', '_')}_{data_hash}.geojson"
    
    # Save to static/maps/
    static_maps_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "static", "maps"
    )
    os.makedirs(static_maps_dir, exist_ok=True)
    
    filepath = os.path.join(static_maps_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(geojson, f)
    
    print(f"Generated correlation map: {filename}")
    
    # Calculate statistics
    entity_counts = entities_gdf['entity_type'].value_counts().to_dict()
    
    # Build summary description
    if 'solar_facility' in entity_counts and 'deforestation_area' in entity_counts:
        solar_correlated = sum(1 for c in last_correlations if c['entity1_type'] == 'solar_facility')
        total_solar = entity_counts['solar_facility']
        pct = round(100 * solar_correlated / total_solar, 1) if total_solar > 0 else 0
        description = f"Multi-layer map: {solar_correlated} of {total_solar} solar facilities ({pct}%) in deforested areas"
    else:
        description = f"Spatial correlation map with {len(last_correlations)} relationships"
    
    return {
        "type": "correlation_map_multilayer",
        "geojson_url": f"/static/maps/{filename}",
        "geojson_filename": filename,
        "summary": {
            "description": description,
            "total_features": len(features),
            "total_correlations": len(last_correlations),
            "entity_counts": entity_counts,
            "layers": [
                {"type": et, "count": ct, "correlated": sum(1 for c in last_correlations 
                 if c['entity1_type'] == et or c['entity2_type'] == et)}
                for et, ct in entity_counts.items()
            ]
        },
        "session_id": session_id
    }


@mcp.tool()
def GetRegisteredEntities(session_id: str = "_default") -> Dict[str, Any]:
    """
    Get summary of all registered entities.
    
    Returns:
        Summary of entities by type
    """
    store = _get_session_store(session_id)
    entities_gdf = store['entities_gdf']
    
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    
    if entities_gdf is None or entities_gdf.empty:
        return {"total": 0, "by_type": {}, "message": "No entities registered", "session_id": session_id}
    
    summary = entities_gdf.groupby('entity_type').size().to_dict()
    
    # Get geometry types
    geom_types = {}
    for entity_type in summary.keys():
        type_gdf = entities_gdf[entities_gdf['entity_type'] == entity_type]
        geom_type = type_gdf.iloc[0].geometry.geom_type if not type_gdf.empty else "Unknown"
        geom_types[entity_type] = geom_type
    
    return {
        "total": len(entities_gdf),
        "by_type": summary,
        "geometry_types": geom_types,
        "entity_types": list(summary.keys()),
        "has_correlations": len(store['last_correlations']) > 0,
        "correlation_count": len(store['last_correlations']),
        "session_id": session_id
    }


@mcp.tool()
def ClearSpatialIndex(session_id: str = "_default") -> Dict[str, Any]:
    """
    Clear all registered entities and correlations.
    Call this between different queries to start fresh.
    
    Returns:
        Clear operation summary
    """
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    store = _get_session_store(session_id)
    entities_gdf = store['entities_gdf']
    last_correlations = store['last_correlations']
    count = len(entities_gdf) if entities_gdf is not None else 0
    correlation_count = len(last_correlations)
    # Reset per-session
    if GEOSPATIAL_AVAILABLE:
        new_gdf = gpd.GeoDataFrame(columns=['entity_id', 'entity_type', 'geometry', 'properties'])
        new_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
    else:
        new_gdf = None
    store['entities_gdf'] = new_gdf
    store['last_correlations'] = []
    print(f"Cleared {count} entities and {correlation_count} correlations for session {session_id}")
    return {
        "cleared_entities": count,
        "cleared_correlations": correlation_count,
        "status": "success",
        "message": "Spatial index cleared - ready for new query",
        "session_id": session_id
    }


if __name__ == "__main__":
    mcp.run()
