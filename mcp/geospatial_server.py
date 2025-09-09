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

# Session-scoped storage (cleared between queries and isolated per session)
SESSIONS: dict = {}

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
    
    # Get entities by type
    entities1 = entities_gdf[entities_gdf['entity_type'] == entity_type1].copy()
    entities2 = entities_gdf[entities_gdf['entity_type'] == entity_type2].copy()
    
    if entities1.empty:
        return {"error": f"No entities of type '{entity_type1}' registered"}
    if entities2.empty:
        return {"error": f"No entities of type '{entity_type2}' registered"}
    
    print(f"Correlating {len(entities1)} {entity_type1} with {len(entities2)} {entity_type2} using method '{method}'")
    
    correlations = []
    
    if method == "within":
        try:
            # Use spatial join for efficient point-in-polygon correlation
            left = entities1.copy()
            right = entities2.copy()
            left = left.set_geometry('geometry')
            right = right.set_geometry('geometry')
            joined = gpd.sjoin(left, right, predicate='within', how='inner', lsuffix='1', rsuffix='2')
            for _, row in joined.iterrows():
                try:
                    props1 = json.loads(row['properties1'])
                    props2 = json.loads(row['properties2'])
                except Exception:
                    props1 = {}
                    props2 = {}
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
            print(f"Error during sjoin within correlation: {e}")
            return {"error": f"sjoin failed: {e}"}
    
    elif method == "intersects":
        # Check for any intersection/overlap
        for idx1, ent1 in entities1.iterrows():
            for idx2, ent2 in entities2.iterrows():
                try:
                    if ent1.geometry.intersects(ent2.geometry):
                        props1 = json.loads(ent1['properties'])
                        props2 = json.loads(ent2['properties'])
                        
                        # Calculate overlap percentage if both are polygons
                        overlap_pct = None
                        if ent1.geometry.geom_type == 'Polygon' and ent2.geometry.geom_type == 'Polygon':
                            intersection = ent1.geometry.intersection(ent2.geometry)
                            overlap_pct = (intersection.area / min(ent1.geometry.area, ent2.geometry.area)) * 100
                        
                        correlation = {
                            "entity1_id": ent1['entity_id'],
                            "entity1_type": entity_type1,
                            "entity1_properties": props1,
                            "entity2_id": ent2['entity_id'],
                            "entity2_type": entity_type2,
                            "entity2_properties": props2,
                            "relationship": "intersects"
                        }
                        
                        if overlap_pct is not None:
                            correlation["overlap_percentage"] = round(overlap_pct, 2)
                        
                        correlations.append(correlation)
                except Exception as e:
                    print(f"Error checking intersection: {e}")
                    continue
    
    elif method == "proximity":
        # Find entities within specified distance
        # Project to metric CRS for accurate distance calculation
        try:
            # Use Web Mercator for distance calculations (not perfect but fast)
            entities1_proj = entities1.to_crs('EPSG:3857')
            entities2_proj = entities2.to_crs('EPSG:3857')
            distance_m = distance_km * 1000
            
            for idx1, ent1 in entities1_proj.iterrows():
                for idx2, ent2 in entities2_proj.iterrows():
                    try:
                        dist = ent1.geometry.distance(ent2.geometry)
                        if dist <= distance_m:
                            props1 = json.loads(entities1.loc[idx1, 'properties'])
                            props2 = json.loads(entities2.loc[idx2, 'properties'])
                            
                            correlations.append({
                                "entity1_id": entities1.loc[idx1, 'entity_id'],
                                "entity1_type": entity_type1,
                                "entity1_properties": props1,
                                "entity2_id": entities2.loc[idx2, 'entity_id'],
                                "entity2_type": entity_type2,
                                "entity2_properties": props2,
                                "relationship": "proximity",
                                "distance_m": round(float(dist), 2),
                                "distance_km": round(float(dist / 1000), 2)
                            })
                    except Exception as e:
                        print(f"Error calculating distance: {e}")
                        continue
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
    
    return {
        "correlations": correlations,
        "total_correlations": len(correlations),
        "method": method,
        "parameters": {
            "distance_km": distance_km if method == "proximity" else None
        },
        "entity_counts": {
            entity_type1: len(entities1),
            entity_type2: len(entities2)
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
            # Point markers for solar facilities
            properties['marker-color'] = '#FFD700' if is_correlated else '#FFA500'  # Gold if correlated, orange if not
            properties['marker-size'] = 'medium' if is_correlated else 'small'
            properties['marker-symbol'] = 'circle'
            
            # Add descriptive title
            name = properties.get('name', 'Solar Facility')
            capacity = properties.get('capacity_mw', 'Unknown')
            status = 'IN deforestation' if is_correlated else 'Clear area'
            properties['title'] = f"{name} ({capacity} MW) - {status}"
            
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
        
        # Create feature
        feature = {
            "type": "Feature",
            "geometry": entity.geometry.__geo_interface__ if hasattr(entity.geometry, '__geo_interface__') else {"type": "Point", "coordinates": []},
            "properties": properties
        }
        features.append(feature)
    
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
            }
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
