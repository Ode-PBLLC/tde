"""
ODE MCP Generic - Geographic Server
Universal mapping and geographic analysis tools for location-based data.
"""
import json
from fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Union, Tuple
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import numpy as np
from datetime import datetime
import hashlib
import time
import requests
import os

mcp = FastMCP("geographic-server")

metadata = {
    "Name": "Generic Geographic Server", 
    "Description": "Provides universal mapping and geographic analysis tools for location-based data",
    "Version": "1.0.0",
    "Author": "ODE Framework"
}

def _generate_map_id(map_type: str, data_hash: str = None) -> str:
    """Generate unique map ID"""
    timestamp = str(int(time.time() * 1000))
    if data_hash:
        return f"{map_type}_{data_hash}_{timestamp}"
    else:
        return f"{map_type}_{timestamp}"

def _create_geojson_feature(lat: float, lon: float, properties: Dict[str, Any]) -> Dict[str, Any]:
    """Create a GeoJSON feature from coordinates and properties"""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]  # GeoJSON uses [lon, lat] order
        },
        "properties": properties
    }

def _create_map_response(map_type: str, geojson_data: Dict, title: str, summary: Dict) -> Dict[str, Any]:
    """Create standardized map response"""
    map_id = _generate_map_id(map_type)
    
    return {
        "type": "map",
        "map_type": map_type,
        "map_id": map_id,
        "title": title,
        "geojson": geojson_data,
        "summary": summary,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "map_library": "geojson",
            "interactive": True
        }
    }

@mcp.tool()
def CreatePointMap(
    locations: List[Dict[str, Any]],
    lat_field: str = "latitude",
    lon_field: str = "longitude", 
    title: str = "Point Map",
    color_field: Optional[str] = None,
    size_field: Optional[str] = None,
    popup_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a point map from location data.
    
    Args:
        locations: List of location dictionaries with lat/lon and other properties
        lat_field: Name of latitude field in location data
        lon_field: Name of longitude field in location data  
        title: Map title
        color_field: Optional field name to use for point colors
        size_field: Optional field name to use for point sizes
        popup_fields: List of field names to include in popup/tooltip
    
    Returns:
        Map data structure with GeoJSON for frontend rendering
    """
    try:
        if not locations:
            return {"error": "Locations data cannot be empty"}
        
        # Validate required fields
        for i, loc in enumerate(locations):
            if lat_field not in loc or lon_field not in loc:
                return {"error": f"Location {i} missing required fields: {lat_field}, {lon_field}"}
            
            try:
                float(loc[lat_field])
                float(loc[lon_field])
            except (ValueError, TypeError):
                return {"error": f"Location {i} has invalid coordinates"}
        
        # Create GeoJSON features
        features = []
        for loc in locations:
            lat = float(loc[lat_field])
            lon = float(loc[lon_field])
            
            # Create properties for the feature
            properties = loc.copy()
            
            # Add display properties
            if color_field and color_field in loc:
                properties["marker_color"] = loc[color_field]
            
            if size_field and size_field in loc:
                properties["marker_size"] = loc[size_field]
            
            # Set popup content
            if popup_fields:
                popup_content = []
                for field in popup_fields:
                    if field in loc:
                        popup_content.append(f"{field}: {loc[field]}")
                properties["popup"] = "<br>".join(popup_content)
            
            features.append(_create_geojson_feature(lat, lon, properties))
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Calculate summary statistics
        lats = [float(loc[lat_field]) for loc in locations]
        lons = [float(loc[lon_field]) for loc in locations]
        
        summary = {
            "total_points": len(locations),
            "bounds": {
                "north": max(lats),
                "south": min(lats),
                "east": max(lons),
                "west": min(lons)
            },
            "center": {
                "latitude": sum(lats) / len(lats),
                "longitude": sum(lons) / len(lons)
            }
        }
        
        if color_field and color_field in locations[0]:
            color_values = [loc.get(color_field) for loc in locations if loc.get(color_field) is not None]
            summary["color_field_stats"] = {
                "field": color_field,
                "unique_values": len(set(color_values)),
                "sample_values": list(set(color_values))[:10]
            }
        
        return _create_map_response("point", geojson_data, title, summary)
        
    except Exception as e:
        return {"error": f"Point map creation failed: {str(e)}"}

@mcp.tool()
def CreateChoroplethMap(
    regions_data: List[Dict[str, Any]],
    region_field: str = "region",
    value_field: str = "value",
    title: str = "Choropleth Map",
    color_scale: str = "Blues"
) -> Dict[str, Any]:
    """
    Create a choropleth map from regional data.
    
    Args:
        regions_data: List of region dictionaries with region names and values
        region_field: Name of field containing region/country names
        value_field: Name of field containing values to visualize
        title: Map title
        color_scale: Color scale for the visualization
    
    Returns:
        Map data structure for frontend rendering
    """
    try:
        if not regions_data:
            return {"error": "Regions data cannot be empty"}
        
        # Validate required fields
        for i, region in enumerate(regions_data):
            if region_field not in region or value_field not in region:
                return {"error": f"Region {i} missing required fields: {region_field}, {value_field}"}
            
            try:
                float(region[value_field])
            except (ValueError, TypeError):
                return {"error": f"Region {i} has invalid value: {region[value_field]}"}
        
        # Create simplified choropleth data structure
        # Note: Actual geometry would need to be loaded from shape files or external API
        choropleth_data = {
            "type": "choropleth",
            "data": regions_data,
            "config": {
                "region_field": region_field,
                "value_field": value_field,
                "color_scale": color_scale
            }
        }
        
        # Calculate summary statistics
        values = [float(region[value_field]) for region in regions_data]
        regions = [region[region_field] for region in regions_data]
        
        summary = {
            "total_regions": len(regions_data),
            "value_range": [min(values), max(values)],
            "regions": regions[:10],  # Sample regions
            "color_scale": color_scale,
            "note": "Choropleth rendering requires region boundary data"
        }
        
        return _create_map_response("choropleth", choropleth_data, title, summary)
        
    except Exception as e:
        return {"error": f"Choropleth map creation failed: {str(e)}"}

@mcp.tool()
def GeocodeLocations(
    addresses: List[str],
    api_key: Optional[str] = None,
    provider: str = "nominatim"
) -> Dict[str, Any]:
    """
    Geocode addresses to latitude/longitude coordinates.
    
    Args:
        addresses: List of address strings to geocode
        api_key: Optional API key for commercial geocoding services
        provider: Geocoding provider ("nominatim", "google", "mapbox")
    
    Returns:
        List of geocoded locations with coordinates and metadata
    """
    try:
        if not addresses:
            return {"error": "Addresses list cannot be empty"}
        
        geocoded_results = []
        
        for i, address in enumerate(addresses):
            if not address.strip():
                geocoded_results.append({
                    "original_address": address,
                    "success": False,
                    "error": "Empty address"
                })
                continue
            
            try:
                if provider == "nominatim":
                    # Use free Nominatim service (OpenStreetMap)
                    result = _geocode_nominatim(address)
                else:
                    # Placeholder for other services
                    result = {
                        "success": False,
                        "error": f"Provider {provider} not implemented. Use 'nominatim' for free geocoding."
                    }
                
                result["original_address"] = address
                geocoded_results.append(result)
                
                # Be respectful to free services
                if provider == "nominatim":
                    time.sleep(1)  # Rate limiting
                
            except Exception as e:
                geocoded_results.append({
                    "original_address": address,
                    "success": False,
                    "error": f"Geocoding failed: {str(e)}"
                })
        
        # Summary statistics
        successful_geocodes = [r for r in geocoded_results if r.get("success")]
        
        summary = {
            "total_addresses": len(addresses),
            "successful_geocodes": len(successful_geocodes),
            "failed_geocodes": len(addresses) - len(successful_geocodes),
            "success_rate": len(successful_geocodes) / len(addresses) * 100,
            "provider": provider
        }
        
        return {
            "type": "geocoding_results",
            "results": geocoded_results,
            "summary": summary
        }
        
    except Exception as e:
        return {"error": f"Geocoding failed: {str(e)}"}

def _geocode_nominatim(address: str) -> Dict[str, Any]:
    """Geocode using Nominatim (OpenStreetMap) service"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "addressdetails": 1
        }
        
        headers = {
            "User-Agent": "ODE-MCP-Generic/1.0"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return {
                "success": False,
                "error": "Address not found"
            }
        
        result = data[0]
        
        return {
            "success": True,
            "latitude": float(result["lat"]),
            "longitude": float(result["lon"]),
            "formatted_address": result.get("display_name", address),
            "country": result.get("address", {}).get("country"),
            "city": result.get("address", {}).get("city"),
            "confidence": result.get("importance", 0)
        }
        
    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Geocoding error: {str(e)}"
        }

@mcp.tool()
def CalculateDistance(
    point1: Dict[str, float],
    point2: Dict[str, float], 
    unit: str = "km"
) -> Dict[str, Any]:
    """
    Calculate distance between two geographic points.
    
    Args:
        point1: Dictionary with "latitude" and "longitude" keys
        point2: Dictionary with "latitude" and "longitude" keys
        unit: Distance unit ("km", "miles", "meters")
    
    Returns:
        Distance calculation results
    """
    try:
        # Validate input points
        for i, point in enumerate([point1, point2], 1):
            if "latitude" not in point or "longitude" not in point:
                return {"error": f"Point {i} missing latitude or longitude"}
            
            try:
                lat = float(point["latitude"])
                lon = float(point["longitude"])
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    return {"error": f"Point {i} has invalid coordinates"}
            except (ValueError, TypeError):
                return {"error": f"Point {i} has non-numeric coordinates"}
        
        # Calculate distance using Haversine formula
        lat1, lon1 = float(point1["latitude"]), float(point1["longitude"])
        lat2, lon2 = float(point2["latitude"]), float(point2["longitude"])
        
        distance_km = _haversine_distance(lat1, lon1, lat2, lon2)
        
        # Convert to requested unit
        if unit == "km":
            distance = distance_km
        elif unit == "miles":
            distance = distance_km * 0.621371
        elif unit == "meters":
            distance = distance_km * 1000
        else:
            return {"error": f"Unsupported unit: {unit}. Use 'km', 'miles', or 'meters'"}
        
        return {
            "distance": round(distance, 2),
            "unit": unit,
            "point1": point1,
            "point2": point2,
            "straight_line_distance": True
        }
        
    except Exception as e:
        return {"error": f"Distance calculation failed: {str(e)}"}

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points using Haversine formula"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    
    return c * r

@mcp.tool()
def FindPointsInRadius(
    center_point: Dict[str, float],
    points: List[Dict[str, Any]],
    radius_km: float,
    lat_field: str = "latitude",
    lon_field: str = "longitude"
) -> Dict[str, Any]:
    """
    Find all points within a specified radius of a center point.
    
    Args:
        center_point: Dictionary with "latitude" and "longitude" keys
        points: List of point dictionaries with location data
        radius_km: Search radius in kilometers
        lat_field: Name of latitude field in points data
        lon_field: Name of longitude field in points data
    
    Returns:
        Points within radius with distance information
    """
    try:
        if "latitude" not in center_point or "longitude" not in center_point:
            return {"error": "Center point missing latitude or longitude"}
        
        if not points:
            return {"error": "Points list cannot be empty"}
        
        center_lat = float(center_point["latitude"])
        center_lon = float(center_point["longitude"])
        
        points_in_radius = []
        
        for point in points:
            if lat_field not in point or lon_field not in point:
                continue
            
            try:
                point_lat = float(point[lat_field])
                point_lon = float(point[lon_field])
                
                distance = _haversine_distance(center_lat, center_lon, point_lat, point_lon)
                
                if distance <= radius_km:
                    point_with_distance = point.copy()
                    point_with_distance["distance_km"] = round(distance, 2)
                    points_in_radius.append(point_with_distance)
                    
            except (ValueError, TypeError):
                continue
        
        # Sort by distance
        points_in_radius.sort(key=lambda p: p["distance_km"])
        
        return {
            "type": "radius_search_results",
            "center_point": center_point,
            "radius_km": radius_km,
            "points_found": len(points_in_radius),
            "points_searched": len(points),
            "results": points_in_radius
        }
        
    except Exception as e:
        return {"error": f"Radius search failed: {str(e)}"}

@mcp.tool()
def CreateBoundingBox(
    points: List[Dict[str, Any]],
    lat_field: str = "latitude",
    lon_field: str = "longitude",
    padding_percent: float = 5.0
) -> Dict[str, Any]:
    """
    Create a bounding box that contains all given points.
    
    Args:
        points: List of point dictionaries with location data
        lat_field: Name of latitude field in points data  
        lon_field: Name of longitude field in points data
        padding_percent: Percentage padding around the bounding box
    
    Returns:
        Bounding box coordinates and metadata
    """
    try:
        if not points:
            return {"error": "Points list cannot be empty"}
        
        valid_points = []
        for point in points:
            if lat_field in point and lon_field in point:
                try:
                    lat = float(point[lat_field])
                    lon = float(point[lon_field])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        valid_points.append((lat, lon))
                except (ValueError, TypeError):
                    continue
        
        if not valid_points:
            return {"error": "No valid points found"}
        
        lats = [p[0] for p in valid_points]
        lons = [p[1] for p in valid_points]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Add padding
        lat_padding = (max_lat - min_lat) * (padding_percent / 100)
        lon_padding = (max_lon - min_lon) * (padding_percent / 100)
        
        bounding_box = {
            "north": max_lat + lat_padding,
            "south": min_lat - lat_padding,
            "east": max_lon + lon_padding,
            "west": min_lon - lon_padding,
            "center": {
                "latitude": (min_lat + max_lat) / 2,
                "longitude": (min_lon + max_lon) / 2
            }
        }
        
        return {
            "type": "bounding_box",
            "bounds": bounding_box,
            "points_processed": len(valid_points),
            "points_total": len(points),
            "padding_percent": padding_percent
        }
        
    except Exception as e:
        return {"error": f"Bounding box creation failed: {str(e)}"}

@mcp.tool()
def GetGeographicCapabilities() -> Dict[str, Any]:
    """
    Get information about available geographic capabilities.
    
    Returns:
        Dictionary describing available geographic tools and their parameters
    """
    return {
        "available_tools": {
            "mapping": {
                "CreatePointMap": "Create maps from point location data",
                "CreateChoroplethMap": "Create regional data visualizations"
            },
            "geocoding": {
                "GeocodeLocations": "Convert addresses to coordinates",
                "supported_providers": ["nominatim"]
            },
            "spatial_analysis": {
                "CalculateDistance": "Calculate distances between points",
                "FindPointsInRadius": "Find points within specified radius",
                "CreateBoundingBox": "Create bounding boxes for point sets"
            }
        },
        "supported_formats": {
            "input": ["geojson", "csv_with_coordinates", "address_strings"],
            "output": ["geojson", "map_data", "analysis_results"]
        },
        "coordinate_systems": ["WGS84 (latitude/longitude)"],
        "distance_units": ["km", "miles", "meters"],
        "limitations": [
            "Choropleth maps require external boundary data",
            "Free geocoding has rate limits",
            "Advanced spatial operations may require additional tools"
        ],
        "server_info": metadata
    }

if __name__ == "__main__":
    print(f"🗺️ {metadata['Name']} v{metadata['Version']}")
    print(f"🌍 Universal mapping and geographic analysis tools")
    print("Available tools:")
    print("  - CreatePointMap: Point-based maps")
    print("  - CreateChoroplethMap: Regional data visualization")
    print("  - GeocodeLocations: Address to coordinates")
    print("  - CalculateDistance: Distance between points")
    print("  - FindPointsInRadius: Spatial proximity search")
    print("  - CreateBoundingBox: Bounding box generation")
    print("  - GetGeographicCapabilities: Server capabilities")
    print("\\nReady for MCP connections!")