import pandas as pd
import numpy as np
from fastmcp import FastMCP
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import folium
import base64
from io import BytesIO
import os
from solar_db import SolarDatabase

mcp = FastMCP("solar-facilities-server")

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize database connection
try:
    db = SolarDatabase()
    print(f"Connected to solar facilities database with {db.get_total_count():,} facilities")
    metadata = {
        "Name": "Solar Facilities Server", 
        "Description": "Global solar facility data from TransitionZero Solar Asset Mapper (TZ-SAM) Q1 2025",
        "Version": "2.0.0 - Database Edition",
        "Author": "Climate Policy Radar Team",
        "Dataset": "TZ-SAM Q1 2025 - Complete Global Dataset (SQLite)",
        **db.get_metadata()
    }
except Exception as e:
    print(f"Error: Could not connect to solar facilities database: {e}")
    db = None
    metadata = {"error": "Database not available"}

@mcp.tool()
def GetSolarFacilitiesByCountry(country: str, min_capacity_mw: float = 0, limit: int = 100) -> Dict[str, Any]:
    """Get solar facilities summary for a country."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facilities = db.get_facilities_by_country(country, limit=limit)
        
        if not facilities:
            return {"error": f"No facilities found for {country}"}
        
        # Return summary
        summary = {
            "country": country,
            "total_facilities": len(facilities),
            "sample_facilities": facilities[:3],  # First 3 as examples
            "coordinate_range": {
                "lat_min": min(f['latitude'] for f in facilities),
                "lat_max": max(f['latitude'] for f in facilities),
                "lon_min": min(f['longitude'] for f in facilities),
                "lon_max": max(f['longitude'] for f in facilities)
            },
            "data_available": True
        }
        
        return summary
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilitiesMapData(country: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Get facilities data for map visualization."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        if country:
            facilities = db.get_facilities_by_country(country, limit=limit)
        else:
            facilities = db.search_facilities(limit=limit)
        
        if not facilities:
            return {"error": f"No facilities found{' for ' + country if country else ''}"}
        
        # Get country statistics for summary
        country_stats = db.get_country_statistics()
        total_facilities = sum(stat['facility_count'] for stat in country_stats)
        countries = [stat['country'] for stat in country_stats]
        
        # Convert to map-ready format
        full_data = []
        for facility in facilities:
            full_data.append({
                "cluster_id": facility.get('cluster_id', ''),
                "latitude": float(facility['latitude']),
                "longitude": float(facility['longitude']),
                "country": facility['country'],
                "source": facility.get('source', 'Unknown'),
                "capacity_mw": facility.get('capacity_mw'),
                "constructed_before": facility.get('constructed_before'),
                "constructed_after": facility.get('constructed_after'),
                "name": f"Solar Facility {facility.get('cluster_id', '')}"
            })
        
        return {
            "type": "map_data_summary",
            "summary": {
                "total_facilities": total_facilities,
                "countries": countries[:10],  # Top 10 countries
                "facilities_shown_on_map": len(facilities),
                "data_from_database": True
            },
            "sample_facilities": facilities[:3],  # First 3 as examples
            "full_data": full_data,
            "metadata": {
                "data_source": "TZ-SAM Q1 2025 (SQLite Database)",
                "geojson_available": True,
                "query_time_optimized": True
            }
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilitiesForGeoJSON(country: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Get facilities for GeoJSON generation."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        if country:
            facilities = db.get_facilities_by_country(country, limit=limit)
        else:
            facilities = db.search_facilities(limit=limit)
        
        if not facilities:
            return {"error": f"No facilities found{' for ' + country if country else ''}"}
        
        # Convert to GeoJSON-ready format
        facilities_list = []
        countries = set()
        
        for facility in facilities:
            facilities_list.append({
                "cluster_id": facility.get('cluster_id', ''),
                "latitude": float(facility['latitude']),
                "longitude": float(facility['longitude']),
                "country": facility['country'],
                "source": facility.get('source', 'Unknown'),
                "capacity_mw": facility.get('capacity_mw'),
                "constructed_before": facility.get('constructed_before'),
                "constructed_after": facility.get('constructed_after'),
                "name": f"Solar Facility {facility.get('cluster_id', '')}"
            })
            countries.add(facility['country'])
        
        return {
            "type": "map",
            "data": facilities_list,
            "metadata": {
                "total_facilities": len(facilities_list),
                "countries": list(countries),
                "data_source": "TZ-SAM Q1 2025 Database",
                "note": "Combined raw and analysis dataset with capacity where available"
            }
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarCapacityByCountry() -> Dict[str, Any]:
    """Get solar facility counts by country."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        country_stats = db.get_country_statistics()
        
        if not country_stats:
            return {"error": "No country statistics available"}
        
        # Sort by facility count
        country_stats.sort(key=lambda x: x['facility_count'], reverse=True)
        
        total_facilities = sum(stat['facility_count'] for stat in country_stats)
        top_country = country_stats[0]
        
        return {
            "countries_analyzed": [stat['country'] for stat in country_stats],
            "total_global_facilities": total_facilities,
            "total_countries": len(country_stats),
            "country_with_most_facilities": top_country['country'],
            "top_10_countries": country_stats[:10],
            "data_available": True,
            "query_optimized": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetDataForDirectVisualization(request_type: str, country: Optional[str] = None) -> str:
    """Get data ID for direct visualization."""
    if not db:
        return "NO_DATA_AVAILABLE"
    
    # Generate a simple identifier that the client can interpret
    if request_type == "country_comparison":
        return "SOLAR_COUNTRY_STATS"
    elif request_type == "facilities_map":
        country_filter = country or 'all'
        return f"SOLAR_MAP_{country_filter.upper()}"
    elif request_type == "capacity_distribution":
        return "SOLAR_CAPACITY_DIST"
    elif request_type == "timeline":
        return "SOLAR_TIMELINE"
    else:
        return f"SOLAR_DATA_{request_type.upper()}"

@mcp.tool()
def GetSolarFacilitiesInRadius(latitude: float, longitude: float, radius_km: float = 50, country: Optional[str] = None) -> Dict[str, Any]:
    """Find facilities within radius of coordinates."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facilities = db.get_facilities_in_radius(
            latitude, longitude, radius_km, country, limit=100
        )
        
        if not facilities:
            return {"error": f"No facilities found within {radius_km}km of ({latitude}, {longitude})"}
        
        return {
            "search_center": [latitude, longitude],
            "radius_km": radius_km,
            "facilities_found": len(facilities),
            "closest_facilities": facilities[:5],  # Top 5 closest
            "data_available": True,
            "query_optimized": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarConstructionTimeline(start_year: int = 2017, end_year: int = 2025, country: Optional[str] = None) -> Dict[str, Any]:
    """Get construction timeline data based on source dates."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        # Get facilities with source date filtering
        if country:
            facilities = db.get_facilities_by_country(country, limit=10000)
        else:
            facilities = db.search_facilities(limit=10000)
        
        # Filter by source date years
        timeline_data = []
        for facility in facilities:
            if facility.get('source_date'):
                try:
                    source_date = pd.to_datetime(facility['source_date'])
                    year = source_date.year
                    if start_year <= year <= end_year:
                        timeline_data.append({
                            **facility,
                            'source_year': year
                        })
                except:
                    continue
        
        if not timeline_data:
            return {"error": f"No facilities with source dates between {start_year}-{end_year}"}
        
        # Group by year
        year_counts = {}
        for facility in timeline_data:
            year = facility['source_year']
            year_counts[year] = year_counts.get(year, 0) + 1
        
        return {
            "period": f"{start_year}-{end_year}",
            "country_filter": country,
            "total_facilities_in_period": len(timeline_data),
            "years_with_data": sorted(year_counts.keys()),
            "yearly_counts": year_counts,
            "data_available": True,
            "note": "Based on source_date field in database"
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetLargestSolarFacilities(limit: int = 20, country: Optional[str] = None) -> Dict[str, Any]:
    """Get largest solar facilities by capacity."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facilities = db.get_largest_facilities(limit=limit, country=country)
        
        if not facilities:
            return {"error": f"No facilities found{' in ' + country if country else ''}"}
        
        # Convert to list format with capacity data
        facilities_list = []
        for facility in facilities:
            facilities_list.append({
                "cluster_id": facility.get('cluster_id', ''),
                "capacity_mw": facility.get('capacity_mw'),
                "constructed_before": facility.get('constructed_before'),
                "constructed_after": facility.get('constructed_after'),
                "latitude": float(facility['latitude']),
                "longitude": float(facility['longitude']),
                "country": facility['country'],
                "name": f"Solar Facility {facility.get('cluster_id', '')}"
            })
        
        return {
            "search_criteria": {"limit": limit, "country": country},
            "facilities_found": len(facilities),
            "top_3_facilities": facilities_list[:3],
            "all_facilities": facilities_list,
            "note": "Sorted by capacity (MW) - combined dataset with capacity where available",
            "data_available": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilityDetails(cluster_id: str) -> Dict[str, Any]:
    """Get details for specific facility."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facility = db.get_facility_by_id(cluster_id)
        
        if not facility:
            return {"error": f"Facility with cluster_id '{cluster_id}' not found"}
        
        # Add derived information if source_date is available
        if facility.get('source_date'):
            try:
                source_date = pd.to_datetime(facility['source_date'])
                facility['source_year'] = source_date.year
                facility['source_date_formatted'] = source_date.strftime('%Y-%m-%d')
            except Exception as e:
                facility['date_parse_error'] = str(e)
        
        return facility
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def SearchSolarFacilitiesByCapacity(min_capacity_mw: float, max_capacity_mw: float, country: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """Search facilities by capacity range (capacity data not available in current schema)."""
    if not db:
        return {"error": "Database not available"}
    
    return {
        "error": "Capacity search not available - current database schema doesn't include capacity_mw field",
        "available_fields": ["cluster_id", "source_id", "source", "source_date", "latitude", "longitude", "country"],
        "suggestion": "Use GetSolarFacilitiesByCountry or GetSolarFacilitiesInRadius instead",
        "note": "Original TZ-SAM dataset doesn't include capacity information in the raw polygons file"
    }

# CreateSolarFacilitiesMap tool removed - was causing context explosion
# Maps are now generated purely client-side

@mcp.tool()
def GetSolarCapacityVisualizationData(visualization_type: str = "by_country") -> Dict[str, Any]:
    """Get data for visualization. Types: by_country, source_timeline."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        if visualization_type == "by_country":
            # Country-level statistics
            country_stats = db.get_country_statistics()
            
            return {
                "visualization_type": "by_country",
                "data": country_stats[:20],  # Top 20 countries
                "chart_config": {
                    "x_axis": "country",
                    "y_axis": "facility_count", 
                    "title": "Solar Facilities by Country",
                    "chart_type": "bar"
                }
            }
        
        elif visualization_type == "source_timeline":
            # Timeline based on source dates - use unlimited data for accurate charts
            facilities = db.search_facilities(limit=None)  # No limit for chart accuracy
            
            # Group by source and year
            timeline_data = {}
            for facility in facilities:
                if facility.get('source_date'):
                    try:
                        source_date = pd.to_datetime(facility['source_date'])
                        year = source_date.year
                        source = facility.get('source', 'Unknown')
                        
                        key = f"{source}_{year}"
                        if key not in timeline_data:
                            timeline_data[key] = {
                                "source": source,
                                "year": year,
                                "facility_count": 0
                            }
                        timeline_data[key]["facility_count"] += 1
                    except:
                        continue
            
            data = list(timeline_data.values())
            
            return {
                "visualization_type": "source_timeline",
                "data": data,
                "chart_config": {
                    "x_axis": "year",
                    "y_axis": "facility_count",
                    "color": "source",
                    "title": "Solar Facilities by Source and Year",
                    "chart_type": "line"
                }
            }
        
        else:
            return {"error": f"Unknown visualization type: {visualization_type}. Available: by_country, source_timeline"}
            
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarDatasetMetadata() -> Dict[str, Any]:
    """Get dataset metadata."""
    if not db:
        return {"error": "Database not available"}
    
    return metadata

if __name__ == "__main__":
    mcp.run()