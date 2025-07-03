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

mcp = FastMCP("solar-facilities-server")

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load filtered solar facilities data
try:
    facilities_df = pd.read_csv(os.path.join(script_dir, "solar_facilities_demo.csv"))
    print(f"Loaded {len(facilities_df):,} solar facilities from demo dataset")
except FileNotFoundError:
    print("Error: solar_facilities_demo.csv not found. Please run data filtering first.")
    facilities_df = pd.DataFrame()

metadata = {
    "Name": "Solar Facilities Server", 
    "Description": "Global solar facility data from TransitionZero Solar Asset Mapper (TZ-SAM) Q1 2025",
    "Version": "1.0.0",
    "Author": "Climate Policy Radar Team",
    "Dataset": "TZ-SAM Q1 2025 - Demo Countries (Brazil, India, South Africa, Vietnam)",
    "Total_Facilities": len(facilities_df) if not facilities_df.empty else 0
}

@mcp.tool()
def GetSolarFacilitiesByCountry(country: str, min_capacity_mw: float = 0, limit: int = 100) -> Dict[str, Any]:
    """
    Get solar facilities summary for a specific country with optional capacity filtering.
    Returns summary statistics instead of full facility list to avoid context explosion.
    
    Parameters:
    - country: Country name (Brazil, India, South Africa, Vietnam)
    - min_capacity_mw: Minimum capacity in MW (default: 0)
    - limit: Maximum number of facilities to return (default: 100)
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    filtered = facilities_df[
        (facilities_df['country'].str.lower() == country.lower()) & 
        (facilities_df['capacity_mw'] >= min_capacity_mw)
    ].head(limit)
    
    if filtered.empty:
        return {"error": f"No facilities found for {country} with min capacity {min_capacity_mw} MW"}
    
    # Return summary instead of full data
    summary = {
        "country": country,
        "total_facilities": len(filtered),
        "total_capacity_mw": round(filtered['capacity_mw'].sum(), 2),
        "avg_capacity_mw": round(filtered['capacity_mw'].mean(), 2),
        "capacity_range": [round(filtered['capacity_mw'].min(), 2), round(filtered['capacity_mw'].max(), 2)],
        "sample_facilities": filtered.head(3)[['cluster_id', 'capacity_mw', 'latitude', 'longitude']].to_dict('records'),
        "data_available": True  # Flag that full data can be accessed for visualization
    }
    
    return summary

@mcp.tool()
def GetSolarFacilitiesMapData(country: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
    """
    Get solar facilities data specifically for map visualization with GeoJSON format.
    Returns summary statistics for LLM and generates GeoJSON file for frontend.
    
    Args:
        country: Filter by specific country (optional)
        limit: Maximum number of facilities to return for map display (default 1000)
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    # Filter by country if specified
    if country:
        filtered = facilities_df[facilities_df['country'].str.lower() == country.lower()]
        if filtered.empty:
            return {"error": f"No facilities found for {country}"}
    else:
        filtered = facilities_df.copy()
    
    # Get summary statistics for all facilities (not limited)
    total_facilities = len(filtered)
    total_capacity = filtered['capacity_mw'].sum()
    countries = list(filtered['country'].unique())
    
    # Sort by capacity (largest first) and limit for map display
    filtered_for_map = filtered.sort_values('capacity_mw', ascending=False)
    if limit is not None:
        filtered_for_map = filtered_for_map.head(limit)
    
    # Convert filtered_for_map to list of facility records for GeoJSON generation
    full_data = []
    for _, facility in filtered_for_map.iterrows():
        full_data.append({
            "cluster_id": facility.get('cluster_id', ''),
            "capacity_mw": float(facility['capacity_mw']),
            "latitude": float(facility['latitude']),
            "longitude": float(facility['longitude']),
            "country": facility['country'],
            "completion_year": facility.get('completion_year', 'Unknown'),
            "name": facility.get('name', f"Solar Facility {facility.get('cluster_id', '')}")
        })
    
    # Return summary data (lightweight for LLM) + full data for GeoJSON generation
    return {
        "type": "map_data_summary",
        "summary": {
            "total_facilities": total_facilities,
            "total_capacity_mw": float(total_capacity),
            "countries": countries,
            "facilities_shown_on_map": len(filtered_for_map),
            "largest_facilities_shown": True if limit else False,
            "capacity_range": {
                "min": float(filtered['capacity_mw'].min()),
                "max": float(filtered['capacity_mw'].max()),
                "avg": float(filtered['capacity_mw'].mean())
            }
        },
        "full_data": full_data,  # Same exact facilities used for summary stats
        "metadata": {
            "data_source": "TZ-SAM Q1 2025",
            "geojson_available": True,
            "truncated_for_display": total_facilities > (limit or 0)
        }
    }

@mcp.tool()
def GetSolarFacilitiesForGeoJSON(country: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
    """
    Get solar facilities data for GeoJSON generation (internal use).
    Returns actual facility records with coordinates for mapping.
    
    Args:
        country: Filter by specific country (optional)
        limit: Maximum number of facilities to return (default 1000)
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    # Filter by country if specified
    if country:
        filtered = facilities_df[facilities_df['country'].str.lower() == country.lower()]
        if filtered.empty:
            return {"error": f"No facilities found for {country}"}
    else:
        filtered = facilities_df.copy()
    
    # Sort by capacity (largest first) and limit results
    filtered = filtered.sort_values('capacity_mw', ascending=False)
    if limit is not None:
        filtered = filtered.head(limit)
    
    # Convert to list of facility records
    facilities_list = []
    for _, facility in filtered.iterrows():
        facilities_list.append({
            "cluster_id": facility.get('cluster_id', ''),
            "capacity_mw": float(facility['capacity_mw']),
            "latitude": float(facility['latitude']),
            "longitude": float(facility['longitude']),
            "country": facility['country'],
            "completion_year": facility.get('completion_year', 'Unknown'),
            "name": facility.get('name', f"Solar Facility {facility.get('cluster_id', '')}")
        })
    
    return {
        "type": "map",
        "data": facilities_list,
        "metadata": {
            "total_facilities": len(facilities_list),
            "total_capacity": float(filtered['capacity_mw'].sum()),
            "countries": filtered['country'].unique().tolist(),
            "capacity_range": [float(filtered['capacity_mw'].min()), float(filtered['capacity_mw'].max())]
        }
    }

@mcp.tool()
def GetSolarCapacityByCountry() -> Dict[str, Any]:
    """
    Get total solar capacity and facility count by country.
    Returns summary statistics instead of detailed records.
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    stats = facilities_df.groupby('country').agg({
        'capacity_mw': ['sum', 'mean', 'count', 'min', 'max'],
        'cluster_id': 'count'
    }).round(2)
    
    stats.columns = ['total_capacity_mw', 'avg_capacity_mw', 'facility_count', 'min_capacity_mw', 'max_capacity_mw', 'total_facilities']
    stats = stats.reset_index()
    
    # Return summary of the summary
    return {
        "countries_analyzed": stats['country'].tolist(),
        "total_global_capacity_mw": round(stats['total_capacity_mw'].sum(), 2),
        "total_global_facilities": int(stats['facility_count'].sum()),
        "country_with_most_capacity": stats.loc[stats['total_capacity_mw'].idxmax(), 'country'],
        "country_with_most_facilities": stats.loc[stats['facility_count'].idxmax(), 'country'],
        "data_available": True  # Flag for direct access
    }

@mcp.tool()
def GetDataForDirectVisualization(request_type: str, country: Optional[str] = None) -> str:
    """
    Get data directly for visualization without going through Claude context.
    Returns a unique identifier that the client can use to fetch full data.
    
    Parameters:
    - request_type: Type of data needed ("country_comparison", "facilities_map", "capacity_distribution", etc.)
    - country: Optional country filter for certain request types
    """
    if facilities_df.empty:
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
    """
    Find solar facilities within a radius of given coordinates.
    
    Parameters:
    - latitude: Center latitude
    - longitude: Center longitude  
    - radius_km: Search radius in kilometers (default: 50)
    - country: Optional country filter
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    # Simple distance calculation (approximate for small distances)
    df_copy = facilities_df.copy()
    
    # Convert to radians
    lat1, lon1 = np.radians(latitude), np.radians(longitude)
    lat2, lon2 = np.radians(df_copy['latitude']), np.radians(df_copy['longitude'])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))  # 6371 km = Earth's radius
    
    # Filter by distance
    in_radius = df_copy[distance_km <= radius_km].copy()
    
    if in_radius.empty:
        return {"error": f"No facilities found within {radius_km}km of ({latitude}, {longitude})"}
    
    in_radius['distance_km'] = distance_km[distance_km <= radius_km].round(2)
    
    # Optional country filter
    if country:
        in_radius = in_radius[in_radius['country'].str.lower() == country.lower()]
        if in_radius.empty:
            return {"error": f"No facilities found in {country} within {radius_km}km"}
    
    # Sort by distance and return summary
    in_radius = in_radius.sort_values('distance_km')
    
    return {
        "search_center": [latitude, longitude],
        "radius_km": radius_km,
        "facilities_found": len(in_radius),
        "closest_facilities": in_radius.head(5)[['cluster_id', 'capacity_mw', 'distance_km', 'country']].to_dict('records'),
        "total_capacity_mw": round(in_radius['capacity_mw'].sum(), 2),
        "data_available": True
    }

@mcp.tool()
def GetSolarConstructionTimeline(start_year: int = 2017, end_year: int = 2025, country: Optional[str] = None) -> Dict[str, Any]:
    """
    Get solar facility construction timeline data.
    
    Parameters:
    - start_year: Start year for timeline (default: 2017)
    - end_year: End year for timeline (default: 2025)
    - country: Optional country filter
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    df_copy = facilities_df.copy()
    
    # Filter by country if specified
    if country:
        df_copy = df_copy[df_copy['country'].str.lower() == country.lower()]
    
    # Convert timestamps to years (handle NaN values)
    df_copy = df_copy.dropna(subset=['constructed_before', 'constructed_after'])
    
    if df_copy.empty:
        return {"error": "No timeline data available for the specified criteria"}
    
    # Extract years from timestamps (fixed: remove unit='s' for ISO 8601 strings)
    df_copy['completion_year'] = pd.to_datetime(df_copy['constructed_before'], errors='coerce').dt.year
    df_copy['start_year'] = pd.to_datetime(df_copy['constructed_after'], errors='coerce').dt.year
    
    # Filter by year range
    df_copy = df_copy[
        (df_copy['completion_year'] >= start_year) & 
        (df_copy['completion_year'] <= end_year)
    ]
    
    if df_copy.empty:
        return {"error": f"No facilities completed between {start_year}-{end_year}"}
    
    # Return summary statistics instead of full timeline
    return {
        "period": f"{start_year}-{end_year}",
        "country_filter": country,
        "total_facilities_completed": len(df_copy),
        "total_capacity_added_mw": round(df_copy['capacity_mw'].sum(), 2),
        "peak_year": int(df_copy.groupby('completion_year')['capacity_mw'].sum().idxmax()),
        "peak_year_capacity_mw": round(df_copy.groupby('completion_year')['capacity_mw'].sum().max(), 2),
        "years_with_data": sorted(df_copy['completion_year'].unique().tolist()),
        "data_available": True
    }

@mcp.tool()
def GetLargestSolarFacilities(limit: int = 20, country: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the largest solar facilities by capacity.
    
    Parameters:
    - limit: Number of facilities to return (default: 20)
    - country: Optional country filter
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    df_copy = facilities_df.copy()
    
    # Filter by country if specified
    if country:
        df_copy = df_copy[df_copy['country'].str.lower() == country.lower()]
        
    if df_copy.empty:
        return {"error": f"No facilities found in {country}" if country else "No facilities found"}
    
    # Sort by capacity and get top facilities
    largest = df_copy.nlargest(limit, 'capacity_mw')
    
    return {
        "search_criteria": {"limit": limit, "country": country},
        "facilities_found": len(largest),
        "largest_facility_mw": round(largest.iloc[0]['capacity_mw'], 2) if not largest.empty else 0,
        "smallest_in_top_mw": round(largest.iloc[-1]['capacity_mw'], 2) if not largest.empty else 0,
        "total_capacity_top_facilities_mw": round(largest['capacity_mw'].sum(), 2),
        "top_3_facilities": largest.head(3)[['cluster_id', 'capacity_mw', 'country', 'latitude', 'longitude']].to_dict('records'),
        "data_available": True
    }

@mcp.tool()
def GetSolarFacilityDetails(cluster_id: str) -> Dict[str, Any]:
    """
    Get detailed information for a specific solar facility.
    
    Parameters:
    - cluster_id: Unique facility identifier
    """
    if facilities_df.empty:
        return {}
    
    facility = facilities_df[facilities_df['cluster_id'] == cluster_id]
    
    if facility.empty:
        return {"error": f"Facility with cluster_id '{cluster_id}' not found"}
    
    facility_dict = facility.iloc[0].to_dict()
    
    # Add derived information (fixed: parse ISO 8601 strings instead of Unix timestamps)
    if not pd.isna(facility_dict.get('constructed_before')) and not pd.isna(facility_dict.get('constructed_after')):
        try:
            start_date = pd.to_datetime(facility_dict['constructed_after'])
            end_date = pd.to_datetime(facility_dict['constructed_before'])
            facility_dict['construction_start'] = start_date.strftime('%Y-%m-%d')
            facility_dict['construction_end'] = end_date.strftime('%Y-%m-%d')
            facility_dict['construction_duration_days'] = (end_date - start_date).days
        except Exception as e:
            print(f"Warning: Could not parse construction dates: {e}")
    
    return facility_dict

@mcp.tool()
def SearchSolarFacilitiesByCapacity(min_capacity_mw: float, max_capacity_mw: float, country: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Search solar facilities within a capacity range.
    
    Parameters:
    - min_capacity_mw: Minimum capacity in MW
    - max_capacity_mw: Maximum capacity in MW
    - country: Optional country filter
    - limit: Maximum number of facilities to return (default: 50)
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    df_copy = facilities_df.copy()
    
    # Filter by capacity range
    filtered = df_copy[
        (df_copy['capacity_mw'] >= min_capacity_mw) & 
        (df_copy['capacity_mw'] <= max_capacity_mw)
    ]
    
    # Filter by country if specified
    if country:
        filtered = filtered[filtered['country'].str.lower() == country.lower()]
    
    if filtered.empty:
        return {"error": f"No facilities found in range {min_capacity_mw}-{max_capacity_mw} MW" + (f" in {country}" if country else "")}
    
    # Sort by capacity (largest first) and limit results
    result = filtered.nlargest(limit, 'capacity_mw')
    
    return {
        "search_criteria": {
            "capacity_range_mw": [min_capacity_mw, max_capacity_mw],
            "country": country,
            "limit": limit
        },
        "facilities_found": len(filtered),
        "facilities_returned": len(result),
        "total_capacity_in_range_mw": round(filtered['capacity_mw'].sum(), 2),
        "avg_capacity_in_range_mw": round(filtered['capacity_mw'].mean(), 2),
        "sample_facilities": result.head(3)[['cluster_id', 'capacity_mw', 'country']].to_dict('records'),
        "data_available": True
    }

# CreateSolarFacilitiesMap tool removed - was causing context explosion
# Maps are now generated purely client-side

@mcp.tool()
def GetSolarCapacityVisualizationData(visualization_type: str = "by_country") -> Dict[str, Any]:
    """
    Get data formatted for specific visualization types.
    
    Parameters:
    - visualization_type: Type of chart ("by_country", "capacity_distribution", "timeline")
    
    Returns structured data ready for plotting
    """
    if facilities_df.empty:
        return {"error": "No facilities data available"}
    
    if visualization_type == "by_country":
        # Country-level statistics
        country_stats = facilities_df.groupby('country').agg({
            'capacity_mw': ['sum', 'mean', 'count'],
            'cluster_id': 'count'
        }).round(2)
        
        country_stats.columns = ['total_capacity_mw', 'avg_capacity_mw', 'facility_count', 'total_facilities']
        data = country_stats.reset_index().to_dict('records')
        
        return {
            "visualization_type": "by_country",
            "data": data,
            "chart_config": {
                "x_axis": "country",
                "y_axis": "total_capacity_mw", 
                "title": "Solar Capacity by Country",
                "chart_type": "bar"
            }
        }
    
    elif visualization_type == "capacity_distribution":
        # Capacity distribution histogram data
        bins = [0, 1, 5, 10, 25, 50, 100, 500, 3000]
        bin_labels = ['<1MW', '1-5MW', '5-10MW', '10-25MW', '25-50MW', '50-100MW', '100-500MW', '>500MW']
        
        facilities_df['capacity_bin'] = pd.cut(facilities_df['capacity_mw'], bins=bins, labels=bin_labels, include_lowest=True)
        dist_data = facilities_df['capacity_bin'].value_counts().sort_index()
        
        data = [{"capacity_range": str(k), "facility_count": v} for k, v in dist_data.items()]
        
        return {
            "visualization_type": "capacity_distribution", 
            "data": data,
            "chart_config": {
                "x_axis": "capacity_range",
                "y_axis": "facility_count",
                "title": "Solar Facilities by Capacity Range",
                "chart_type": "bar"
            }
        }
    
    elif visualization_type == "timeline":
        # Construction timeline (if available)
        timeline_df = facilities_df.dropna(subset=['constructed_before'])
        if timeline_df.empty:
            return {"error": "No timeline data available"}
        
        timeline_df['completion_year'] = pd.to_datetime(timeline_df['constructed_before'], errors='coerce').dt.year
        timeline_data = timeline_df.groupby(['completion_year', 'country']).agg({
            'capacity_mw': 'sum',
            'cluster_id': 'count'
        }).reset_index()
        
        data = timeline_data.to_dict('records')
        
        return {
            "visualization_type": "timeline",
            "data": data, 
            "chart_config": {
                "x_axis": "completion_year",
                "y_axis": "capacity_mw",
                "color": "country",
                "title": "Solar Capacity Installation Timeline",
                "chart_type": "line"
            }
        }
    
    else:
        return {"error": f"Unknown visualization type: {visualization_type}"}

@mcp.tool()
def GetSolarDatasetMetadata() -> Dict[str, Any]:
    """
    Get metadata about the solar facilities dataset.
    """
    return metadata

if __name__ == "__main__":
    mcp.run()