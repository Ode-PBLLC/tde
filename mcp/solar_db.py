"""
Database utility functions for solar facilities data.
Provides fast, indexed access to solar facility information.
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
import math
from contextlib import contextmanager

class SolarDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default path relative to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            db_path = os.path.join(project_root, "data", "solar_facilities.db")
        
        self.db_path = db_path
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Solar database not found at {self.db_path}. Run migration script first.")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def get_facilities_by_country(self, country: str, min_capacity: float = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get solar facilities for a specific country."""
        # Handle common country name variants
        country_variants = {
            'usa': 'United States of America',
            'united states': 'United States of America',
            'us': 'United States of America',
            'america': 'United States of America',
            'uk': 'United Kingdom',
            'britain': 'United Kingdom',
            'england': 'United Kingdom'
        }
        
        normalized_country = country_variants.get(country.lower(), country)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT cluster_id, capacity_mw, constructed_before, constructed_after, latitude, longitude, country
            FROM solar_facilities 
            WHERE country = ? COLLATE NOCASE
            AND (capacity_mw IS NULL OR capacity_mw >= ?)
            LIMIT ?
            """
            
            cursor.execute(query, (normalized_country, min_capacity, limit))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_facilities_in_radius(self, latitude: float, longitude: float, radius_km: float = 50, 
                                country: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find facilities within radius of coordinates using Haversine formula."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Approximate bounding box for faster initial filtering
            lat_delta = radius_km / 111.0  # Rough km per degree latitude
            lon_delta = radius_km / (111.0 * math.cos(math.radians(latitude)))
            
            lat_min, lat_max = latitude - lat_delta, latitude + lat_delta
            lon_min, lon_max = longitude - lon_delta, longitude + lon_delta
            
            # SQL query with Haversine distance calculation
            # Use subquery for HAVING clause to work properly
            query = """
            SELECT * FROM (
                SELECT cluster_id, capacity_mw, constructed_before, constructed_after, latitude, longitude, country,
                       (6371 * acos(cos(radians(?)) * cos(radians(latitude)) * 
                       cos(radians(longitude) - radians(?)) + sin(radians(?)) * 
                       sin(radians(latitude)))) AS distance_km
                FROM solar_facilities 
                WHERE latitude BETWEEN ? AND ? 
                AND longitude BETWEEN ? AND ?
            """
            
            params = [latitude, longitude, latitude, lat_min, lat_max, lon_min, lon_max]
            
            if country:
                query += " AND country = ? COLLATE NOCASE"
                params.append(country)
            
            query += ") WHERE distance_km <= ? ORDER BY distance_km LIMIT ?"
            params.extend([radius_km, limit])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_country_statistics(self) -> List[Dict[str, Any]]:
        """Get facility counts and statistics by country."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT country, 
                   COUNT(*) as facility_count,
                   COUNT(capacity_mw) as facilities_with_capacity,
                   SUM(capacity_mw) as total_capacity_mw,
                   AVG(capacity_mw) as avg_capacity_mw,
                   MIN(capacity_mw) as min_capacity_mw,
                   MAX(capacity_mw) as max_capacity_mw,
                   MIN(latitude) as min_lat, MAX(latitude) as max_lat,
                   MIN(longitude) as min_lon, MAX(longitude) as max_lon
            FROM solar_facilities 
            GROUP BY country 
            ORDER BY total_capacity_mw DESC NULLS LAST
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_facilities_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get facilities by data source."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT cluster_id, source_id, source, source_date, latitude, longitude, country
            FROM solar_facilities 
            WHERE source = ? COLLATE NOCASE
            ORDER BY source_date DESC
            LIMIT ?
            """
            
            cursor.execute(query, (source, limit))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_facility_by_id(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Get specific facility by cluster ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT cluster_id, capacity_mw, constructed_before, constructed_after, latitude, longitude, country
            FROM solar_facilities 
            WHERE cluster_id = ?
            """
            
            cursor.execute(query, (cluster_id,))
            row = cursor.fetchone()
            
            return dict(row) if row else None
    
    def search_facilities(self, country: Optional[str] = None,
                         lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                         lon_min: Optional[float] = None, lon_max: Optional[float] = None,
                         limit: Optional[int] = 100) -> List[Dict[str, Any]]:
        """Search facilities with multiple filter criteria."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT cluster_id, capacity_mw, constructed_before, constructed_after, latitude, longitude, country FROM solar_facilities WHERE 1=1"
            params = []
            
            if country:
                query += " AND country = ? COLLATE NOCASE"
                params.append(country)
            
            if lat_min is not None:
                query += " AND latitude >= ?"
                params.append(lat_min)
            
            if lat_max is not None:
                query += " AND latitude <= ?"
                params.append(lat_max)
            
            if lon_min is not None:
                query += " AND longitude >= ?"
                params.append(lon_min)
            
            if lon_max is not None:
                query += " AND longitude <= ?"
                params.append(lon_max)
            
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_largest_facilities(self, limit: int = 100, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get largest facilities by capacity."""
        # Handle common country name variants
        if country:
            country_variants = {
                'usa': 'United States of America',
                'united states': 'United States of America',
                'us': 'United States of America',
                'america': 'United States of America',
                'uk': 'United Kingdom',
                'britain': 'United Kingdom',
                'england': 'United Kingdom'
            }
            country = country_variants.get(country.lower(), country)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT cluster_id, capacity_mw, constructed_before, constructed_after, latitude, longitude, country
            FROM solar_facilities 
            WHERE capacity_mw IS NOT NULL
            """
            params = []
            
            if country:
                query += " AND country = ? COLLATE NOCASE"
                params.append(country)
            
            query += " ORDER BY capacity_mw DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def search_by_capacity(self, min_capacity: float, max_capacity: float, 
                          country: Optional[str] = None, limit: int = 100, include_null: bool = False) -> List[Dict[str, Any]]:
        """Search facilities by capacity range."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if include_null:
                query = """
                SELECT cluster_id, capacity_mw, constructed_before, constructed_after, latitude, longitude, country
                FROM solar_facilities 
                WHERE (capacity_mw BETWEEN ? AND ? OR capacity_mw IS NULL)
                """
            else:
                query = """
                SELECT cluster_id, capacity_mw, constructed_before, constructed_after, latitude, longitude, country
                FROM solar_facilities 
                WHERE capacity_mw BETWEEN ? AND ?
                """
            
            params = [min_capacity, max_capacity]
            
            if country:
                query += " AND country = ? COLLATE NOCASE"
                params.append(country)
            
            query += " ORDER BY capacity_mw DESC NULLS LAST LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_total_count(self) -> int:
        """Get total number of facilities in database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM solar_facilities")
            return cursor.fetchone()[0]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT key, value FROM dataset_metadata")
            metadata_rows = cursor.fetchall()
            metadata = {row['key']: row['value'] for row in metadata_rows}
            
            # Add real-time stats
            cursor.execute("SELECT COUNT(*) FROM solar_facilities")
            total_facilities = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT country) FROM solar_facilities")
            total_countries = cursor.fetchone()[0]
            
            return {
                **metadata,
                'total_facilities': total_facilities,
                'total_countries': total_countries,
                'database_path': self.db_path
            }