#!/usr/bin/env python3
"""
Migration script to convert solar facilities CSV to SQLite database.
This will significantly improve query performance from 30+ seconds to under 1 second.
"""

import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime

def create_database_schema(db_path):
    """Create the solar facilities database schema with indexes."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the main table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS solar_facilities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cluster_id TEXT NOT NULL UNIQUE,
        source_id INTEGER,
        source TEXT,
        source_date TEXT,
        capacity_mw REAL NULL,
        constructed_before TEXT NULL,
        constructed_after TEXT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        country TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Create indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cluster_id ON solar_facilities(cluster_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_country ON solar_facilities(country);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_latitude ON solar_facilities(latitude);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_longitude ON solar_facilities(longitude);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON solar_facilities(source);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_capacity ON solar_facilities(capacity_mw);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON solar_facilities(latitude, longitude);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_country_lat_lon ON solar_facilities(country, latitude, longitude);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_country_capacity ON solar_facilities(country, capacity_mw);")
    
    # Create metadata table for tracking dataset info
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset_metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database schema created successfully")

def import_combined_csv_data(raw_csv_path, analysis_csv_path, db_path):
    """Import combined CSV data from both raw and analysis datasets."""
    if not os.path.exists(raw_csv_path):
        print(f"❌ Error: Raw CSV file not found at {raw_csv_path}")
        return False
    
    if not os.path.exists(analysis_csv_path):
        print(f"❌ Error: Analysis CSV file not found at {analysis_csv_path}")
        return False
    
    print(f"📁 Loading combined data from:")
    print(f"   Raw dataset: {raw_csv_path}")
    print(f"   Analysis dataset: {analysis_csv_path}")
    
    raw_size = os.path.getsize(raw_csv_path) / (1024*1024)
    analysis_size = os.path.getsize(analysis_csv_path) / (1024*1024)
    print(f"📊 File sizes: Raw: {raw_size:.1f} MB, Analysis: {analysis_size:.1f} MB")
    
    chunk_size = 10000
    total_rows = 0
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Step 1: Load analysis dataset first to create capacity lookup
        print(f"🔍 Step 1: Loading analysis dataset for capacity data...")
        analysis_df = pd.read_csv(analysis_csv_path)
        
        # Clean analysis data
        analysis_df = analysis_df.dropna(subset=['latitude', 'longitude', 'country'])
        analysis_df['capacity_mw'] = pd.to_numeric(analysis_df['capacity_mw'], errors='coerce')
        
        # Create capacity lookup by cluster_id
        capacity_lookup = {}
        for _, row in analysis_df.iterrows():
            if pd.notna(row.get('cluster_id')) and pd.notna(row.get('capacity_mw')):
                capacity_lookup[row['cluster_id']] = {
                    'capacity_mw': row['capacity_mw'],
                    'constructed_before': row.get('constructed_before'),
                    'constructed_after': row.get('constructed_after')
                }
        
        print(f"✅ Created capacity lookup for {len(capacity_lookup):,} facilities from analysis dataset")
        
        # Step 2: Process raw dataset with capacity enrichment
        print(f"🔍 Step 2: Processing raw dataset with capacity enrichment...")
        
        # Get total row count for progress tracking
        with open(raw_csv_path, 'r') as f:
            total_lines = sum(1 for line in f) - 1  # Subtract header
        
        print(f"📈 Processing {total_lines:,} raw facilities in chunks of {chunk_size:,}")
        
        # Process raw data in chunks
        for chunk_num, chunk in enumerate(pd.read_csv(raw_csv_path, chunksize=chunk_size)):
            # Clean and validate raw data
            chunk = chunk.dropna(subset=['latitude', 'longitude', 'country'])
            
            # Convert to proper data types
            chunk['latitude'] = pd.to_numeric(chunk['latitude'], errors='coerce')
            chunk['longitude'] = pd.to_numeric(chunk['longitude'], errors='coerce')
            
            # Remove rows with invalid coordinates
            chunk = chunk.dropna(subset=['latitude', 'longitude'])
            
            # Enrich with capacity data where available
            enriched_rows = []
            for _, row in chunk.iterrows():
                cluster_id = row.get('cluster_id')
                
                # Check if we have capacity data for this facility
                if cluster_id in capacity_lookup:
                    capacity_data = capacity_lookup[cluster_id]
                    enriched_row = {
                        'cluster_id': cluster_id,
                        'capacity_mw': capacity_data['capacity_mw'],
                        'constructed_before': capacity_data['constructed_before'],
                        'constructed_after': capacity_data['constructed_after'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'country': row['country']
                    }
                else:
                    # No capacity data available - use NULL
                    enriched_row = {
                        'cluster_id': cluster_id,
                        'capacity_mw': None,
                        'constructed_before': None,
                        'constructed_after': None,
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'country': row['country']
                    }
                
                enriched_rows.append(enriched_row)
            
            # Convert to DataFrame and remove duplicates within this chunk
            enriched_df = pd.DataFrame(enriched_rows)
            enriched_df = enriched_df.drop_duplicates(subset=['cluster_id'], keep='first')
            
            # Insert into database using INSERT OR REPLACE to handle duplicates across chunks
            cursor = conn.cursor()
            for _, row in enriched_df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO solar_facilities 
                    (cluster_id, capacity_mw, constructed_before, constructed_after, latitude, longitude, country)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['cluster_id'],
                    row['capacity_mw'],
                    row['constructed_before'],
                    row['constructed_after'],
                    row['latitude'],
                    row['longitude'],
                    row['country']
                ))
            
            total_rows += len(enriched_df)
            progress = (chunk_num + 1) * chunk_size / total_lines * 100
            print(f"⏳ Progress: {progress:.1f}% - Imported {total_rows:,} rows", end='\r')
        
        print(f"\n✅ Successfully imported {total_rows:,} solar facilities")
        
        # Get final statistics
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM solar_facilities WHERE capacity_mw IS NOT NULL")
        facilities_with_capacity = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM solar_facilities WHERE capacity_mw IS NULL")
        facilities_without_capacity = cursor.fetchone()[0]
        
        print(f"📊 Final statistics:")
        print(f"   Total facilities: {total_rows:,}")
        print(f"   With capacity data: {facilities_with_capacity:,}")
        print(f"   Without capacity data: {facilities_without_capacity:,}")
        
        # Update metadata
        cursor.execute("INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)", 
                      ('total_facilities', str(total_rows)))
        cursor.execute("INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)", 
                      ('facilities_with_capacity', str(facilities_with_capacity)))
        cursor.execute("INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)", 
                      ('facilities_without_capacity', str(facilities_without_capacity)))
        cursor.execute("INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)", 
                      ('raw_source_file', raw_csv_path))
        cursor.execute("INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)", 
                      ('analysis_source_file', analysis_csv_path))
        cursor.execute("INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)", 
                      ('migration_date', datetime.now().isoformat()))
        cursor.execute("INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)", 
                      ('dataset_version', 'TZ-SAM Q1 2025 - Combined Raw + Analysis Polygons'))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"❌ Error importing combined data: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        conn.close()

def verify_database(db_path):
    """Verify the database was created correctly."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check total count
        cursor.execute("SELECT COUNT(*) FROM solar_facilities")
        total_count = cursor.fetchone()[0]
        print(f"📊 Total facilities in database: {total_count:,}")
        
        # Check country distribution
        cursor.execute("SELECT country, COUNT(*) as count FROM solar_facilities GROUP BY country ORDER BY count DESC LIMIT 10")
        countries = cursor.fetchall()
        print(f"🌍 Top countries by facility count:")
        for country, count in countries:
            print(f"   {country}: {count:,}")
        
        # Check coordinate ranges
        cursor.execute("SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude) FROM solar_facilities")
        lat_min, lat_max, lon_min, lon_max = cursor.fetchone()
        print(f"🗺️  Coordinate ranges:")
        print(f"   Latitude: {lat_min:.2f} to {lat_max:.2f}")
        print(f"   Longitude: {lon_min:.2f} to {lon_max:.2f}")
        
        # Test a sample query performance
        import time
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM solar_facilities WHERE country = 'China'")
        china_count = cursor.fetchone()[0]
        query_time = time.time() - start_time
        print(f"⚡ Sample query (China facilities): {china_count:,} results in {query_time:.3f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False
    finally:
        conn.close()

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    raw_csv_path = os.path.join(project_root, "data", "tz-sam-q1-2025", "tz-sam-runs_2025-Q1_outputs_external_raw_polygons.csv")
    analysis_csv_path = os.path.join(project_root, "data", "tz-sam-q1-2025", "tz-sam-runs_2025-Q1_outputs_external_analysis_polygons.csv")
    db_path = os.path.join(project_root, "data", "solar_facilities.db")
    
    print("🚀 Starting Solar Facilities Database Migration - Combined Dataset")
    print(f"📂 Raw CSV: {raw_csv_path}")
    print(f"📂 Analysis CSV: {analysis_csv_path}")
    print(f"💾 Target DB: {db_path}")
    
    # Check if CSV files exist
    if not os.path.exists(raw_csv_path):
        print(f"❌ Error: Raw CSV file not found at {raw_csv_path}")
        return
    
    if not os.path.exists(analysis_csv_path):
        print(f"❌ Error: Analysis CSV file not found at {analysis_csv_path}")
        return
    
    # Check if database already exists
    if os.path.exists(db_path):
        response = input(f"Database already exists at {db_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return
        os.remove(db_path)
    
    # Step 1: Create schema
    print("\n1️⃣ Creating database schema...")
    create_database_schema(db_path)
    
    # Step 2: Import combined data
    print("\n2️⃣ Importing combined CSV data...")
    success = import_combined_csv_data(raw_csv_path, analysis_csv_path, db_path)
    
    if not success:
        print("❌ Migration failed!")
        return
    
    # Step 3: Verify
    print("\n3️⃣ Verifying database...")
    verify_database(db_path)
    
    # Display final info
    db_size = os.path.getsize(db_path) / (1024*1024)
    print(f"\n🎉 Combined dataset migration completed successfully!")
    print(f"💾 Database size: {db_size:.1f} MB")
    print(f"📍 Database location: {db_path}")
    print(f"\nNext steps:")
    print(f"1. Update solar_facilities_server.py to use the database")
    print(f"2. Test query performance improvements")
    print(f"3. Verify NULL capacity handling in queries")

if __name__ == "__main__":
    main()