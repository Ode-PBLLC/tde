# Large Dataset Deployment Strategy

## Dataset Profile Analysis

### Current Datasets (~400MB total):
1. **TZ-SAM Raw Polygons** - 268MB GeoPackage (CRITICAL: This needs special handling)
2. **TZ-SAM Analysis Polygons** - 48MB GeoPackage  
3. **Policy Data** - 11MB Excel/CSV files
4. **LSE Modules** - ~1MB policy documents

## Critical Issues & Solutions

### 1. 268MB GeoPackage Problem

**Issue:** Loading 268MB polygon data into memory will:
- Consume 1-4GB RAM (with GeoPandas overhead)
- Take 30+ seconds to load
- Crash on concurrent requests

**Solutions:**

#### Option A: PostGIS Database (Recommended)
```bash
# Install PostGIS
sudo apt install postgresql postgresql-contrib postgis

# Load GeoPackage to PostGIS
ogr2ogr -f PostgreSQL PG:"dbname=climate_data user=postgres" \
    tz-sam-runs_2025-Q1_outputs_external_raw_polygons.gpkg \
    -nln solar_polygons
```

#### Option B: Spatial Indexing + Chunking
```python
# Load only what's needed for current viewport
import geopandas as gpd
from shapely.geometry import box

def get_polygons_in_bounds(north, south, east, west, dataset='raw'):
    # Use spatial index to filter before loading
    bounds_geom = box(west, south, east, north)
    
    # Only load polygons that intersect viewport
    gdf = gpd.read_file(
        f"data/tz-sam-q1-2025/tz-sam-runs_2025-Q1_outputs_external_{dataset}_polygons.gpkg",
        bbox=(west, south, east, north)  # Spatial filter at read time
    )
    return gdf.head(1000)  # Limit for performance
```

#### Option C: Pre-processed Tile Cache
```python
# Pre-generate map tiles at different zoom levels
# Store as smaller GeoJSON files organized by geographic tiles

def generate_tile_cache():
    gdf = gpd.read_file("large_dataset.gpkg")
    
    # Create 10x10 grid of tiles
    for x in range(10):
        for y in range(10):
            tile_bounds = calculate_tile_bounds(x, y)
            tile_data = gdf.cx[tile_bounds[0]:tile_bounds[2], tile_bounds[1]:tile_bounds[3]]
            
            tile_file = f"static/tiles/tile_{x}_{y}.geojson"
            tile_data.to_file(tile_file, driver='GeoJSON')
```

### 2. Memory Management Strategy

```python
# Lazy loading with LRU cache
from functools import lru_cache
import gc

@lru_cache(maxsize=3)  # Only keep 3 datasets in memory
def load_dataset(dataset_name):
    if dataset_name == 'solar_polygons_raw':
        # Special handling for large dataset
        return load_large_dataset_chunked(dataset_name)
    else:
        return pd.read_csv(f"data/{dataset_name}.csv")

def load_large_dataset_chunked(dataset_name):
    # Only load what's absolutely necessary
    # Use spatial/temporal filters
    return gpd.read_file(dataset_file, rows=slice(0, 10000))

# Memory cleanup after operations
def cleanup_memory():
    gc.collect()  # Force garbage collection
```

### 3. Updated Server Configuration

#### Enhanced Systemd Service
```ini
[Service]
# Increase memory limits for large datasets
MemoryHigh=12G
MemoryMax=14G

# Increase timeouts for heavy operations
TimeoutStartSec=120
TimeoutStopSec=60

# Environment for large datasets
Environment=GDAL_CACHEMAX=1024
Environment=GDAL_MAX_DATASET_POOL_SIZE=100
```

#### Nginx Optimizations
```nginx
# Increase timeouts for heavy operations
proxy_connect_timeout 300s;
proxy_send_timeout 300s;
proxy_read_timeout 300s;

# Increase client body size for uploads
client_max_body_size 50M;

# Rate limiting for heavy endpoints
location /query {
    limit_req zone=heavy_api burst=3 nodelay;
    proxy_pass http://127.0.0.1:8099;
}
```

### 4. Cost-Optimized Alternative

#### Option: Use AWS RDS PostGIS
- **RDS db.t3.medium** (~$40/month)
- **EC2 t3.medium** (~$30/month)
- **Total:** ~$70/month (vs $120 for t3.xlarge)

Benefits:
- Managed database with automatic backups
- Spatial queries handled by PostGIS
- Can scale database separately from application
- Better resource utilization

## Deployment Decision Matrix

| Option | Monthly Cost | Complexity | Performance | Scalability |
|--------|-------------|------------|-------------|-------------|
| t3.xlarge | $120 | Low | High | Medium |
| t3.large + PostGIS | $60 | Medium | High | High |
| t3.medium + RDS | $70 | Medium | Medium | High |

## Recommendation: t3.large + PostGIS

**Why:**
1. **Cost-effective** - $60/month 
2. **Handles large datasets** - Spatial indexing
3. **Scalable** - Add read replicas if needed
4. **Professional** - Industry standard for geospatial data

**Implementation plan:**
1. Start with t3.large + PostGIS
2. Load 268MB GeoPackage into PostGIS
3. Implement spatial query endpoints
4. Monitor performance and scale if needed