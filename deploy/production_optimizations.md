# Production Optimizations for Multiple Datasets

## 1. Memory Optimization

### Lazy Loading Pattern
```python
# Instead of loading all datasets at startup
datasets = {
    'solar': None,  # Load on first request
    'wind': None,   
    'hydro': None
}

def get_dataset(name):
    if datasets[name] is None:
        datasets[name] = pd.read_csv(f"data/{name}_facilities.csv")
    return datasets[name]
```

### Dataset Chunking
```python
# For very large datasets, process in chunks
def process_large_dataset(df, chunk_size=1000):
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        yield process_chunk(chunk)
```

## 2. Caching Strategy

### Redis Cache Layer
```bash
# Install Redis
sudo apt install redis-server

# Cache GeoJSON generation
@cache_result(ttl=3600)  # 1 hour cache
def generate_geojson(country, dataset_type):
    # Expensive GeoPandas operation
    return geojson_data
```

### Smart File Caching
```python
# Cache generated GeoJSON files by content hash
def get_cached_geojson(facilities_data):
    data_hash = hashlib.md5(str(facilities_data).encode()).hexdigest()
    cache_file = f"static/maps/cached_{data_hash}.geojson"
    
    if os.path.exists(cache_file):
        return f"/static/maps/cached_{data_hash}.geojson"
    
    # Generate new file
    return generate_and_save_geojson(facilities_data, cache_file)
```

## 3. Database Migration

### For Large Datasets (>100MB)
```python
# Migrate from CSV to PostgreSQL with PostGIS
# Benefits:
# - Spatial indexing for geographic queries
# - Memory-efficient queries
# - Better concurrent access

DATABASE_URL = "postgresql://user:pass@localhost/climate_data"

def get_facilities_in_bounds(north, south, east, west):
    query = """
    SELECT * FROM solar_facilities 
    WHERE ST_Within(
        ST_Point(longitude, latitude),
        ST_MakeEnvelope(%s, %s, %s, %s, 4326)
    )
    LIMIT 500
    """
    return pd.read_sql(query, DATABASE_URL, params=[west, south, east, north])
```

## 4. Async Processing

### Background GeoJSON Generation
```python
from celery import Celery

@celery.task
def generate_geojson_async(country, dataset_type):
    # Heavy processing in background
    return generate_geojson(country, dataset_type)

# API returns immediately with status
@app.post("/generate-map")
async def generate_map(request):
    task = generate_geojson_async.delay(request.country, request.dataset)
    return {"task_id": task.id, "status": "processing"}
```

## 5. Monitoring Setup

### Resource Monitoring
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Setup Prometheus + Grafana for metrics
# Monitor:
# - Memory usage per dataset
# - GeoJSON generation time
# - Cache hit rates
# - API response times
```

## 6. Scaling Triggers

### When to upgrade from t3.medium:

**Memory Issues:**
- CSV loading failures
- GeoPandas OOM errors
- Swap usage > 50%

**CPU Issues:**
- API response time > 10 seconds
- High CPU usage during concurrent requests
- Map generation timeouts

**Storage Issues:**
- Disk usage > 80%
- GeoJSON cache growing > 5GB
- Application logs filling disk

### Auto-scaling with ECS/EKS (Advanced)
- Container-based deployment
- Auto-scale based on CPU/memory
- Load balancer for multiple instances
- Shared EFS for static files