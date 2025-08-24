# Port Migration Summary

## Changes Made

All client configurations have been updated to use port 8098 to align with your nginx SSL setup.

### Updated Files:
1. **streamlit_api_demo.py** - API_BASE_URL changed from 8099 → 8098
2. **scripts/generate_featured_cache.py** - API_BASE_URL changed from 8099 → 8098  
3. **test_simple_api.py** - API URL changed from 8099 → 8098
4. **CLAUDE.md** - Documentation updated to reflect port 8098

### Architecture:
- **Main API Server** (`api_server.py`): Runs on port 8098
- **KG Visualization Server** (`kg_visualization_server.py`): Runs on port 8100 (internal only)

### Your nginx setup:
```
External traffic → CloudFlare SSL → nginx (port 80/443) → localhost:8098
```

## To Start Services:

1. Start the main API server:
   ```bash
   python api_server.py
   ```
   This will start on port 8098.

2. (Optional) Start the KG visualization server:
   ```bash
   python kg_visualization_server.py
   ```
   This will start on port 8100 (internal service only).

## Testing:

Run the test script to verify everything is working:
```bash
python test_port_config.py
```

Or test manually:
```bash
curl -X POST http://localhost:8098/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is climate change?", "include_thinking": false}'
```

## Notes:
- The KG server on port 8100 is only called internally by the API server
- Only port 8098 needs to be exposed through nginx
- Your SSL termination at CloudFlare + nginx proxy to 8098 is the correct setup