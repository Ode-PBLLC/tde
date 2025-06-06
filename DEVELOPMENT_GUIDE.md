# Development Guide

## Quick Start

### Local Development Setup

1. **Clone and Setup Environment**
```bash
git clone https://github.com/Ode-PBLLC/tde.git
cd tde

# Create conda environment
conda create -n tde-api python=3.11
conda activate tde-api

# Install dependencies
pip install -r requirements.txt
```

2. **Configure Environment Variables**
```bash
# Create .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "OPENAI_API_KEY=sk-..." >> .env
```

3. **Run the API Server**
```bash
python api_server.py
# Server starts on http://localhost:8099
```

4. **Test the API**
```bash
# Health check
curl http://localhost:8099/health

# Test query
curl -X POST http://localhost:8099/query \
  -H "Content-Type: application/json" \
  -d '{"query": "solar energy in Brazil"}'
```

### Development Workflow

**File Structure**:
```
tde/
├── api_server.py              # FastAPI application
├── mcp/                       # MCP servers
│   ├── mcp_chat.py           # Orchestration layer  
│   ├── cpr_kg_server.py      # Knowledge graph
│   ├── solar_facilities_server.py  # Geographic data
│   └── response_formatter_server.py # Response structuring
├── data/                      # Raw datasets
├── extras/                    # Knowledge graph data
├── static/                    # Generated files
└── requirements.txt
```

**Development Commands**:
```bash
# Run with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8099

# Test specific endpoints
curl -X POST localhost:8099/thorough-response \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'

# Check generated files
ls -la static/maps/
```

## Extending the System

### Adding New MCP Servers

MCP (Model Context Protocol) servers provide specialized data access. Here's how to add new ones:

#### 1. Create New MCP Server

```python
# mcp/my_new_server.py
import asyncio
from mcp import Application, Tool
from mcp.server.models import InitializeRequest
from mcp.types import TextContent, Tool, JSONSchema
import pandas as pd

app = Application()

# Load your dataset
my_data = pd.read_csv("data/my_dataset.csv")

@app.tool()
def get_my_data(query: str) -> list[TextContent]:
    """Tool to access my custom dataset."""
    # Your data processing logic here
    results = my_data[my_data['field'].str.contains(query, case=False)]
    
    return [TextContent(
        type="text", 
        text=results.to_json(orient='records')
    )]

@app.tool() 
def get_my_visualization_data(category: str) -> list[TextContent]:
    """Generate visualization data for my dataset."""
    # Your visualization logic here
    chart_data = {
        "type": "bar",
        "data": {
            "labels": [...],
            "datasets": [...]
        }
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(chart_data)
    )]

if __name__ == "__main__":
    app.run()
```

#### 2. Register Server in Orchestration Layer

```python
# In mcp/mcp_chat.py, add to run_query():

async def run_query(q: str):
    async with MultiServerClient() as client:
        # Existing servers
        await client.connect_to_server("kg", os.path.join(mcp_dir, "cpr_kg_server.py"))
        await client.connect_to_server("solar", os.path.join(mcp_dir, "solar_facilities_server.py"))
        await client.connect_to_server("formatter", os.path.join(mcp_dir, "response_formatter_server.py"))
        
        # Add your new server
        await client.connect_to_server("mydata", os.path.join(mcp_dir, "my_new_server.py"))
        
        # Update system prompt to include new tools
        system_prompt += """
        
        My Data Server Tools:
        - `get_my_data`: Access custom dataset
        - `get_my_visualization_data`: Generate charts for my data
        """
```

#### 3. Link to Knowledge Graph

```python
# In mcp/cpr_kg_server.py, add dataset connections:

def add_dataset_connections(G):
    # Add your dataset node
    my_dataset_node_id = "MY_CUSTOM_DATASET"
    my_dataset_label = "My Custom Dataset"
    
    if not G.has_node(my_dataset_node_id):
        G.add_node(
            my_dataset_node_id,
            kind="Dataset", 
            label=my_dataset_label,
            description="Description of my custom dataset",
            server_name="mydata",
            total_records=len(my_data)
        )
    
    # Link to relevant concepts
    my_concept_id = _concept_id("my topic")
    if my_concept_id and G.has_node(my_concept_id):
        if not G.has_edge(my_concept_id, my_dataset_node_id):
            G.add_edge(my_concept_id, my_dataset_node_id, type="HAS_DATASET_ABOUT")
```

### Adding New Concepts

To add new concepts to the knowledge graph:

#### 1. Add to Concepts CSV

```python
# Add row to extras/concepts.csv
import pandas as pd

concepts = pd.read_csv("extras/concepts.csv")
new_concept = {
    'id': 'Q999',
    'preferred_label': 'my new concept',
    'alternative_labels': 'alternative1,alternative2',
    'description': 'Description of the concept',
    'definition': 'Detailed definition...'
}

concepts = pd.concat([concepts, pd.DataFrame([new_concept])], ignore_index=True)
concepts.to_csv("extras/concepts.csv", index=False)
```

#### 2. Generate Embeddings

The system will automatically generate embeddings for new concepts when the server starts.

#### 3. Link to Datasets

Use the approach shown above to link your concept to relevant datasets.

### Adding New Data Sources

#### Option 1: File-Based Datasets

```python
# Place files in data/ directory
data/
├── my_dataset/
│   ├── main_data.csv
│   ├── supplementary.json
│   └── metadata.txt

# Create MCP server to access them
@app.tool()
def get_my_file_data(filename: str) -> list[TextContent]:
    file_path = f"data/my_dataset/{filename}"
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
        return [TextContent(type="text", text=df.to_json())]
    # Handle other formats...
```

#### Option 2: API-Based Data Sources

```python
# Create MCP server that calls external APIs
@app.tool()
def get_external_api_data(query: str) -> list[TextContent]:
    import requests
    
    response = requests.get(f"https://api.example.com/data?q={query}")
    data = response.json()
    
    return [TextContent(type="text", text=json.dumps(data))]
```

#### Option 3: Database Integration

```python
# Database-backed MCP server
import sqlite3

@app.tool() 
def query_database(sql_query: str) -> list[TextContent]:
    conn = sqlite3.connect("data/my_database.db")
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    
    return [TextContent(type="text", text=df.to_json())]
```

### Creating New Response Module Types

The system supports extensible module types for different visualizations:

#### 1. Add Module Type to Formatter

```python
# In mcp/response_formatter_server.py

@app.tool()
def create_my_custom_module(data: str, title: str) -> list[TextContent]:
    """Create a custom module type."""
    
    module = {
        "type": "my_custom_type",
        "heading": title,
        "custom_data": json.loads(data),
        "render_config": {
            "width": 800,
            "height": 400,
            "interactive": True
        }
    }
    
    return [TextContent(type="text", text=json.dumps(module))]
```

#### 2. Update Frontend Handling

Your frontend will need to handle the new module type:

```javascript
// Frontend rendering logic
function renderModule(module) {
    switch(module.type) {
        case 'text':
            return renderTextModule(module);
        case 'map': 
            return renderMapModule(module);
        case 'my_custom_type':
            return renderMyCustomModule(module);
        default:
            return renderFallback(module);
    }
}
```

### Performance Optimization

#### 1. Caching Strategies

```python
# Add caching to expensive operations
from functools import lru_cache
import time

@lru_cache(maxsize=100)
def expensive_computation(query):
    # Cache results for repeated queries
    time.sleep(2)  # Simulate expensive operation
    return results

# Or use file-based caching
import joblib

def cached_api_call(query):
    cache_file = f"cache/{hash(query)}.pkl"
    if os.path.exists(cache_file):
        return joblib.load(cache_file)
    
    result = make_expensive_api_call(query)
    joblib.dump(result, cache_file)
    return result
```

#### 2. Async Processing

```python
# Make I/O operations async
import aiohttp
import asyncio

async def fetch_multiple_sources(queries):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results
```

#### 3. Data Loading Optimization

```python
# Lazy loading for large datasets
class LazyDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self._data = None
    
    @property
    def data(self):
        if self._data is None:
            self._data = pd.read_csv(self.file_path)
        return self._data

# Use in MCP servers
data_loader = LazyDataLoader("data/large_dataset.csv")

@app.tool()
def query_large_dataset(query):
    # Data only loaded when first accessed
    return data_loader.data.query(query)
```

## Testing Strategies

### Unit Testing

```python
# tests/test_mcp_servers.py
import pytest
from mcp_chat import MultiServerClient

@pytest.mark.asyncio
async def test_knowledge_graph_server():
    async with MultiServerClient() as client:
        await client.connect_to_server("kg", "mcp/cpr_kg_server.py")
        
        result = await client.call_tool(
            "GetPassagesMentioningConcept", 
            {"concept": "climate change"},
            "kg"
        )
        
        assert len(result) > 0
        assert "climate" in result[0].text.lower()

def test_solar_data_loading():
    from mcp.solar_facilities_server import load_solar_data
    
    data = load_solar_data()
    assert len(data) > 0
    assert 'capacity_mw' in data.columns
```

### Integration Testing

```python
# tests/test_api_integration.py
import requests
import pytest

BASE_URL = "http://localhost:8099"

def test_health_endpoint():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_endpoint():
    payload = {"query": "solar energy"}
    response = requests.post(f"{BASE_URL}/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "modules" in data
    assert len(data["modules"]) > 0

@pytest.mark.slow
def test_complex_query():
    payload = {"query": "extreme weather and show me data"}
    response = requests.post(f"{BASE_URL}/thorough-response", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "raw_mcp_response" in data
    assert data["raw_mcp_response"]["chart_data"] is not None
```

### Load Testing

```python
# tests/load_test.py
import asyncio
import aiohttp
import time

async def make_request(session, query):
    payload = {"query": query}
    async with session.post("http://localhost:8099/query", json=payload) as response:
        return await response.json()

async def load_test():
    queries = ["solar energy", "climate change", "extreme weather"] * 10
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    successful = len([r for r in results if not isinstance(r, Exception)])
    print(f"Completed {successful}/{len(queries)} requests in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(load_test())
```

## Debugging and Monitoring

### Logging Setup

```python
# Add to api_server.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    
    try:
        result = await process_query(request)
        logger.info(f"Query completed successfully")
        return result
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise
```

### Performance Monitoring

```python
# Add timing decorators
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

@timing_decorator
async def process_query(request):
    # Your processing logic
    pass
```

### Health Monitoring

```python
# Enhanced health check
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check MCP servers
    try:
        async with MultiServerClient() as client:
            await client.connect_to_server("kg", "mcp/cpr_kg_server.py")
            health_status["components"]["knowledge_graph"] = "healthy"
    except Exception as e:
        health_status["components"]["knowledge_graph"] = f"unhealthy: {e}"
        health_status["status"] = "degraded"
    
    # Check data files
    if os.path.exists("extras/concepts.csv"):
        health_status["components"]["concepts_data"] = "healthy"
    else:
        health_status["components"]["concepts_data"] = "missing"
        health_status["status"] = "degraded"
    
    return health_status
```

## Deployment Guide

### Local Deployment

```bash
# Development server
python api_server.py

# Production-like server
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8099
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8099
CMD ["python", "api_server.py"]
```

```bash
# Build and run
docker build -t climate-api .
docker run -p 8099:8099 --env-file .env climate-api
```

### Production Deployment

```bash
# On EC2/server
git clone https://github.com/Ode-PBLLC/tde.git
cd tde

# Setup environment
conda create -n tde-api python=3.11
conda activate tde-api
pip install -r requirements.txt

# Configure environment
echo "ANTHROPIC_API_KEY=your-key" > .env
echo "OPENAI_API_KEY=your-key" >> .env

# Run with systemd (see deploy/climate-api.service)
sudo systemctl start climate-api
sudo systemctl enable climate-api
```

This development guide provides comprehensive instructions for extending and maintaining the climate policy API system. Each section includes practical examples and code snippets to help developers get started quickly.