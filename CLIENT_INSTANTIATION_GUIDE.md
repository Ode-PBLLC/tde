# Client Instantiation Guide

**For internal development team: How to turn the generic MCP repo into a client-specific product**

## Overview

This guide covers the end-to-end process of taking this generic MCP API framework and instantiating it for a specific client with their data, domain, and requirements.

## Phase 1: Client Requirements Gathering

### Data Discovery
**What to collect from client:**
- [ ] **Data samples** - Representative files, database exports, API responses
- [ ] **Data schemas** - Column definitions, relationships, constraints
- [ ] **Data volume** - How much data, update frequency, growth expectations
- [ ] **Data quality** - Known issues, cleaning requirements, missing values
- [ ] **Access patterns** - How users currently find and use this data

### Domain Understanding
- [ ] **Key entities** - What are the main "things" in their domain (facilities, policies, companies, etc.)
- [ ] **Relationships** - How entities connect (geographic, temporal, hierarchical)
- [ ] **Business rules** - Domain-specific logic, calculations, constraints
- [ ] **Terminology** - Industry-specific language, abbreviations, classifications

### Use Case Definition
- [ ] **Query examples** - 10-20 realistic questions users would ask
- [ ] **Output preferences** - Maps, charts, tables, exports - what matters most?
- [ ] **User personas** - Who uses this? Analysts, executives, researchers?
- [ ] **Integration needs** - Existing tools, authentication systems, export formats

### Technical Requirements
- [ ] **Authentication** - SSO, API keys, role-based access
- [ ] **Performance** - Response time expectations, concurrent users
- [ ] **Deployment** - Cloud platform, scaling requirements, monitoring
- [ ] **Compliance** - Data privacy, retention, audit requirements

## Phase 2: Architecture Planning

### MCP Server Design
Based on client data, determine which MCP servers to build:

**Common patterns:**
- **Geographic server** - For location-based data (facilities, regions, coordinates)
- **Time-series server** - For temporal data (metrics over time, events)
- **Knowledge graph server** - For document/policy data with relationships
- **Calculation server** - For domain-specific computations and aggregations
- **External API server** - For third-party data integration

**Decision matrix:**
```
Data Type           → Server Pattern
Geospatial data     → Geographic server (inherit from geographic_server.py)
CSV/database tables → Data server (inherit from base_data_manager.py)  
Documents/policies  → Knowledge server (custom, likely external integration)
Real-time APIs      → API proxy server (inherit from base_server.py)
```

### Configuration Strategy
Plan the layered configuration:

1. **servers.base.json** - Generic MCP servers (stays in generic repo)
2. **servers.{client}.json** - Client-specific servers and overrides
3. **servers.local.json** - Developer testing (gitignored)

Example structure:
```json
{
  "servers": {
    "client_facilities": {
      "command": "python",
      "args": ["mcp/client_facilities_server.py"],
      "env": {
        "DATABASE_PATH": "data/facilities.db",
        "API_KEY_ENV": "CLIENT_API_KEY"
      }
    },
    "client_policies": {
      "command": "python", 
      "args": ["mcp/client_knowledge_server.py"],
      "env": {
        "KNOWLEDGE_BASE_URL": "https://client-docs.example.com/api"
      }
    }
  }
}
```

## Phase 3: Development Workflow

### 1. Repository Setup
```bash
# Clone generic repo for client
git clone <generic-mcp-repo> client-project-name
cd client-project-name

# Set up client-specific remote (optional)
git remote add client https://github.com/client/their-repo.git

# Create client configuration
cp config/servers.example.json config/servers.client.json
```

### 2. Data Integration
```bash
# Create client data directory (gitignored)
mkdir -p data/client-name

# Set up data ingestion scripts if needed
mkdir -p scripts/data-ingestion
```

### 3. MCP Server Development

**For each data source, create a custom MCP server:**

```python
# mcp/client_facilities_server.py
from base_data_manager import BaseDataManager

class ClientFacilitiesServer(BaseDataManager):
    def __init__(self):
        super().__init__(
            db_path=os.getenv("DATABASE_PATH", "data/facilities.db"),
            table_name="facilities"
        )
    
    @self.server.call_tool()
    async def get_facilities_by_region(self, region: str, limit: int = 100):
        # Client-specific business logic
        return await self.search_records(
            filters={"region": region},
            limit=limit,
            sort_by="capacity_mw DESC"
        )
```

**Development checklist per server:**
- [ ] Inherits from appropriate base class
- [ ] Implements client-specific query methods
- [ ] Handles client data formats and edge cases
- [ ] Includes proper error handling
- [ ] Has comprehensive docstrings
- [ ] Follows client terminology and business rules

### 4. Configuration Files

**Update config/servers.client.json:**
```json
{
  "servers": {
    "client_facilities": {
      "command": "python",
      "args": ["mcp/client_facilities_server.py"],
      "env": {
        "DATABASE_PATH": "data/client/facilities.db"
      }
    }
  }
}
```

**Create .env file:**
```bash
# Client-specific configuration
API_TITLE="Client Name Data API"
SERVICE_NAME="Client Name Service"
CLIENT_API_KEY=<secure-key>
DATA_DIR=./data/client
```

### 5. Testing and Validation

**Unit testing:**
```bash
# Test individual MCP servers
python -m pytest test_scripts/test_client_facilities.py

# Test API integration
python test_scripts/test_client_api_integration.py
```

**Query validation:**
- [ ] Test all client-provided example queries
- [ ] Verify outputs match expected format
- [ ] Check performance with client data volumes
- [ ] Validate business logic and calculations

**Data quality checks:**
- [ ] Verify data ingestion accuracy
- [ ] Check for missing or malformed data
- [ ] Validate geographic coordinates, dates, etc.
- [ ] Test edge cases and error conditions

## Phase 4: Deployment Preparation

### Environment Configuration
```bash
# Production environment variables
export API_TITLE="Client Production API"
export DATA_DIR="/opt/client-data" 
export CONFIG_DIR="/opt/client-config"
export PORT=8080
```

### Documentation
Create client-specific documentation:
- [ ] **API Documentation** - Available endpoints, example queries
- [ ] **Data Dictionary** - What data is available, how it's structured
- [ ] **User Guide** - Common query patterns, interpreting results
- [ ] **Maintenance Guide** - Data updates, configuration changes

### Handoff Package
Provide client with:
- [ ] **Deployed API** - Running service with their data
- [ ] **Configuration files** - Environment settings, server configs
- [ ] **Documentation** - User guides, API reference
- [ ] **Update scripts** - How to pull generic improvements
- [ ] **Support contact** - Who to reach for issues/changes

## Phase 5: Ongoing Maintenance

### Update Workflow
```bash
# Pull improvements from generic repo
scripts/pull_generic.sh

# Test with client data
python test_scripts/test_full_integration.py

# Deploy updates
./deploy.sh
```

### Monitoring
- [ ] Set up health checks on client endpoints
- [ ] Monitor query performance and errors
- [ ] Track data freshness and update success
- [ ] Alert on service issues

## Common Patterns and Templates

### Geographic Data Server Template
```python
from geographic_server import GeographicServer

class ClientLocationServer(GeographicServer):
    def __init__(self):
        super().__init__(
            data_path=os.getenv("LOCATIONS_DATA", "data/client/locations.geojson")
        )
    
    @self.server.call_tool()
    async def get_facilities_near_city(self, city: str, radius_km: float = 50):
        # Client-specific geographic queries
        pass
```

### Time Series Data Template
```python
from base_data_manager import BaseDataManager

class ClientTimeSeriesServer(BaseDataManager):
    @self.server.call_tool()
    async def get_trend_analysis(self, metric: str, start_date: str, end_date: str):
        # Time-based analysis specific to client domain
        pass
```

### External API Integration Template
```python
from base_server import BaseServer
import aiohttp

class ClientExternalServer(BaseServer):
    @self.server.call_tool()
    async def fetch_external_data(self, query_params: dict):
        # Integrate with client's existing APIs
        pass
```

## Quality Checklist

Before handoff, ensure:
- [ ] All client query examples work correctly
- [ ] Performance meets requirements (< 5s response time)
- [ ] Error handling is comprehensive
- [ ] Documentation is complete and accurate
- [ ] Security/authentication is properly configured
- [ ] Data updates/ingestion process is documented
- [ ] Monitoring and alerting is set up
- [ ] Client team has been trained on basic usage

## Troubleshooting Common Issues

**MCP server connection failures:**
- Check server configuration in servers.client.json
- Verify environment variables are set
- Test server startup independently

**Data query errors:**
- Validate data format matches server expectations
- Check for missing required fields
- Verify database/file permissions

**Performance issues:**
- Add database indexes for common queries
- Implement result caching where appropriate
- Consider data pre-aggregation for large datasets

---

*This guide should be updated as we learn from each client instantiation to capture new patterns and improve the process.*