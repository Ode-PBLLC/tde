# Climate Policy Radar API (v2)

AI-powered API that provides comprehensive climate policy analysis with real-time data discovery, interactive visualizations, and intelligent reasoning.

**Version 2.0** - Consolidated v2 architecture with simplified orchestration and enhanced performance.

## 🌟 Key Features

- **🤖 Intelligent Analysis**: Claude Sonnet 4 powered reasoning and synthesis
- **📊 Automatic Data Discovery**: Finds and surfaces relevant datasets automatically
- **🗺️ Interactive Visualizations**: Real-time maps, charts, and tables
- **⚡ Streaming Responses**: Live progress indicators and results
- **🔗 Multi-Source Integration**: Policy documents + structured datasets + geographic data
- **📱 Frontend Ready**: JSON modules optimized for web applications
- **🚀 V2 Architecture**: Single consolidated orchestrator for faster, cleaner responses
- **💬 Multi-turn Conversations**: Context-aware follow-up queries with session management
- **📝 Analytics-Ready Logging**: Comprehensive conversation tracking to CSV
- **⚡ Featured Query Cache**: Pre-recorded streams for 50-100x faster responses
- **🗺️ Dynamic Artifacts**: On-demand generation of maps, KGs, and visualizations

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda or venv for environment management
- **Required**: Anthropic API key
- Optional: OpenAI API key (for certain features)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ode-PBLLC/tde.git
cd tde

# Create and activate environment
conda create -n tde-api python=3.11
conda activate tde-api

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Required Environment Variables

See `.env.example` for all configuration options. At minimum, you need:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Optional variables:
- `OPENAI_API_KEY` - For OpenAI features (if used)
- `API_BASE_URL` - Custom API base URL (defaults to http://localhost:8098)
- `API_PORT` - API server port (defaults to 8098)

### Start the Server

```bash
python api_server.py
```

The API will be available at `http://localhost:8098` (default port)

### Verify Installation

```bash
# Run smoke tests
./scripts/smoke_test.sh

# Health check
curl http://localhost:8098/health

# Stream a test query
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Brazil'\''s renewable energy targets?"}'
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[CHANGELOG.md](CHANGELOG.md)** | Version history and release notes |
| **[API_GUIDE.md](API_GUIDE.md)** | Complete developer guide with examples |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Production deployment guide |
| **[data/README.md](data/README.md)** | Dataset documentation and regeneration |
| **[docs/](docs/)** | Technical implementation guides |

## 🏗️ V2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (v2.0.0)                    │
│                      (api_server.py)                           │
├─────────────────────────────────────────────────────────────────┤
│  • POST /query/stream       - Streaming responses              │
│  • POST /query              - Synchronous responses            │
│  • GET  /featured-queries   - Curated query gallery            │
│  • GET  /health             - Health check                     │
│  • Static files at /static  - Generated maps/charts/KGs        │
│  • Session management       - Multi-turn conversations         │
│  • Stream caching           - Pre-recorded featured queries    │
│  • Conversation logging     - Analytics to CSV                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────────┐
│              MCP V2 Orchestration Layer                        │
│                  (mcp_chat_v2.py)                              │
├─────────────────────────────────────────────────────────────────┤
│  • FastMCP singleton client (5-10x faster TTFT)                │
│  • Query enrichment (domain context injection)                 │
│  • Server tool planning (AI-powered routing)                   │
│  • Fact ordering & narrative synthesis                         │
│  • Citation registry & evidence tracking                       │
│  • RunQueryResponse contract validation                        │
│  • Streaming progress updates                                  │
└─────────┬──────────┬──────────┬──────────┬─────────────────────┘
          │          │          │          │
          v          v          v          v
    ┌─────────┐ ┌────────┐ ┌──────┐ ┌──────────┐ ┌──────────────┐
    │   CPR   │ │  Solar │ │ LSE  │ │   GIST   │ │ + 7 more     │
    │ Server  │ │ Server │ │Server│ │  Server  │ │ servers      │
    └─────────┘ └────────┘ └──────┘ └──────────┘ └──────────────┘
                                                   (11 total)
```

### Key V2 Improvements

- **Single orchestrator**: Consolidated from 3 legacy orchestrators to 1
- **Faster startup**: Singleton MCP client pre-connects all servers
- **Cleaner contracts**: All servers implement `RunQueryResponse` standard
- **Better testing**: 15 automated smoke tests verify integrity
- **Smaller codebase**: Removed 13,000+ lines of legacy code

### Core Architecture Components

#### **api_server.py** (1,600 lines)
The FastAPI server providing REST endpoints and stateful session management.

**Key Features**:
- **SessionStore**: Multi-turn conversation state with 20-minute TTL
- **StreamCache**: Pre-recorded SSE events for featured queries (50-100x faster) (Unfortunately currently broken on deploy)
- **ConversationLogger**: Analytics tracking to `conversation_logs.csv`
- **Dynamic Artifacts**: On-demand GeoJSON generation and serving
- **Lifecycle Management**: MCP client warmup on startup for faster first requests
- **Environment Portability**: Dynamic URL rewriting for dev/prod compatibility

**Endpoints**: `/query/stream`, `/query`, `/featured-queries`, `/health`, `/kg/{id}`, `/static/*`

#### **mcp_chat_v2.py** (2,000+ lines)
The intelligent orchestration layer coordinating AI reasoning with data retrieval.

**Key Components**:
- **QueryEnricher**: Adds Brazilian environmental domain context to queries
- **ServerToolPlanner**: AI-powered selection of relevant MCP servers
- **FactOrderer**: Sequences evidence for coherent narrative flow
- **NarrativeSynthesizer**: Composes final response with inline citations
- **CitationRegistry**: Tracks and validates evidence references
- **SimpleOrchestrator**: Main coordination loop with streaming support
- **MultiServerClient**: FastMCP singleton managing 11 server connections

**Models Used**: Claude Sonnet 4 (fact ordering, narrative synthesis), Haiku (query enrichment)

#### **V2 MCP Servers** (11 servers, ~18,500 lines)

All servers implement the `RunQueryResponse` contract with standardized `run_query` tool:

| Server | Dataset | Scope |
|--------|---------|-------|
| **cpr_server_v2.py** | Climate Policy Knowledge Graph | 1,325 concepts, semantic search |
| **solar_server_v2.py** | Global Solar Facilities | 8,319 facilities, 124.9 GW |
| **solar_clay_server_v2.py** | TZ-SAM Analysis | Transition zone spatial analysis |
| **deforestation_server_v2.py** | PRODES/MapBiomas | Spatial polygons, 2GB dataset |
| **lse_server_v2.py** | LSE Climate Policies | NDCs, governance, subnational |
| **gist_server_v2.py** | IPCC Chapters 11 & 12 | Processed summaries |
| **extreme_heat_server_v2.py** | Heat Index Data | Brazilian municipalities |
| **brazilian_admin_server_v2.py** | Admin Boundaries | Municipal/state GeoJSON |
| **wmo_cli_server_v2.py** | WMO Climate Index | Climate adaptation data |
| **spa_server_v2.py** | Sectoral Policy Analysis | Cross-sector policy data |
| **mb_deforest_server_v2.py** | MapBiomas Deforestation | Centroid-based queries |
| **meta_server_v2.py** | Cross-dataset Metadata | Aggregated stats |

Each server handles domain-specific queries and returns structured responses (facts, citations, artifacts).

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query/stream` | POST | **Primary endpoint** - Streaming analysis with progress |
| `/query` | POST | Synchronous analysis (simple integration) |
| `/featured-queries` | GET | Curated queries for frontend gallery |
| `/featured-queries/{id}/cached` | GET | Pre-recorded SSE stream replay (50-100x faster) |
| `/health` | GET | System health check |
| `/kg/{kg_id}` | GET | Knowledge graph visualization |
| `/static/*` | GET | Generated artifacts (maps, KGs, charts, images) |

### Response Modules

The API returns structured **modules** ready for frontend rendering:

- **Text**: Analysis content with inline citations `^1,2^`
- **Charts**: Chart.js compatible data (bar, line, pie)
- **Tables**: Structured data with columns and rows
- **Maps**: GeoJSON with interactive markers
- **Knowledge Graphs**: Interactive network visualizations
- **Citations**: References table (always last)

## 💡 Example Queries

### Geographic Analysis
```bash
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Where are the largest solar facilities in Brazil?"}'
```
**Returns**: Policy analysis + interactive map + facility data table

### Policy Research
```bash
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Brazil'\''s NDC commitments for renewable energy?"}'
```
**Returns**: NDC analysis + governance data + targets

### Multi-Dataset Discovery
```bash
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Climate risks from extreme heat in Brazilian municipalities"}'
```
**Returns**: Heat stress data + geographic analysis + policy context

## 💬 Multi-turn Conversations

The API maintains conversation context for follow-up queries, enabling natural multi-turn interactions.

### How It Works

**Session Management**:
- Each query can include an optional `conversation_id` parameter
- Sessions expire after 20 minutes of inactivity (configurable)
- Context includes last 2 conversation turns (configurable)
- Automatic garbage collection of expired sessions

### Example Usage

```bash
# First query creates a session
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Brazil'\''s solar energy targets?"}'

# Response includes conversation_id
# {"conversation_id": "abc123XYZ", "modules": [...]}

# Follow-up query uses the conversation_id
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which states are leading in solar deployment?",
    "conversation_id": "abc123XYZ"
  }'

# Another follow-up
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me a map of those facilities",
    "conversation_id": "abc123XYZ"
  }'
```

### Configuration

In `api_server.py`:
```python
SESSION_TTL_SECONDS = 20 * 60  # 20 minutes
MAX_CONTEXT_TURNS = 2          # Number of previous turns to include
```

### Conversation Logging

All queries are logged to `conversation_logs.csv` with analytics metadata:
- **timestamp**: Query time (ISO format)
- **conversation_id**: Session identifier
- **turn_number**: Position in conversation (1, 2, 3...)
- **message_type**: `new_conversation`, `continuation`, or `reset`
- **query**: User's query text
- **response_summary**: Extracted text from response modules
- **response_modules**: Types of modules returned (text, map, chart, etc.)
- **context_included**: Number of previous turns included
- **session_duration_seconds**: Time since session started
- **tokens_used**: Total tokens consumed
- **error_flag**: 1 if error occurred, 0 otherwise

**Use Cases**:
- Usage analytics and query pattern analysis
- Performance monitoring and optimization
- User behavior insights
- Dataset popularity tracking

## 📁 Static Artifacts & Content Serving

The API serves dynamically generated and cached content from the `/static` directory (~30MB).

### Directory Structure

```
static/
├── maps/           # 8MB - GeoJSON files (dynamic + cached)
│   ├── solar_facilities_*.geojson
│   ├── correlation_*.geojson
│   └── deforestation_*.geojson
├── images/         # 21MB - Featured query gallery images
│   ├── brazil-solar-expansion.jpg
│   ├── brazil-climate-goals.jpg
│   └── brazil-climate-risks.jpg
├── kg/             # 92KB - Interactive knowledge graph embeds
│   └── *.html (D3.js visualizations)
├── stream_cache/   # 168KB - Pre-recorded SSE events
│   ├── brazil-solar-expansion.jsonl
│   ├── brazil-climate-goals.jsonl
│   └── brazil-climate-risks.jsonl
├── meta/           # 1.3MB - Cross-dataset metadata
│   └── deforestation_by_year.json
└── featured_queries.json  # Gallery configuration
```

### Dynamic Map Generation

Maps are generated on-demand and cached to disk:
- **GeoJSON format** with feature properties and styling
- **Automatic correlation overlays** (e.g., solar facilities + deforestation)
- **Environment-aware URLs**: Dynamic rewriting for dev/prod portability
- **Lazy generation**: Only created when requested, then cached

### Stream Cache System

Featured queries are pre-recorded for instant replay:

**Performance**:
- **Normal queries**: 2-5s time to first token
- **Cached queries**: ~100ms time to first token
- **50-100x faster** for featured queries

**How It Works**:
1. Exact string match against `FEATURED_QUERY_CACHE_MAP` in `api_server.py`
2. If cache exists, replay pre-recorded SSE events with original timing
3. **Completely transparent** to clients (identical API contract)
4. **Dynamic URL rewriting** ensures environment portability

**Generate/Update Caches**:
```bash
# Ensure API server is running
python api_server.py

# In another terminal, record featured streams
python scripts/record_featured_streams.py
```

Cache files (`.jsonl` format) contain:
- `type`: Event type (thinking, tool_call, content, complete)
- `data`: Event payload
- `timestamp_ms`: Timing offset from start
- `recorded_at`: Recording timestamp

### Featured Queries Gallery

The `/featured-queries` endpoint serves a pseudo-CMS system for curated content:

**Files**:
- `static/featured_queries.json` - Query definitions
- `static/images/` - Gallery thumbnails

**To Update**:
1. Edit `featured_queries.json` with new query definitions
2. Add corresponding images to `images/` directory
3. Optionally record cache: `python scripts/record_featured_streams.py`
4. Restart API server (or auto-reload in development)

See `static/README.md` for detailed gallery management guide.

## 🐳 Docker & AWS Deployment

### For DevOps/Infrastructure Engineers

This section contains essential information for containerizing and deploying the TDE API to AWS.

#### Architecture Requirements

**Runtime**:
- Python 3.11+ base image recommended (`python:3.11-slim` or `python:3.11-alpine`)
- Multi-stage build recommended to minimize image size

**Resource Requirements**:
- **CPU**: 2+ vCPUs recommended (handles concurrent MCP server processes)
- **Memory**: 16GB minimum (semantic search indexes can cause OOM with 8GB under load)
- **Disk**: 10GB minimum (code + dependencies ~2GB, datasets ~2.8GB)
- **Network**: Outbound HTTPS to Anthropic API (api.anthropic.com)

**Port Configuration**:
- Default API port: `8098` (configurable via `API_PORT` env var)
- Optional KG visualization server: `8100` (internal, can be disabled)
- No inbound ports required beyond the API endpoint

#### Docker Setup

**Sample Dockerfile** (multi-stage):

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create directories for generated assets (not in repo)
RUN mkdir -p static/kg static/maps static/cache logs data

EXPOSE 8098

CMD ["python", "api_server.py"]
```

**Environment Variables for Container**:
```bash
ANTHROPIC_API_KEY=<required>
API_PORT=8098
PYTHONUNBUFFERED=1  # For proper log streaming
```

**Data Acquisition Strategy**:

The `data/` directory (~2.8GB) is **not** in the repo (gitignored). You have 3 options:

**Option 1: Copy from Existing Server (Recommended for initial setup)**
```bash
# On your current production server
cd /home/ubuntu/tde
tar -czf tde-data.tar.gz data/

# Download to local machine
scp ubuntu@your-server:/home/ubuntu/tde/tde-data.tar.gz .

# Use in Docker build or upload to S3 (see below)
```

**Option 2: Bake into Docker Image** (Simplest, but larger image)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
# Copy your extracted data directory
COPY data/ /app/data/
# Image size: ~4GB (code + deps + data)
```

**Option 3: Mount from S3/EFS** (Production recommended)
```bash
# 1. Upload data to S3
aws s3 sync ./data/ s3://your-bucket/tde-data/

# 2. In your container entrypoint or init script:
#!/bin/bash
if [ ! -d "/app/data/solar_facilities.db" ]; then
  echo "Downloading datasets from S3..."
  aws s3 sync s3://your-bucket/tde-data/ /app/data/
fi
python api_server.py

# OR use EFS mount:
# Mount EFS volume at /app/data in ECS task definition
```

**Option 4: Generate from Source** (Advanced, requires source data)
```bash
# If you have source data files, regenerate datasets
python scripts/migrate_solar_to_db.py
python scripts/precompute_deforestation_overlays.py
# See data/README.md for full regeneration guide
```

**Recommended Approach for Production**:
- **Development**: Copy data/ into Docker image (Option 2)
- **Production**: Use S3 + EFS or persistent EBS volume (Option 3)
- **Benefits**: Smaller images, easier updates, shared across instances

#### AWS Deployment Options

**Option 1: ECS Fargate** (Recommended for simplicity)
```yaml
# Task Definition essentials:
CPU: 4096 (4 vCPU)
Memory: 16384 (16 GB)
Port Mapping: 8098:8098
Environment Variables:
  - ANTHROPIC_API_KEY (from Secrets Manager)
Health Check:
  Command: ["CMD-SHELL", "curl -f http://localhost:8098/health || exit 1"]
  Interval: 30s
  Timeout: 5s
  Retries: 3
```

**Option 2: EC2 with Docker Compose** (Current production setup)
- Use `t3.xlarge` or larger (4 vCPU, 16GB RAM minimum)
- Avoid `t3.large` (8GB) - can cause OOM under load
- Docker Compose for orchestration
- EBS volume for data persistence
- Application Load Balancer for SSL termination

**Option 3: EKS** (For larger scale)
- Kubernetes deployment with HPA (Horizontal Pod Autoscaling)
- Use PVC (Persistent Volume Claim) for shared `data/` directory
- ConfigMap for non-sensitive config, Secrets for API keys

#### Database & Persistence

The application uses **file-based storage** (no external database required):

- SQLite databases in `data/` directory (e.g., `solar_facilities.db` - 44MB)
- JSON indexes for semantic search
- GeoJSON/HTML generated to `static/` directory (ephemeral, can be regenerated)

**Persistent Storage Needs**:
- `data/` - Core datasets (~2.8GB) - **must persist**
- `static/kg/`, `static/maps/`, `static/cache/` - Generated assets (safe to clear on redeploy)
- `logs/` - Application logs (optional persistence)

#### Health Checks & Monitoring

**Health Endpoint**: `GET /health`
```json
{
  "status": "ok",
  "version": "2.0.0",
  "orchestrator": "mcp_chat_v2"
}
```

**Key Metrics to Monitor**:
- Response time (P50, P95, P99) - expect 3-20s depending on query complexity
- Anthropic API rate limits (100K tokens/min)
- Memory usage (watch for semantic index loading)
- Error rate (4xx, 5xx responses)

**Logging**:
- Application logs to stdout (PYTHONUNBUFFERED=1 recommended)
- Structured JSON logs can be enabled if needed
- CloudWatch Logs integration recommended for AWS deployments

#### Scaling Considerations

**Horizontal Scaling**:
- Application is **stateless** (no session storage)
- Safe to run multiple replicas behind ALB/NLB
- Each instance pre-loads MCP clients on startup (~5-10s cold start)
- No shared state between instances

**Vertical Scaling**:
- Memory usage scales with semantic index size
- CPU usage spikes during query processing (Claude API calls)
- Larger instances (4+ vCPU) improve concurrent query handling

**Cost Optimization**:
- Use Fargate Spot for dev/staging (70% savings)
- Consider reserved instances for production EC2
- Monitor Anthropic API usage (primary cost driver)

#### Security Checklist

- [ ] API keys in AWS Secrets Manager (not environment variables in task definition)
- [ ] VPC with private subnets for application tier
- [ ] Security group: only port 8098 from ALB
- [ ] IAM role with minimal permissions (Secrets Manager read, CloudWatch Logs write)
- [ ] HTTPS termination at ALB (ACM certificate)
- [ ] Consider WAF for rate limiting and bot protection
- [ ] No authentication currently built-in - add ALB auth or API Gateway if needed

#### Deployment Workflow

**Step-by-Step First Deployment**:

```bash
# 1. Get the data from your current server
ssh ubuntu@your-current-server
cd /home/ubuntu/tde
tar -czf tde-data.tar.gz data/
exit

scp ubuntu@your-current-server:/home/ubuntu/tde/tde-data.tar.gz .
tar -xzf tde-data.tar.gz  # Extract locally

# 2. Upload data to S3 (one-time setup)
aws s3 mb s3://your-company-tde-data
aws s3 sync ./data/ s3://your-company-tde-data/

# 3. Build Docker image with init script
# (Use Dockerfile that downloads from S3 on startup - see Data Acquisition above)
docker build -t tde-api:v2.0.0 .

# 4. Test locally with data volume
docker run -v $(pwd)/data:/app/data \
  -e ANTHROPIC_API_KEY=your_key \
  -p 8098:8098 \
  tde-api:v2.0.0

# 5. Verify it works
curl http://localhost:8098/health

# 6. Push to ECR
aws ecr create-repository --repository-name tde-api
docker tag tde-api:v2.0.0 <account>.dkr.ecr.<region>.amazonaws.com/tde-api:v2.0.0
docker push <account>.dkr.ecr.<region>.amazonaws.com/tde-api:v2.0.0

# 7. Deploy to ECS/Fargate with S3 data sync or EFS mount
```

**Ongoing CI/CD** (after initial setup):
```bash
1. Build Docker image
2. Run smoke tests: docker run <image> ./scripts/smoke_test.sh
3. Push to ECR
4. Update ECS task definition with new image
5. Deploy with rolling update strategy
6. Monitor health checks and rollback if needed
```

**Quick Validation**:
```bash
# After deployment
curl https://your-api.com/health
curl -X POST https://your-api.com/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "test deployment"}'
```

#### Known Issues for Deployment

1. **Rate Limits**: Complex queries can exceed Anthropic's 100K tokens/min limit
   - Mitigation: Implement request queuing or rate limiting at ALB

2. **Cold Start**: First query after deployment takes 5-10s (MCP client initialization)
   - Mitigation: Health check warmup after deployment

3. **Large Responses**: Some queries generate large GeoJSON files (>10MB)
   - Mitigation: Ensure adequate network bandwidth and timeouts

4. **Dataset Availability**: Missing datasets cause server warnings but don't crash
   - Mitigation: Verify `data/` directory contents match `data/README.md`

See `CHANGELOG.md` and `CLAUDE.md` for full issue tracking.

---

## 📊 Data Sources

### Current Datasets (~2.8GB total)

- **Climate Policy Knowledge Graph**: 1,325 concepts, 6,813 passages from CPR
- **Solar Facilities**: 8,319 facilities, 124.9 GW capacity (TransitionZero)
- **LSE Climate Policies**: NDCs, governance, subnational policies
- **GIST (IPCC Data)**: Chapters 11 & 12 processed summaries
- **Deforestation Data**: Spatial polygons and analysis (2GB)
- **Extreme Heat**: Heat index data for Brazil
- **Brazilian Admin Boundaries**: Municipal/state boundaries

See `data/README.md` for full dataset documentation and regeneration instructions.

## 🧪 Development & Testing

```bash
# Run with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8098

# Run smoke tests (validates v2 architecture)
./scripts/smoke_test.sh

# Test specific servers
python test_scripts/test_cpr_server_v2.py
python test_scripts/test_lse_semantic_index.py

# Inspect datasets
python scripts/inspect_solar_db.py
python scripts/inspect_lse.py
```

## 📈 Performance

**Query Response Times**:
- **Simple queries**: 3-5 seconds
- **Geographic queries**: 10-15 seconds
- **Complex multi-dataset queries**: 15-20 seconds
- **Cached featured queries**: ~100ms time to first token (50-100x faster)

**System Performance**:
- **V2 improvement**: 5-10x faster time-to-first-token vs v1
- **Cold start**: ~5-10s (MCP client initialization)
- **Warm start**: <1s (singleton client pre-connected)
- **Session overhead**: ~20ms (lookup and context retrieval)
- **Concurrent capacity**: ~10 concurrent queries per instance

**Optimization Features**:
- FastMCP singleton client (startup warmup)
- Stream cache for featured queries
- Lazy artifact generation with disk caching
- Multi-turn context limited to 2 turns (configurable)

## 🗂️ Repository Structure

```
tde/
├── README.md                    # This file - project overview
├── CHANGELOG.md                # Version history and release notes
├── CLAUDE.md                   # Development notes and known issues
├── requirements.txt            # Python dependencies (v2 cleaned)
├── .env.example                # Environment variable template
├── Dockerfile                  # Container build configuration
│
├── api_server.py               # FastAPI server (1,600 lines)
│                                # • 7 endpoints, session store
│                                # • Conversation logging, stream cache
│                                # • Dynamic artifact serving
│
├── stream_cache_manager.py     # Cache recording & replay (200 lines)
├── kg_embed_generator.py       # Knowledge graph visualizations (700 lines)
├── kg_visualization_server.py  # KG embed server (optional, 900 lines)
│
├── conversation_logs.csv       # Analytics logging (generated)
│
├── mcp/
│   ├── mcp_chat_v2.py          # V2 orchestrator (2,000+ lines)
│   │                            # • Query enrichment, server planning
│   │                            # • Fact ordering, narrative synthesis
│   │                            # • Citation registry, streaming
│   ├── contracts_v2.py         # Response contracts (RunQueryResponse)
│   ├── url_utils.py            # URL handling utilities
│   └── servers_v2/             # 11 MCP servers (~18,500 lines total)
│       ├── base.py             # Base classes and mixins
│       ├── cpr_server_v2.py    # Climate Policy Knowledge Graph
│       ├── solar_server_v2.py  # Global solar facilities (SQLite)
│       ├── solar_clay_server_v2.py  # TZ-SAM analysis
│       ├── deforestation_server_v2.py  # PRODES/MapBiomas polygons
│       ├── lse_server_v2.py    # LSE climate policies
│       ├── gist_server_v2.py   # IPCC Chapters 11 & 12
│       ├── extreme_heat_server_v2.py  # Heat index data
│       ├── brazilian_admin_server_v2.py  # Admin boundaries
│       ├── wmo_cli_server_v2.py  # WMO climate index
│       ├── spa_server_v2.py    # Sectoral policy analysis
│       ├── mb_deforest_server_v2.py  # MapBiomas centroids
│       └── meta_server_v2.py   # Cross-dataset metadata
│
├── data/                       # Datasets (~2.8GB, not in repo)
│   ├── README.md               # Dataset documentation
│   ├── solar_facilities.db     # SQLite (44MB)
│   ├── deforestation/          # PRODES polygons (2GB)
│   ├── lse/                    # LSE policy data (1.1MB)
│   ├── gist/                   # IPCC data (11MB)
│   ├── heat_stress/            # Heat index (308MB)
│   └── ...                     # Additional datasets
│
├── static/                     # Generated artifacts (~30MB)
│   ├── README.md               # Static content guide
│   ├── maps/                   # GeoJSON files (8MB)
│   ├── images/                 # Gallery images (21MB)
│   ├── kg/                     # Knowledge graph embeds (92KB)
│   ├── stream_cache/           # Pre-recorded SSE events (168KB)
│   │   ├── brazil-solar-expansion.jsonl
│   │   ├── brazil-climate-goals.jsonl
│   │   └── brazil-climate-risks.jsonl
│   ├── meta/                   # Cross-dataset metadata (1.3MB)
│   └── featured_queries.json   # Gallery configuration
│
├── scripts/                    # Utilities and maintenance
│   ├── smoke_test.sh           # Pre-deployment verification
│   ├── record_featured_streams.py  # Cache generation
│   ├── inspect_*.py            # Dataset inspection tools
│   ├── precompute_*.py         # Dataset preprocessing
│   └── migrate_*.py            # Data migration scripts
│
├── test_scripts/               # Server-specific test scripts
│   └── test_*_server_v2.py     # Individual server tests
│
└── docs/                       # Technical documentation
    ├── API_GUIDE.md            # Complete API reference
    ├── DEPLOYMENT.md           # Production deployment guide
    ├── MULTI_TURN_CONVERSATION_API.md  # Conversation system docs
    ├── FEATURED_QUERIES_CACHE.md  # Cache system docs
    └── ...                     # Additional technical guides
```

## 🔍 Key Innovation: Automatic Dataset Discovery

Unlike traditional APIs that require explicit data requests, this system **automatically discovers and surfaces relevant datasets**:

**Traditional**: `"extreme weather"` → Only text response
**Our System**: `"extreme weather"` → Text + data + visualizations automatically

Achieved through AI reasoning + knowledge graph relationships + automatic tool discovery via the v2 orchestration layer.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Run `./scripts/smoke_test.sh` to verify
5. Update documentation
6. Submit a pull request

See the `docs/` directory for detailed technical documentation.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For questions, issues, or contributions:

- **Issues**: Create a GitHub issue
- **API Integration**: See [API_GUIDE.md](API_GUIDE.md)
- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Dataset Documentation**: See `data/README.md`
- **V1 → V2 Migration**: See `docs/V1_TO_V2_MIGRATION.md`

---

**Built with ❤️ for climate policy intelligence and research**

**Version 2.0.0** - October 2025
