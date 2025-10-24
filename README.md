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
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────────┐
│              MCP V2 Orchestration Layer                        │
│                  (mcp_chat_v2.py)                              │
├─────────────────────────────────────────────────────────────────┤
│  • FastMCP singleton client (5-10x faster TTFT)                │
│  • AI reasoning with Claude Sonnet 4                           │
│  • Automatic server selection                                  │
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

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query/stream` | POST | **Primary endpoint** - Streaming analysis with progress |
| `/query` | POST | Synchronous analysis (simple integration) |
| `/featured-queries` | GET | Curated queries for frontend gallery |
| `/health` | GET | System health check |
| `/kg/{kg_id}` | GET | Knowledge graph visualization |

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

## 🐳 Docker & AWS Deployment

### For DevOps/Infrastructure Engineers

This section contains essential information for containerizing and deploying the TDE API to AWS.

#### Architecture Requirements

**Runtime**:
- Python 3.11+ base image recommended (`python:3.11-slim` or `python:3.11-alpine`)
- Multi-stage build recommended to minimize image size

**Resource Requirements**:
- **CPU**: 2+ vCPUs recommended (handles concurrent MCP server processes)
- **Memory**: 8GB minimum, 16GB recommended (for semantic search indexes)
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

**Data Volume Considerations**:
- The `data/` directory (~2.8GB) is **not** in the repo (gitignored)
- Options:
  1. **Bake into image**: Copy datasets during build (increases image size to ~4GB)
  2. **EFS/EBS volume**: Mount `data/` from persistent storage
  3. **S3 + init**: Download datasets on container startup (slower start, smaller image)
- See `data/README.md` for regeneration scripts if datasets are missing

#### AWS Deployment Options

**Option 1: ECS Fargate** (Recommended for simplicity)
```yaml
# Task Definition essentials:
CPU: 2048 (2 vCPU)
Memory: 8192 (8 GB)
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
- Use `t3.large` or larger (2 vCPU, 8GB RAM)
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

**Recommended CI/CD**:
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

- **Simple queries**: 3-5 seconds
- **Geographic queries**: 10-15 seconds
- **Complex multi-dataset queries**: 15-20 seconds
- **V2 improvement**: 5-10x faster time-to-first-token vs v1
- **Concurrent capacity**: ~10 concurrent queries per instance

## 🗂️ Repository Structure

```
tde/
├── README.md                 # This file - project overview
├── CHANGELOG.md             # Version history and release notes
├── API_GUIDE.md             # Complete API documentation
├── DEPLOYMENT.md            # Production deployment guide
├── CLAUDE.md                # Development notes and known issues
├── requirements.txt         # Python dependencies (v2 cleaned)
├── .env.example             # Environment variable template
├── api_server.py           # Main FastAPI application
├── kg_embed_generator.py   # Knowledge graph visualization utilities
├── mcp/
│   ├── mcp_chat_v2.py      # V2 orchestrator (single source of truth)
│   ├── contracts_v2.py     # Response contracts
│   └── servers_v2/         # 11 MCP servers
├── data/                   # Datasets (~2.8GB, not in repo)
│   └── README.md           # Dataset documentation
├── static/                 # Generated assets (maps, charts, KGs)
├── scripts/                # Utilities and maintenance
│   ├── smoke_test.sh       # Pre-deployment verification
│   └── inspect_*.py        # Dataset inspection tools
├── tests/                  # Unit and integration tests
├── test_scripts/           # Server-specific test scripts
└── docs/                   # Technical documentation
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
