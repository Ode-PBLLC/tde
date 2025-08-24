# Climate Policy Radar API

AI-powered API that provides comprehensive climate policy analysis with real-time data discovery, interactive visualizations, and intelligent reasoning.

## 🌟 Key Features

- **🤖 Intelligent Analysis**: Claude Sonnet 4 powered reasoning and synthesis
- **📊 Automatic Data Discovery**: Finds and surfaces relevant datasets automatically  
- **🗺️ Interactive Visualizations**: Real-time maps, charts, and tables
- **⚡ Streaming Responses**: Live progress indicators and results
- **🔗 Multi-Source Integration**: Policy documents + structured datasets + geographic data
- **📱 Frontend Ready**: JSON modules optimized for web applications

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda or venv for environment management
- Anthropic API key
- OpenAI API key (for embeddings)

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
echo "ANTHROPIC_API_KEY=your-anthropic-key" > .env
echo "OPENAI_API_KEY=your-openai-key" >> .env

# Start the API server
python api_server.py
```

The API will be available at `http://localhost:8099`

### Basic Usage

```bash
# Health check
curl http://localhost:8099/health

# Stream a query (recommended)
curl -X POST http://localhost:8099/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Brazilian oil companies environmental risks"}'

# Get featured queries for frontend gallery
curl http://localhost:8099/featured-queries
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[API_GUIDE.md](API_GUIDE.md)** | Complete developer guide with examples and integration code |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Local setup, production deployment, and maintenance |
| **[docs/](docs/)** | Detailed implementation guides and technical documentation |
| `static/README.md` | Content management for featured queries |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                          │
│                      (api_server.py)                           │
├─────────────────────────────────────────────────────────────────┤
│  • POST /query              - Structured responses             │
│  • POST /thorough-response  - Raw MCP data                     │
│  • GET  /health             - Health check                     │
│  • Static files at /static  - Generated maps/charts            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────────┐
│                   MCP Orchestration Layer                      │
│                     (mcp_chat.py)                              │
├─────────────────────────────────────────────────────────────────┤
│  • Multi-server client management                              │
│  • AI reasoning with Claude Sonnet 4                           │
│  • Automatic dataset discovery                                 │
│  • Response formatting and structuring                         │
└─────────┬───────────┬───────────┬───────────────────────────────┘
          │           │           │
          v           v           v
┌─────────────┐ ┌─────────────┐ ┌─────────────────────────────────┐
│   Knowledge │ │    Solar    │ │      Response Formatter        │
│    Graph    │ │ Facilities  │ │         Server                 │
│   Server    │ │   Server    │ │                                │
│             │ │             │ │ • Structures raw data          │
│ • 1,325     │ │ • 8,319     │ │ • Creates visualizations       │
│   concepts  │ │   facilities│ │ • Formats for frontend         │
│ • 6,813     │ │ • 4 countries│ │                                │
│   passages  │ │ • 124.9 GW  │ │                                │
└─────────────┘ └─────────────┘ └─────────────────────────────────┘
```

## 🔧 Core Components

### Knowledge Graph Server
- **1,325 climate concepts** with semantic relationships
- **6,813 labeled passages** from policy documents
- **200+ policy documents** from CCLW, UNFCCC, CPR, GCF
- Automatic concept-to-dataset linking

### Solar Facilities Server
- **8,319 solar facilities** across Brazil, India, South Africa, Vietnam
- **124.9 GW total capacity** with geographic coordinates
- Interactive map generation with GeoJSON output
- Capacity analysis and visualization tools

### Response Formatter
- Converts raw data into frontend-ready modules
- Supports text, maps, charts, and tables
- Consistent JSON structure for easy integration

## 💡 Example Queries

### Geographic Analysis
```json
{"query": "solar facilities in Brazil"}
```
**Returns**: Policy analysis + interactive map + facility data table

### Policy Research
```json
{"query": "climate legislation and adaptation policies"}
```
**Returns**: Policy document analysis + related datasets

### Data Discovery
```json
{"query": "extreme weather and show me data"}
```
**Returns**: Policy context + structured datasets + visualizations

### Cross-Referenced Intelligence
```json
{"query": "renewable energy investment in developing countries"}
```
**Returns**: Policy analysis + solar facility data + economic context

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query/stream` | POST | **Primary endpoint** - Streaming analysis with progress |
| `/query` | POST | Synchronous analysis (simple integration) |
| `/featured-queries` | GET | Curated queries for frontend gallery |
| `/health` | GET | System health check |

### Response Modules

The API returns structured **modules** ready for frontend rendering:

- **Text**: Analysis content with inline citations `^1,2^`
- **Charts**: Chart.js compatible data (bar, line, pie)  
- **Tables**: Structured data with columns and rows
- **Maps**: GeoJSON with interactive markers
- **Citations**: References table (always last)

## 🌐 Production Deployment

The system is currently deployed at:
**http://54.146.227.119:8099**

### Deployment Features
- AWS EC2 hosting with Ubuntu
- Conda environment management  
- Git-based deployment workflow
- Static file serving for maps/charts
- Cross-platform compatibility

## 📈 Performance

- **Simple queries**: 3-5 seconds
- **Geographic queries**: 10-15 seconds  
- **Complex multi-dataset queries**: 15-20 seconds
- **Dataset size**: 1,325 concepts + 8,319 facilities + 200+ documents

## 🔒 Security

- No authentication currently required
- CORS enabled for frontend integration
- API keys stored in environment variables
- No user data persistence
- Input validation and error sanitization

## 🔍 Key Innovation: Automatic Dataset Discovery

Unlike traditional APIs that require explicit data requests, this system **automatically discovers and surfaces relevant datasets**:

**Traditional**: `"extreme weather"` → Only text response  
**Our System**: `"extreme weather"` → Text + data + visualizations automatically

Achieved through AI reasoning + knowledge graph relationships + automatic tool discovery.

## 🗂️ Repository Structure

```
tde/
├── README.md                 # This file - project overview
├── API_GUIDE.md             # Complete API documentation
├── DEPLOYMENT.md            # Production deployment guide
├── CLAUDE.md                # Development notes and configuration
├── requirements.txt         # Python dependencies
├── api_server.py           # Main FastAPI application
├── kg_embed_generator.py   # Knowledge graph embedding utilities
├── kg_visualization_server.py # KG visualization server
├── mcp/                    # MCP integration layer
├── data/                   # Datasets (solar, climate, policy)
├── static/                 # Frontend assets (maps, charts, cache)
├── scripts/                # Essential utilities and maintenance
├── deploy/                 # Production deployment configurations
├── docs/                   # Detailed implementation documentation
├── knowledge-graph/        # Core knowledge graph system
└── geocode/               # Geographic data utilities
```

## 🧪 Development & Testing

```bash
# Run with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8099

# Test streaming endpoint
curl -X POST http://localhost:8099/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# Essential test scripts are in test_scripts/ directory
```

## 📊 Data Sources

### Current Datasets
- **Climate Policy Knowledge Graph**: 1,325 concepts, 6,813 passages
- **Solar Facilities**: 8,319 facilities from TransitionZero
- **Policy Documents**: CCLW, UNFCCC, CPR, GCF collections
- **Extreme Weather Demo**: Sample structured event data

### Potential Extensions
- **LSE Climate Data**: NDCs, institutions, subnational policies
- **Climate Finance**: Project funding and investment data
- **Real-time Weather**: Current conditions and alerts
- **Economic Analysis**: Cost-benefit calculations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Update documentation
5. Submit a pull request

See the `docs/` directory for detailed technical documentation and implementation guides.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For questions, issues, or contributions:

- **Issues**: Create a GitHub issue  
- **API Integration**: See [API_GUIDE.md](API_GUIDE.md)
- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Content Management**: See `static/README.md`

## 🎯 Future Roadmap

- [ ] Authentication and rate limiting
- [ ] Additional data source integrations
- [ ] Advanced visualization types
- [ ] Multi-language support
- [ ] Real-time data streaming
- [ ] Machine learning insights
- [ ] Collaborative features

---

**Built with ❤️ for climate policy intelligence and research**