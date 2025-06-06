# Climate Policy API System

A comprehensive AI-powered API system that combines knowledge graphs, structured datasets, and intelligent reasoning to provide comprehensive responses about climate policy topics. The system automatically discovers and surfaces relevant data, policy documents, and visualizations based on user queries.

## 🌟 Key Features

- **🤖 Intelligent Query Processing**: Uses Claude Sonnet 4 for sophisticated reasoning and synthesis
- **📊 Automatic Dataset Discovery**: Proactively finds and surfaces relevant structured data
- **🗺️ Interactive Visualizations**: Generates maps, charts, and tables dynamically
- **🔗 Multi-Source Integration**: Combines policy documents, geographic data, and structured datasets
- **⚡ Real-Time Responses**: Fast API responses with comprehensive analysis
- **🔍 Debug-Friendly**: Complete transparency into AI reasoning and data processing

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

# Query example
curl -X POST http://localhost:8099/query \
  -H "Content-Type: application/json" \
  -d '{"query": "solar energy in Brazil"}'

# Debug example  
curl -X POST http://localhost:8099/thorough-response \
  -H "Content-Type: application/json" \
  -d '{"query": "extreme weather data"}'
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and component overview |
| [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) | Development setup and workflows |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API documentation with examples |
| [EXTENSION_EXAMPLES.md](EXTENSION_EXAMPLES.md) | Practical examples for extending the system |

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

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Structured responses for frontend |
| `/thorough-response` | POST | Complete raw MCP data for debugging |
| `/health` | GET | System health check |
| `/example-response` | GET | Sample response format |
| `/static/maps/{file}` | GET | Generated GeoJSON map files |

### Response Format

```json
{
  "query": "user query",
  "modules": [
    {
      "type": "text|map|table|chart",
      "heading": "Module Title",
      "data": "Module-specific content"
    }
  ],
  "metadata": {
    "modules_count": 3,
    "has_maps": true,
    "has_charts": false,
    "has_tables": true
  }
}
```

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

## 🛠️ Extending the System

### Adding New Data Sources

1. **Create MCP Server**: Implement new data access tools
2. **Link to Knowledge Graph**: Connect datasets to relevant concepts  
3. **Register in Orchestration**: Add server to query processing
4. **Update System Prompt**: Include new tools in AI instructions

### Example: Adding Climate Finance Data

```python
# 1. Create mcp/climate_finance_server.py
@app.tool()
def get_climate_finance_projects(country: str) -> list[TextContent]:
    # Your implementation here
    pass

# 2. Link in cpr_kg_server.py
G.add_edge("green finance", "CLIMATE_FINANCE_DATASET", type="HAS_DATASET_ABOUT")

# 3. Register in mcp_chat.py
await client.connect_to_server("finance", "mcp/climate_finance_server.py")
```

See [EXTENSION_EXAMPLES.md](EXTENSION_EXAMPLES.md) for detailed implementation examples.

## 🔍 Key Innovation: Automatic Dataset Discovery

Unlike traditional APIs that require explicit data requests, this system **automatically discovers and surfaces relevant datasets** for any concept:

**Traditional Approach**:
```
Query: "extreme weather" → Only text response
Query: "extreme weather data" → Text + data
```

**Our Approach**:
```
Query: "extreme weather" → Automatically includes text + data + visualizations
```

This is achieved through:
1. Enhanced AI system prompt mandating dataset discovery
2. Knowledge graph relationships linking concepts to datasets
3. Automatic tool calling based on available connections

## 🧪 Development & Testing

### Local Development
```bash
# Run with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8099

# Test endpoints
python test_integration.py

# Debug MCP servers individually
python mcp/cpr_kg_server.py
```

### Adding Tests
```python
# tests/test_new_feature.py
import pytest
from api_server import app

def test_new_endpoint():
    # Your test implementation
    pass
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

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for detailed contribution guidelines.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For questions, issues, or contributions:

- **Issues**: Create a GitHub issue
- **Documentation**: Check the `/docs` directory
- **Development**: See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md)

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