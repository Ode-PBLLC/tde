# Changelog

All notable changes to the TDE (Transition Digital Engine) project.

## [2.0.0] - 2025-10-24

### Major Changes - V2 Architecture

**Consolidated to Single Orchestrator**
- Retired legacy orchestrators (`mcp_chat.py`, `mcp_chat_redo.py`, `mcp_chat_plan_execute.py`)
- Consolidated to `mcp_chat_v2.py` as the single orchestrator
- Removed all Streamlit UI code (11,650+ lines removed)
- API server now exclusively uses v2 architecture
- Bumped API version to 2.0.0

**Cleaned MCP Servers**
- Removed duplicate server files (`cpr_server_v2 copy.py`)
- Removed test files from servers_v2/ directory
- Moved server documentation to `docs/mcp_servers/`
- All 11 production servers verified to implement `RunQueryResponse` contract

**Dependencies & Configuration**
- Removed Streamlit and streamlit-folium dependencies
- Added FastAPI, uvicorn, aiohttp, pydantic to requirements
- Updated `.env.example` with clearer required vs optional variables
- Removed duplicate `requirements copy.txt`

**Testing & Tooling**
- Consolidated all test files into `tests/` directory
- Created `scripts/smoke_test.sh` for pre-deployment verification
- 15 automated checks for architecture integrity

**Documentation**
- Added comprehensive `data/README.md` documenting ~2.8GB of datasets
- Updated configuration documentation
- Consolidated scattered markdown into organized structure

### LSE Server Search Improvements

**Enhanced Search Capabilities**
- Implemented combined semantic + token search for better recall
- Added n-gram token matching alongside semantic search
- LSE server now set to "always_include" for all Brazil climate queries
- Search performance: <50ms typical query time

**Query Processing**
- Added `_tokenize()` method for text normalization
- Implemented `_token_search()` for n-gram matching
- Combined semantic and token results with deduplication
- Improved relevance scoring

**Server Descriptions**
- Enhanced LSE server description for better orchestrator prioritization
- Clarified coverage: NDC commitments, governance, subnational policies
- Updated capability descriptions for more accurate routing

See `LSE_ENERGY_QUERY_ANALYSIS.md` and `LSE_RUN_QUERY_FIX.md` for detailed analysis.

### Performance Optimizations

**FastMCP Client Singleton** (July 8, 2025)
- Implemented singleton pattern with FastAPI lifecycle hooks
- Global MCP client initialized once at startup
- All 5 servers pre-connected (kg, solar, gist, lse, formatter)
- Thread-safe singleton with `asyncio.Lock()`
- 5-10x faster time to first token

### Repository Cleanup

**Removed Generated Artifacts**
- Cleared 451 generated files (~1.5M lines)
- Added `.gitignore` rules for: `data/`, `static/kg/`, `static/maps/`, `static/cache/`, `logs/`
- Preserved directory structure for regeneration
- All generated assets now excluded from version control

**Data Management**
- Documented all datasets in `data/README.md`
- Categorized core vs optional datasets
- Listed regeneration scripts for each dataset
- Total data size: ~2.8GB (not pushed to repo)

### Known Issues

**Rate Limit Error with Complex Queries**
- Featured query "Water Stress Impacts on Financial Sector" exceeds rate limits
- Status: 429 (100,000 input tokens/minute limit)
- Workaround: Break into smaller queries or simplify scope
- Future: Implement query optimization and caching

See `CLAUDE.md` for full issue details.

---

## [1.0.0] - Previous Versions

See git history for earlier changes before v2 consolidation.
