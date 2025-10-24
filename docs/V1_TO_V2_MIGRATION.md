# V1 to V2 Migration Guide

This guide explains the changes between TDE v1 and v2, and how to migrate if you were using the v1 architecture.

## Overview of Changes

**TDE v2.0** is a major consolidation release that simplifies the architecture while improving performance.

### What Changed

**Removed (13,000+ lines)**:
- ❌ `mcp/mcp_chat.py` - Streamlit-based orchestrator
- ❌ `mcp/mcp_chat_redo.py` - 3-phase redo orchestrator
- ❌ `mcp/mcp_chat_plan_execute.py` - Plan-execute orchestrator
- ❌ All Streamlit UI code and dependencies

**Consolidated**:
- ✅ Single orchestrator: `mcp/mcp_chat_v2.py`
- ✅ Clean contracts: All servers implement `RunQueryResponse`
- ✅ Faster startup: Singleton MCP client (5-10x TTFT improvement)

**Improved**:
- 📚 Better documentation (CHANGELOG.md, updated README)
- 🧪 Automated testing (`scripts/smoke_test.sh`)
- 🐳 Docker/AWS deployment guide in README
- 📊 Dataset documentation in `data/README.md`

---

## API Compatibility

### ✅ Fully Compatible

The **public API remains unchanged**. If you were using the HTTP endpoints, no code changes are needed:

```bash
# These endpoints work exactly the same in v2
POST /query/stream
POST /query
GET /featured-queries
GET /health
```

**Response format is identical** - same JSON module structure (text, charts, tables, maps, citations).

### ⚠️ Breaking Changes

**If you were importing orchestrator functions directly:**

**v1 (multiple options)**:
```python
# Any of these worked in v1
from mcp.mcp_chat import process_chat_query
from mcp.mcp_chat_redo import process_chat_query
from mcp.mcp_chat_plan_execute import process_chat_query
```

**v2 (single option)**:
```python
# Only this works in v2
from mcp.mcp_chat_v2 import process_chat_query, stream_chat_query
```

**If you were using Streamlit UI:**

The Streamlit interface has been removed. Use the API endpoints instead:

```python
# v1: Streamlit UI
# No longer available

# v2: Use HTTP API
import requests
response = requests.post(
    "http://localhost:8098/query/stream",
    json={"query": "your query here"}
)
```

---

## Migration Steps

### Step 1: Update Dependencies

```bash
# Pull latest code
git pull origin main

# Reinstall dependencies (streamlit removed, fastapi added)
pip install -r requirements.txt
```

### Step 2: Update Environment Variables

Check `.env.example` for the latest required variables:

```bash
# Required (was optional in v1)
ANTHROPIC_API_KEY=your_key_here

# Optional (behavior unchanged)
OPENAI_API_KEY=your_key_here
API_BASE_URL=http://localhost:8098
```

### Step 3: Update Code (if importing directly)

**Before (v1)**:
```python
from mcp.mcp_chat_redo import MCPChatSystem

chat = MCPChatSystem()
result = await chat.process_query("your query")
```

**After (v2)**:
```python
from mcp.mcp_chat_v2 import process_chat_query

result = await process_chat_query(
    query="your query",
    session_id="optional-session-id"
)
```

### Step 4: Verify Installation

```bash
# Run automated smoke tests
./scripts/smoke_test.sh

# Should see:
# ✓ All 15 tests passed
```

### Step 5: Test Your Integration

```bash
# Start the server
python api_server.py

# Test a query
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

---

## Configuration Changes

### Environment Variables

| Variable | v1 | v2 | Notes |
|----------|----|----|-------|
| `ANTHROPIC_API_KEY` | Optional | **Required** | Now mandatory for v2 orchestrator |
| `ORCHESTRATOR` | Used to switch | **Removed** | Only v2 orchestrator exists |
| `API_PORT` | 8099 default | 8098 default | Port changed |

### Removed Configuration

- `ORCHESTRATOR` env var (no longer needed - only one orchestrator)
- Streamlit config files (no Streamlit UI)

---

## Performance Improvements

### Faster Time-to-First-Token

**v1**: Each request created new MCP client connections (250-700ms overhead)

**v2**: Singleton client pre-connects all servers at startup (5-10x faster)

```
v1: 500ms cold start + 3s query = 3.5s total
v2: 0ms cold start + 3s query = 3s total
```

### Cleaner Server Contracts

All 11 MCP servers now implement the standardized `RunQueryResponse` contract:

```python
from mcp.contracts_v2 import RunQueryResponse

class MyServer:
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        return RunQueryResponse(
            server="my-server",
            query=query,
            facts=[...],
            citations=[...],
            artifacts=[...]
        )
```

This ensures:
- Type safety and validation
- Consistent response structure
- Better error messages when contracts are violated

---

## Testing Changes

### Old Test Files

Test files were scattered in the root directory:

```
test_queries.py
test_streaming_api.py
test_deep_dive.py
...
```

### New Organization

All tests consolidated:

```
tests/                 # Main test directory
├── test_queries.py
├── test_streaming_api.py
└── ...

test_scripts/          # Server-specific tests
├── test_cpr_server_v2.py
├── test_lse_semantic_index.py
└── ...

scripts/smoke_test.sh  # Pre-deployment validation
```

### Running Tests

```bash
# Quick validation (recommended before deploying)
./scripts/smoke_test.sh

# Run specific tests
python tests/test_api_uses_mcp_v2.py
python test_scripts/test_cpr_server_v2.py

# With pytest (if installed)
pytest tests/
```

---

## Docker/Deployment Changes

### Updated Ports

- **v1**: Default port 8099
- **v2**: Default port 8098

Update your Docker configs, load balancers, and firewall rules accordingly.

### New Docker Guide

See the comprehensive **"Docker & AWS Deployment"** section in [README.md](../README.md) for:

- Sample Dockerfile (multi-stage build)
- ECS Fargate task definitions
- Health check configuration
- Resource requirements
- Security checklist

### Data Persistence

The `data/` directory (~2.8GB) is now fully documented in `data/README.md`:

- Core vs optional datasets
- Regeneration scripts
- Volume mount recommendations for Docker

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'mcp.mcp_chat_redo'"

**Cause**: Your code is importing from a removed orchestrator.

**Fix**: Update imports to use `mcp.mcp_chat_v2`:

```python
# Old
from mcp.mcp_chat_redo import process_chat_query

# New
from mcp.mcp_chat_v2 import process_chat_query
```

### "ModuleNotFoundError: No module named 'streamlit'"

**Cause**: Streamlit was removed from dependencies.

**Fix**: Reinstall dependencies:

```bash
pip install -r requirements.txt
```

If you have old cached packages:

```bash
pip uninstall streamlit streamlit-folium
pip install -r requirements.txt
```

### "Port 8099 connection refused"

**Cause**: Default port changed from 8099 to 8098 in v2.

**Fix**: Update your client code or set custom port:

```bash
API_PORT=8099 python api_server.py
```

### Server starts but queries fail with "no module named X"

**Cause**: Missing dependencies after upgrade.

**Fix**: Clean reinstall:

```bash
pip freeze > old_requirements.txt
pip uninstall -r old_requirements.txt -y
pip install -r requirements.txt
```

---

## Rollback (if needed)

If you need to roll back to v1:

```bash
# Checkout the commit before v2 upgrade
git checkout <commit-before-v2>

# Reinstall v1 dependencies
pip install -r requirements.txt

# Restart server (port 8099)
python api_server.py
```

To find the restore point:

```bash
git log --oneline | grep "v2"
# Look for commit just before v2 changes
```

---

## Getting Help

- **Migration issues**: Create a GitHub issue with "v2 migration" tag
- **API questions**: See [API_GUIDE.md](../API_GUIDE.md)
- **Deployment**: See [README.md](../README.md) Docker section
- **v2 changes**: See [CHANGELOG.md](../CHANGELOG.md)

---

## Summary: Should You Migrate?

✅ **Yes, migrate to v2 if**:
- You're using the HTTP API endpoints (no code changes needed)
- You want faster response times (5-10x TTFT improvement)
- You want better testing and documentation
- You're deploying to production (cleaner, more maintainable)

⚠️ **Stay on v1 temporarily if**:
- You're using Streamlit UI (need to migrate to API first)
- You're importing orchestrator internals directly (need code changes)
- You're mid-deployment and can't test v2 immediately

**Recommendation**: Migrate to v2. It's faster, cleaner, and better documented. The public API is 100% compatible.
