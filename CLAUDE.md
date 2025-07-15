# CLAUDE.md - Development Notes

This file contains important information for Claude Code and future development work.

## Known Issues

### Rate Limit Error with Financial Water Stress Query

**Date**: June 24, 2025
**Issue**: The featured query "Water Stress Impacts on Financial Sector" fails due to Anthropic API rate limits.

**Query that causes the issue**:
```
Analyze water stress exposure for major Brazilian banks and financial institutions. Show asset-level risk data, percentage of assets at risk, and how water stress correlates with financial performance.
```

**Error Details**:
- Status: 500 Internal Server Error
- Error Type: `anthropic.RateLimitError` (429)
- Message: Request would exceed rate limit of 100,000 input tokens per minute
- Organization ID: 913bc629-fd72-4ddd-ae40-36405c7177b5

**Why this happens**: 
This query likely triggers extensive searches across multiple MCP tools, generating a large context that exceeds the token limit.

**Potential Solutions**:
1. Simplify the query to request less comprehensive data
2. Break it into multiple smaller queries
3. Implement query optimization to reduce tool calls
4. Add rate limit handling with exponential backoff
5. Cache intermediate tool results to reduce repeated API calls

**Temporary Fix**: 
Replace this query with a simpler alternative that requests less comprehensive analysis.

## Performance Optimizations

### FastMCP Client Singleton Implementation (July 8, 2025)
**Status**: ✅ IMPLEMENTED

**Problem**: Every API request created new `MultiServerClient()` with 5 subprocess connections, causing 250-700ms cold-start overhead.

**Solution**: Implemented singleton pattern with FastAPI lifecycle hooks:
- Global MCP client initialized once at startup
- All 5 servers (kg, solar, gist, lse, formatter) pre-connected
- Thread-safe singleton with `asyncio.Lock()`
- Graceful startup/shutdown handling

**Performance Impact**: 5-10x faster time to first token

**Files Modified**:
- `mcp/mcp_chat.py` - Added singleton client functions
- `api_server.py` - Added startup/shutdown hooks

**Testing**: Use `curl -X POST http://localhost:8099/query/stream` to test streaming performance

## Configuration Notes

### API Ports
- Default API server port: 8098 (configured in api_server.py)
- KG visualization server port: 8100 (internal service only)
- Update `API_BASE_URL` in the following files when changing ports:
  - `streamlit_api_demo.py`
  - `scripts/generate_featured_cache.py`
  - Test scripts

### Cache System
- Featured queries cache location: `static/cache/`
- Cache generation script: `scripts/generate_featured_cache.py`
- Failed queries during cache generation are logged but don't block other queries