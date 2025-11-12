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

### Stream Cache URL Environment Handling

**Date**: November 11, 2025
**Status**: ✅ RESOLVED

**Issue**: Cached stream files contained environment-specific URLs that wouldn't work across dev/prod deployments.

**Problem Details**:
The cache files in `static/stream_cache/*.jsonl` contained hardcoded URLs from cache generation time:
- Initially: `http://localhost:8098/static/...` (broke in all deployments)
- Then: `https://dev-tde.sunship.one/static/...` (breaks if accessed from prod)

When caches are replayed, the frontend receives these hardcoded URLs instead of environment-appropriate URLs.

**Root Cause**:
1. Cache files store complete event payloads with URLs
2. URLs are generated during cache recording based on recording environment
3. Cache replay was sending exact cached content without URL adaptation
4. `API_BASE_URL` was only set inside `event_generator()`, not for cached responses

**Solution Implemented**:
Dynamic URL rewriting during cache replay:

1. **api_server.py**: Moved `API_BASE_URL` detection before cache check (lines 1311-1327)
   - Extracts base URL from request headers (`x-forwarded-host`, `x-forwarded-proto`)
   - Works for both proxied and direct requests
   - Passes `base_url` parameter to `stream_cache.replay_stream()`

2. **stream_cache_manager.py**: Added `_rewrite_urls_in_dict()` helper (lines 65-89)
   - Recursively walks event dictionaries/lists
   - Replaces `http://localhost:8098` → `{request_base_url}`
   - Also replaces `https://dev-tde.sunship.one` → `{request_base_url}`
   - Handles KG embeds, maps, and all static asset URLs

3. **stream_cache_manager.py**: Updated `replay_stream()` to accept `base_url` (lines 91-133)
   - Applies URL rewriting if `base_url` provided
   - Maintains backward compatibility (works without `base_url`)

**Benefits**:
- Single cache works across all environments (dev, prod, staging)
- URLs automatically adapt to request environment
- No need to regenerate caches per environment
- Works with any domain configuration

**Files Modified**:
- `api_server.py` - Moved API_BASE_URL detection, pass to cache replay
- `stream_cache_manager.py` - Added URL rewriting functionality

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

**Testing**: Use `curl -X POST http://localhost:8098/query/stream` to test streaming performance

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