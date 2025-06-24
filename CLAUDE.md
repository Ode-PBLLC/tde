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

## Configuration Notes

### API Ports
- Default API server port: 8099 (configured in streamlit_api_demo.py)
- Update `API_BASE_URL` in the following files when changing ports:
  - `streamlit_api_demo.py`
  - `scripts/generate_featured_cache.py`
  - Test scripts

### Cache System
- Featured queries cache location: `static/cache/`
- Cache generation script: `scripts/generate_featured_cache.py`
- Failed queries during cache generation are logged but don't block other queries