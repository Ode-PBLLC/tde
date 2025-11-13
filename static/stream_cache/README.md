# Stream Cache

This directory contains pre-recorded SSE event streams for featured queries.

## How It Works

When a user sends one of the 3 featured queries (exact string match), the API server:
1. Checks if a `.jsonl` cache file exists for that query
2. If found, replays the cached events with original timing
3. Client receives identical stream as if it was processed live
4. **Caching is completely transparent to the client**

## Files

- `brazil-solar-expansion.jsonl` - "How fast is Brazil's solar energy capacity expanding..."
- `brazil-climate-goals.jsonl` - "How is Brazil currently performing on its national climate goals..."
- `brazil-climate-risks.jsonl` - "What are the main climate risks facing different regions of Brazil..."

Each `.jsonl` file contains one JSON event per line with:
- `type`: Event type (thinking, tool_call, content, complete, etc.)
- `data`: Event payload
- `timestamp_ms`: Timing offset from start (for replay)
- `recorded_at`: When the event was recorded

## Generating Caches

To record/update the cache files:

```bash
# Make sure API server is running
python api_server.py

# In another terminal, run the recorder
python scripts/record_featured_streams.py
```

This will:
- Send each featured query to the live API
- Record all SSE events to `.jsonl` files
- Preserve timing for realistic replay

## Testing

Test a cached query replay:

```bash
# Test replay locally
python stream_cache_manager.py replay brazil-solar-expansion

# Test via API
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How fast is Brazil'\''s solar energy capacity expanding, and how does this align with national climate targets and land-use priorities?"}'
```

## Performance

- **Cached queries**: ~100ms time to first token
- **Normal queries**: ~2-5s time to first token
- **50-100x faster** for featured queries

## Notes

- Cache is exact string match only (no fuzzy matching)
- Client cannot detect if response is cached
- Cache files are gitignored (too large)
- Regenerate caches after data updates
