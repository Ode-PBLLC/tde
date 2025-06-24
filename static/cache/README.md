# Featured Query Cache

This directory contains pre-computed responses for featured queries to improve loading performance.

## Structure
- Each cached response is stored as `{query-id}.json`
- Files include the full API response plus metadata about when the cache was generated

## Updating Cache
Run the cache generation script to update all cached responses:
```bash
python scripts/generate_featured_cache.py
```

## Cache Format
Each cached file contains:
- `cached_at`: ISO timestamp of when the cache was generated
- `query_id`: The ID from featured_queries.json
- `response`: The full API response from the /query endpoint