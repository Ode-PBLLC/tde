# Featured Queries Cache System

This document describes the caching system for featured queries in the Climate Policy Radar API.

## Overview

The featured queries system provides a curated list of example queries that showcase the API's capabilities. To improve performance and user experience, we cache the API responses for these queries.

## Structure

```
static/
├── featured_queries.json    # Featured queries definitions
├── cache/                   # Cached API responses
│   ├── README.md
│   ├── brazil-oil-environmental-risks.json
│   ├── global-solar-capacity.json
│   └── ... (one file per featured query)
└── images/                  # Featured query preview images
    └── README.md
```

## Usage

### 1. Generating Cache

Run the cache generation script to pre-compute responses for all featured queries:

```bash
# Make sure the API server is running first
python api_server.py

# In another terminal, generate the cache
python scripts/generate_featured_cache.py
```

This script will:
- Read all featured queries from `static/featured_queries.json`
- Call the API for each query
- Save responses to `static/cache/{query-id}.json`
- Update metadata in the featured queries file

### 2. API Endpoints

#### Get Featured Queries List
```bash
# Without cached responses (lightweight)
GET /featured-queries

# With cached responses included
GET /featured-queries?include_cached=true
```

#### Get Specific Cached Query
```bash
GET /featured-queries/{query-id}/cached
```

Example:
```bash
curl http://localhost:8000/featured-queries/brazil-oil-environmental-risks/cached
```

### 3. Testing

Test the caching system:
```bash
python test_featured_cache.py
```

## Featured Queries

The system includes 10 featured queries:

1. **Brazilian Oil Companies Environmental Risk Analysis** - Environmental risk exposure of oil and gas companies
2. **Global Solar Capacity Analysis** - Solar facilities and capacity data across multiple countries
3. **Climate Policy Effectiveness by Country** - Cross-country comparison of climate policies
4. **Water Stress Impacts on Financial Sector** - Water stress risk assessment for banking
5. **Scope 3 Emissions Trends Analysis** - Time series analysis of corporate emissions
6. **Brazil's Renewable Energy Policy Framework** - Deep dive into renewable energy policies
7. **Mining Sector Climate Risks in Africa** - Climate vulnerability for mining operations
8. **Deforestation Policy Effectiveness in Amazon Countries** - Amazon deforestation prevention policies
9. **Global Carbon Pricing Mechanisms Comparison** - Overview of carbon pricing instruments
10. **Electric Vehicle Policy Adoption in Emerging Markets** - EV policy frameworks analysis

## Adding New Featured Queries

1. Edit `static/featured_queries.json` to add your query
2. Create a preview image and save it to `static/images/`
3. Run `python scripts/generate_featured_cache.py` to generate the cache
4. The new query will automatically appear in the API responses

## Cache Refresh

To refresh the cache (e.g., after data updates):

```bash
# Regenerate all caches
python scripts/generate_featured_cache.py

# Or set up a cron job for automatic updates
0 2 * * * cd /path/to/project && python scripts/generate_featured_cache.py
```

## Performance Benefits

- **Instant responses**: Cached queries return immediately without processing
- **Reduced server load**: No need to run complex queries repeatedly
- **Better UX**: Users can preview example outputs instantly
- **Scalability**: Serve thousands of requests without computational overhead

## Notes

- Cache files include timestamps for freshness tracking
- The cache directory is gitignored to avoid repository bloat
- Failed cache generations are logged but don't block other queries
- The system gracefully handles missing cache files