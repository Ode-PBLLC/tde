# Static Files Directory

This directory is for domain-specific static files that will be served by the API server.

## Directory Structure

```
static/
├── images/          # Images for featured queries and visualizations
├── cache/           # Cached query responses (auto-generated)
├── maps/            # Generated map files (GeoJSON, etc.)
├── data/            # Public data files
└── docs/            # Static documentation files
```

## Usage

### Featured Query Images
Place images referenced in your `config/featured_queries.json` here:

```json
{
  "id": "example-query",
  "title": "Example Analysis",
  "image": "/static/images/example-analysis.jpg"
}
```

**Recommended image specifications:**
- Size: 1200x630 pixels
- Format: JPG or PNG
- Content: Relevant visualization or domain imagery

### Generated Files
The system will automatically create files in subdirectories:
- `cache/` - Cached query responses
- `maps/` - Generated GeoJSON and map files

### Public Data Files
Place any public data files that should be accessible via the API:
- CSV files for download
- Documentation files
- Sample datasets

## File Serving

Files in this directory are served at `/static/` URLs:
- `/static/images/chart.png` → `static/images/chart.png`
- `/static/data/sample.csv` → `static/data/sample.csv`

## Security Notes

- Only place files here that should be publicly accessible
- Do not store sensitive data or API keys
- Large files may impact API performance
- Consider using external storage for very large datasets

## Domain Examples

### Finance Domain
```
static/
├── images/
│   ├── portfolio-analysis.jpg
│   ├── market-trends.jpg
│   └── risk-assessment.png
└── data/
    ├── sample-portfolio.csv
    └── market-indices.json
```

### Environmental Domain  
```
static/
├── images/
│   ├── air-quality-map.jpg
│   ├── climate-trends.png
│   └── pollution-analysis.jpg
└── maps/
    ├── monitoring-stations.geojson
    └── pollution-zones.geojson
```

## Setup Instructions

1. Create subdirectories as needed for your domain
2. Add appropriate images for featured queries
3. Update `.gitignore` if needed to exclude generated files
4. Test file serving with: `curl http://localhost:8098/static/your-file.jpg`