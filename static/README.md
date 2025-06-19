# Static Content Management

This directory contains static content served by the API, including the pseudo-CMS system for featured queries.

## Featured Queries System

### Overview
The `/featured-queries` endpoint serves curated content for the frontend gallery. This allows you to maintain featured queries without changing backend code.

### Files
- `featured_queries.json` - Main content file
- `images/` - Directory for featured query images

### Updating Featured Queries

To add, modify, or remove featured queries:

1. **Edit `featured_queries.json`**
2. **Add corresponding images to `images/` directory**  
3. **Restart the API server** (or it will auto-reload in development)

### JSON Structure

```json
{
  "featured_queries": [
    {
      "id": "unique-slug",
      "title": "Display Title",
      "query": "Full query text that will be sent to /query/stream",
      "image": "/static/images/image-filename.jpg",
      "category": "Category Name",
      "description": "Brief description for frontend display"
    }
  ],
  "metadata": {
    "last_updated": "2024-01-15T10:00:00Z",
    "total_queries": 6,
    "categories": ["Category1", "Category2"]
  }
}
```

### Image Guidelines

- **Format**: JPG, PNG, or WebP
- **Size**: Recommended 400x300px or similar 4:3 aspect ratio
- **File naming**: Use kebab-case matching the query ID
- **Location**: Must be in `static/images/` directory
- **Reference**: Use path `/static/images/filename.jpg` in JSON

### Example Workflow

1. **Create new query**:
   ```json
   {
     "id": "new-analysis-topic",
     "title": "New Analysis Topic",
     "query": "Analyze something interesting about climate data...",
     "image": "/static/images/new-analysis-topic.jpg",
     "category": "Analysis",
     "description": "Description of what this analysis covers"
   }
   ```

2. **Add image**: Save `new-analysis-topic.jpg` in `images/` directory

3. **Update metadata**: Increment `total_queries` and add category if new

4. **Test**: Visit `http://localhost:8099/featured-queries` to verify

### Categories
Current categories in use:
- Environmental Risk
- Renewable Energy  
- Policy Analysis
- Financial Risk
- Emissions Analysis

Feel free to add new categories as needed.

### Error Handling
- If `featured_queries.json` is malformed, the endpoint returns a 500 error
- If file is missing, returns empty array with error message
- Individual missing images will show broken image in frontend

### Best Practices
- Keep query text focused and specific
- Use descriptive titles that indicate what the analysis will show
- Choose images that visually represent the topic
- Test queries in the main API before adding to featured list
- Keep descriptions concise (1-2 sentences)
- Use consistent category naming