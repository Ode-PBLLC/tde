# Frontend Integration Notes

## Backend Changes Made (2025)

### 1. Fixed Mixed Content Error ✅
**Issue**: Frontend was receiving HTTP URLs (`http://3.222.23.240:8098`) for GeoJSON files when served over HTTPS
**Solution**: Updated `API_BASE_URL` environment variable to use `https://api.transitiondigital.org`
**Files Modified**:
- `/home/ubuntu/tde/restart_servers.sh` - Changed API_BASE_URL to HTTPS
- `/home/ubuntu/tde/mcp/response_formatter.py` - Updated default fallback URL

### 2. CORS Configuration ✅
**Status**: Already properly configured
- Backend allows all origins (`*`) in CORS middleware
- If frontend still sees CORS errors for `/featured-queries`, check:
  - Frontend is calling the correct endpoint: `https://api.transitiondigital.org/featured-queries`
  - No proxy/CDN is blocking the requests

### 3. Facts Summary Event Type
**Issue**: Frontend shows warning "Unknown event type: facts_summary"
**Explanation**: This is an informational streaming event sent during query processing
**Frontend Action Required**: 
```javascript
// In Chat.ts handleStreamEvent function, add:
case 'facts_summary':
  // Optional: Use this data to show progress to user
  // data contains: { phase: 1|2, total: number, by_server: object }
  console.log('Facts collected:', event.data);
  break;
```

### 4. Missing Mapbox CSS
**Issue**: "This page appears to be missing CSS declarations for Mapbox GL JS"
**Frontend Action Required**:
Add to your HTML head or import in your main CSS:
```html
<link href='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css' rel='stylesheet' />
```

## API Endpoints Available

### Core Endpoints
- `POST /query` - Process query with session management
- `POST /query/stream` - Stream query with real-time updates
- `GET /featured-queries` - Get featured queries list
- `GET /featured-queries/{query_id}/cached` - Get cached response for featured query
- `GET /static/maps/{filename}` - Get GeoJSON map data

### Static Files
- GeoJSON files are served from: `https://api.transitiondigital.org/static/maps/`
- Featured queries cache: `https://api.transitiondigital.org/static/cache/`

## Environment Variables
The backend uses `API_BASE_URL` environment variable to generate absolute URLs for resources.
Current value: `https://api.transitiondigital.org`

## Session Management
- Sessions use conversation_id for multi-turn conversations
- Session TTL: 20 minutes of inactivity
- Include `conversation_id` in request body to continue a conversation

## Streaming Response Format
The `/query/stream` endpoint sends Server-Sent Events with these types:
- `thinking` - LLM thinking process
- `facts_summary` - Summary of facts collected (can be ignored or used for progress)
- `response` - Partial response text
- `complete` - Final complete response with all modules

## Notes for Frontend Developers
1. Always use HTTPS URLs when connecting to the API
2. The `facts_summary` event is optional - you can safely ignore it or use it for progress indication
3. Ensure Mapbox CSS is loaded before initializing maps
4. All GeoJSON URLs now use HTTPS to prevent mixed content errors