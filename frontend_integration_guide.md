# Frontend Integration Guide

## API Endpoints

### POST /query
Process climate policy queries and get structured responses.

```javascript
// Basic query
const response = await fetch('/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "Show me solar facilities in Brazil",
    include_thinking: false
  })
});

const data = await response.json();
```

### Response Format

```typescript
interface QueryResponse {
  query: string;
  modules: Module[];
  thinking_process?: string;
  metadata: {
    modules_count: number;
    has_maps: boolean;
    has_charts: boolean;
    has_tables: boolean;
  };
}

type Module = TextModule | ChartModule | TableModule | MapModule;
```

## Map Module with Embedded GeoJSON

The map module contains a complete GeoJSON FeatureCollection that you can directly use with any mapping library:

```json
{
  "type": "map",
  "mapType": "geojson",
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [-38.5014, -12.9734]  // [longitude, latitude]
        },
        "properties": {
          "facility_id": "BR_FAC_001",
          "capacity_mw": 2511.1,
          "country": "Brazil",
          "popup_title": "Large Solar Complex",
          "popup_content": "Capacity: 2,511.1 MW<br>Location: Bahia State",
          "marker_color": "#4CAF50",
          "marker_size": 20,
          "marker_opacity": 0.8
        }
      }
    ]
  },
  "viewState": {
    "center": [-51.9253, -14.235],
    "zoom": 6,
    "bounds": { "north": -12.97, "south": -19.92, "east": -38.50, "west": -43.93 }
  },
  "legend": {
    "title": "Solar Facilities", 
    "items": [{"label": "Brazil", "color": "#4CAF50"}]
  }
}
```

## Frontend Implementation Examples

### Leaflet.js
```javascript
function renderMapModule(mapModule) {
  const map = L.map('map').setView(
    [mapModule.viewState.center[1], mapModule.viewState.center[0]], 
    mapModule.viewState.zoom
  );
  
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
  
  // Add GeoJSON layer
  L.geoJSON(mapModule.geojson, {
    pointToLayer: (feature, latlng) => {
      const props = feature.properties;
      return L.circleMarker(latlng, {
        radius: props.marker_size,
        fillColor: props.marker_color,
        color: props.marker_color,
        fillOpacity: props.marker_opacity
      });
    },
    onEachFeature: (feature, layer) => {
      layer.bindPopup(`<b>${feature.properties.popup_title}</b><br>${feature.properties.popup_content}`);
    }
  }).addTo(map);
}
```

### Mapbox GL JS
```javascript
function renderMapModule(mapModule) {
  const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: mapModule.viewState.center,
    zoom: mapModule.viewState.zoom
  });
  
  map.on('load', () => {
    // Add GeoJSON source
    map.addSource('facilities', {
      type: 'geojson',
      data: mapModule.geojson
    });
    
    // Add circle layer
    map.addLayer({
      id: 'facilities-circle',
      type: 'circle',
      source: 'facilities',
      paint: {
        'circle-radius': ['get', 'marker_size'],
        'circle-color': ['get', 'marker_color'],
        'circle-opacity': ['get', 'marker_opacity']
      }
    });
    
    // Add popup on click
    map.on('click', 'facilities-circle', (e) => {
      const props = e.features[0].properties;
      new mapboxgl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(`<b>${props.popup_title}</b><br>${props.popup_content}`)
        .addTo(map);
    });
  });
}
```

### React + Deck.gl
```jsx
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { Map } from 'react-map-gl';

function MapComponent({ mapModule }) {
  const layers = [
    new ScatterplotLayer({
      id: 'facilities',
      data: mapModule.geojson.features,
      getPosition: d => d.geometry.coordinates,
      getRadius: d => d.properties.marker_size * 100,
      getFillColor: d => hexToRgb(d.properties.marker_color),
      pickable: true,
      onHover: ({object, x, y}) => {
        // Show tooltip
      }
    })
  ];

  return (
    <DeckGL
      initialViewState={{
        longitude: mapModule.viewState.center[0],
        latitude: mapModule.viewState.center[1], 
        zoom: mapModule.viewState.zoom
      }}
      controller={true}
      layers={layers}
    >
      <Map mapStyle="mapbox://styles/mapbox/light-v11" />
    </DeckGL>
  );
}
```

## Chart Module (Chart.js Compatible)

```javascript
function renderChartModule(chartModule) {
  const ctx = document.getElementById('chart').getContext('2d');
  new Chart(ctx, {
    type: chartModule.chartType, // 'bar', 'line', etc.
    data: chartModule.data,      // Ready to use with Chart.js
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' }
      }
    }
  });
}
```

## Complete Integration Example

```javascript
async function processQuery(query) {
  const response = await fetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      query: query,
      include_thinking: true 
    })
  });
  
  const data = await response.json();
  
  // Render each module
  data.modules.forEach(module => {
    switch(module.type) {
      case 'text':
        renderTextModule(module);
        break;
      case 'chart':
        renderChartModule(module);
        break;
      case 'table':
        renderTableModule(module);
        break;
      case 'map':
        renderMapModule(module);
        break;
    }
  });
  
  // Show thinking process if requested
  if (data.thinking_process) {
    document.getElementById('thinking').innerHTML = data.thinking_process;
  }
}
```

## Benefits for Your Frontend

✅ **Standard GeoJSON** - Works with every mapping library  
✅ **No coordinate conversion** - Ready to use  
✅ **Rich metadata** - Popup content, styling, legends included  
✅ **Performance optimized** - Pre-limited to 500 features  
✅ **Bounds included** - For automatic map fitting  
✅ **Thinking process** - Optional AI reasoning for transparency