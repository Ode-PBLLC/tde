import mapboxgl from 'mapbox-gl';
import { fullscreenToggle } from './fullscreenToggle';

export const createMap = (wrapper: any, module: any) => {
    mapboxgl.accessToken = !!location.hostname.match(/(localhost|\.lh|192\.168\.)/g) ? 'pk.eyJ1IjoibWlrb2xhai1odW5jd290IiwiYSI6ImNtMng3ZDV0djAxamoybHF6emU1MTh2ZmoifQ.AMKdpHxNeZWa3AMukQOWFQ' : 'pk.eyJ1IjoiaHVuY3dvdHkiLCJhIjoiY21hdjVjdWdjMDB4cDJzcjRpaXFjMTBncSJ9.r6VUFh6jrz47vuQbpJQUzw';

    const { geojson_url, legend, viewState } = module;

    const mapContainer = document.createElement('div');
    wrapper.appendChild(mapContainer);

    const padding = 1.5;

    const maxBounds: any = [
        [viewState.bounds.west - padding, viewState.bounds.south - padding],
        [viewState.bounds.east + padding, viewState.bounds.north + padding]
    ];

    const map = new mapboxgl.Map({
        container: mapContainer,
        style: 'mapbox://styles/mapbox/light-v11',
        center: viewState.center,
        zoom: viewState.zoom,
        projection: 'mercator',
        maxBounds
    });

    fullscreenToggle(mapContainer);

    map.on('load', async () => {
        map.resize();

        let geojson_data;

        try {
            const response = await fetch(geojson_url);

            console.log("geojson_url", geojson_url)

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! Status: ${response.status}. Response: ${errorText}`);
            }

            geojson_data = await response.json();

            console.log("geojson_data", geojson_data)
        } catch (error) {
            console.error("Error loading GeoJSON data:", error);
            return;
        }

        const countryColorMap: Record<string, string> = {};
        legend.items.forEach((item: any) => {
            countryColorMap[item.label.toLowerCase()] = item.color;
        });

        map.addSource('solar-facilities', {
            type: 'geojson',
            data: geojson_data
        });

        if (module.hasOwnProperty('geometry_type') && module.geometry_type === 'polygon') {
            map.addLayer({
                id: 'solar-facilities-fill',
                type: 'fill',
                source: 'solar-facilities',
                paint: {
                    'fill-color': [
                        'match',
                        ['downcase', ['string', ['get', 'country']]],
                        ...legend.items.flatMap((item: any) => [item.label.toLowerCase(), item.color]),
                        '#4CAF50'
                    ],
                    'fill-opacity': 0.6,
                    'fill-outline-color': '#FFFFFF'
                }
            });

            map.on('click', 'solar-facilities-fill', (e) => {
                if (e.features && e.features.length > 0) {
                    const feature = e.features[0];
                    let description = `<h3>${feature.properties?.name || 'Solar Facility'}</h3>`;

                    if (feature.properties?.capacity_mw) {
                        description += `<p>Capacity: ${feature.properties.capacity_mw} MW</p>`;
                    }

                    if (feature.properties?.country) {
                        description += `<p>Country: ${feature.properties.country}</p>`;
                    }

                    new mapboxgl.Popup()
                        .setLngLat(e.lngLat)
                        .setHTML(description)
                        .addTo(map);
                }
            });

            map.on('mouseenter', 'solar-facilities-fill', () => map.getCanvas().style.cursor = 'pointer');
            map.on('mouseleave', 'solar-facilities-fill', () => map.getCanvas().style.cursor = '');
        } else {

            map.addLayer({
                id: 'solar-facilities-points',
                type: 'circle',
                source: 'solar-facilities',
                paint: {
                    'circle-color': [
                        'match',
                        ['downcase', ['string', ['get', 'country']]],
                        ...legend.items.flatMap((item: any) => [item.label.toLowerCase(), item.color]),
                        '#4CAF50'
                    ],
                    'circle-radius': [
                        'interpolate',
                        ['linear'],
                        ['get', 'capacity_mw'],
                        0, 4, 500, 8, 2000, 12, 5000, 16, 10000, 20
                    ],
                    'circle-stroke-color': '#FFFFFF',
                    'circle-stroke-width': 1
                }
            });

            map.on('click', 'solar-facilities-points', (e) => {
                if (e.features && e.features.length > 0) {
                    const feature = e.features[0];
                    let description = `<h3>${feature.properties?.name || 'Solar Facility'}</h3>`;

                    if (feature.properties?.capacity_mw) {
                        description += `<p>Capacity: ${feature.properties.capacity_mw} MW</p>`;
                    }

                    if (feature.properties?.country) {
                        description += `<p>Country: ${feature.properties.country}</p>`;
                    }

                    new mapboxgl.Popup()
                        .setLngLat(e.lngLat)
                        .setHTML(description)
                        .addTo(map);
                }
            });

            map.on('mouseenter', 'solar-facilities-points', () => map.getCanvas().style.cursor = 'pointer');
            map.on('mouseleave', 'solar-facilities-points', () => map.getCanvas().style.cursor = '');
        }

    });

    const legendContainer = document.createElement('div');
    legendContainer.className = 'map-legend';
    const legendItemsHtml = legend.items.map((item: any) => `
        <li class="map-legend--item">
            <span class="map-legend--item--circle" style="background: ${item.color};"></span>
            <span class="map-legend--item--label">${item.label}</span>
            <span class="map-legend--item--description">(${item.description})</span>
        </li>
    `).join('');

    legendContainer.innerHTML = `<ul>${legendItemsHtml}</ul>`;
    mapContainer.appendChild(legendContainer);
};
