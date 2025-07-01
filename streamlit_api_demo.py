#!/usr/bin/env python3
"""
Comprehensive Streamlit App for Climate Policy Radar API
Demonstrates all API functionality including text, charts, tables, and maps.
"""

import streamlit as st
import asyncio
import aiohttp
import json
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import re
from typing import Dict, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Climate Policy Radar API Demo",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
/* Citation superscript styling */
.citation {
    font-size: 0.8em;
    vertical-align: super;
    color: #1f77b4;
    font-weight: bold;
    cursor: pointer;
    text-decoration: none;
}

/* Module containers */
.module-container {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    background-color: #fafafa;
}

/* Thinking process styling */
.thinking-process {
    background-color: #f0f8ff;
    border-left: 4px solid #1f77b4;
    padding: 10px;
    margin: 5px 0;
    font-family: monospace;
    font-size: 0.9em;
    color: #333333;
    border-radius: 4px;
}

/* Citation table styling */
.citation-table {
    background-color: #fff8dc;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
}

/* Featured query cards */
.featured-query-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f9f9f9;
    cursor: pointer;
    transition: all 0.3s ease;
}

.featured-query-card:hover {
    border-color: #1f77b4;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Gallery card */
.gallery-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.3s ease;
}

.gallery-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
}

.gallery-card h4 {
    color: #1a1a1a;
    margin-bottom: 10px;
}

.gallery-card p {
    color: #4a4a4a;
    font-size: 0.9em;
    flex-grow: 1;
}

.gallery-card .category-badge {
    background-color: #1f77b4;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    display: inline-block;
}

/* Status indicators */
.status-thinking {
    color: #ff6600;
    font-weight: bold;
}

.status-complete {
    color: #28a745;
    font-weight: bold;
}

.status-error {
    color: #dc3545;
    font-weight: bold;
}

/* Table styling */
.ranking-table {
    background-color: #f0f8ff;
}

.comparison-table {
    background-color: #f0fff0;
}

.trend-table {
    background-color: #fffacd;
}

/* Map container */
.map-container {
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    background-color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8098"

def format_text_with_citations(text: str) -> str:
    """Convert citation markers to HTML superscript links."""
    citation_pattern = r'\^([0-9,]+)\^'
    
    def citation_replacer(match):
        citation_nums = match.group(1)
        return f'<span class="citation" title="See reference {citation_nums}">{citation_nums}</span>'
    
    return re.sub(citation_pattern, citation_replacer, text)

def create_plotly_chart(module: Dict[str, Any]) -> Optional[go.Figure]:
    """Create a Plotly chart from module data."""
    chart_type = module.get("chartType", "bar")
    data = module.get("data", {})
    heading = module.get("heading", "Chart")
    
    if not data or "labels" not in data or "datasets" not in data:
        return None
    
    labels = data["labels"]
    datasets = data["datasets"]
    
    if not datasets:
        return None
    
    try:
        if chart_type == "bar":
            fig = go.Figure()
            for dataset in datasets:
                fig.add_trace(go.Bar(
                    x=labels,
                    y=dataset["data"],
                    name=dataset.get("label", ""),
                    marker_color=dataset.get("backgroundColor", "#1f77b4")
                ))
            fig.update_layout(
                title=heading,
                xaxis_title="",
                yaxis_title=datasets[0].get("label", "Value") if datasets else "Value",
                showlegend=len(datasets) > 1
            )
            
        elif chart_type == "line":
            fig = go.Figure()
            for dataset in datasets:
                fig.add_trace(go.Scatter(
                    x=labels,
                    y=dataset["data"],
                    mode='lines+markers',
                    name=dataset.get("label", ""),
                    line=dict(color=dataset.get("borderColor", "#ff6384"))
                ))
            fig.update_layout(
                title=heading,
                xaxis_title="",
                yaxis_title="Value",
                showlegend=len(datasets) > 1
            )
            
        elif chart_type == "pie":
            dataset = datasets[0]  # Pie charts typically have one dataset
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=dataset["data"],
                marker_colors=dataset.get("backgroundColor", None)
            )])
            fig.update_layout(title=heading)
            
        else:
            return None
            
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def create_map_from_geojson(module: Dict[str, Any]) -> Optional[folium.Map]:
    """Create a Folium map from GeoJSON data."""
    try:
        # Get map configuration
        view_state = module.get("viewState", {})
        center = view_state.get("center", [0, 0])
        zoom = view_state.get("zoom", 2)
        
        # Create base map
        m = folium.Map(
            location=[center[1], center[0]],  # Folium uses [lat, lon]
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Handle different map data formats
        if "geojson_url" in module:
            # Load GeoJSON from URL
            geojson_url = module["geojson_url"]
            
            try:
                # Fetch GeoJSON data from the API
                import requests
                full_url = f"{API_BASE_URL}{geojson_url}"
                response = requests.get(full_url, timeout=10)
                
                if response.status_code == 200:
                    geojson_data = response.json()
                    
                    # Add GeoJSON layer with popup information
                    folium.GeoJson(
                        geojson_data,
                        name="Solar Facilities",
                        style_function=lambda feature: {
                            'fillColor': feature['properties'].get('marker_color', '#4CAF50'),
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.7,
                        },
                        popup=folium.GeoJsonPopup(
                            fields=['name', 'country', 'capacity_mw', 'technology'],
                            aliases=['Name:', 'Country:', 'Capacity (MW):', 'Technology:'],
                            localize=True
                        )
                    ).add_to(m)
                else:
                    # Fallback to note if GeoJSON can't be fetched
                    folium.Marker(
                        [center[1], center[0]],
                        popup=f"GeoJSON data available at: {geojson_url} (Status: {response.status_code})",
                        icon=folium.Icon(color='orange', icon='exclamation-sign')
                    ).add_to(m)
                    
            except Exception as e:
                # Fallback to note if there's an error
                folium.Marker(
                    [center[1], center[0]],
                    popup=f"Error loading GeoJSON: {str(e)}",
                    icon=folium.Icon(color='red', icon='exclamation-sign')
                ).add_to(m)
            
        elif "geojson" in module:
            # Direct GeoJSON data
            geojson_data = module["geojson"]
            
            # Add GeoJSON layer
            folium.GeoJson(
                geojson_data,
                name="Solar Facilities",
                style_function=lambda feature: {
                    'fillColor': feature['properties'].get('marker_color', '#4CAF50'),
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7,
                }
            ).add_to(m)
            
        elif "markers" in module:
            # Legacy marker format
            markers = module["markers"]
            for marker in markers[:100]:  # Limit markers for performance
                folium.Marker(
                    [marker["latitude"], marker["longitude"]],
                    popup=folium.Popup(
                        f"<b>{marker['popup']['title']}</b><br>{marker['popup']['content']}",
                        max_width=200
                    ),
                    icon=folium.Icon(
                        color='green' if marker.get('country', '').lower() == 'brazil' else 'blue',
                        icon='solar-panel' if 'solar' in marker.get('popup', {}).get('title', '').lower() else 'info-sign'
                    )
                ).add_to(m)
        
        # Add legend if available
        legend = module.get("legend", {})
        if legend and "items" in legend:
            legend_html = f'''
            <div style="position: fixed; 
                        top: 10px; right: 10px; width: 200px; height: auto; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p style="margin: 0; font-weight: bold;">{legend.get("title", "Legend")}</p>
            '''
            for item in legend["items"]:
                legend_html += f'<p style="margin: 5px 0;"><span style="color: {item["color"]};">●</span> {item["label"]}</p>'
            legend_html += '</div>'
            m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
        
    except Exception as e:
        st.error(f"Error creating map: {e}")
        return None

def display_module(module: Dict[str, Any], module_index: int):
    """Display a single module with appropriate formatting."""
    module_type = module.get("type", "unknown")
    heading = module.get("heading", f"Module {module_index}")
    
    if module_type == "text":
        st.markdown(f"### {heading}")
        texts = module.get("texts", [])
        for text in texts:
            formatted_text = format_text_with_citations(text)
            st.markdown(formatted_text, unsafe_allow_html=True)
            
    elif module_type == "chart":
        fig = create_plotly_chart(module)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"📊 Chart visualization: {module.get('chartType', 'unknown')} chart")
            
    elif module_type == "map":
        st.markdown(f"### {heading}")
        map_obj = create_map_from_geojson(module)
        if map_obj:
            st_folium(map_obj, width=None, height=400, returned_objects=[])
        else:
            st.info("🗺️ Map visualization")
        
        # Show metadata
        metadata = module.get("metadata", {})
        if metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Facilities", metadata.get("total_facilities", "N/A"))
            with col2:
                st.metric("Total Capacity (MW)", f"{metadata.get('total_capacity_mw', 0):,.0f}")
            with col3:
                st.metric("Countries", len(metadata.get("countries", [])))
                
    elif module_type in ["table", "ranking_table", "comparison_table", "trend_table", 
                         "summary_table", "detail_table", "geographic_table"]:
        st.markdown(f"### {heading}")
        columns = module.get("columns", [])
        rows = module.get("rows", [])
        
        if columns and rows:
            df = pd.DataFrame(rows, columns=columns)
            
            # Apply styling based on table type
            if module_type == "ranking_table":
                st.markdown('<div class="ranking-table">', unsafe_allow_html=True)
            elif module_type == "comparison_table":
                st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
            elif module_type == "trend_table":
                st.markdown('<div class="trend-table">', unsafe_allow_html=True)
                
            st.dataframe(df, use_container_width=True)
            
            if module_type in ["ranking_table", "comparison_table", "trend_table"]:
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Show metadata if available
            metadata = module.get("metadata", {})
            if metadata:
                with st.expander("📊 Table Metadata"):
                    st.json(metadata)
                    
    elif module_type == "numbered_citation_table":
        st.markdown("### 📚 References")
        columns = module.get("columns", [])
        rows = module.get("rows", [])
        
        if columns and rows:
            df = pd.DataFrame(rows, columns=columns)
            st.markdown('<div class="citation-table">', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    elif module_type == "source_table":
        st.markdown(f"### {heading}")
        columns = module.get("columns", [])
        rows = module.get("rows", [])
        
        if columns and rows:
            df = pd.DataFrame(rows, columns=columns)
            st.dataframe(df, use_container_width=True)
            
    else:
        st.info(f"Module type: {module_type}")
        st.json(module)

async def fetch_featured_queries(include_cached=True):
    """Fetch featured queries from the API."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{API_BASE_URL}/featured-queries"
            if include_cached:
                url += "?include_cached=true"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
    except Exception as e:
        st.error(f"Error fetching featured queries: {e}")
    return {"featured_queries": [], "metadata": {}}

async def run_query(query: str, include_thinking: bool = False):
    """Run a query against the API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/query",
                json={"query": query, "include_thinking": include_thinking}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    st.error(f"API Error: {error_text}")
                    return None
    except Exception as e:
        st.error(f"Error running query: {e}")
        return None

async def run_streaming_query(query: str, thinking_placeholder, modules_placeholder):
    """Run a streaming query and update UI in real-time."""
    thinking_events = []
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/query/stream",
                json={"query": query}
            ) as response:
                async for line in response.content:
                    if line:
                        text = line.decode('utf-8').strip()
                        if text.startswith('data: '):
                            event_data = text[6:]
                            try:
                                event = json.loads(event_data)
                                event_type = event.get("type", "unknown")
                                data = event.get("data", {})
                                
                                if event_type in ["thinking", "tool_call", "tool_result"]:
                                    thinking_events.append(event)
                                    
                                    # Update thinking display
                                    with thinking_placeholder.container():
                                        st.markdown("### 🧠 AI Thinking Process")
                                        
                                        for evt in thinking_events[-10:]:  # Show last 10 events
                                            t_type = evt.get("type", "")
                                            t_data = evt.get("data", {})
                                            
                                            if t_type == "thinking":
                                                message = t_data.get("message", "")
                                                st.markdown(f'<div class="thinking-process"><span class="status-thinking">🤔</span> {message}</div>', 
                                                           unsafe_allow_html=True)
                                            elif t_type == "tool_call":
                                                tool = t_data.get("tool", "")
                                                st.markdown(f'<div class="thinking-process"><span class="status-thinking">🔧</span> Calling {tool}...</div>', 
                                                           unsafe_allow_html=True)
                                            elif t_type == "tool_result":
                                                st.markdown(f'<div class="thinking-process"><span class="status-complete">✅</span> Tool completed</div>', 
                                                           unsafe_allow_html=True)
                                
                                elif event_type == "complete":
                                    # Final response with modules
                                    modules = data.get("modules", [])
                                    metadata = data.get("metadata", {})
                                    
                                    # Update modules display
                                    with modules_placeholder.container():
                                        st.markdown("### 📄 Response")
                                        
                                        # Display metadata
                                        if metadata:
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Modules", metadata.get("modules_count", 0))
                                            with col2:
                                                st.metric("Has Maps", "✅" if metadata.get("has_maps") else "❌")
                                            with col3:
                                                st.metric("Has Charts", "✅" if metadata.get("has_charts") else "❌")
                                            with col4:
                                                st.metric("Has Tables", "✅" if metadata.get("has_tables") else "❌")
                                        
                                        # Display modules
                                        for i, module in enumerate(modules):
                                            with st.expander(f"📄 {module.get('heading', f'Module {i+1}')}", expanded=True):
                                                display_module(module, i+1)
                                    
                                    break
                                    
                                elif event_type == "error":
                                    st.error(f"Error: {data.get('message', 'Unknown error')}")
                                    break
                                    
                            except json.JSONDecodeError:
                                pass
                                
    except Exception as e:
        st.error(f"Streaming error: {e}")
        import traceback
        st.code(traceback.format_exc())

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🌍 Climate Policy Radar API Demo")
    st.markdown("**Comprehensive demonstration of all API features including text, charts, tables, and maps**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Settings")
        
        use_streaming = st.checkbox("Use Streaming Mode", value=True, 
                                   help="Show real-time thinking process")
        include_thinking = st.checkbox("Include Thinking Process", value=False,
                                      help="Include thinking in non-streaming mode")
        
        st.markdown("---")
        
        st.markdown("### 📊 API Status")
        api_status = st.empty()
        
        # Check API status
        try:
            import requests
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                api_status.success("✅ API is running")
            else:
                api_status.error("❌ API is not responding")
        except:
            api_status.warning("⚠️ Cannot connect to API")
            st.info("Make sure the API server is running on port 8099")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Query Interface", "🌟 Featured Queries", "📚 Documentation"])
    
    with tab1:
        # Query input
        query = st.text_area(
            "Enter your query:",
            value="Show me solar capacity by country with maps and charts",
            height=100,
            help="Ask questions about climate policy, renewable energy, emissions, or environmental risks"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            run_button = st.button("🚀 Run Query", type="primary", use_container_width=True)
        
        # Example queries
        st.markdown("#### 💡 Example Queries")
        example_cols = st.columns(3)
        
        examples = [
            "Show me solar facilities in Brazil with maps",
            "Compare climate policies across major economies",
            "Analyze environmental risk for oil companies",
            "Show emissions trends with charts",
            "Display water stress data for financial sector",
            "Compare renewable energy policies"
        ]
        
        for i, example in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(f"📝 {example[:25]}...", key=f"ex_{i}", use_container_width=True):
                    st.session_state.query = example
                    st.rerun()
        
        # Update query from session state
        if 'query' in st.session_state:
            query = st.session_state.query
        
        # Results area
        if run_button and query.strip():
            st.markdown("---")
            
            if use_streaming:
                # Streaming mode
                thinking_placeholder = st.empty()
                modules_placeholder = st.empty()
                
                # Run streaming query
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run_streaming_query(query, thinking_placeholder, modules_placeholder))
                
            else:
                # Non-streaming mode
                with st.spinner("Processing query..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(run_query(query, include_thinking))
                    
                    if result:
                        # Display thinking process if included
                        if include_thinking and "thinking_process" in result:
                            with st.expander("🧠 AI Thinking Process", expanded=False):
                                st.text(result["thinking_process"])
                        
                        # Display modules
                        modules = result.get("modules", [])
                        metadata = result.get("metadata", {})
                        
                        # Display metadata
                        if metadata:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Modules", metadata.get("modules_count", 0))
                            with col2:
                                st.metric("Has Maps", "✅" if metadata.get("has_maps") else "❌")
                            with col3:
                                st.metric("Has Charts", "✅" if metadata.get("has_charts") else "❌")
                            with col4:
                                st.metric("Has Tables", "✅" if metadata.get("has_tables") else "❌")
                        
                        # Display modules
                        st.markdown("### 📄 Response")
                        for i, module in enumerate(modules):
                            with st.expander(f"📄 {module.get('heading', f'Module {i+1}')}", expanded=True):
                                display_module(module, i+1)
    
    with tab2:
        st.markdown("### 🌟 Featured Queries")
        st.markdown("Explore our curated collection of example queries showcasing the API's capabilities")
        
        # Add cache status indicator and view mode
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            view_mode = st.radio("View Mode", ["List", "Gallery"], horizontal=True)
        with col3:
            use_cache = st.checkbox("Use Cached Responses", value=True, help="Load pre-computed responses for instant display")
        
        # Fetch and display featured queries
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        featured_data = loop.run_until_complete(fetch_featured_queries(include_cached=use_cache))
        featured_queries = featured_data.get("featured_queries", [])
        metadata = featured_data.get("metadata", {})
        
        # Show cache metadata if available
        if metadata.get("cache_generated_at"):
            st.info(f"📅 Cache last updated: {metadata.get('cache_generated_at', 'Unknown')}")
        
        if featured_queries:
            if view_mode == "Gallery":
                # Gallery View
                st.markdown("#### 🎨 Gallery View")
                
                # Create a grid layout
                cols = st.columns(3)
                for idx, query in enumerate(featured_queries):
                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div class="gallery-card">
                            <div>
                                <h4>{query.get('title', 'Untitled')}</h4>
                                <p>{query.get('description', '')[:100]}...</p>
                            </div>
                            <div>
                                <span class="category-badge">{query.get('category', 'Other')}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🚀 Run", key=f"gallery_run_{query.get('id', '')}"):
                                st.session_state.query = query.get('query', '')
                                st.session_state.active_tab = 0
                                st.rerun()
                        
                        with col2:
                            if use_cache and query.get('cached_response'):
                                if st.button("📋 View", key=f"gallery_view_{query.get('id', '')}"):
                                    st.session_state.selected_featured_query = query
                                    st.session_state.show_modal = True
                
                # Show modal for cached response
                if st.session_state.get('show_modal', False) and st.session_state.get('selected_featured_query'):
                    selected_query = st.session_state.selected_featured_query
                    st.markdown("---")
                    st.markdown(f"### 📋 {selected_query.get('title', 'Query Response')}")
                    
                    if st.button("❌ Close", key="close_modal"):
                        st.session_state.show_modal = False
                        st.session_state.selected_featured_query = None
                        st.rerun()
                    
                    cached_response = selected_query.get('cached_response', {})
                    modules = cached_response.get('modules', [])
                    
                    if modules:
                        for i, module in enumerate(modules):
                            with st.container():
                                display_module(module, i+1)
                    else:
                        st.warning("No cached response available")
                        
            else:
                # List View (existing code)
                # Create tabs for categories
                all_categories = list(set(q.get("category", "Other") for q in featured_queries))
                category_tabs = st.tabs(["All"] + all_categories)
                
                # All queries tab
                with category_tabs[0]:
                    for query in featured_queries:
                        with st.expander(f"**{query.get('title', 'Untitled')}** - {query.get('category', 'Other')}", expanded=False):
                            st.markdown(f"**Description:** {query.get('description', '')}")
                            st.markdown(f"**Query:** `{query.get('query', '')}`")
                            
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                if st.button("🚀 Run Live Query", key=f"run_all_{query.get('id', '')}"):
                                    st.session_state.query = query.get('query', '')
                                    st.session_state.active_tab = 0  # Switch to Query Interface tab
                                    st.rerun()
                            
                            with col2:
                                if use_cache and query.get('cached_response'):
                                    if st.button("📋 View Cached Response", key=f"view_all_{query.get('id', '')}"):
                                        st.session_state[f"show_cached_{query.get('id')}" ] = True
                            
                            with col3:
                                if query.get('cached_at'):
                                    st.caption(f"Cached: {query.get('cached_at', '').split('T')[0]}")
                            
                            # Show cached response if button clicked
                            if st.session_state.get(f"show_cached_{query.get('id')}", False):
                                st.markdown("---")
                                st.markdown("#### 📋 Cached Response")
                                
                                cached_response = query.get('cached_response', {})
                                modules = cached_response.get('modules', [])
                                
                                if modules:
                                    for i, module in enumerate(modules):
                                        with st.container():
                                            display_module(module, i+1)
                                else:
                                    st.warning("No cached response available")
            
                # Category-specific tabs
                for idx, category in enumerate(all_categories):
                    with category_tabs[idx + 1]:
                        category_queries = [q for q in featured_queries if q.get("category") == category]
                        
                        for query in category_queries:
                            with st.expander(f"**{query.get('title', 'Untitled')}**", expanded=False):
                                st.markdown(f"**Description:** {query.get('description', '')}")
                                st.markdown(f"**Query:** `{query.get('query', '')}`")
                                
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1:
                                    if st.button("🚀 Run Live Query", key=f"run_{category}_{query.get('id', '')}"):
                                        st.session_state.query = query.get('query', '')
                                        st.session_state.active_tab = 0  # Switch to Query Interface tab
                                        st.rerun()
                                
                                with col2:
                                    if use_cache and query.get('cached_response'):
                                        if st.button("📋 View Cached Response", key=f"view_{category}_{query.get('id', '')}"):
                                            st.session_state[f"show_cached_{category}_{query.get('id')}" ] = True
                                
                                with col3:
                                    if query.get('cached_at'):
                                        st.caption(f"Cached: {query.get('cached_at', '').split('T')[0]}")
                                
                                # Show cached response if button clicked
                                if st.session_state.get(f"show_cached_{category}_{query.get('id')}", False):
                                    st.markdown("---")
                                    st.markdown("#### 📋 Cached Response")
                                    
                                    cached_response = query.get('cached_response', {})
                                    modules = cached_response.get('modules', [])
                                    
                                    if modules:
                                        for i, module in enumerate(modules):
                                            with st.container():
                                                display_module(module, i+1)
                                    else:
                                        st.warning("No cached response available for this query")
            
                # Add refresh cache instructions
                with st.expander("ℹ️ How to Update Cache", expanded=False):
                    st.markdown("""
                    To update the cached responses:
                    1. Ensure the API server is running
                    2. Run: `python scripts/generate_featured_cache.py`
                    3. Refresh this page to see updated responses
                    """)
        else:
            st.info("No featured queries available. Make sure the API server is running.")
    
    with tab3:
        st.markdown("### 📚 API Documentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Response Types
            
            **📝 Text Modules**
            - Formatted text with paragraphs
            - Inline citations with superscripts
            - Automatic citation numbering
            
            **📊 Chart Modules**
            - Bar charts for comparisons
            - Line charts for trends
            - Pie charts for distributions
            - Interactive Plotly visualizations
            
            **🗺️ Map Modules**
            - GeoJSON-based maps
            - Interactive markers
            - Facility locations
            - Country-based coloring
            """)
        
        with col2:
            st.markdown("""
            #### Table Types
            
            **📋 Standard Tables**
            - Data presentation
            - Sortable columns
            - Export capabilities
            
            **🏆 Specialized Tables**
            - Ranking tables (top emitters, etc.)
            - Comparison tables (countries, sectors)
            - Trend tables (time series data)
            - Geographic tables (location data)
            
            **📚 Citation Tables**
            - Numbered references
            - Source attribution
            - Document IDs and passages
            """)
        
        st.markdown("---")
        
        st.markdown("""
        #### 🔧 Advanced Features
        
        - **Streaming Mode**: Real-time thinking process visualization
        - **Citation System**: Automatic inline citations with numbered references
        - **Multi-modal Responses**: Combines text, data, and visualizations
        - **Narrative Organization**: AI-powered content organization
        - **Source Attribution**: Comprehensive source tracking
        
        #### 🌐 Data Sources
        
        - Climate Policy Radar Knowledge Graph
        - GIST Environmental Database
        - TransitionZero Solar Asset Mapper
        - Brazilian State Climate Policies
        - And more...
        """)

if __name__ == "__main__":
    main()