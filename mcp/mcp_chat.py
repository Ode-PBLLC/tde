import asyncio
import anthropic
from fastmcp import Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import Optional, Dict, List, Any
import time
import os
from dotenv import load_dotenv
import json
from textwrap import fill
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
import re


load_dotenv()

MAX_CTX_CHARS = 18_000          # hard cap – keep below Claude-Haiku context
WIDE = 88          # tweak for your terminal width
SUMMARY_PROMPT = (
    "You are a climate-policy expert. "
    "Assume the reader wants the big picture and key linkages."
    "You should always look for passages that are relevant to the user's query."
    "You should look for data to support the user's query, especially when they ask for it."
    "If given multiple concepts, you should look for passages that are relevant to all of them."
    "You should call at least one of the following tools AT LEAST ONCE for every query: GetPassagesMentioningConcept, PassagesMentioningBothConcepts"
    "You MUST ALWAYS CALL THE ALWAYSRUN TOOL FOR EVERY QUERY"
    "Passages are not Datasets. They are distinct. You find datasets by looking for 'HAS_DATASET_ABOUT' or 'DATASET_ON_TOPIC' edges."
)

def _fmt_sources(sources):
    """
    Return a pretty string for `sources` which may be a mix of:
      • plain strings (legacy metadata)
      • dicts like {"doc_id": "D123", "passage_id": "P456", "text": "..."}
    """
    if not sources:
        return "— no sources captured —"

    rows = []
    has_structured_source = False
    for src in sources:
        if isinstance(src, dict) and ("passage_id" in src or "doc_id" in src):
            has_structured_source = True
            doc_id = src.get('doc_id', '?')
            passage_id = src.get('passage_id', 'N/A') # N/A if it's a doc-level source without specific passage
            text_snippet = src.get('text', 'No text snippet available')
            if text_snippet and len(text_snippet) > 150: # Truncate long snippets
                text_snippet = text_snippet[:147] + "..."
            rows.append(f"DOC ID: {doc_id:<10} PASSAGE ID: {passage_id:<10}\nTEXT: {text_snippet}")
            rows.append("-"*WIDE) # Add a separator line
        else:                                # plain str or other legacy format
            rows.append(str(src))
            rows.append("-"*WIDE)
    
    if has_structured_source:
        # Header already included in each entry for clarity, or we can add a general one.
        # For now, the per-entry labels DOC ID/PASSAGE ID/TEXT should be clear.
        pass # No additional header needed if using per-entry labels
    
    # Remove the last separator if it exists
    if rows and rows[-1] == "-"*WIDE:
        rows.pop()
        
    return "\n".join(rows)

def _render_formatted_modules(formatted_response: Dict[str, Any]):
    """Render the structured modules in Streamlit."""
    modules = formatted_response.get("modules", [])
    
    for module in modules:
        module_type = module.get("type", "")
        
        if module_type == "text":
            heading = module.get("heading", "")
            texts = module.get("texts", [])
            
            if heading:
                st.markdown(f"### {heading}")
            
            for text in texts:
                st.markdown(text)
        
        elif module_type == "chart":
            chart_type = module.get("chartType", "bar")
            chart_data = module.get("data", {})
            
            if chart_data:
                # Convert Chart.js format to Plotly
                labels = chart_data.get("labels", [])
                datasets = chart_data.get("datasets", [])
                
                if datasets and labels:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    
                    for dataset in datasets:
                        dataset_label = dataset.get("label", "Data")
                        dataset_data = dataset.get("data", [])
                        
                        if chart_type == "line":
                            fig.add_trace(go.Scatter(
                                x=labels, 
                                y=dataset_data,
                                mode='lines+markers',
                                name=dataset_label
                            ))
                        else:  # bar chart default
                            fig.add_trace(go.Bar(
                                x=labels,
                                y=dataset_data, 
                                name=dataset_label
                            ))
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif module_type == "table":
            heading = module.get("heading", "")
            columns = module.get("columns", [])
            rows = module.get("rows", [])
            
            if heading:
                st.markdown(f"### {heading}")
            
            if columns and rows:
                df = pd.DataFrame(rows, columns=columns)
                st.dataframe(df, use_container_width=True)
        
        elif module_type == "map":
            # Render interactive map using GeoJSON data
            map_type = module.get("mapType", "geojson")
            geojson_data = module.get("geojson", {})
            view_state = module.get("viewState", {})
            legend = module.get("legend", {})
            map_metadata = module.get("metadata", {})
            
            st.markdown("### Interactive Map")
            
            # Display map metadata
            if map_metadata:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Facilities", map_metadata.get("total_facilities", "N/A"))
                with col2:
                    st.metric("Total Capacity", f"{map_metadata.get('total_capacity_mw', 0):.0f} MW")
                with col3:
                    st.metric("Countries", len(map_metadata.get("countries", [])))
            
            # Create Folium map from GeoJSON data
            if geojson_data and view_state:
                import folium
                
                center = view_state.get("center", [0, 0])
                m = folium.Map(
                    location=[center[1], center[0]],  # Folium uses [lat, lng]
                    zoom_start=view_state.get("zoom", 6)
                )
                
                # Add GeoJSON data to map
                for feature in geojson_data.get("features", []):
                    coords = feature["geometry"]["coordinates"]
                    props = feature["properties"]
                    
                    folium.CircleMarker(
                        location=[coords[1], coords[0]],  # Folium uses [lat, lng]
                        radius=props.get("marker_size", 8),
                        popup=f"<b>{props.get('popup_title', 'Facility')}</b><br>{props.get('popup_content', '')}",
                        color=props.get("marker_color", "#blue"),
                        fill=True,
                        fillOpacity=props.get("marker_opacity", 0.8),
                        weight=2
                    ).add_to(m)
                
                # Add legend if available
                if legend and legend.get("items"):
                    legend_html = f'''
                    <div style="position: fixed; 
                                bottom: 50px; left: 50px; width: 200px; background-color: white; 
                                border:2px solid grey; z-index:9999; font-size:14px; padding: 10px">
                    <p><b>{legend.get("title", "Legend")}</b></p>
                    '''
                    for item in legend["items"]:
                        legend_html += f'<p><i class="fa fa-circle" style="color:{item["color"]}"></i> {item["label"]}</p>'
                    legend_html += '</div>'
                    m.get_root().html.add_child(folium.Element(legend_html))
                
                # Display the map
                components.html(m._repr_html_(), height=500)
        
        elif module_type == "image":
            src = module.get("src", "")
            alt = module.get("alt", "Image")
            
            if src:
                st.image(src, caption=alt)

def create_dataset_citation(tool_name: str, tool_args: dict, result_content: list) -> dict:
    """
    Create dataset citations for data tools (maps, charts, tables).
    Returns a standardized citation record for non-passage data sources.
    """
    
    # Define data source mappings for each tool
    tool_sources = {
        # Solar Facilities Tools
        "GetSolarFacilitiesMapData": {
            "source_type": "Dataset",
            "source_name": "TZ-SAM Q1 2025 Solar Facilities Database", 
            "description": "TransitionZero Solar Asset Mapper - Global solar facility locations and capacity data",
            "provider": "TransitionZero",
            "coverage": "Brazil, India, South Africa, Vietnam"
        },
        "GetSolarFacilitiesByCountry": {
            "source_type": "Dataset",
            "source_name": "TZ-SAM Q1 2025 Solar Database",
            "description": "Solar facility summary statistics by country",
            "provider": "TransitionZero", 
            "coverage": "Global"
        },
        "GetSolarCapacityByCountry": {
            "source_type": "Dataset", 
            "source_name": "TZ-SAM Solar Capacity Database",
            "description": "Solar capacity statistics and aggregations",
            "provider": "TransitionZero",
            "coverage": "Global"
        },
        "GetSolarCapacityVisualizationData": {
            "source_type": "Dataset",
            "source_name": "TZ-SAM Visualization Dataset", 
            "description": "Structured solar data for charts and visualizations",
            "provider": "TransitionZero",
            "coverage": "Multiple countries"
        },
        
        # GIST Environmental Tools
        "GetGistAssetsMapData": {
            "source_type": "Database",
            "source_name": "GIST Environmental Database - Assets",
            "description": "Corporate asset locations and environmental risk data (40K+ assets)",
            "provider": "GIST Environmental Research",
            "coverage": "100+ companies, 5 sectors"
        },
        "GetGistEmissionsTrends": {
            "source_type": "Database", 
            "source_name": "GIST Scope 3 Emissions Database",
            "description": "Corporate Scope 3 emissions data with multi-year trends (2016-2024)",
            "provider": "GIST Environmental Research",
            "coverage": "100+ companies"
        },
        "GetGistBiodiversityImpacts": {
            "source_type": "Database",
            "source_name": "GIST Biodiversity Impact Database", 
            "description": "Corporate biodiversity footprint data (PDF, CO2E, LCE metrics)",
            "provider": "GIST Environmental Research",
            "coverage": "Multiple companies"
        },
        "GetGistVisualizationData": {
            "source_type": "Database",
            "source_name": "GIST Sustainability Dashboard Data",
            "description": "Structured corporate sustainability data for visualizations",
            "provider": "GIST Environmental Research", 
            "coverage": "100+ companies"
        },
        
        # LSE Climate Policy Tools
        "GetStateClimatePolicy": {
            "source_type": "Database",
            "source_name": "LSE Climate Governance Database",
            "description": "State-level climate policy analysis and governance frameworks",
            "provider": "London School of Economics",
            "coverage": "Brazilian states (27 states)"
        },
        "GetNDCOverviewData": {
            "source_type": "Database", 
            "source_name": "LSE NDC Analysis Framework",
            "description": "Nationally Determined Contributions analysis and policy comparisons",
            "provider": "London School of Economics",
            "coverage": "Multiple countries"
        },
        "CompareBrazilianStates": {
            "source_type": "Database",
            "source_name": "LSE Brazilian State Policy Database",
            "description": "Comparative analysis of climate policies across Brazilian states",
            "provider": "London School of Economics", 
            "coverage": "All 27 Brazilian states"
        },
        
        # Knowledge Graph Tools  
        "GetDatasetContent": {
            "source_type": "Dataset",
            "source_name": "Climate Policy Knowledge Graph - Datasets",
            "description": "Structured datasets from climate policy knowledge graph",
            "provider": "Climate Policy Radar",
            "coverage": "Various climate policy datasets"
        }
    }
    
    if tool_name not in tool_sources:
        return None
        
    source_info = tool_sources[tool_name]
    
    # Try to extract additional details from the result
    data_description = source_info["description"]
    if result_content and isinstance(result_content, list) and len(result_content) > 0:
        first_content = result_content[0]
        if hasattr(first_content, 'text'):
            try:
                data = json.loads(first_content.text)
                
                # Extract specific details for different tool types
                if isinstance(data, list) and len(data) > 0:
                    data_description += f" ({len(data)} records retrieved)"
                elif isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], list):
                        data_description += f" ({len(data['data'])} facilities)"
                    if 'metadata' in data:
                        metadata = data['metadata']
                        if 'total_facilities' in metadata:
                            data_description += f" (Total: {metadata['total_facilities']} facilities)"
                        if 'data_source' in metadata:
                            data_description += f" - {metadata['data_source']}"
                            
            except (json.JSONDecodeError, AttributeError):
                pass
    
    # Add tool arguments context
    args_context = ""
    if tool_args:
        if 'country' in tool_args:
            args_context += f" for {tool_args['country']}"
        if 'concept' in tool_args:
            args_context += f" related to '{tool_args['concept']}'"
        if 'company_code' in tool_args:
            args_context += f" for company {tool_args['company_code']}"
            
    data_description += args_context
    
    return {
        "source_type": source_info["source_type"],
        "source_name": source_info["source_name"],
        "provider": source_info["provider"],
        "description": data_description,
        "tool_used": tool_name,
        "coverage": source_info["coverage"],
        "citation_id": f"{tool_name}_{hash(str(tool_args)) % 10000}"
    }

def harvest_sources(payload):
    """
    Enhanced function to extract sources from tool results.
    Accepts result.content (could be list/dict/str) and
    returns a list of {doc_id, passage_id, text, title, date} records.
    """
    out = []
    print(f"HARVEST_SOURCES DEBUG: Processing payload type: {type(payload)}")
    
    if isinstance(payload, list):
        for i, item in enumerate(payload):
            print(f"HARVEST_SOURCES DEBUG: Processing item {i}, type: {type(item)}")
            try: 
                # Handle TextContent objects with .text attribute
                if hasattr(item, 'text'):
                    print(f"HARVEST_SOURCES DEBUG: Item has .text attribute, parsing JSON...")
                    try:
                        data = json.loads(item.text)
                        print(f"HARVEST_SOURCES DEBUG: Parsed JSON, type: {type(data)}")
                        
                        # Handle list of passages (e.g., from GetPassagesMentioningConcept)
                        if isinstance(data, list):
                            print(f"HARVEST_SOURCES DEBUG: Found list with {len(data)} items")
                            for j, passage in enumerate(data):
                                if isinstance(passage, dict) and "passage_id" in passage:
                                    source_record = {
                                        "doc_id": passage.get("doc_id") or passage.get("document_id"),
                                        "passage_id": passage["passage_id"],
                                        "text": passage.get("text", ""),
                                        "title": passage.get("title", ""),
                                        "date": passage.get("date", ""),
                                        "type": passage.get("type", "document")
                                    }
                                    out.append(source_record)
                                    print(f"HARVEST_SOURCES DEBUG: Added source {j+1}: {source_record['doc_id']}")
                        
                        # Handle single passage dict
                        elif isinstance(data, dict) and "passage_id" in data:
                            source_record = {
                                "doc_id": data.get("doc_id") or data.get("document_id"),
                                "passage_id": data["passage_id"],
                                "text": data.get("text", ""),
                                "title": data.get("title", ""),
                                "date": data.get("date", ""),
                                "type": data.get("type", "document")
                            }
                            out.append(source_record)
                            print(f"HARVEST_SOURCES DEBUG: Added single source: {source_record['doc_id']}")
                        
                        else:
                            print(f"HARVEST_SOURCES DEBUG: Data format not recognized, keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                            
                            # Handle responses that might contain passage data in different formats
                            if isinstance(data, dict):
                                # Check if this is a metadata response that we should ignore
                                if 'available_files' in data or 'content_summary' in data:
                                    print(f"HARVEST_SOURCES DEBUG: Skipping metadata response")
                                # Check for passages in nested structures
                                elif 'passages' in data:
                                    passages = data['passages']
                                    if isinstance(passages, list):
                                        print(f"HARVEST_SOURCES DEBUG: Found nested passages: {len(passages)}")
                                        for passage in passages:
                                            if isinstance(passage, dict) and "passage_id" in passage:
                                                source_record = {
                                                    "doc_id": passage.get("doc_id") or passage.get("document_id"),
                                                    "passage_id": passage["passage_id"],
                                                    "text": passage.get("text", ""),
                                                    "title": passage.get("title", ""),
                                                    "date": passage.get("date", ""),
                                                    "type": passage.get("type", "document")
                                                }
                                                out.append(source_record)
                                                print(f"HARVEST_SOURCES DEBUG: Added nested source: {source_record['doc_id']}")
                                # Check for any field that might contain passage-like data
                                else:
                                    for key, value in data.items():
                                        if isinstance(value, list):
                                            for item in value:
                                                if isinstance(item, dict) and "passage_id" in item:
                                                    source_record = {
                                                        "doc_id": item.get("doc_id") or item.get("document_id"),
                                                        "passage_id": item["passage_id"],
                                                        "text": item.get("text", ""),
                                                        "title": item.get("title", ""),
                                                        "date": item.get("date", ""),
                                                        "type": item.get("type", "document")
                                                    }
                                                    out.append(source_record)
                                                    print(f"HARVEST_SOURCES DEBUG: Added source from {key}: {source_record['doc_id']}")
                    
                    except json.JSONDecodeError as e:
                        print(f"HARVEST_SOURCES DEBUG: JSON decode failed: {e}")
                        print(f"HARVEST_SOURCES DEBUG: Raw text: {item.text[:200]}...")
                
                # Handle direct dict items
                elif isinstance(item, dict) and "passage_id" in item:
                    source_record = {
                        "doc_id": item.get("doc_id") or item.get("document_id"),
                        "passage_id": item["passage_id"],
                        "text": item.get("text", ""),
                        "title": item.get("title", ""),
                        "date": item.get("date", ""),
                        "type": item.get("type", "document")
                    }
                    out.append(source_record)
                    print(f"HARVEST_SOURCES DEBUG: Added direct dict source: {source_record['doc_id']}")
                    
            except Exception as e:
                print(f"HARVEST_SOURCES ERROR: {e}")
                import traceback
                traceback.print_exc()
                
    elif isinstance(payload, dict):
        if "passage_id" in payload:
            source_record = {
                "doc_id": payload.get("doc_id") or payload.get("document_id"),
                "passage_id": payload["passage_id"],
                "text": payload.get("text", ""),
                "title": payload.get("title", ""),
                "date": payload.get("date", ""),
                "type": payload.get("type", "document")
            }
            out.append(source_record)
            print(f"HARVEST_SOURCES DEBUG: Added single dict source: {source_record['doc_id']}")
    
    print(f"HARVEST_SOURCES DEBUG: Returning {len(out)} sources")
    return out


async def get_solar_data_direct(client, data_type: str, **kwargs):
    """
    Directly fetch solar data for visualization without going through Claude.
    This bypasses the LLM context to avoid token limits.
    """
    import pandas as pd  # Move import to top to fix the error
    import os
    
    # Get project root path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    try:
        if data_type == "country_comparison":
            # Get country comparison data
            result = await client.call_tool("GetSolarCapacityByCountry", {}, "solar")
            # Parse the result and return structured data
            
            # Load the CSV directly for visualization
            csv_path = os.path.join(project_root, "mcp", "solar_facilities_demo.csv")
            df = pd.read_csv(csv_path) 
            stats = df.groupby('country').agg({
                'capacity_mw': ['sum', 'mean', 'count', 'min', 'max']
            }).round(2)
            
            stats.columns = ['total_capacity_mw', 'avg_capacity_mw', 'facility_count', 'min_capacity_mw', 'max_capacity_mw']
            stats = stats.reset_index()
            
            return {
                "type": "country_comparison",
                "data": stats.to_dict('records'),
                "chart_config": {
                    "x": "country",
                    "y": "total_capacity_mw", 
                    "title": "Solar Capacity by Country",
                    "chart_type": "bar"
                }
            }
            
        elif data_type == "facilities_map":
            # Create map data
            df = pd.read_csv(os.path.join(project_root, "mcp", "solar_facilities_demo.csv"))
            country_filter = kwargs.get('country')
            if country_filter:
                df = df[df['country'].str.lower() == country_filter.lower()]
            
            # Limit to 500 facilities for performance
            if len(df) > 500:
                df = df.nlargest(500, 'capacity_mw')
                
            return {
                "type": "map",
                "data": df.to_dict('records'),
                "metadata": {
                    "total_facilities": len(df),
                    "total_capacity": df['capacity_mw'].sum(),
                    "countries": df['country'].unique().tolist()
                }
            }
            
        elif data_type == "capacity_distribution":
            df = pd.read_csv(os.path.join(project_root, "mcp", "solar_facilities_demo.csv"))
            
            # Create capacity bins
            bins = [0, 1, 5, 10, 25, 50, 100, 500, 3000]
            bin_labels = ['<1MW', '1-5MW', '5-10MW', '10-25MW', '25-50MW', '50-100MW', '100-500MW', '>500MW']
            
            df['capacity_bin'] = pd.cut(df['capacity_mw'], bins=bins, labels=bin_labels, include_lowest=True)
            dist_data = df['capacity_bin'].value_counts().sort_index()
            
            chart_data = [{"capacity_range": str(k), "facility_count": v} for k, v in dist_data.items()]
            
            return {
                "type": "capacity_distribution",
                "data": chart_data,
                "chart_config": {
                    "x": "capacity_range",
                    "y": "facility_count",
                    "title": "Solar Facilities by Capacity Range",
                    "chart_type": "bar"
                }
            }
            
    except Exception as e:
        print(f"Error in direct data access: {e}")
        return None

class MultiServerClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.anthropic = anthropic.Anthropic()
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # shuts down all stdio_clients, ClientSessions, etc. in the *same* task
        await self.exit_stack.aclose()

    async def connect_to_server(self, server_name: str, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        print(f"Connecting to {server_name} server at {server_script_path}")
        
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        
        self.sessions[server_name] = session
        print(f"Initialized {server_name} session")

    async def call_tool(self, tool_name: str, tool_args: dict, server_name: str = "kg"):
        """Call a tool on a specific server. Defaults to KG server for backward compatibility."""
        if server_name not in self.sessions:
            raise ValueError(f"Server '{server_name}' not connected. Available servers: {list(self.sessions.keys())}")
        
        session = self.sessions[server_name]
        return await session.call_tool(tool_name, tool_args)

    async def get_all_available_tools(self) -> Dict[str, List[dict]]:
        """Get all tools from all connected servers."""
        all_tools = {}
        for server_name, session in self.sessions.items():
            response = await session.list_tools()
            tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
                "server": server_name
            } for tool in response.tools]
            all_tools[server_name] = tools
        return all_tools

    def _determine_server_for_tool(self, tool_name: str, all_tools: Dict[str, List[dict]]) -> str:
        """Determine which server hosts a specific tool."""
        for server_name, tools in all_tools.items():
            for tool in tools:
                if tool["name"] == tool_name:
                    return server_name
        return "kg"  # Default to KG server for backward compatibility

        
    async def process_query(self, query: str):
        # --- Temporary hardcoding for chart visualization --- 
        DUMMY_DATASET_ID = "DUMMY_DATASET_EXTREME_WEATHER"
        if query.lower() == "show dummy chart":
            print(f"Hardcoded trigger: Querying for dataset ID: {DUMMY_DATASET_ID}")
            # Fall through to the main logic, but ensure the AI targets this ID if it uses GetDatasetContent
            # This specific hardcoding will now rely on the general GetDatasetContent parsing below.
            # For true hardcoding, we'd simulate the AI making the *correct* call.
            # To ensure it uses the GetDatasetContent, we can subtly change the query for the AI if needed,
            # or rely on the fact that the user *wants* this specific dataset for this query.
            # For now, we assume if query is "show dummy chart", the AI will be guided/already knows.
            # The main purpose of this block is now just for a print statement.
            pass # Let the normal flow attempt to get this via AI if possible, or adjust query for AI.

        messages = [
            {"role": "user", 
            "content": query} # Original query
        ]
        
        # Get tools from all connected servers
        all_tools = await self.get_all_available_tools()
        available_tools = []
        for server_name, tools in all_tools.items():
            # Remove server field before sending to Claude API
            clean_tools = []
            for tool in tools:
                clean_tool = {
                    "name": tool["name"],
                    "description": tool["description"], 
                    "input_schema": tool["input_schema"]
                }
                clean_tools.append(clean_tool)
            available_tools.extend(clean_tools)
        
        system_prompt = """
            You are a climate policy expert. Assume the reader wants the big picture and key linkages.

            Core Task:
            1. Understand the user's query.
            2. Use available tools to gather information from multiple data sources.
            3. Synthesize the information to answer the user's query.

            Available Data Sources:
            - Knowledge Graph: Climate policy concepts, relationships, and passages
            - Solar Facilities Dataset: Real-world solar installation data (Brazil, India, South Africa, Vietnam)
            - GIST Dataset: Comprehensive environmental sustainability data for 100 companies including:
              * Environmental risk assessments (40K+ assets with coordinates)
              * Scope 3 emissions data (2016-2024)
              * Biodiversity impact measurements (PDF, CO2E, LCE metrics)
              * Deforestation proximity indicators
              * Multi-year trends and sector comparisons
            - LSE Climate Policy Dataset: London School of Economics climate governance analysis including:
              * NDC overview and domestic policy comparisons
              * Institutions and processes analysis
              * Plans and policies assessment
              * Brazilian state-level climate governance (all 27 states)
              * TPI graphical data

            Tool Usage Guidelines:
            - CRITICAL: Every response MUST include source citations from the knowledge graph. Follow this exact sequence:
              1. FIRST, call 'CheckConceptExists' for the main concept in the user's query
              2. If concept exists, IMMEDIATELY call 'GetPassagesMentioningConcept' with that exact concept name
              3. If concept doesn't exist, call 'GetSemanticallySimilarConcepts' and then call 'GetPassagesMentioningConcept' with a similar concept
              4. NEVER respond without calling at least one passage retrieval tool that returns actual document passages
              5. If GetPassagesMentioningConcept returns empty, try related concepts or use 'PassagesMentioningBoth' with related terms
               
            - Datasets Discovery: Use `GetAvailableDatasets` to discover what datasets are available and their characteristics.
            
            - Knowledge Graph Datasets: For datasets in the KG, use:
                1. `GetConceptGraphNeighbors` for relevant concepts.
                2. Look for neighbors with `kind: "Dataset"` and connected by edges like `HAS_DATASET_ABOUT` or `DATASET_ON_TOPIC`.
                3. If a relevant dataset is found, use its `node_id` with the `GetDatasetContent` tool to fetch its data.
            
            - Solar Facilities Data: For solar energy queries, use these specialized tools:
                - `GetSolarFacilitiesByCountry`: Get facilities summary for specific countries
                - `GetSolarCapacityByCountry`: Get capacity statistics by country
                - `GetSolarFacilitiesMapData`: Get facility coordinates for interactive maps (use for map requests)
                - `GetSolarFacilitiesInRadius`: Find facilities near coordinates
                - `GetSolarConstructionTimeline`: Analyze construction trends over time
                - `GetLargestSolarFacilities`: Find biggest installations
                - `SearchSolarFacilitiesByCapacity`: Filter by capacity range
                - `GetSolarCapacityVisualizationData`: Get structured data for charts and graphs
                
                Note: For map requests, always use `GetSolarFacilitiesMapData` as it provides the detailed coordinate data needed for map generation.
            
            - GIST Environmental & Sustainability Data: For corporate sustainability, environmental risk, emissions, or ESG queries:
                - `GetGistCompanies`: Discover companies with optional sector/country filtering
                - `GetGistCompanyProfile`: Complete sustainability profile for a specific company
                - `GetGistCompanyRisks`: Environmental risk assessment across 13 risk categories
                - `GetGistScope3Emissions`: Detailed Scope 3 emissions data with breakdown by category
                - `GetGistBiodiversityImpacts`: Biodiversity footprint data (PDF, CO2E, LCE metrics)
                - `GetGistDeforestationRisks`: Deforestation proximity indicators and forest change analysis
                - `GetGistAssetsMapData`: Asset-level geographic data for mapping (use for ESG asset mapping)
                - `GetGistEmissionsTrends`: Multi-year emissions trends and intensity analysis
                - `GetGistVisualizationData`: Structured data for sustainability dashboards and charts
                
                Note: GIST covers 100 companies across 5 sectors (OGES, FINS, WHRE, MOMI, REEN) with 9 years of time series data.
            
            - LSE Climate Policy Analysis: For climate governance, policy analysis, NDC, or Brazilian state policy queries:
                - `GetLSEDatasetOverview`: Get overview of all available LSE policy modules
                - `GetBrazilianStatesOverview`: Get overview of Brazilian state climate governance
                - `GetStateClimatePolicy`: Get detailed climate policy info for specific Brazilian states
                - `CompareBrazilianStates`: Compare climate policies across multiple states
                - `GetNDCOverviewData`: Get NDC (Nationally Determined Contributions) analysis
                - `GetInstitutionsProcessesData`: Get climate governance institutions analysis
                - `GetPlansAndPoliciesData`: Get climate plans and policies assessment
                - `SearchLSEContent`: Search across all LSE content for specific terms
                - `GetLSEVisualizationData`: Get structured data for policy analysis charts
                
                Note: LSE covers NDC analysis, institutional frameworks, and comprehensive Brazilian state-level governance (all 27 states).
            
            - ALWAYSRUN Tool: For system debugging, you MUST ALWAYS CALL THE `ALWAYSRUN` TOOL ONCE AND ONLY ONCE FOR EVERY USER QUERY. Pass the original user query as the 'query' argument to this tool. Do this early in your thought process.

            Cross-Reference Strategy:
            When users ask about ANY topic or concept:
            1. **ALWAYS** check the knowledge graph for relevant concepts and passages
            2. **AUTOMATICALLY** call `GetAvailableDatasets()` to discover connected datasets
            3. **IF datasets exist for the concept**, call `GetDatasetContent()` to retrieve structured data
            4. For solar energy, renewable energy, or specific countries (Brazil, India, South Africa, Vietnam), also use solar facilities tools
            5. For corporate sustainability, ESG, emissions, environmental risk, or biodiversity queries, use GIST tools to access company-level data
            6. For climate governance, policy analysis, NDC, or Brazilian state-level queries, use LSE tools to access governance frameworks and policy assessments
            7. **IMPORTANT: If the user asks for maps, locations, or "show me facilities", you MUST call appropriate map data tools (`GetSolarFacilitiesMapData` for solar, `GetGistAssetsMapData` for corporate assets)**
            8. **Combine** policy text + structured data + geographic data + sustainability metrics + governance analysis in comprehensive answers

            Enhanced Data Discovery:
            - After getting concept passages, ALWAYS check for connected datasets using `GetAvailableDatasets()`
            - Look for concepts with "HAS_DATASET_ABOUT" relationships in the knowledge graph
            - Proactively surface both textual insights AND structured data when available
            - Include data tables and visualizations when datasets are connected to the queried concept
            - For corporate/company queries, automatically check GIST data for sustainability metrics and environmental risk assessments
            - For governance/policy/Brazilian state queries, automatically check LSE data for institutional frameworks and policy implementation status
            - This ensures users get complete information: policy context + real data + geographic context + corporate sustainability data + governance frameworks

            Visualization Capabilities:
            - Interactive maps and charts may be automatically generated for certain datasets
            - If visualizations are available for the current query, this will be indicated in the context
            - Only reference visualizations if explicitly mentioned in the tool results or context

            Output Format:
            - After completing all necessary tool calls, synthesize the gathered information into a single, comprehensive response to the user. 
            - Do NOT narrate your tool calling process (e.g., avoid phrases like "First, I will call...", "Next, I found..."). 
            - Present the final answer as if you are directly answering the user's query based on the knowledge you have acquired.
            - When presenting solar facility data, include specific numbers and context.
            - Only mention maps or visualizations if they are explicitly confirmed as available in the context.

            Respond to the user based on the information gathered from the tools.
            """

        if query.lower() == "show dummy chart": # Override messages for dummy chart query
             messages = [
                {"role": "user", "content": "Get the content of dataset DUMMY_DATASET_EXTREME_WEATHER."}
            ]

        response = self.anthropic.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
            tools=available_tools # TODO Add lower temperature for tool calling LLMs
        )

        final_text = []
        sources_used = []
        context_chunks = []   # every tool_result.content goes in here
        passage_sources = []        # each element: {"doc_id": …, "passage_id": …}
        chart_data = None           # To store data for charting
        map_data = None             # To store map HTML and metadata
        visualization_data = None   # To store structured visualization data
        all_tool_outputs_for_debug = [] # For Feature 2
        
        intermediate_ai_text_parts = [] # Collect all AI text parts during the process
        last_assistant_text = "" # Store the last piece of text from the assistant

        while True:
            assistant_message_content = []
            current_turn_text_parts = [] # Collect text from the current turn

            for content in response.content:
                if content.type == "text":
                    current_turn_text_parts.append(content.text)
                    intermediate_ai_text_parts.append(content.text) # Also add to the comprehensive list
                    assistant_message_content.append(content)
                elif content.type == "tool_use":
                    tool_name = content.name
                    tool_args = content.input
                    
                    # Determine which server to route this tool call to
                    server_name = self._determine_server_for_tool(tool_name, all_tools)
                    
                    # Pretty print of the tool and its arguments
                    print(f"Calling tool {tool_name} on {server_name} server with args {tool_args}")
                    
                    result = await self.call_tool(tool_name, tool_args, server_name)

                    # For Feature 2: Collect all tool outputs for debugging
                    all_tool_outputs_for_debug.append({
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_result_content": result.content 
                    })

                    # Parse map data from GetSolarFacilitiesMapData
                    if tool_name == "GetSolarFacilitiesMapData":
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            first_content_block = result.content[0]
                            if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                                try:
                                    parsed_content = json.loads(first_content_block.text)
                                    if isinstance(parsed_content, dict) and parsed_content.get("type") == "map":
                                        map_data = parsed_content
                                        print(f"Successfully parsed map data from {tool_name}: {len(parsed_content.get('data', []))} facilities")
                                    else:
                                        print(f"Map data from {tool_name} is not in expected format: {first_content_block.text[:100]}...")
                                except json.JSONDecodeError:
                                    print(f"Map content from {tool_name} is not valid JSON: {first_content_block.text[:100]}...")
                    
                    # Parse visualization data from different tools
                    elif tool_name in ["GetDatasetContent", "GetSolarFacilitiesByCountry", "GetSolarCapacityByCountry"]:
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            first_content_block = result.content[0]
                            if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                                try:
                                    parsed_content = json.loads(first_content_block.text)
                                    if isinstance(parsed_content, list):
                                        # Check if it's a list of dicts (actual data)
                                        if all(isinstance(item, dict) for item in parsed_content):
                                            chart_data = parsed_content # Assign to chart_data
                                            print(f"Successfully parsed chart data from {tool_name}: {len(chart_data)} records")
                                        else:
                                            print(f"Content from {tool_name} is a list, but not of dictionaries: {first_content_block.text[:100]}...")
                                    else:
                                        # This means the .text was valid JSON, but not a list (e.g. a string like "Dataset not found")
                                        print(f"Parsed JSON from {tool_name} is not a list: {first_content_block.text[:100]}...")
                                except json.JSONDecodeError:
                                    # This means .text was not valid JSON (e.g. plain string "Dataset not found")
                                    print(f"Content from {tool_name} is not valid JSON: {first_content_block.text[:100]}...")
                            else:
                                print(f"Content block from {tool_name} is not TextContent or lacks .text attribute.")
                        else:
                            print(f"{tool_name} did not return expected content structure: {result.content}")
                    
                    # Map generation removed - now done purely client-side
                    
                    # Parse structured visualization data
                    elif tool_name == "GetSolarCapacityVisualizationData":
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            first_content_block = result.content[0]
                            if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                                try:
                                    parsed_content = json.loads(first_content_block.text)
                                    if isinstance(parsed_content, dict) and "data" in parsed_content:
                                        visualization_data = parsed_content
                                        print(f"Successfully parsed visualization data from {tool_name}: {parsed_content.get('visualization_type', 'unknown')}")
                                    else:
                                        print(f"Visualization data from {tool_name} missing data field: {str(parsed_content)[:100]}...")
                                except json.JSONDecodeError:
                                    print(f"Visualization content from {tool_name} is not valid JSON: {first_content_block.text[:100]}...")
                    
                    # Add to context but aggressively limit size to prevent token explosion
                    try:
                        #content_str = json.dumps(result.content, ensure_ascii=False)
                        # Just pass text string for now
                        content_str = result.content[0].text
                        print(content_str)
                        # VERY aggressive truncation - max 1000 characters per chunk
                        if len(content_str) > 1000:
                            content_str = content_str[:1000] + "... [truncated]"
                        context_chunks.append(content_str)
                    except Exception:
                        pass

                    # 1) legacy metadata capture
                    if tool_name.lower() == "getmetadata":
                        sources_used.append(result.content)

                    # 2) NEW: collect passage/document IDs anywhere they appear  
                    print(f"DEBUG: About to call harvest_sources for tool: {tool_name}")
                    print(f"DEBUG: result.content type: {type(result.content)}")
                    if result.content and isinstance(result.content, list) and len(result.content) > 0:
                        print(f"DEBUG: First content item type: {type(result.content[0])}")
                        if hasattr(result.content[0], 'text'):
                            print(f"DEBUG: First 200 chars: {result.content[0].text[:200]}...")
                    
                    # Extract passage sources (traditional approach)
                    new_passage_sources = harvest_sources(result.content)
                    passage_sources.extend(new_passage_sources)
                    print(f"DEBUG: harvest_sources returned {len(new_passage_sources)} passage sources for tool {tool_name}")
                    
                    # 3) NEW: Extract dataset citations for data tools
                    dataset_citation = create_dataset_citation(tool_name, tool_args, result.content)
                    if dataset_citation:
                        # Convert dataset citation to source format for backward compatibility
                        dataset_source = {
                            "doc_id": dataset_citation["citation_id"],
                            "passage_id": dataset_citation["tool_used"], 
                            "text": dataset_citation["description"],
                            "title": dataset_citation["source_name"],
                            "date": "",
                            "type": dataset_citation["source_type"],
                            "provider": dataset_citation["provider"],
                            "coverage": dataset_citation["coverage"]
                        }
                        passage_sources.append(dataset_source)
                        print(f"DEBUG: Added dataset citation for {tool_name}: {dataset_citation['source_name']}")
                    
                    print(f"DEBUG: Total sources now: {len(passage_sources)} (passages + datasets)")

                    # Attach tool_use to assistant message
                    assistant_message_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_message_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }]
                    })

                    # Break early and send updated messages to Claude
                    break
            else:
                # No tool_use found → conversation complete
                messages.append({"role": "assistant", "content": assistant_message_content})
                break

            # Ask Claude for the next step
            response = self.anthropic.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
                tools=available_tools,
            )

        # After the loop, process the final response
        if current_turn_text_parts: # This will be the text from the AI's last turn
            last_assistant_text = "\n".join(current_turn_text_parts)

        # --- final synthesis / response construction ---------------------------------
        final_response_text = ""
        if len(context_chunks) > 0:
            # AGGRESSIVE context trimming to prevent token explosion
            joined_ctx = "\n\n".join(context_chunks)
            # Much smaller limit to stay well under token limits
            SAFE_CTX_LIMIT = 8000  # Very conservative limit
            if len(joined_ctx) > SAFE_CTX_LIMIT:
                joined_ctx = joined_ctx[:SAFE_CTX_LIMIT] + "\n\n[truncated for safety]"
            
            # System prompt for final answer synthesis using context
            synthesis_system_prompt = (SUMMARY_PROMPT + # Use the original SUMMARY_PROMPT here
                "Based on the following context from various tools, synthesize a comprehensive answer to the user's original query. "
                "Present it as a direct answer, not a summary of the context itself. Avoid narrating the tool process. "
                "If the context mentions that visualizations are available, you may reference them appropriately.")

            synthesis_messages = [
                {"role": "user", "content": f"User's original query: {query}\n\nTool-derived Context:\n{joined_ctx}"}
            ]
            
            summary_resp = self.anthropic.messages.create(
                model="claude-3-5-haiku-20241022", # Or a more powerful model for synthesis
                max_tokens=1000, # Allow more tokens for a full answer, not just a summary
                system=synthesis_system_prompt,
                messages=synthesis_messages,
            )
            final_response_text = summary_resp.content[0].text.strip()
        elif last_assistant_text: # No context chunks, but AI provided a final text response
            final_response_text = last_assistant_text
        else: # Should not happen if AI always responds with text eventually
            final_response_text = "I was unable to produce a response."

        # de-dupe sources
        uniq_passages = {(p["doc_id"], p["passage_id"]): p for p in passage_sources}
        print(f"DEBUG: Found these unique passages: {uniq_passages}")
        if uniq_passages:
            sources_used.extend(uniq_passages.values())

        return {
            "response": final_response_text,
            "sources": sources_used or ["No source captured"],
            "chart_data": chart_data,  # Legacy chart data for backward compatibility
            "map_data": map_data,  # Map HTML and metadata
            "visualization_data": visualization_data,  # Structured chart data
            "all_tool_outputs_for_debug": all_tool_outputs_for_debug, # For Feature 2
            "ai_thought_process": "\n".join(intermediate_ai_text_parts) # Add the collected thoughts
        }
    
    async def process_query_streaming(self, query: str):
        """
        Process a query with streaming capability - yields events as they happen.
        """
        import json
        
        def translate_tool_to_action(tool_name: str, tool_args: dict) -> dict:
            """Convert technical tool calls to user-friendly action messages."""
            
            # Extract common arguments
            concept = tool_args.get('concept', '')
            country = tool_args.get('country', '')
            query_arg = tool_args.get('query', '')
            
            # Knowledge Graph Tools
            if tool_name == "ALWAYSRUN":
                return {
                    "message": "🚀 Initializing search across all databases...",
                    "category": "initialization"
                }
            elif tool_name == "CheckConceptExists":
                return {
                    "message": f"🔍 Checking our knowledge base for information about {concept}...",
                    "category": "search"
                }
            elif tool_name == "GetSemanticallySimilarConcepts":
                return {
                    "message": f"🔗 Finding concepts related to {concept}...",
                    "category": "search"
                }
            elif tool_name == "GetPassagesMentioningConcept":
                return {
                    "message": f"📚 Searching policy documents that mention {concept}...",
                    "category": "document_search"
                }
            elif tool_name == "GetDescription":
                return {
                    "message": f"📖 Getting detailed definition of {concept}...",
                    "category": "information"
                }
            elif tool_name == "GetRelatedConcepts":
                return {
                    "message": f"🌐 Exploring connections to {concept}...",
                    "category": "exploration"
                }
            elif tool_name == "GetAvailableDatasets":
                return {
                    "message": "📊 Discovering available datasets...",
                    "category": "data_discovery"
                }
            elif tool_name == "GetDatasetContent":
                dataset_id = tool_args.get('dataset_id', 'dataset')
                return {
                    "message": f"📈 Loading data from {dataset_id}...",
                    "category": "data_loading"
                }
            
            # Solar Facilities Tools
            elif tool_name == "GetSolarFacilitiesByCountry":
                if country:
                    return {
                        "message": f"🏭 Looking up solar facilities in {country}...",
                        "category": "solar_data"
                    }
                else:
                    return {
                        "message": "🏭 Gathering solar facility information...",
                        "category": "solar_data"
                    }
            elif tool_name == "GetSolarCapacityByCountry":
                return {
                    "message": "⚡ Analyzing global solar energy capacity by country...",
                    "category": "solar_analysis"
                }
            elif tool_name == "GetSolarFacilitiesMapData":
                if country:
                    return {
                        "message": f"🗺️ Mapping solar facilities in {country}...",
                        "category": "mapping"
                    }
                else:
                    return {
                        "message": "🗺️ Generating solar facility map data...",
                        "category": "mapping"
                    }
            elif tool_name == "GetLargestSolarFacilities":
                return {
                    "message": "🏗️ Finding the largest solar installations...",
                    "category": "solar_analysis"
                }
            elif tool_name == "GetSolarConstructionTimeline":
                return {
                    "message": "📅 Analyzing solar construction trends over time...",
                    "category": "temporal_analysis"
                }
            elif tool_name == "GetSolarCapacityVisualizationData":
                return {
                    "message": "📊 Preparing solar capacity visualization data...",
                    "category": "visualization"
                }
            elif tool_name == "SearchSolarFacilitiesByCapacity":
                min_cap = tool_args.get('min_capacity_mw', '')
                max_cap = tool_args.get('max_capacity_mw', '')
                if min_cap and max_cap:
                    return {
                        "message": f"🔍 Searching for solar facilities between {min_cap}-{max_cap} MW...",
                        "category": "filtered_search"
                    }
                else:
                    return {
                        "message": "🔍 Filtering solar facilities by capacity...",
                        "category": "filtered_search"
                    }
            
            # Response Formatting Tools
            elif tool_name == "FormatResponseAsModules":
                return {
                    "message": "✨ Formatting response for optimal presentation...",
                    "category": "formatting"
                }
            
            # Default fallback
            else:
                # Convert camelCase to human readable
                readable_name = tool_name.replace('Get', '').replace('Search', 'Searching ')
                return {
                    "message": f"⚙️ Running {readable_name}...",
                    "category": "processing"
                }
        
        def _get_completion_message(tool_name: str, tool_args: dict, result) -> str:
            """Generate user-friendly completion messages based on tool results."""
            concept = tool_args.get('concept', '')
            country = tool_args.get('country', '')
            
            try:
                # Try to get meaningful info from result
                result_text = result.content[0].text if result.content and hasattr(result.content[0], 'text') else str(result.content)
                
                if tool_name == "CheckConceptExists":
                    if "true" in result_text.lower():
                        return f"✅ Found detailed information about {concept}"
                    else:
                        return f"💡 Exploring alternative approaches for {concept}"
                        
                elif tool_name == "GetSemanticallySimilarConcepts":
                    try:
                        concepts = json.loads(result_text)
                        if isinstance(concepts, list) and len(concepts) > 1:
                            return f"🔗 Found {len(concepts)} related concepts to explore"
                        else:
                            return f"🔗 Identified related concepts for {concept}"
                    except:
                        return f"🔗 Found related concepts for {concept}"
                        
                elif tool_name == "GetSolarCapacityByCountry":
                    try:
                        data = json.loads(result_text)
                        if isinstance(data, dict) and "total_global_capacity_mw" in data:
                            capacity = data["total_global_capacity_mw"]
                            facilities = data.get("total_global_facilities", 0)
                            return f"⚡ Found {facilities:,} solar facilities with {capacity:,.0f} MW total capacity"
                        else:
                            return "⚡ Retrieved global solar capacity data"
                    except:
                        return "⚡ Retrieved solar capacity information"
                        
                elif tool_name == "GetSolarFacilitiesByCountry":
                    if country:
                        return f"🏭 Retrieved solar facility data for {country}"
                    else:
                        return "🏭 Retrieved solar facility information"
                        
                elif tool_name == "GetSolarFacilitiesMapData":
                    try:
                        data = json.loads(result_text)
                        if isinstance(data, dict) and data.get("type") == "map" and "data" in data:
                            facility_count = len(data["data"])
                            if country:
                                return f"🗺️ Generated map data for {facility_count:,} solar facilities in {country}"
                            else:
                                return f"🗺️ Generated map data for {facility_count:,} solar facilities"
                        else:
                            return f"🗺️ Generated solar facility map data"
                    except:
                        return f"🗺️ Generated solar facility map data"
                        
                elif tool_name == "GetPassagesMentioningConcept":
                    try:
                        passages = json.loads(result_text)
                        if isinstance(passages, list):
                            count = len(passages)
                            if count > 0:
                                return f"📚 Found {count} relevant policy document{'s' if count != 1 else ''}"
                            else:
                                return f"📚 Searched policy documents (exploring alternative sources)"
                        else:
                            return f"📚 Searched policy documents about {concept}"
                    except:
                        return f"📚 Completed document search for {concept}"
                        
                elif tool_name == "FormatResponseAsModules":
                    return "✨ Response ready for presentation"
                    
                else:
                    # Generic completion message
                    return f"✅ Completed {tool_name.replace('Get', '').replace('Search', 'search for ')}"
                    
            except Exception:
                # Fallback for any errors
                return f"✅ Completed {tool_name}"
        
        # --- Temporary hardcoding for chart visualization --- 
        DUMMY_DATASET_ID = "DUMMY_DATASET_EXTREME_WEATHER"
        if query.lower() == "show dummy chart":
            print(f"Hardcoded trigger: Querying for dataset ID: {DUMMY_DATASET_ID}")

        messages = [
            {"role": "user", 
            "content": query}
        ]
        
        # Get tools from all connected servers
        all_tools = await self.get_all_available_tools()
        available_tools = []
        for server_name, tools in all_tools.items():
            clean_tools = []
            for tool in tools:
                clean_tool = {
                    "name": tool["name"],
                    "description": tool["description"], 
                    "input_schema": tool["input_schema"]
                }
                clean_tools.append(clean_tool)
            available_tools.extend(clean_tools)
        
        system_prompt = """
            You are a climate policy expert. Assume the reader wants the big picture and key linkages.

            Core Task:
            1. Understand the user's query.
            2. Use available tools to gather information from multiple data sources.
            3. Synthesize the information to answer the user's query.

            Available Data Sources:
            - Knowledge Graph: Climate policy concepts, relationships, and passages
            - Solar Facilities Dataset: Real-world solar installation data (Brazil, India, South Africa, Vietnam)
            - GIST Dataset: Comprehensive environmental sustainability data for 100 companies including:
              * Environmental risk assessments (40K+ assets with coordinates)
              * Scope 3 emissions data (2016-2024)
              * Biodiversity impact measurements (PDF, CO2E, LCE metrics)
              * Deforestation proximity indicators
              * Multi-year trends and sector comparisons

            Tool Usage Guidelines:
            - CRITICAL: Every response MUST include source citations from the knowledge graph. Follow this exact sequence:
              1. FIRST, call 'CheckConceptExists' for the main concept in the user's query
              2. If concept exists, IMMEDIATELY call 'GetPassagesMentioningConcept' with that exact concept name
              3. If concept doesn't exist, call 'GetSemanticallySimilarConcepts' and then call 'GetPassagesMentioningConcept' with a similar concept
              4. NEVER respond without calling at least one passage retrieval tool that returns actual document passages
              5. If GetPassagesMentioningConcept returns empty, try related concepts or use 'PassagesMentioningBoth' with related terms
               
            - Datasets Discovery: Use `GetAvailableDatasets` to discover what datasets are available and their characteristics.
            
            - Knowledge Graph Datasets: For datasets in the KG, use:
                1. `GetConceptGraphNeighbors` for relevant concepts.
                2. Look for neighbors with `kind: "Dataset"` and connected by edges like `HAS_DATASET_ABOUT` or `DATASET_ON_TOPIC`.
                3. If a relevant dataset is found, use its `node_id` with the `GetDatasetContent` tool to fetch its data.
            
            - Solar Facilities Data: For solar energy queries, use these specialized tools:
                - `GetSolarFacilitiesByCountry`: Get facilities summary for specific countries
                - `GetSolarCapacityByCountry`: Get capacity statistics by country
                - `GetSolarFacilitiesMapData`: Get facility coordinates for interactive maps (use for map requests)
                - `GetSolarFacilitiesInRadius`: Find facilities near coordinates
                - `GetSolarConstructionTimeline`: Analyze construction trends over time
                - `GetLargestSolarFacilities`: Find biggest installations
                - `SearchSolarFacilitiesByCapacity`: Filter by capacity range
                - `GetSolarCapacityVisualizationData`: Get structured data for charts and graphs
                
                Note: For map requests, always use `GetSolarFacilitiesMapData` as it provides the detailed coordinate data needed for map generation.
            
            - GIST Environmental & Sustainability Data: For corporate sustainability, environmental risk, emissions, or ESG queries:
                - `GetGistCompanies`: Discover companies with optional sector/country filtering
                - `GetGistCompanyProfile`: Complete sustainability profile for a specific company
                - `GetGistCompanyRisks`: Environmental risk assessment across 13 risk categories
                - `GetGistScope3Emissions`: Detailed Scope 3 emissions data with breakdown by category
                - `GetGistBiodiversityImpacts`: Biodiversity footprint data (PDF, CO2E, LCE metrics)
                - `GetGistDeforestationRisks`: Deforestation proximity indicators and forest change analysis
                - `GetGistAssetsMapData`: Asset-level geographic data for mapping (use for ESG asset mapping)
                - `GetGistEmissionsTrends`: Multi-year emissions trends and intensity analysis
                - `GetGistVisualizationData`: Structured data for sustainability dashboards and charts
                
                Note: GIST covers 100 companies across 5 sectors (OGES, FINS, WHRE, MOMI, REEN) with 9 years of time series data.
            
            - ALWAYSRUN Tool: For system debugging, you MUST ALWAYS CALL THE `ALWAYSRUN` TOOL ONCE AND ONLY ONCE FOR EVERY USER QUERY. Pass the original user query as the 'query' argument to this tool. Do this early in your thought process.

            Cross-Reference Strategy:
            When users ask about ANY topic or concept:
            1. **ALWAYS** check the knowledge graph for relevant concepts and passages
            2. **AUTOMATICALLY** call `GetAvailableDatasets()` to discover connected datasets
            3. **IF datasets exist for the concept**, call `GetDatasetContent()` to retrieve structured data
            4. For solar energy, renewable energy, or specific countries (Brazil, India, South Africa, Vietnam), also use solar facilities tools
            5. For corporate sustainability, ESG, emissions, environmental risk, or biodiversity queries, use GIST tools to access company-level data
            6. **IMPORTANT: If the user asks for maps, locations, or "show me facilities", you MUST call appropriate map data tools (`GetSolarFacilitiesMapData` for solar, `GetGistAssetsMapData` for corporate assets)**
            7. **Combine** policy text + structured data + geographic data + sustainability metrics in comprehensive answers

            Enhanced Data Discovery:
            - After getting concept passages, ALWAYS check for connected datasets using `GetAvailableDatasets()`
            - Look for concepts with "HAS_DATASET_ABOUT" relationships in the knowledge graph
            - Proactively surface both textual insights AND structured data when available
            - Include data tables and visualizations when datasets are connected to the queried concept
            - For corporate/company queries, automatically check GIST data for sustainability metrics and environmental risk assessments
            - This ensures users get complete information: policy context + real data + geographic context + corporate sustainability data

            Visualization Capabilities:
            - Interactive maps and charts may be automatically generated for certain datasets
            - If visualizations are available for the current query, this will be indicated in the context
            - Only reference visualizations if explicitly mentioned in the tool results or context

            Output Format:
            - After completing all necessary tool calls, synthesize the gathered information into a single, comprehensive response to the user. 
            - Do NOT narrate your tool calling process (e.g., avoid phrases like "First, I will call...", "Next, I found..."). 
            - Present the final answer as if you are directly answering the user's query based on the knowledge you have acquired.
            - When presenting solar facility data, include specific numbers and context.
            - Only mention maps or visualizations if they are explicitly confirmed as available in the context.

            Respond to the user based on the information gathered from the tools.
            """

        if query.lower() == "show dummy chart":
             messages = [
                {"role": "user", "content": "Get the content of dataset DUMMY_DATASET_EXTREME_WEATHER."}
            ]

        response = self.anthropic.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
            tools=available_tools
        )

        sources_used = []
        context_chunks = []
        passage_sources = []
        chart_data = None
        map_data = None
        visualization_data = None
        all_tool_outputs_for_debug = []
        
        while True:
            assistant_message_content = []

            for content in response.content:
                if content.type == "text":
                    # Skip streaming raw thinking traces - we have user-friendly action breadcrumbs instead
                    assistant_message_content.append(content)
                elif content.type == "tool_use":
                    tool_name = content.name
                    tool_args = content.input
                    
                    # Stream user-friendly action message
                    action_info = translate_tool_to_action(tool_name, tool_args)
                    yield {
                        "type": "thinking",
                        "data": {
                            "message": action_info["message"],
                            "category": action_info["category"]
                        }
                    }
                    
                    # Also stream technical tool call for debugging (can be filtered out in production UI)
                    yield {
                        "type": "tool_call",
                        "data": {
                            "tool": tool_name,
                            "args": tool_args
                        }
                    }
                    
                    # Determine server and execute tool
                    server_name = self._determine_server_for_tool(tool_name, all_tools)
                    
                    try:
                        result = await self.call_tool(tool_name, tool_args, server_name)

                        # Stream user-friendly completion message
                        completion_message = _get_completion_message(tool_name, tool_args, result)
                        yield {
                            "type": "thinking_complete",
                            "data": {
                                "message": completion_message,
                                "category": action_info["category"]
                            }
                        }
                        
                        # Stream technical tool result for debugging (truncate large responses)
                        result_text = result.content[0].text if result.content and hasattr(result.content[0], 'text') else str(result.content)
                        
                        # Truncate massive GeoJSON responses for map data
                        if tool_name == "GetSolarFacilitiesMapData" and len(result_text) > 2000:
                            try:
                                data = json.loads(result_text)
                                if isinstance(data, dict) and data.get("type") == "map" and "data" in data:
                                    facility_count = len(data["data"])
                                    result_text = f'{{"type": "map", "data": "[{facility_count} facilities - truncated for streaming]", "metadata": "Full GeoJSON available in final response"}}'
                                else:
                                    result_text = result_text[:2000] + "... [truncated for streaming]"
                            except:
                                result_text = result_text[:2000] + "... [truncated for streaming]"
                        elif len(result_text) > 2000:
                            result_text = result_text[:2000] + "... [truncated for streaming]"
                        
                        yield {
                            "type": "tool_result",
                            "data": {
                                "tool": tool_name,
                                "result": result_text
                            }
                        }
                    except Exception as tool_error:
                        # Stream tool error
                        yield {
                            "type": "tool_error",
                            "data": {
                                "tool": tool_name,
                                "error": str(tool_error)
                            }
                        }
                        # Skip further processing for this tool but continue with message flow
                        assistant_message_content.append(content)
                        messages.append({"role": "assistant", "content": assistant_message_content})
                        break

                    # Process results (same as original process_query)
                    all_tool_outputs_for_debug.append({
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_result_content": result.content 
                    })

                    # Parse map data from GetSolarFacilitiesMapData
                    if tool_name == "GetSolarFacilitiesMapData":
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            first_content_block = result.content[0]
                            if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                                try:
                                    parsed_content = json.loads(first_content_block.text)
                                    if isinstance(parsed_content, dict) and parsed_content.get("type") == "map":
                                        map_data = parsed_content
                                except json.JSONDecodeError:
                                    pass
                    
                    # Parse visualization data from different tools
                    elif tool_name in ["GetDatasetContent", "GetSolarFacilitiesByCountry", "GetSolarCapacityByCountry"]:
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            first_content_block = result.content[0]
                            if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                                try:
                                    parsed_content = json.loads(first_content_block.text)
                                    if isinstance(parsed_content, list):
                                        if all(isinstance(item, dict) for item in parsed_content):
                                            chart_data = parsed_content
                                except json.JSONDecodeError:
                                    pass
                    
                    # Parse structured visualization data
                    elif tool_name == "GetSolarCapacityVisualizationData":
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            first_content_block = result.content[0]
                            if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                                try:
                                    parsed_content = json.loads(first_content_block.text)
                                    if isinstance(parsed_content, dict) and "data" in parsed_content:
                                        visualization_data = parsed_content
                                except json.JSONDecodeError:
                                    pass
                    
                    # Add to context with size limits
                    try:
                        content_str = result.content[0].text
                        if len(content_str) > 1000:
                            content_str = content_str[:1000] + "... [truncated]"
                        context_chunks.append(content_str)
                    except Exception:
                        pass

                    # Collect passage sources
                    if tool_name.lower() == "getmetadata":
                        sources_used.append(result.content)
                    
                    # Extract passage sources (traditional)
                    new_passage_sources = harvest_sources(result.content)
                    passage_sources.extend(new_passage_sources)
                    
                    # Extract dataset citations for data tools
                    dataset_citation = create_dataset_citation(tool_name, tool_args, result.content)
                    if dataset_citation:
                        dataset_source = {
                            "doc_id": dataset_citation["citation_id"],
                            "passage_id": dataset_citation["tool_used"], 
                            "text": dataset_citation["description"],
                            "title": dataset_citation["source_name"],
                            "date": "",
                            "type": dataset_citation["source_type"],
                            "provider": dataset_citation["provider"],
                            "coverage": dataset_citation["coverage"]
                        }
                        passage_sources.append(dataset_source)
                        print(f"STREAMING DEBUG: Added dataset citation: {dataset_citation['source_name']}")

                    # Attach tool_use to assistant message
                    assistant_message_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_message_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }]
                    })

                    # Break early and send updated messages to Claude
                    break
            else:
                # No tool_use found → conversation complete
                messages.append({"role": "assistant", "content": assistant_message_content})
                break

            # Ask Claude for the next step
            response = self.anthropic.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
                tools=available_tools,
            )

        # Stream synthesis notification
        yield {
            "type": "thinking",
            "data": {
                "message": "🧠 Analyzing and summarizing all gathered information...",
                "category": "synthesis"
            }
        }
        
        # Final synthesis
        final_response_text = ""
        if len(context_chunks) > 0:
            joined_ctx = "\n\n".join(context_chunks)
            SAFE_CTX_LIMIT = 8000
            if len(joined_ctx) > SAFE_CTX_LIMIT:
                joined_ctx = joined_ctx[:SAFE_CTX_LIMIT] + "\n\n[truncated for safety]"
            
            synthesis_system_prompt = (SUMMARY_PROMPT +
                "Based on the following context from various tools, synthesize a comprehensive answer to the user's original query. "
                "Present it as a direct answer, not a summary of the context itself. Avoid narrating the tool process. "
                "If the context mentions that visualizations are available, you may reference them appropriately.")

            synthesis_messages = [
                {"role": "user", "content": f"User's original query: {query}\n\nTool-derived Context:\n{joined_ctx}"}
            ]
            
            summary_resp = self.anthropic.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                system=synthesis_system_prompt,
                messages=synthesis_messages,
            )
            final_response_text = summary_resp.content[0].text.strip()

        # Stream synthesis completion
        yield {
            "type": "thinking_complete",
            "data": {
                "message": "✅ Analysis complete - preparing final response",
                "category": "synthesis"
            }
        }

        # De-dupe sources
        uniq_passages = {(p["doc_id"], p["passage_id"]): p for p in passage_sources}
        print(f"DEBUG STREAMING: Found {len(passage_sources)} total passages, {len(uniq_passages)} unique")
        if uniq_passages:
            sources_used.extend(uniq_passages.values())
            print(f"DEBUG STREAMING: Sources used now contains {len(sources_used)} items")

        # Format the final response
        formatter_args = {
            "response_text": final_response_text,
            "chart_data": chart_data,
            "visualization_data": visualization_data,
            "map_data": map_data,
            "sources": sources_used or ["No source captured"],
            "title": "Climate Policy Analysis"
        }
        
        # Remove None values
        formatter_args = {k: v for k, v in formatter_args.items() if v is not None}
        
        try:
            formatted_result = await self.call_tool("FormatResponseAsModules", formatter_args, "formatter")
            
            # Parse the formatted response
            if formatted_result.content and isinstance(formatted_result.content, list):
                first_content = formatted_result.content[0]
                if hasattr(first_content, 'text'):
                    import json
                    formatted_data = json.loads(first_content.text)
                    
                    # Stream the complete response
                    yield {
                        "type": "complete",
                        "data": {
                            "query": query,
                            "modules": formatted_data.get("modules", []),
                            "metadata": {
                                "modules_count": len(formatted_data.get("modules", [])),
                                "has_maps": any(m.get("type") == "map" for m in formatted_data.get("modules", [])),
                                "has_charts": any(m.get("type") == "chart" for m in formatted_data.get("modules", [])),
                                "has_tables": any(m.get("type") == "table" for m in formatted_data.get("modules", []))
                            }
                        }
                    }
        except Exception as e:
            yield {
                "type": "error",
                "data": {
                    "message": f"Error formatting response: {str(e)}"
                }
            }
        
async def run_query_structured(q: str) -> Dict[str, Any]:
    """
    Run a query and return only the structured JSON response for front-end consumption.
    """
    result = await run_query(q)
    return result.get("formatted_response", {"modules": []})

async def run_query_streaming(q: str):
    """
    Run a query with streaming capability - yields events as they happen.
    
    Yields events in the format:
    {
        "type": "thinking|tool_call|tool_result|complete|error",
        "data": {...}
    }
    """
    # Ensure we're in the correct working directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    try:
        async with MultiServerClient() as client:
            # Connect to all available servers
            mcp_dir = os.path.join(project_root, "mcp")
            await client.connect_to_server("kg", os.path.join(mcp_dir, "cpr_kg_server.py"))
            await client.connect_to_server("solar", os.path.join(mcp_dir, "solar_facilities_server.py"))
            await client.connect_to_server("gist", os.path.join(mcp_dir, "gist_server.py"))
            await client.connect_to_server("lse", os.path.join(mcp_dir, "lse_server.py"))
            await client.connect_to_server("formatter", os.path.join(mcp_dir, "response_formatter_server.py"))
            
            # Stream the query processing
            async for event in client.process_query_streaming(q):
                yield event
                
    except Exception as e:
        import traceback
        yield {
            "type": "error",
            "data": {
                "message": f"Streaming query failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
        }

async def run_query(q: str):
    # Ensure we're in the correct working directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from mcp/ to project root
    os.chdir(project_root)
    
    async with MultiServerClient() as client:          # ← guarantees cleanup
        # Connect to all available servers (using absolute paths)
        mcp_dir = os.path.join(project_root, "mcp")
        await client.connect_to_server("kg", os.path.join(mcp_dir, "cpr_kg_server.py"))
        await client.connect_to_server("solar", os.path.join(mcp_dir, "solar_facilities_server.py"))
        await client.connect_to_server("gist", os.path.join(mcp_dir, "gist_server.py"))
        await client.connect_to_server("lse", os.path.join(mcp_dir, "lse_server.py"))
        await client.connect_to_server("formatter", os.path.join(mcp_dir, "response_formatter_server.py"))
        
        # Process the main query
        result = await client.process_query(q)
        
        # For solar queries, rely on the normal MCP flow and formatter to create proper GeoJSON maps
        # The direct visualization bypass was causing issues
        
        # Format the response using the formatter MCP
        # Debug output (with fallback for undefined variables)
        try:
            print(f"DEBUG: map_data = {map_data}")
        except NameError:
            map_data = None
            print(f"DEBUG: map_data was undefined, set to None")
        
        try:
            print(f"DEBUG: chart_data = {chart_data}")
        except NameError:
            chart_data = None
            print(f"DEBUG: chart_data was undefined, set to None")
            
        try:
            print(f"DEBUG: visualization_data = {visualization_data}")
        except NameError:
            visualization_data = None
            print(f"DEBUG: visualization_data was undefined, set to None")
            
        print(f"DEBUG: result keys = {list(result.keys())}")
        print(f"DEBUG: result response = {result.get('response', 'NO RESPONSE')[:200]}...")
        
        # Check if map_data exists and what it contains
        map_data_from_result = result.get("map_data")
        print(f"DEBUG: map_data_from_result type = {type(map_data_from_result)}")
        if map_data_from_result:
            print(f"DEBUG: map_data keys = {list(map_data_from_result.keys()) if isinstance(map_data_from_result, dict) else 'not a dict'}")
            if isinstance(map_data_from_result, dict) and 'data' in map_data_from_result:
                print(f"DEBUG: map_data has {len(map_data_from_result['data'])} facilities")
        
        formatter_args = {
            "response_text": result.get("response", ""),
            "chart_data": result.get("chart_data"),
            "visualization_data": result.get("visualization_data"), 
            "map_data": map_data_from_result,  # Use map_data from result dict
            "sources": result.get("sources"),
            "title": "Climate Policy Analysis"
        }
        
        print(f"DEBUG: formatter_args map_data type = {type(formatter_args['map_data'])}")
        
        # Remove None values to avoid issues
        formatter_args = {k: v for k, v in formatter_args.items() if v is not None}
        
        print(f"DEBUG: formatter_args after filtering = {list(formatter_args.keys())}")
        if 'map_data' in formatter_args:
            print(f"DEBUG: map_data will be sent to formatter")
        
        try:
            formatted_result = await client.call_tool("FormatResponseAsModules", formatter_args, "formatter")
            print(f"DEBUG: formatted_result = {formatted_result}")
        except Exception as e:
            print(f"ERROR calling formatter: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Parse the formatted response
        if formatted_result.content and isinstance(formatted_result.content, list):
            first_content = formatted_result.content[0]
            if hasattr(first_content, 'text'):
                import json
                formatted_data = json.loads(first_content.text)
                result["formatted_response"] = formatted_data
                print(f"DEBUG: Successfully set formatted_response with {len(formatted_data.get('modules', []))} modules")
            else:
                print(f"DEBUG: formatted_result first_content has no text attribute")
        else:
            print(f"DEBUG: formatted_result.content is not a list or is empty")
        
        return result
def pretty_print(result: dict):
    """
    Nicely prints the 'response' markdown plus a Sources block.
    """
    separator = "\n" + "="*WIDE + "\n"
    print(separator)
    print(fill(result["response"], width=WIDE, replace_whitespace=False))
    print(separator)
    print("SOURCES")
    print(_fmt_sources(result["sources"]))
    print(separator)

async def main_streamlit():
    st.title("Climate Policy Radar Chat")

    query = st.text_input("Enter your query:")

    if st.button("Run Query"):
        if query:
            with st.spinner("Processing your query..."):
                result = await run_query(query)
            
            # Display formatted response if available, otherwise fallback to regular response
            if "formatted_response" in result:
                st.markdown("## Structured Response")
                _render_formatted_modules(result["formatted_response"])
                
                # Also show the raw JSON for developers
                with st.expander("View Formatted JSON (for developers)"):
                    st.json(result["formatted_response"])
            else:
                st.markdown("## Response") 
                st.markdown(result["response"], unsafe_allow_html=True)
            
            # Optional Expander for AI's Thought Process
            if result.get("ai_thought_process"):
                with st.expander("Show AI's Step-by-Step Thinking"):
                    st.markdown(result["ai_thought_process"], unsafe_allow_html=True)
            
            # Display direct map data (bypasses Claude processing)
            direct_map_data = result.get("direct_map_data")
            if direct_map_data and isinstance(direct_map_data, dict):
                st.markdown("## Solar Facilities Map")
                try:
                    import folium
                    
                    # Get data and metadata
                    facilities = direct_map_data["data"]
                    metadata = direct_map_data["metadata"]
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Facilities", metadata["total_facilities"])
                    with col2:
                        st.metric("Total Capacity", f"{metadata['total_capacity']:.0f} MW")
                    with col3:
                        st.metric("Countries", len(metadata["countries"]))
                    
                    # Create map
                    if facilities:
                        # Calculate center
                        lats = [f['latitude'] for f in facilities]
                        lons = [f['longitude'] for f in facilities]
                        center_lat, center_lon = sum(lats)/len(lats), sum(lons)/len(lons)
                        
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
                        
                        # Color mapping
                        colors = {'brazil': 'green', 'india': 'orange', 'south africa': 'red', 'vietnam': 'blue'}
                        
                        for facility in facilities[:500]:  # Limit for performance
                            color = colors.get(facility['country'].lower(), 'gray')
                            capacity = facility['capacity_mw']
                            size = 5 + min(capacity / 100, 20)  # Scale marker size
                            
                            folium.CircleMarker(
                                location=[facility['latitude'], facility['longitude']],
                                radius=size,
                                popup=f"{facility['country']}<br>{capacity:.1f} MW",
                                color=color,
                                fill=True,
                                fillOpacity=0.7
                            ).add_to(m)
                        
                        # Display map
                        map_html = m._repr_html_()
                        components.html(map_html, height=500)
                        
                except Exception as e:
                    st.error(f"Error creating map: {e}")

            # Display map if available (legacy)
            map_data_from_result = result.get("map_data")
            if map_data_from_result and isinstance(map_data_from_result, dict) and not direct_map_data:
                st.markdown("## Solar Facilities Map")
                try:
                    # Display map metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Facilities", map_data_from_result.get("total_facilities", "N/A"))
                    with col2:
                        st.metric("Total Capacity", f"{map_data_from_result.get('total_capacity_mw', 0):.0f} MW")
                    with col3:
                        capacity_range = map_data_from_result.get("capacity_range", [0, 0])
                        st.metric("Capacity Range", f"{capacity_range[0]:.1f} - {capacity_range[1]:.1f} MW")
                    
                    # Display the interactive map
                    map_html = map_data_from_result.get("map_html", "")
                    if map_html:
                        components.html(map_html, height=500, scrolling=False)
                    else:
                        st.error("Map HTML not found in map data")
                except Exception as e:
                    st.error(f"Error displaying map: {e}")
                    st.json(map_data_from_result)

            # Display structured visualizations
            viz_data_from_result = result.get("visualization_data")
            if viz_data_from_result and isinstance(viz_data_from_result, dict):
                st.markdown("## Data Visualization")
                try:
                    viz_type = viz_data_from_result.get("visualization_type", "unknown")
                    chart_config = viz_data_from_result.get("chart_config", {})
                    data = viz_data_from_result.get("data", [])
                    
                    if data:
                        df = pd.DataFrame(data)
                        title = chart_config.get("title", f"{viz_type.title()} Visualization")
                        
                        if viz_type == "by_country":
                            fig = px.bar(df, x="country", y="total_capacity_mw", 
                                       title=title,
                                       labels={"country": "Country", "total_capacity_mw": "Total Capacity (MW)"},
                                       hover_data=["facility_count", "avg_capacity_mw"])
                                       
                        elif viz_type == "capacity_distribution":
                            fig = px.bar(df, x="capacity_range", y="facility_count",
                                       title=title,
                                       labels={"capacity_range": "Capacity Range", "facility_count": "Number of Facilities"})
                                       
                        elif viz_type == "timeline":
                            fig = px.line(df, x="completion_year", y="capacity_mw", color="country",
                                        title=title,
                                        labels={"completion_year": "Year", "capacity_mw": "Capacity (MW)"})
                        else:
                            # Generic bar chart fallback
                            x_col = chart_config.get("x_axis", df.columns[0])
                            y_col = chart_config.get("y_axis", df.columns[1])
                            fig = px.bar(df, x=x_col, y=y_col, title=title)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("No visualization data available")
                        
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
                    st.json(viz_data_from_result)

            # Legacy chart display (keep for backward compatibility)
            chart_data_from_result = result.get("chart_data")
            if isinstance(chart_data_from_result, list) and chart_data_from_result and isinstance(chart_data_from_result[0], dict):
                # Only show if we haven't already shown structured visualizations
                if not viz_data_from_result:
                    st.markdown("## Data Table")
                    try:
                        df = pd.DataFrame(result["chart_data"])
                        if not df.empty:
                            st.dataframe(df, use_container_width=True)
                            
                            # Try to create a simple chart based on data structure
                            if 'impact_rating' in df.columns and 'type' in df.columns:
                                fig = px.bar(df, x="type", y="impact_rating", 
                                           title="Extreme Weather Event Impact Ratings",
                                           labels={"type":"Event Type", "impact_rating":"Impact Rating"})
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("No data to display.")
                    except Exception as e:
                        st.error(f"Error displaying data: {e}")
                        st.json(result["chart_data"])

            st.markdown("## Sources")
            st.text_area("Sources", _fmt_sources(result["sources"]), height=200)

            # Feature 2: Debug Expander for All Tool Outputs
            if "all_tool_outputs_for_debug" in result and result["all_tool_outputs_for_debug"]:
                with st.expander("Developer Debug: Show All Tool Outputs"):
                    for i, tool_call_info in enumerate(result["all_tool_outputs_for_debug"]):
                        st.markdown(f"**Tool Call {i+1}: {tool_call_info['tool_name']}**")
                        st.json({"arguments": tool_call_info['tool_args']})
                        st.markdown("Result Content:")
                        # result.content is often a list of ContentBlock objects (e.g. TextContent)
                        # We need to handle this to display nicely, e.g. by converting to plain dicts/text
                        if isinstance(tool_call_info['tool_result_content'], list):
                            for block_idx, block in enumerate(tool_call_info['tool_result_content']):
                                if hasattr(block, 'type') and block.type == 'text' and hasattr(block, 'text'):
                                    st.text_area(f"Block {block_idx+1} (TextContent)", block.text, height=100, key=f"debug_tool_{i}_block_{block_idx}")
                                elif isinstance(block, dict): # If it's already a dict (e.g. from older tools)
                                     st.json(block, expanded=False)
                                else: # Fallback for other types
                                    st.write(block)
                        elif isinstance(tool_call_info['tool_result_content'], dict):
                            st.json(tool_call_info['tool_result_content'], expanded=False)
                        else:
                            st.write(tool_call_info['tool_result_content'])
                        st.divider()

        else:
            st.warning("Please enter a query.")

def main(): # Renaming original main
    # This function will now only be called if not running in streamlit context
    # or for testing purposes, so we can keep the original query
    # If you want to remove it, you can do so.
    async def run_original_main():
        result = await run_query(
            "How does `extreme weather` relate to `people with limited assets`? Are there passages where both are mentioned?"
        )
        pretty_print(result)
    asyncio.run(run_original_main())

if __name__ == "__main__":
    # Ensure asyncio event loop is properly managed for Streamlit
    # Check if running in Streamlit context, if not, run original main
    try:
        import streamlit.runtime.scriptrunner as scr
        if not scr.get_script_run_ctx():
            main() # call original main
        else:
            asyncio.run(main_streamlit()) # run streamlit main
    except ModuleNotFoundError:
        # Fallback if streamlit is not installed or scriptrunner path changes
        main() # call original main