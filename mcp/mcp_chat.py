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
import aiohttp

def create_citation_info_for_tool(tool_name: str, tool_args: dict = None) -> dict:
    """
    Create citation metadata for a given tool call.
    
    Args:
        tool_name: Name of the tool called
        tool_args: Arguments passed to the tool
        
    Returns:
        Citation info dictionary with source_name, provider, etc.
    """
    # Tool name to citation info mapping
    citation_mapping = {
        "CheckConceptExists": {
            "source_name": "Climate Policy Knowledge Graph - Concepts",
            "provider": "Climate Policy Radar",
            "spatial_coverage": "Global climate policy concepts",
            "temporal_coverage": "Current",
            "source_url": "https://climatepolicyradar.org"
        },
        "GetPassagesMentioningConcept": {
            "source_name": "Climate Policy Knowledge Graph - Passages", 
            "provider": "Climate Policy Radar",
            "spatial_coverage": "Global policy documents",
            "temporal_coverage": "Historical and current",
            "source_url": "https://climatepolicyradar.org"
        },
        "GetSolarFacilitiesByCountry": {
            "source_name": "TZ-SAM Q1 2025 Solar Database",
            "provider": "TransitionZero",
            "spatial_coverage": tool_args.get("country", "Global") if tool_args else "Global",
            "temporal_coverage": "Q1 2025",
            "source_url": "https://transitionzero.org"
        },
        "GetSolarFacilitiesMapData": {
            "source_name": "TZ-SAM Q1 2025 Solar Facilities Database",
            "provider": "TransitionZero", 
            "spatial_coverage": "Brazil, India, South Africa, Vietnam",
            "temporal_coverage": "Q1 2025",
            "source_url": "https://transitionzero.org"
        },
        "GetSolarCapacityByCountry": {
            "source_name": "TZ-SAM Solar Capacity Database",
            "provider": "TransitionZero",
            "spatial_coverage": "Global",
            "temporal_coverage": "Q1 2025", 
            "source_url": "https://transitionzero.org"
        },
        "GetSolarFacilityDetails": {
            "source_name": "TZ-SAM Q1 2025 Solar Facility Details",
            "provider": "TransitionZero",
            "spatial_coverage": "Global solar facilities",
            "temporal_coverage": "Q1 2025",
            "source_url": "https://transitionzero.org"
        },
        "GetGistCompanies": {
            "source_name": "GIST Corporate ESG Database",
            "provider": "GIST",
            "spatial_coverage": "Global corporations",
            "temporal_coverage": "2016-2024",
            "source_url": "https://gist.com"
        },
        "GetBrazilianStatesOverview": {
            "source_name": "LSE Brazilian Climate Governance Database",
            "provider": "London School of Economics",
            "spatial_coverage": "Brazil (all 27 states)",
            "temporal_coverage": "Current",
            "source_url": "https://lse.ac.uk"
        }
    }
    
    # Get citation info for this tool, with fallback
    citation_info = citation_mapping.get(tool_name, {
        "source_name": f"{tool_name} Dataset",
        "provider": "Unknown Provider",
        "spatial_coverage": "Unknown Coverage", 
        "temporal_coverage": "Unknown Period",
        "source_url": ""
    })
    
    return citation_info

def wrap_tool_result_with_citation(tool_name: str, tool_result_text: str, tool_args: dict = None) -> dict:
    """
    Wrap a tool result with citation information in the expected format.
    
    Args:
        tool_name: Name of the tool that generated the result
        tool_result_text: Raw JSON text result from the tool
        tool_args: Arguments passed to the tool
        
    Returns:
        Dictionary in {"fact": result, "citation_info": {...}} format
    """
    # CITATION DEBUG: Analyze tool result for errors/failures
    tool_success = True
    error_reasons = []
    
    # Check for common failure patterns
    if "Error" in tool_result_text:
        tool_success = False
        error_reasons.append("Error in result")
    elif tool_result_text.strip() in ["[]", "{}", '""']:
        tool_success = False
        error_reasons.append("Empty result")
    # Note: "false" is a valid boolean response, not a failure
    elif "null" in tool_result_text.lower():
        tool_success = False
        error_reasons.append("Null result")
    
    # Log the analysis
    status = "SUCCESS" if tool_success else "FAILED"
    print(f"🔍 CITATION DEBUG - Tool: {tool_name}, Status: {status}")
    if not tool_success:
        print(f"   ❌ Failure reasons: {', '.join(error_reasons)}")
        print(f"   📄 Result snippet: {tool_result_text[:100]}...")
    
    citation_info = create_citation_info_for_tool(tool_name, tool_args)
    
    # Add success/failure metadata to citation info
    citation_info["tool_success"] = tool_success
    citation_info["error_reasons"] = error_reasons
    
    return {
        "fact": tool_result_text,  # Keep original JSON string as the fact
        "citation_info": citation_info
    }

def convert_citation_format(text: str) -> str:
    """
    Convert [citation_X] format to ^X^ format for frontend compatibility.
    
    Args:
        text: Text containing [citation_1], [citation_2] style citations
        
    Returns:
        Text with ^1^, ^2^ style citations
    """
    if not text:
        return text
        
    # Replace [citation_1] with ^1^, [citation_2] with ^2^, etc.
    def replace_citation(match):
        citation_id = match.group(1)  # Extract the number part
        # Handle both citation_1 and just 1 formats
        if citation_id.startswith('citation_'):
            number = citation_id.replace('citation_', '')
        else:
            number = citation_id
        return f"^{number}^"
    
    # Pattern matches [citation_1], [citation_2], etc.
    pattern = r'\[citation_(\w+)\]'
    converted_text = re.sub(pattern, replace_citation, text)
    
    print(f"CITATION FORMAT DEBUG: Converted citations in text: {len(re.findall(pattern, text))} found")
    return converted_text


load_dotenv()

# Global singleton client for performance optimization
_global_client = None
_client_lock = asyncio.Lock()

async def get_global_client():
    """Get or create the global singleton MCP client."""
    global _global_client
    
    async with _client_lock:
        if _global_client is None:
            _global_client = MultiServerClient()
            await _global_client.__aenter__()
            
            # Connect to all servers
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            mcp_dir = os.path.join(project_root, "mcp")
            
            try:
                await _global_client.connect_to_server("kg", os.path.join(mcp_dir, "cpr_kg_server.py"))
                await _global_client.connect_to_server("solar", os.path.join(mcp_dir, "solar_facilities_server.py"))
                await _global_client.connect_to_server("gist", os.path.join(mcp_dir, "gist_server.py"))
                await _global_client.connect_to_server("lse", os.path.join(mcp_dir, "lse_server.py"))
                await _global_client.connect_to_server("formatter", os.path.join(mcp_dir, "response_formatter_server.py"))
                print("Global MCP client initialized successfully")
            except Exception as e:
                print(f"Error initializing global MCP client: {e}")
                # Clean up on failure
                await _global_client.__aexit__(None, None, None)
                _global_client = None
                raise
                
    return _global_client

async def cleanup_global_client():
    """Clean up the global singleton client."""
    global _global_client
    
    async with _client_lock:
        if _global_client is not None:
            await _global_client.__aexit__(None, None, None)
            _global_client = None
            print("Global MCP client cleaned up")

class CitationRegistry:
    """
    Manages citation numbering and tracking for inline citations.
    
    Assigns unique citation numbers to sources and tracks which modules use which citations.
    """
    
    def __init__(self):
        self.citations = {}  # source_key -> citation_number
        self.citation_counter = 1
        self.module_citations = {}  # module_id -> list of citation numbers
        self.citation_details = {}  # citation_number -> full source dict
        
    def add_source(self, source: Dict[str, Any], module_id: str = None) -> int:
        """
        Add a source and return its citation number.
        
        Args:
            source: Source dictionary (passage or dataset)
            module_id: Optional module identifier for tracking
            
        Returns:
            Citation number for this source
        """
        # Create unique key for this source
        source_key = self._generate_source_key(source)
        
        # Check if we already have this source
        if source_key in self.citations:
            citation_num = self.citations[source_key]
        else:
            # Assign new citation number
            citation_num = self.citation_counter
            self.citations[source_key] = citation_num
            self.citation_details[citation_num] = source
            self.citation_counter += 1
        
        # Track module association
        if module_id:
            if module_id not in self.module_citations:
                self.module_citations[module_id] = []
            if citation_num not in self.module_citations[module_id]:
                self.module_citations[module_id].append(citation_num)
        
        return citation_num
    
    def _generate_source_key(self, source: Dict[str, Any]) -> str:
        """Generate unique key for source deduplication."""
        if isinstance(source, dict):
            # For passage sources
            if "doc_id" in source and "passage_id" in source:
                return f"passage_{source['doc_id']}_{source['passage_id']}"
            # For dataset sources
            elif "citation_id" in source:
                return f"dataset_{source['citation_id']}"
            # For tool-based sources
            elif "tool_used" in source and "source_name" in source:
                return f"tool_{source['tool_used']}_{source['source_name']}"
        
        # Fallback for other source types
        return f"generic_{hash(str(source))}"
    
    def get_module_citations(self, module_id: str) -> List[int]:
        """Get all citation numbers used by a specific module."""
        return self.module_citations.get(module_id, [])
    
    def get_citation_details(self, citation_num: int) -> Dict[str, Any]:
        """Get full source details for a citation number."""
        return self.citation_details.get(citation_num, {})
    
    def get_all_citations(self) -> Dict[int, Dict[str, Any]]:
        """Get all citations ordered by number."""
        return {num: self.citation_details[num] for num in sorted(self.citation_details.keys())}
    
    def format_citation_superscript(self, citation_nums: List[int]) -> str:
        """Format citation numbers as superscript text."""
        if not citation_nums:
            return ""
        if len(citation_nums) == 1:
            return f"^{citation_nums[0]}^"
        else:
            return f"^{','.join(map(str, sorted(citation_nums)))}^"

MAX_CTX_CHARS = 18_000          # hard cap – keep below Claude-Haiku context
WIDE = 88          # tweak for your terminal width
SUMMARY_PROMPT = (
    "You are a climate-policy expert. "
    "Assume the reader wants the big picture and key linkages."
    "You should always look for passages that are relevant to the user's query."
    "You should look for data to support the user's query, especially when they ask for it."
    "If given multiple concepts, you should look for passages that are relevant to all of them."
    "You should call at least one of the following tools AT LEAST ONCE for every query: GetPassagesMentioningConcept, PassagesMentioningBothConcepts"
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
        "GetLargestSolarFacilities": {
            "source_type": "Dataset",
            "source_name": "TZ-SAM Large Solar Facilities Database",
            "description": "Database of the world's largest solar installations by capacity",
            "provider": "TransitionZero",
            "coverage": "Global - Top 20 facilities"
        },
        
        # GIST Environmental Tools
        "GetGistAssetsMapData": {
            "source_type": "Database",
            "source_name": "GIST Environmental Database - Assets",
            "description": "Corporate asset locations and environmental risk data (40K+ assets)",
            "provider": "GIST Environmental Research",
            "coverage": "100+ companies, 5 sectors"
        },
        "GetGistCompanies": {
            "source_type": "Database",
            "source_name": "GIST Corporate Directory",
            "description": "Directory of companies with sustainability and environmental data",
            "provider": "GIST Environmental Research",
            "coverage": "100+ companies across 5 sectors"
        },
        "GetGistCompanyProfile": {
            "source_type": "Database",
            "source_name": "GIST Company Profiles Database",
            "description": "Complete sustainability profiles and environmental metrics for companies",
            "provider": "GIST Environmental Research",
            "coverage": "100+ companies"
        },
        "GetGistCompanyRisks": {
            "source_type": "Database",
            "source_name": "GIST Environmental Risk Database",
            "description": "Environmental risk assessments across 13 risk categories",
            "provider": "GIST Environmental Research",
            "coverage": "Corporate assets and facilities"
        },
        "GetGistEmissionsTrends": {
            "source_type": "Database", 
            "source_name": "GIST Scope 3 Emissions Database",
            "description": "Corporate Scope 3 emissions data with multi-year trends (2016-2024)",
            "provider": "GIST Environmental Research",
            "coverage": "100+ companies"
        },
        "GetGistScope3Emissions": {
            "source_type": "Database",
            "source_name": "GIST Scope 3 Emissions Database",
            "description": "Detailed Scope 3 emissions data with breakdown by category",
            "provider": "GIST Environmental Research",
            "coverage": "Corporate emissions data"
        },
        "GetGistBiodiversityImpacts": {
            "source_type": "Database",
            "source_name": "GIST Biodiversity Impact Database", 
            "description": "Corporate biodiversity footprint data (PDF, CO2E, LCE metrics)",
            "provider": "GIST Environmental Research",
            "coverage": "Multiple companies"
        },
        "GetGistDeforestationRisks": {
            "source_type": "Database",
            "source_name": "GIST Deforestation Risk Database",
            "description": "Deforestation proximity indicators and forest change analysis",
            "provider": "GIST Environmental Research",
            "coverage": "Asset-level deforestation risk data"
        },
        "GetGistVisualizationData": {
            "source_type": "Database",
            "source_name": "GIST Sustainability Dashboard Data",
            "description": "Structured corporate sustainability data for visualizations",
            "provider": "GIST Environmental Research", 
            "coverage": "100+ companies"
        },
        
        # LSE Climate Policy Tools
        "GetLSEDatasetOverview": {
            "source_type": "Database",
            "source_name": "LSE Climate Policy Analysis Database",
            "description": "Overview of all available LSE policy modules and datasets",
            "provider": "London School of Economics",
            "coverage": "Global climate governance data"
        },
        "GetBrazilianStatesOverview": {
            "source_type": "Database",
            "source_name": "LSE Brazilian State Governance Database",
            "description": "Overview of Brazilian state climate governance and policy frameworks",
            "provider": "London School of Economics",
            "coverage": "All 27 Brazilian states"
        },
        "GetStateClimatePolicy": {
            "source_type": "Database",
            "source_name": "LSE Climate Governance Database",
            "description": "State-level climate policy analysis and governance frameworks",
            "provider": "London School of Economics",
            "coverage": "Brazilian states (27 states)"
        },
        "CompareBrazilianStates": {
            "source_type": "Database",
            "source_name": "LSE Brazilian State Policy Database",
            "description": "Comparative analysis of climate policies across Brazilian states",
            "provider": "London School of Economics", 
            "coverage": "All 27 Brazilian states"
        },
        "GetNDCOverviewData": {
            "source_type": "Database", 
            "source_name": "LSE NDC Analysis Framework",
            "description": "Nationally Determined Contributions analysis and policy comparisons",
            "provider": "London School of Economics",
            "coverage": "Multiple countries"
        },
        
        # Knowledge Graph Tools  
        "GetDatasetContent": {
            "source_type": "Dataset",
            "source_name": "Climate Policy Knowledge Graph - Datasets",
            "description": "Structured datasets from climate policy knowledge graph",
            "provider": "Climate Policy Radar",
            "coverage": "Various climate policy datasets"
        },
        "GetPassagesMentioningConcept": {
            "source_type": "Document",
            "source_name": "Climate Policy Knowledge Graph - Documents",
            "description": "Policy documents and text passages mentioning specific concepts",
            "provider": "Climate Policy Radar",
            "coverage": "Global climate policy documents"
        },
        "CheckConceptExists": {
            "source_type": "Knowledge Base",
            "source_name": "Climate Policy Knowledge Graph - Concepts",
            "description": "Concept verification in climate policy knowledge graph",
            "provider": "Climate Policy Radar",
            "coverage": "Climate policy concepts and relationships"
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
    
    def _append_with_spans(base_record: dict, item: dict):
        """Append one or multiple source records depending on span availability."""
        spans = item.get("spans") if isinstance(item, dict) else None
        if isinstance(spans, list) and spans:
            for s in spans:
                rec = dict(base_record)
                # Carry span details through for richer citation descriptions
                rec["labelled_text"] = s.get("labelled_text")
                rec["start_index"] = s.get("start_index")
                rec["end_index"] = s.get("end_index")
                rec["concept_id"] = s.get("concept_id")
                rec["concept_label"] = s.get("concept_label")
                rec["labellers"] = s.get("labellers")
                rec["timestamps"] = s.get("timestamps")
                out.append(rec)
        else:
            out.append(base_record)

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
                                    _append_with_spans(source_record, passage)
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
                            _append_with_spans(source_record, data)
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
                                                _append_with_spans(source_record, passage)
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
            
            # Use database for visualization (capacity data not available in current schema)
            from solar_db import SolarDatabase
            db = SolarDatabase()
            stats = db.get_country_statistics()
            
            return {
                "type": "country_comparison",
                "data": stats[:20],  # Top 20 countries
                "chart_config": {
                    "x": "country",
                    "y": "facility_count", 
                    "title": "Solar Facilities by Country",
                    "chart_type": "bar"
                },
                "note": "Updated to use SQLite database - capacity data not available in current schema"
            }
            
        elif data_type == "facilities_map":
            # Create map data using database
            from solar_db import SolarDatabase
            db = SolarDatabase()
            country_filter = kwargs.get('country')
            
            if country_filter:
                facilities = db.get_facilities_by_country(country_filter, limit=1000)
            else:
                facilities = db.search_facilities(limit=1000)
                
            countries = list(set(f['country'] for f in facilities))
                
            return {
                "type": "map",
                "data": facilities,
                "metadata": {
                    "total_facilities": len(facilities),
                    "countries": countries,
                    "data_source": "SQLite Database"
                },
                "note": "Updated to use SQLite database for faster queries"
            }
            
        elif data_type == "capacity_distribution":
            # Capacity distribution not available in current database schema
            return {
                "type": "capacity_distribution",
                "error": "Capacity distribution not available - current database schema doesn't include capacity_mw field",
                "alternative": "Use country distribution or source timeline instead",
                "note": "Original TZ-SAM raw polygons file doesn't contain capacity information"
            }
            
    except Exception as e:
        print(f"Error in direct data access: {e}")
        return None

def _tool_produces_table_data(tool_name: str, parsed_data: any) -> bool:
    """Check if a tool produces data suitable for table generation."""
    
    # List of tools that produce tabular data
    table_tools = [
        "GetGistCompaniesBySector", "GetGistTopEmitters", "GetGistHighRiskCompanies",
        "GetSolarCapacityByCountry", "GetLargestSolarFacilities", "GetSolarConstructionTimeline", 
        "GetBrazilianStatesOverview", "CompareBrazilianStates", "GetGistEmissionsBySector",
        "GetGistRiskByCategory", "GetGistEmissionsTrends", "GetGistAssetsByCountry",
        "GetGistScope3Emissions", "GetGistCompanyProfile", "GetAvailableDatasets",
        "GetInstitutionsProcessesData", "GetPlansAndPoliciesData", "GetGistCompanies",
        "GetSolarFacilitiesByCountry", "GetGistBiodiversityTrends", "GetGistBiodiversityWorstPerformers",
        "GetGistVisualizationData", "GetStateClimatePolicy", "GetGistCompanyRisks"
    ]
    
    if tool_name not in table_tools:
        return False
    
    # Check if data structure is suitable for tables
    if isinstance(parsed_data, dict):
        # Look for common data patterns in dict structures
        if "companies" in parsed_data and isinstance(parsed_data["companies"], list):
            return True
        if "sectors" in parsed_data and isinstance(parsed_data["sectors"], dict):
            return True
        if "emissions_by_year" in parsed_data and isinstance(parsed_data["emissions_by_year"], list):
            return True
        if "top_emitters" in parsed_data and isinstance(parsed_data["top_emitters"], list):
            return True
        if "states" in parsed_data and isinstance(parsed_data["states"], dict):
            return True
        if "sector_emissions" in parsed_data and isinstance(parsed_data["sector_emissions"], list):
            return True
        # Check for other common patterns that indicate tabular data
        if any(key in parsed_data for key in ["data", "facilities", "countries_analyzed", "worst_performers"]):
            return True
    elif isinstance(parsed_data, list) and len(parsed_data) > 0:
        # Direct list of records
        return True
    
    return False

def _extract_table_data(tool_name: str, parsed_data: any) -> List[Dict]:
    """Extract table-ready data from tool results."""
    
    try:
        # Handle different data structures based on tool type
        if tool_name == "GetGistCompaniesBySector":
            # Extract companies from sector groups
            table_data = []
            if "sectors" in parsed_data:
                for sector_code, sector_info in parsed_data["sectors"].items():
                    for company in sector_info.get("companies", []):
                        table_data.append({
                            "sector": sector_code,
                            "company_code": company.get("company_code", ""),
                            "company_name": company.get("company_name", ""),
                            "country": company.get("country", "")
                        })
            return table_data[:20]  # Limit for display
        
        elif tool_name in ["GetGistTopEmitters", "GetGistHighRiskCompanies", "GetLargestSolarFacilities"]:
            # Direct list of ranked items
            if isinstance(parsed_data, dict):
                # Check multiple possible list fields
                for key in ["top_emitters", "companies", "worst_performers", "top_3_facilities", "facilities_found"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        return parsed_data[key][:20]
                # Also check for the data itself if it's a list
                if "data" in parsed_data and isinstance(parsed_data["data"], list):
                    return parsed_data["data"][:20]
            elif isinstance(parsed_data, list):
                return parsed_data[:20]
        
        elif tool_name == "GetSolarCapacityByCountry":
            # Solar capacity data by country
            if isinstance(parsed_data, list):
                return parsed_data
            elif isinstance(parsed_data, dict) and "data" in parsed_data:
                return parsed_data["data"]
        
        elif tool_name == "GetGistEmissionsBySector":
            # Sector emissions data
            if isinstance(parsed_data, dict):
                # Check for sector emissions in various formats
                for key in ["sector_emissions", "sector_analysis", "sectors", "data"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        return parsed_data[key][:20]
            elif isinstance(parsed_data, list):
                return parsed_data[:20]
        
        elif tool_name in ["GetSolarConstructionTimeline", "GetGistEmissionsTrends"]:
            # Time series data
            if isinstance(parsed_data, dict):
                if "timeline_data" in parsed_data:
                    return parsed_data["timeline_data"]
                elif "emissions_by_year" in parsed_data:
                    return parsed_data["emissions_by_year"]
                elif "data" in parsed_data:
                    return parsed_data["data"]
            elif isinstance(parsed_data, list):
                return parsed_data
        
        elif tool_name == "GetBrazilianStatesOverview":
            # Brazilian states policy data
            if isinstance(parsed_data, dict) and "states" in parsed_data:
                return parsed_data["states"]
            elif isinstance(parsed_data, list):
                return parsed_data
        
        elif tool_name == "GetGistCompanies":
            # Company directory
            if isinstance(parsed_data, dict) and "companies" in parsed_data:
                return parsed_data["companies"][:25]
            elif isinstance(parsed_data, list):
                return parsed_data[:25]
        
        elif tool_name in ["GetGistBiodiversityWorstPerformers", "GetGistVisualizationData"]:
            # Biodiversity and visualization data
            if isinstance(parsed_data, dict):
                for key in ["worst_performers", "data", "companies_analyzed"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        return parsed_data[key][:20]
            elif isinstance(parsed_data, list):
                return parsed_data[:20]
        
        elif tool_name in ["CompareBrazilianStates", "GetStateClimatePolicy"]:
            # Policy comparison data
            if isinstance(parsed_data, dict):
                for key in ["comparison_matrix", "policy_areas", "states_compared", "data"]:
                    if key in parsed_data:
                        if isinstance(parsed_data[key], list):
                            return parsed_data[key][:20]
                        elif isinstance(parsed_data[key], dict):
                            # Convert dict to list of records
                            return [{"state": k, **v} for k, v in parsed_data[key].items()][:20]
            elif isinstance(parsed_data, list):
                return parsed_data[:20]
        
        else:
            # Generic extraction for other tools
            if isinstance(parsed_data, list):
                return parsed_data[:20]
            elif isinstance(parsed_data, dict):
                # Look for common list fields
                for key in ["data", "results", "items", "records", "companies", "facilities"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        return parsed_data[key][:20]
        
        return []
        
    except Exception as e:
        print(f"Error extracting table data from {tool_name}: {e}")
        return []

class MultiServerClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.anthropic = anthropic.Anthropic()
        self.citation_registry = CitationRegistry()
        
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
            
            - Solar Facilities Data: Use GetSolarFacilitiesMapData for maps, GetSolarCapacityByCountry for stats, other tools for specific queries.
            
            - GIST Environmental Data: Use GetGistCompanyProfile for company data, GetGistAssetsMapData for asset maps, other tools for specific ESG metrics.
            
            - LSE Climate Policy Data: Use GetBrazilianStatesOverview for Brazilian states, GetNDCOverviewData for NDCs, other tools for specific policy analysis.
            
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

            Enhanced Data Discovery & Multi-Table Strategy:
            - After getting concept passages, ALWAYS check for connected datasets using `GetAvailableDatasets()`
            - Look for concepts with "HAS_DATASET_ABOUT" relationships in the knowledge graph
            - Proactively surface both textual insights AND structured data when available
            - **CRITICAL: Generate multiple data tables for comprehensive analysis. When users ask broad questions, call multiple related tools to create a rich data dashboard.**
            
            Multi-Table Strategy: For comprehensive queries, call 4-6 complementary tools from different data sources (KG passages + quantitative data + geographic data).

            Visualization: Use visualization tools (GetSolarCapacityVisualizationData, GetGistVisualizationData, GetLSEVisualizationData) for chart requests. System auto-generates maps for facility/location queries.

            Output: Synthesize tool results into comprehensive response. Don't narrate tool calling process. Include specific numbers and context.
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
        chart_data_tool = None      # Track which tool generated chart_data
        map_data = None             # To store map HTML and metadata
        visualization_data = None   # To store structured visualization data
        all_tool_outputs_for_debug = [] # For Feature 2
        
        # NEW: Multi-table data collection
        table_ready_data = []       # List of {tool_name, data} for table generation
        tool_results_for_tables = [] # Structured tool results for enhanced table creation
        
        # CITATION_FIX: Structured fact collection
        intermediate_facts_with_citations = []  # Facts with citation info from tools
        
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
                                    if isinstance(parsed_content, dict) and parsed_content.get("type") in ["map", "map_data_summary"]:
                                        map_data = parsed_content
                                        if parsed_content.get("type") == "map_data_summary":
                                            print(f"Successfully parsed map data summary from {tool_name}: {parsed_content.get('summary', {}).get('total_facilities', 0)} total facilities")
                                        else:
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
                                            chart_data_tool = tool_name  # Track which tool generated it
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
                    
                    # Parse structured visualization data from multiple tools
                    elif tool_name in ["GetSolarCapacityVisualizationData", "GetGistVisualizationData", "GetLSEVisualizationData"]:
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

                    # CITATION_FIX: Wrap tool result with citation info for enhanced synthesis
                    try:
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            tool_result_text = result.content[0].text
                            wrapped_result = wrap_tool_result_with_citation(tool_name, tool_result_text, tool_args)
                            intermediate_facts_with_citations.append({
                                "fact": wrapped_result["fact"],
                                "citation_info": wrapped_result["citation_info"],
                                "tool_context": {
                                    "tool_name": tool_name,
                                    "tool_args": tool_args
                                }
                            })
                            print(f"CITATION_FIX DEBUG: Wrapped tool result from {tool_name} (total facts: {len(intermediate_facts_with_citations)})")
                        else:
                            print(f"CITATION_FIX DEBUG: Skipped {tool_name} - no content or malformed result")
                    except Exception as e:
                        print(f"CITATION_FIX DEBUG: Error wrapping tool result from {tool_name}: {e}")

                    # CITATION_FIX: Check for structured fact/citation format from tools (legacy)
                    parsed_tool_result = None
                    citation_info = None
                    
                    # Check if the tool result has structured citation format
                    if result.content and isinstance(result.content, list) and len(result.content) > 0:
                        first_content_block = result.content[0]
                        if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                            try:
                                # Attempt to parse the tool result as JSON
                                parsed_tool_result = json.loads(first_content_block.text)

                                if isinstance(parsed_tool_result, dict) and "fact" in parsed_tool_result:
                                    # Handle a single fact result
                                    facts_to_process = [parsed_tool_result]
                                elif isinstance(parsed_tool_result, list):
                                    # Handle a list of facts - check if any have the fact/citation_info structure
                                    facts_to_process = [item for item in parsed_tool_result if isinstance(item, dict) and "fact" in item]
                                else:
                                    # Not a recognized format, so skip processing for citations
                                    facts_to_process = []
                                
                                # Process the list of facts (even if it's just one)
                                for fact_item in facts_to_process:
                                    fact_text = fact_item.get("fact", "")
                                    citation_info = fact_item.get("citation_info", {})
                                
                                    # Append to our intermediate facts list
                                    if fact_text and citation_info:
                                        intermediate_facts_with_citations.append({
                                            "fact": fact_text,
                                            "citation_info": citation_info,
                                            "tool_context": {
                                                "tool_name": tool_name,
                                                "tool_args": tool_args
                                            }
                                        })
                                        print(f"CITATION_FIX DEBUG: Added structured fact from {tool_name}")

                            except json.JSONDecodeError:
                                # Handle cases where the tool result is not JSON - continue with existing flow
                                print(f"CITATION_FIX DEBUG: Tool result from {tool_name} is not valid JSON. Using fallback approach.")

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
                    
                    # CITATION REGISTRY: Add passage sources to citation registry
                    for passage_source in new_passage_sources:
                        citation_num = self.citation_registry.add_source(passage_source, module_id=f"tool_{tool_name}")
                        print(f"CITATION DEBUG: Added passage source to registry as citation #{citation_num}")
                    
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
                        
                        # CITATION REGISTRY: Add dataset source to citation registry
                        citation_num = self.citation_registry.add_source(dataset_source, module_id=f"tool_{tool_name}")
                        print(f"CITATION DEBUG: Added dataset citation #{citation_num} for {tool_name}: {dataset_citation['source_name']}")
                    else:
                        print(f"DEBUG: No dataset citation created for {tool_name}")
                    
                    print(f"DEBUG: Total sources now: {len(passage_sources)} (passages + datasets)")
                    print(f"CITATION DEBUG: Citation registry now has {len(self.citation_registry.get_all_citations())} total citations")

                    # NEW: Collect data for multi-table generation
                    if result.content and isinstance(result.content, list) and len(result.content) > 0:
                        first_content_block = result.content[0]
                        if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                            try:
                                parsed_data = json.loads(first_content_block.text)
                                
                                # Check if this tool produces table-ready data
                                if _tool_produces_table_data(tool_name, parsed_data):
                                    table_data = _extract_table_data(tool_name, parsed_data)
                                    if table_data:
                                        tool_results_for_tables.append({
                                            "tool_name": tool_name,
                                            "data": table_data,
                                            "args": tool_args
                                        })
                                        print(f"DEBUG: Collected table data from {tool_name}: {len(table_data)} records")
                                        
                            except json.JSONDecodeError:
                                # Not JSON data, skip table collection
                                pass

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

        # CITATION_FIX: Consolidate citations and prepare facts for LLM
        final_citations_list = []
        citation_id_map = {}
        facts_for_llm = []
        
        if len(intermediate_facts_with_citations) > 0:
            print(f"CITATION_FIX DEBUG: Processing {len(intermediate_facts_with_citations)} structured facts with citations")
            
            # Consolidate citations by deduplicating based on source_name and source_url
            citation_id_counter = 1
            
            for item in intermediate_facts_with_citations:
                citation_info = item["citation_info"]
                citation_key = (citation_info.get('source_name', ''), citation_info.get('source_url', ''))

                if citation_key not in citation_id_map:
                    new_id = f"citation_{citation_id_counter}"
                    citation_id_map[citation_key] = new_id
                    final_citations_list.append({
                        "id": new_id,
                        "source_name": citation_info.get('source_name', 'Unknown Source'),
                        "provider": citation_info.get('provider', 'Unknown Provider'),
                        "spatial_coverage": citation_info.get('spatial_coverage', ''),
                        "temporal_coverage": citation_info.get('temporal_coverage', ''),
                        "source_url": citation_info.get('source_url', '')
                    })
                    citation_id_counter += 1
                    print(f"CITATION_FIX DEBUG: Added new citation {new_id}: {citation_info.get('source_name', 'Unknown')}")
            
            # Tag the facts with their corresponding citation IDs
            for item in intermediate_facts_with_citations:
                citation_info = item["citation_info"]
                citation_key = (citation_info.get('source_name', ''), citation_info.get('source_url', ''))
                final_id = citation_id_map.get(citation_key, 'citation_unknown')

                facts_for_llm.append({
                    "fact": item["fact"],
                    "citation_id": final_id,
                    "tool_context": item["tool_context"]
                })
            
            print(f"CITATION_FIX DEBUG: Created {len(final_citations_list)} unique citations and {len(facts_for_llm)} tagged facts")

        # --- final synthesis / response construction ---------------------------------
        final_response_text = ""
        
        # CITATION_FIX: Use enhanced synthesis with structured facts when available
        if len(facts_for_llm) > 0:
            print(f"CITATION_FIX DEBUG: Using enhanced synthesis with {len(facts_for_llm)} structured facts")
            
            # Enhanced synthesis system prompt for structured facts
            synthesis_system_prompt = (SUMMARY_PROMPT +
                "Based on the following structured facts with citation IDs, synthesize a comprehensive answer to the user's original query. "
                "Present it as a direct answer, not a summary of the facts themselves. Avoid narrating the tool process. "
                "IMPORTANT: When referring to information from the facts, include the citation number in caret format like ^1^ at the relevant point in your text. Use the numeric part of the citation_id (e.g., for 'citation_1', use ^1^). "
                "Use citations judiciously - typically one per paragraph or logical section, not after every single fact. "
                "Place citations where they naturally support key claims or data points. "
                "If the facts mention that visualizations are available, you may reference them appropriately.")
            
            # DEBUG: Log what we're sending to LLM for regular function
            print(f"CITATION_FIX LLM DEBUG (regular): Sending {len(facts_for_llm)} facts to LLM synthesis:")
            for i, fact in enumerate(facts_for_llm, 1):
                print(f"  Fact {i}: citation_id={fact['citation_id']}, tool={fact['tool_context']['tool_name']}")
                print(f"    fact preview: {fact['fact'][:100]}...")
            
            synthesis_messages = [
                {"role": "user", "content": f"User's original query: {query}\n\nFacts with Citation IDs:\n{json.dumps(facts_for_llm, indent=2)}"}
            ]
            
            summary_resp = self.anthropic.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                system=synthesis_system_prompt,
                messages=synthesis_messages,
            )
            final_response_text = summary_resp.content[0].text.strip()
            # CITATION_FIX: Convert citation format for frontend compatibility
            final_response_text = convert_citation_format(final_response_text)
            
            # CITATION_FIX: Filter and renumber citations based on what's actually used in text
            if final_citations_list:
                import re
                # Find all ^1^, ^2^, ^3^ citations in the text
                citation_pattern = r'\^(\d+)\^'
                used_citation_numbers = set()
                for match in re.finditer(citation_pattern, final_response_text):
                    used_citation_numbers.add(int(match.group(1)))
                
                if used_citation_numbers:
                    # Create mapping of old citation numbers to new sequential numbers
                    old_to_new = {}
                    new_citation_counter = 1
                    used_citations = []
                    
                    # Build filtered citation list with only used citations, in order of appearance
                    sorted_used = sorted(used_citation_numbers)
                    for old_num in sorted_used:
                        if old_num <= len(final_citations_list):
                            old_to_new[old_num] = new_citation_counter
                            citation_copy = final_citations_list[old_num - 1].copy()  # Convert to 0-based index
                            citation_copy['id'] = f"citation_{new_citation_counter}"
                            used_citations.append(citation_copy)
                            new_citation_counter += 1
                    
                    # Update text with renumbered citations  
                    def renumber_citation(match):
                        old_num = int(match.group(1))
                        new_num = old_to_new.get(old_num, old_num)
                        return f"^{new_num}^"
                    
                    final_response_text = re.sub(citation_pattern, renumber_citation, final_response_text)
                    final_citations_list = used_citations
                    
                    print(f"CITATION_FIX DEBUG: Filtered from {len(facts_for_llm)} facts to {len(final_citations_list)} used citations: {sorted_used}")
                else:
                    # No citations found in text, clear the citation list
                    final_citations_list = []
                    print("CITATION_FIX DEBUG: No citations found in text, cleared citation list")
            
        elif len(context_chunks) > 0:
            # Fallback to traditional synthesis for backwards compatibility
            print("CITATION_FIX DEBUG: Using traditional synthesis (no structured facts available)")
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
            # CITATION_FIX: Convert citation format for frontend compatibility (fallback case)
            final_response_text = convert_citation_format(final_response_text)
        elif last_assistant_text: # No context chunks, but AI provided a final text response
            final_response_text = last_assistant_text
            # CITATION_FIX: Convert citation format for frontend compatibility (last assistant text)
            final_response_text = convert_citation_format(final_response_text)
        else: # Should not happen if AI always responds with text eventually
            final_response_text = "I was unable to produce a response."

        # de-dupe sources
        uniq_passages = {(p["doc_id"], p["passage_id"]): p for p in passage_sources}
        print(f"DEBUG: Found these unique passages: {uniq_passages}")
        if uniq_passages:
            sources_used.extend(uniq_passages.values())

        # NEW: Generate multiple tables from collected tool results (MISSING IN REGULAR FUNCTION!)
        additional_table_data = []
        print(f"🔍 REGULAR QUERY DEBUG: tool_results_for_tables contains {len(tool_results_for_tables)} items")
        
        if tool_results_for_tables:
            print(f"🔍 REGULAR QUERY DEBUG: Calling CreateMultipleTablesFromToolResults for regular query")
            try:
                multi_table_result = await self.call_tool(
                    "CreateMultipleTablesFromToolResults", 
                    {
                        "tool_results": tool_results_for_tables, 
                        "query_context": query,
                        "citation_registry": {
                            "citations": self.citation_registry.get_all_citations(),
                            "module_citations": self.citation_registry.module_citations
                        }
                    }, 
                    "formatter"
                )
                
                if multi_table_result.content and isinstance(multi_table_result.content, list):
                    first_content = multi_table_result.content[0]
                    if hasattr(first_content, 'text'):
                        multi_table_data = json.loads(first_content.text)
                        additional_table_data = multi_table_data.get("modules", [])
                        print(f"🔍 REGULAR QUERY DEBUG: Generated {len(additional_table_data)} additional table modules")
                        
            except Exception as e:
                print(f"🔍 REGULAR QUERY DEBUG: Error generating multiple tables: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"🔍 REGULAR QUERY DEBUG: No tool_results_for_tables - skipping multi-table generation")

        return {
            "response": final_response_text,
            "sources": sources_used or ["No source captured"],
            "chart_data": chart_data,  # Legacy chart data for backward compatibility
            "chart_data_tool": chart_data_tool,  # Track which tool generated chart_data
            "map_data": map_data,  # Map HTML and metadata
            "visualization_data": visualization_data,  # Structured chart data
            "all_tool_outputs_for_debug": all_tool_outputs_for_debug, # For Feature 2
            "ai_thought_process": "\n".join(intermediate_ai_text_parts), # Add the collected thoughts
            "additional_table_data": additional_table_data,  # NEW: Multi-table data for formatter integration
            "final_citations_list": final_citations_list,  # CITATION_FIX: Include structured citations
            "citation_registry": {  # NEW: Citation registry for KG generation
                "citations": self.citation_registry.get_all_citations(),
                "module_citations": self.citation_registry.module_citations
            }
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
            
            - Solar Facilities Data: Use GetSolarFacilitiesMapData for maps, GetSolarCapacityByCountry for stats, other tools for specific queries.
            
            - GIST Environmental Data: Use GetGistCompanyProfile for company data, GetGistAssetsMapData for asset maps, other tools for specific ESG metrics.
            
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

            Output: Synthesize tool results into comprehensive response. Don't narrate tool calling process. Include specific numbers and context.
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
        chart_data_tool = None      # Track which tool generated chart_data
        map_data = None
        visualization_data = None
        all_tool_outputs_for_debug = []
        
        # NEW: Multi-table data collection for streaming
        tool_results_for_tables = []
        
        # CITATION_FIX: Structured fact collection for streaming
        intermediate_facts_with_citations = []
        final_citations_list = []  # Initialize to prevent errors, full implementation later
        
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

                    # Parse map data from GetSolarFacilitiesMapData and generate GeoJSON file
                    if tool_name == "GetSolarFacilitiesMapData":
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            first_content_block = result.content[0]
                            if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                                try:
                                    parsed_content = json.loads(first_content_block.text)
                                    if isinstance(parsed_content, dict) and parsed_content.get("type") in ["map", "map_data_summary"]:
                                        map_data = parsed_content
                                        
                                        # Generate GeoJSON file from full_data if available
                                        if "full_data" in parsed_content and parsed_content["full_data"]:
                                            facilities = parsed_content["full_data"]
                                            print(f"STREAMING DEBUG: Generating GeoJSON from {len(facilities)} facilities")
                                            
                                            # Generate GeoJSON structure
                                            geojson = {
                                                "type": "FeatureCollection",
                                                "features": []
                                            }
                                            
                                            # Color mapping by country
                                            country_colors = {
                                                'brazil': '#4CAF50',
                                                'india': '#FF9800', 
                                                'south africa': '#F44336',
                                                'vietnam': '#2196F3'
                                            }
                                            
                                            for facility in facilities:
                                                feature = {
                                                    "type": "Feature",
                                                    "geometry": {
                                                        "type": "Point",
                                                        "coordinates": [float(facility["longitude"]), float(facility["latitude"])]
                                                    },
                                                    "properties": {
                                                        "name": facility.get("name", f"Solar Facility {facility.get('cluster_id', '')}"),
                                                        "capacity_mw": float(facility.get("capacity_mw", 0)) if facility.get("capacity_mw") is not None else 0.0,
                                                        "country": facility["country"],
                                                        "completion_year": facility.get("completion_year", "Unknown"),
                                                        "cluster_id": facility.get("cluster_id", ""),
                                                        "technology": "Solar PV",
                                                        "marker_color": country_colors.get(facility["country"].lower(), "#9E9E9E")
                                                    }
                                                }
                                                geojson["features"].append(feature)
                                            
                                            # Save GeoJSON file
                                            try:
                                                import os
                                                import hashlib
                                                
                                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                                project_root = os.path.dirname(script_dir)  # Go up from mcp/ to project root
                                                static_maps_dir = os.path.join(project_root, "static", "maps")
                                                
                                                # Ensure directory exists
                                                os.makedirs(static_maps_dir, exist_ok=True)
                                                
                                                # Create unique filename
                                                countries = parsed_content.get("summary", {}).get("countries", ["world"])
                                                countries_str = "_".join([c.lower().replace(" ", "_") for c in countries[:4]])
                                                data_hash = hashlib.md5(str(len(facilities)).encode()).hexdigest()[:8]
                                                filename = f"solar_facilities_{countries_str}_{data_hash}.geojson"
                                                
                                                # Write GeoJSON file
                                                geojson_path = os.path.join(static_maps_dir, filename)
                                                with open(geojson_path, 'w') as f:
                                                    json.dump(geojson, f, indent=2)
                                                
                                                print(f"STREAMING DEBUG: ✅ Generated GeoJSON file: {geojson_path}")
                                                print(f"STREAMING DEBUG: File contains {len(geojson['features'])} features")
                                                
                                                # Add geojson_url to map_data for formatter
                                                import os
                                                base_url = os.getenv('API_BASE_URL', 'https://api.transitiondigital.org')
                                                map_data["geojson_url"] = f"{base_url}/static/maps/{filename}"
                                                map_data["geojson_filename"] = filename
                                                
                                            except Exception as e:
                                                print(f"STREAMING DEBUG: ❌ Error generating GeoJSON file: {e}")
                                                import traceback
                                                traceback.print_exc()
                                                
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
                                            chart_data_tool = tool_name  # Track which tool generated it
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

                    # CITATION_FIX: Wrap tool result with citation info for enhanced synthesis (streaming)
                    try:
                        if result.content and isinstance(result.content, list) and len(result.content) > 0:
                            tool_result_text = result.content[0].text
                            wrapped_result = wrap_tool_result_with_citation(tool_name, tool_result_text, tool_args)
                            intermediate_facts_with_citations.append({
                                "fact": wrapped_result["fact"],
                                "citation_info": wrapped_result["citation_info"],
                                "tool_context": {
                                    "tool_name": tool_name,
                                    "tool_args": tool_args
                                }
                            })
                            print(f"STREAMING CITATION_FIX DEBUG: Wrapped tool result from {tool_name} (total facts: {len(intermediate_facts_with_citations)})")
                        else:
                            print(f"STREAMING CITATION_FIX DEBUG: Skipped {tool_name} - no content or malformed result")
                    except Exception as e:
                        print(f"STREAMING CITATION_FIX DEBUG: Error wrapping tool result from {tool_name}: {e}")

                    # CITATION_FIX: Check for structured fact/citation format from tools (streaming) (legacy)
                    if result.content and isinstance(result.content, list) and len(result.content) > 0:
                        first_content_block = result.content[0]
                        if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                            try:
                                # Attempt to parse the tool result as JSON
                                parsed_tool_result = json.loads(first_content_block.text)

                                if isinstance(parsed_tool_result, dict) and "fact" in parsed_tool_result:
                                    # Handle a single fact result
                                    facts_to_process = [parsed_tool_result]
                                elif isinstance(parsed_tool_result, list):
                                    # Handle a list of facts - check if any have the fact/citation_info structure
                                    facts_to_process = [item for item in parsed_tool_result if isinstance(item, dict) and "fact" in item]
                                else:
                                    # Not a recognized format, so skip processing for citations
                                    facts_to_process = []
                                
                                # Process the list of facts (even if it's just one)
                                for fact_item in facts_to_process:
                                    fact_text = fact_item.get("fact", "")
                                    citation_info = fact_item.get("citation_info", {})
                                
                                    # Append to our intermediate facts list
                                    if fact_text and citation_info:
                                        intermediate_facts_with_citations.append({
                                            "fact": fact_text,
                                            "citation_info": citation_info,
                                            "tool_context": {
                                                "tool_name": tool_name,
                                                "tool_args": tool_args
                                            }
                                        })
                                        print(f"STREAMING CITATION_FIX DEBUG: Added structured fact from {tool_name}")

                            except json.JSONDecodeError:
                                # Handle cases where the tool result is not JSON - continue with existing flow
                                print(f"STREAMING CITATION_FIX DEBUG: Tool result from {tool_name} is not valid JSON. Using fallback approach.")

                    # Collect passage sources
                    if tool_name.lower() == "getmetadata":
                        sources_used.append(result.content)
                    
                    # Extract passage sources (traditional)
                    new_passage_sources = harvest_sources(result.content)
                    passage_sources.extend(new_passage_sources)
                    
                    # CITATION REGISTRY: Add passage sources to citation registry (streaming)
                    for passage_source in new_passage_sources:
                        citation_num = self.citation_registry.add_source(passage_source, module_id=f"stream_tool_{tool_name}")
                        print(f"STREAMING CITATION DEBUG: Added passage source to registry as citation #{citation_num}")
                    
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
                        
                        # CITATION REGISTRY: Add dataset source to citation registry (streaming)
                        citation_num = self.citation_registry.add_source(dataset_source, module_id=f"stream_tool_{tool_name}")
                        print(f"STREAMING CITATION DEBUG: Added dataset citation #{citation_num}: {dataset_citation['source_name']}")

                    # NEW: Collect data for multi-table generation in streaming
                    if result.content and isinstance(result.content, list) and len(result.content) > 0:
                        first_content_block = result.content[0]
                        if hasattr(first_content_block, 'type') and first_content_block.type == 'text' and hasattr(first_content_block, 'text'):
                            try:
                                parsed_data = json.loads(first_content_block.text)
                                
                                # Check if this tool produces table-ready data
                                print(f"🔍 DATA COLLECTION DEBUG: Checking {tool_name} for table data...")
                                
                                if _tool_produces_table_data(tool_name, parsed_data):
                                    print(f"🔍 DATA COLLECTION DEBUG: {tool_name} produces table data, extracting...")
                                    table_data = _extract_table_data(tool_name, parsed_data)
                                    
                                    if table_data:
                                        tool_results_for_tables.append({
                                            "tool_name": tool_name,
                                            "data": table_data,
                                            "args": tool_args
                                        })
                                        print(f"🔍 DATA COLLECTION DEBUG: ✅ Collected {len(table_data)} records from {tool_name}")
                                        # Handle both list and dict formats
                                        if isinstance(table_data, list) and len(table_data) > 0:
                                            print(f"  - Sample record keys: {list(table_data[0].keys()) if isinstance(table_data[0], dict) else 'not a dict'}")
                                        elif isinstance(table_data, dict) and len(table_data) > 0:
                                            first_key = list(table_data.keys())[0]
                                            print(f"  - Sample record keys: {list(table_data[first_key].keys()) if isinstance(table_data[first_key], dict) else 'not a dict'}")
                                        else:
                                            print(f"  - Sample record keys: none")
                                    else:
                                        print(f"🔍 DATA COLLECTION DEBUG: ❌ {tool_name} extraction returned empty data")
                                else:
                                    print(f"🔍 DATA COLLECTION DEBUG: ❌ {tool_name} does not produce table data")
                                    print(f"  - Data type: {type(parsed_data)}")
                                    if isinstance(parsed_data, dict):
                                        print(f"  - Dict keys: {list(parsed_data.keys())[:5]}...")
                                        
                            except json.JSONDecodeError as jde:
                                print(f"🔍 DATA COLLECTION DEBUG: ❌ {tool_name} - JSON decode error: {jde}")
                                print(f"  - Raw text (first 100 chars): {first_content_block.text[:100]}...")

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

        # CITATION_FIX: Consolidate citations and prepare facts for LLM (streaming) - MOVED BEFORE SYNTHESIS
        citation_id_map = {}
        facts_for_llm = []
        
        if len(intermediate_facts_with_citations) > 0:
            print(f"STREAMING CITATION_FIX DEBUG: Processing {len(intermediate_facts_with_citations)} structured facts with citations")
            
            # Consolidate citations by deduplicating based on source_name and source_url
            citation_id_counter = 1
            final_citations_list = []  # Reset the placeholder initialization
            
            for item in intermediate_facts_with_citations:
                citation_info = item["citation_info"]
                citation_key = (citation_info.get('source_name', ''), citation_info.get('source_url', ''))

                if citation_key not in citation_id_map:
                    new_id = f"citation_{citation_id_counter}"
                    citation_id_map[citation_key] = new_id
                    final_citations_list.append({
                        "id": new_id,
                        "source_name": citation_info.get('source_name', 'Unknown Source'),
                        "provider": citation_info.get('provider', 'Unknown Provider'),
                        "spatial_coverage": citation_info.get('spatial_coverage', ''),
                        "temporal_coverage": citation_info.get('temporal_coverage', ''),
                        "source_url": citation_info.get('source_url', '')
                    })
                    citation_id_counter += 1
                    print(f"STREAMING CITATION_FIX DEBUG: Added new citation {new_id}: {citation_info.get('source_name', 'Unknown')}")
            
            # Tag the facts with their corresponding citation IDs
            for item in intermediate_facts_with_citations:
                citation_info = item["citation_info"]
                citation_key = (citation_info.get('source_name', ''), citation_info.get('source_url', ''))
                final_id = citation_id_map.get(citation_key, 'citation_unknown')

                facts_for_llm.append({
                    "fact": item["fact"],
                    "citation_id": final_id,
                    "tool_context": item["tool_context"]
                })
            
            print(f"STREAMING CITATION_FIX DEBUG: Created {len(final_citations_list)} unique citations and {len(facts_for_llm)} tagged facts")
        
        else:
            # If no structured facts, keep the empty initialization
            final_citations_list = []
            print("STREAMING CITATION_FIX DEBUG: No intermediate facts with citations found - final_citations_list remains empty")

        # Stream synthesis notification
        yield {
            "type": "thinking",
            "data": {
                "message": "🧠 Analyzing and summarizing all gathered information...",
                "category": "synthesis"
            }
        }
        
        # Final synthesis - prioritize structured facts when available (streaming)
        final_response_text = ""
        
        # CITATION_FIX: Use enhanced synthesis with structured facts when available (streaming)
        if len(facts_for_llm) > 0:
            print(f"STREAMING CITATION_FIX DEBUG: Using enhanced synthesis with {len(facts_for_llm)} structured facts")
            
            # Enhanced synthesis system prompt for structured facts
            synthesis_system_prompt = (SUMMARY_PROMPT +
                "Based on the following structured facts with citation IDs, synthesize a comprehensive answer to the user's original query. "
                "Present it as a direct answer, not a summary of the facts themselves. Avoid narrating the tool process. "
                "IMPORTANT: When referring to information from the facts, include the citation number in caret format like ^1^ at the relevant point in your text. Use the numeric part of the citation_id (e.g., for 'citation_1', use ^1^). "
                "Use citations judiciously - typically one per paragraph or logical section, not after every single fact. "
                "Place citations where they naturally support key claims or data points. "
                "If the facts mention that visualizations are available, you may reference them appropriately.")
            
            # DEBUG: Log what we're sending to LLM for streaming function
            print(f"STREAMING CITATION_FIX LLM DEBUG: Sending {len(facts_for_llm)} facts to LLM synthesis:")
            for i, fact in enumerate(facts_for_llm, 1):
                print(f"  Fact {i}: citation_id={fact['citation_id']}, tool={fact['tool_context']['tool_name']}")
                print(f"    fact preview: {fact['fact'][:100]}...")
            
            synthesis_messages = [
                {"role": "user", "content": f"User's original query: {query}\n\nFacts with Citation IDs:\n{json.dumps(facts_for_llm, indent=2)}"}
            ]
            
        elif len(context_chunks) > 0:
            print("STREAMING CITATION_FIX DEBUG: Using traditional synthesis (no structured facts available)")
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
        else:
            # No context at all - should not happen if AI always responds with text eventually
            final_response_text = "I was unable to produce a response."
            synthesis_messages = None
        
        # Only run synthesis if we have messages to process
        if synthesis_messages:
            summary_resp = self.anthropic.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                system=synthesis_system_prompt,
                messages=synthesis_messages,
            )
            final_response_text = summary_resp.content[0].text.strip()
            # CITATION_FIX: Convert citation format for frontend compatibility (streaming)
            final_response_text = convert_citation_format(final_response_text)
            
            # CITATION_FIX: Filter and renumber citations based on what's actually used in text (streaming)
            if final_citations_list:
                import re
                # Find all ^1^, ^2^, ^3^ citations in the text
                citation_pattern = r'\^(\d+)\^'
                used_citation_numbers = set()
                for match in re.finditer(citation_pattern, final_response_text):
                    used_citation_numbers.add(int(match.group(1)))
                
                if used_citation_numbers:
                    # Create mapping of old citation numbers to new sequential numbers
                    old_to_new = {}
                    new_citation_counter = 1
                    used_citations = []
                    
                    # Build filtered citation list with only used citations, in order of appearance
                    sorted_used = sorted(used_citation_numbers)
                    for old_num in sorted_used:
                        if old_num <= len(final_citations_list):
                            old_to_new[old_num] = new_citation_counter
                            citation_copy = final_citations_list[old_num - 1].copy()  # Convert to 0-based index
                            citation_copy['id'] = f"citation_{new_citation_counter}"
                            used_citations.append(citation_copy)
                            new_citation_counter += 1
                    
                    # Update text with renumbered citations  
                    def renumber_citation(match):
                        old_num = int(match.group(1))
                        new_num = old_to_new.get(old_num, old_num)
                        return f"^{new_num}^"
                    
                    final_response_text = re.sub(citation_pattern, renumber_citation, final_response_text)
                    final_citations_list = used_citations
                    
                    print(f"STREAMING CITATION_FIX DEBUG: Filtered from {len(facts_for_llm)} facts to {len(final_citations_list)} used citations: {sorted_used}")
                else:
                    # No citations found in text, clear the citation list
                    final_citations_list = []
                    print("STREAMING CITATION_FIX DEBUG: No citations found in text, cleared citation list")

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


        # NEW: Generate multiple tables from collected tool results
        additional_modules = []
        print(f"🔍 MULTI-TABLE DEBUG: tool_results_for_tables contains {len(tool_results_for_tables)} items")
        
        if tool_results_for_tables:
            # Enhanced debug logging for multi-table generation
            print(f"🔍 MULTI-TABLE DEBUG: Tool results collected:")
            for i, result in enumerate(tool_results_for_tables):
                tool_name = result.get("tool_name", "unknown")
                data_len = len(result.get("data", [])) if isinstance(result.get("data"), list) else "not a list"
                print(f"  [{i+1}] {tool_name}: {data_len} records")
            
            print(f"🔍 MULTI-TABLE DEBUG: Calling CreateMultipleTablesFromToolResults with query context: '{query[:50]}...'")
            
            # Debug citation registry before passing to formatter
            all_citations = self.citation_registry.get_all_citations()
            module_citations = self.citation_registry.module_citations
            
            try:
                multi_table_result = await self.call_tool(
                    "CreateMultipleTablesFromToolResults", 
                    {
                        "tool_results": tool_results_for_tables, 
                        "query_context": query,
                        "citation_registry": {
                            "citations": all_citations,
                            "module_citations": module_citations
                        }
                    }, 
                    "formatter"
                )
                
                print(f"🔍 MULTI-TABLE DEBUG: CreateMultipleTablesFromToolResults returned:")
                print(f"  - Result type: {type(multi_table_result)}")
                print(f"  - Has content: {bool(multi_table_result.content)}")
                
                if multi_table_result.content and isinstance(multi_table_result.content, list):
                    first_content = multi_table_result.content[0]
                    print(f"  - First content type: {type(first_content)}")
                    print(f"  - Has text attr: {hasattr(first_content, 'text')}")
                    
                    if hasattr(first_content, 'text'):
                        try:
                            multi_table_data = json.loads(first_content.text)
                            additional_modules = multi_table_data.get("modules", [])
                            print(f"🔍 MULTI-TABLE DEBUG: Successfully parsed {len(additional_modules)} additional table modules")
                            
                            # Log details of each generated table
                            for i, module in enumerate(additional_modules):
                                table_type = module.get("type", "unknown")
                                heading = module.get("heading", "no heading")
                                row_count = len(module.get("rows", [])) if "rows" in module else "no rows"
                                print(f"  [{i+1}] {table_type}: '{heading}' ({row_count} rows)")
                                
                        except json.JSONDecodeError as je:
                            print(f"🔍 MULTI-TABLE DEBUG: JSON decode error: {je}")
                            print(f"  - Raw text (first 200 chars): {first_content.text[:200]}")
                    else:
                        print(f"🔍 MULTI-TABLE DEBUG: First content has no text attribute")
                else:
                    print(f"🔍 MULTI-TABLE DEBUG: multi_table_result.content is not a list or is empty")
                        
            except Exception as e:
                print(f"🔍 MULTI-TABLE DEBUG: Error generating multiple tables: {e}")
                import traceback
                print(f"🔍 MULTI-TABLE DEBUG: Full traceback:")
                traceback.print_exc()
        else:
            print(f"🔍 MULTI-TABLE DEBUG: No tool_results_for_tables - skipping multi-table generation")

        # Format the final response
        module_citations: Dict[str, List[int]] = {
            module_id: citations.copy()
            for module_id, citations in self.citation_registry.module_citations.items()
        }

        paragraphs = [p.strip() for p in final_response_text.split("\n\n") if p.strip()]
        for idx, paragraph in enumerate(paragraphs):
            citation_matches = re.findall(r"\^([0-9,]+)\^", paragraph)
            if not citation_matches:
                continue

            inline_numbers: set[int] = set()
            for match in citation_matches:
                for part in match.split(','):
                    part = part.strip()
                    if part.isdigit():
                        inline_numbers.add(int(part))

            if not inline_numbers:
                continue

            module_id = f"text_module_{idx}"
            existing = module_citations.get(module_id, [])
            combined = sorted(set(existing).union(inline_numbers))
            module_citations[module_id] = combined

        formatter_args = {
            "response_text": final_response_text,
            "chart_data": chart_data,
            "chart_data_tool": chart_data_tool,  # Pass tool name for chart_data citations
            "visualization_data": visualization_data,
            "map_data": map_data,
            "sources": sources_used or ["No source captured"],
            "title": "Climate Policy Analysis",
            "citation_registry": {
                "citations": self.citation_registry.get_all_citations(),
                "module_citations": module_citations,
            }
        }
        
        # CITATION_FIX: Add final_citations_list when available (keep original sources)
        if final_citations_list:
            print(f"CITATION_FIX DEBUG: Adding {len(final_citations_list)} structured citations to formatter args")
            # Add structured citations as a separate field, keep original sources intact
            formatter_args["structured_citations"] = final_citations_list
            print(f"CITATION_FIX DEBUG: Added structured_citations field")
        
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
                    
                    # Merge additional table modules with main response
                    all_modules = formatted_data.get("modules", [])
                    if additional_modules:
                        # Insert additional tables before the source table
                        source_table_index = None
                        for i, module in enumerate(all_modules):
                            if module.get("type") == "numbered_citation_table":
                                source_table_index = i
                                break
                        
                        if source_table_index is not None:
                            # Insert before source table
                            all_modules = all_modules[:source_table_index] + additional_modules + all_modules[source_table_index:]
                        else:
                            # Append at end
                            all_modules.extend(additional_modules)
                    
                    # Fetch KG data for streaming response
                    kg_data = await _fetch_kg_data_for_streaming(query)
                    
                    # Generate static KG visualization file for streaming response
                    kg_embed_path = None
                    try:
                        # Import here to avoid circular imports
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from kg_embed_generator import KGEmbedGenerator
                        
                        # Create structured response for KG generation
                        structured_response = {"modules": all_modules}
                        
                        # Extract citation registry
                        citation_registry = {
                            "citations": self.citation_registry.get_all_citations(),
                            "module_citations": self.citation_registry.module_citations
                        }
                        
                        # Generate enhanced KG embed with MCP response data
                        kg_generator = KGEmbedGenerator()
                        kg_embed_result = await kg_generator.generate_embed(
                            query, 
                            structured_response,
                            citation_registry
                        )
                        
                        # Extract path info from result
                        if kg_embed_result:
                            if isinstance(kg_embed_result, dict):
                                kg_embed_path = kg_embed_result["relative_path"]
                                kg_embed_absolute_path = kg_embed_result["absolute_path"]
                                kg_embed_url = kg_embed_result["url_path"]
                                kg_embed_filename = kg_embed_result["filename"]
                                print(f"📊 Generated KG embed for streaming: {kg_embed_absolute_path}")
                            else:
                                # Backward compatibility
                                kg_embed_path = kg_embed_result
                                kg_embed_absolute_path = None
                                kg_embed_url = f"/static/{kg_embed_path}"
                                kg_embed_filename = None
                        else:
                            kg_embed_path = None
                            kg_embed_absolute_path = None
                            kg_embed_url = None
                            kg_embed_filename = None
                    except Exception as e:
                        print(f"Failed to generate KG embed for streaming: {e}")
                        import traceback
                        print(f"Streaming KG generation traceback: {traceback.format_exc()}")
                        kg_embed_path = None
                        kg_embed_absolute_path = None
                        kg_embed_url = None
                        kg_embed_filename = None
                    
                    # Stream the complete response with enhanced tables and KG data
                    yield {
                        "type": "complete",
                        "data": {
                            "query": query,
                            "modules": all_modules,
                            "concepts": kg_data["concepts"],
                            "relationships": kg_data["relationships"],
                            "metadata": {
                                "modules_count": len(all_modules),
                                "has_maps": any(m.get("type") == "map" for m in all_modules),
                                "has_charts": any(m.get("type") == "chart" for m in all_modules),
                                "has_tables": any(m.get("type") in ["table", "numbered_citation_table", "comparison_table", "ranking_table", "trend_table", "summary_table", "detail_table", "geographic_table"] for m in all_modules),
                                "table_types": list(set(m.get("type") for m in all_modules if m.get("type", "").endswith("_table"))),
                                "enhanced_tables_count": len(additional_modules),
                                "kg_visualization_url": "/kg-viz",
                                "kg_query_url": f"/kg-viz?query={query.replace(' ', '%20')}",
                                # Compatibility: expose both `kg_embed` and `kg_embed_url`
                                "kg_embed": kg_embed_url,
                                "kg_embed_url": kg_embed_url,
                                "kg_embed_path": kg_embed_path,
                                "kg_embed_absolute_path": kg_embed_absolute_path,
                                "kg_embed_filename": kg_embed_filename
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

async def _fetch_kg_data_for_streaming(query: str) -> Dict[str, Any]:
    """
    Fetch concepts and relationships data from the KG visualization server for streaming.
    Returns empty data if KG server is unavailable.
    """
    kg_server_url = "http://localhost:8100/api/kg/query-subgraph"
    
    payload = {
        "query": query,
        "depth": 2,
        "max_nodes": 80,
        "include_datasets": True,
        "include_passages": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(kg_server_url, json=payload, timeout=aiohttp.ClientTimeout(total=8)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "concepts": data.get("concepts", []),
                        "relationships": data.get("relationships", [])
                    }
                else:
                    print(f"KG server returned status {response.status}")
                    return {"concepts": [], "relationships": []}
    except asyncio.TimeoutError:
        print("KG server timeout - continuing without KG data")
        return {"concepts": [], "relationships": []}
    except Exception as e:
        print(f"Error fetching KG data for streaming: {e}")
        return {"concepts": [], "relationships": []}
        
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
        # Use the global singleton client instead of creating a new one
        client = await get_global_client()
        
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
    
    # Use the global singleton client instead of creating a new one
    client = await get_global_client()
    
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
        chart_data_tool = None
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
        "chart_data_tool": result.get("chart_data_tool"),  # Pass tool name for chart_data citations
        "visualization_data": result.get("visualization_data"), 
        "map_data": map_data_from_result,  # Use map_data from result dict
        "sources": result.get("sources"),
        "structured_citations": result.get("final_citations_list"),  # CITATION_FIX: Pass structured citations
        "title": "Climate Policy Analysis",
        "citation_registry": {
            "citations": client.citation_registry.get_all_citations(),
            "module_citations": client.citation_registry.module_citations
        }
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
    
    # Parse the formatted response and integrate additional tables
    if formatted_result.content and isinstance(formatted_result.content, list):
        first_content = formatted_result.content[0]
        if hasattr(first_content, 'text'):
            import json
            formatted_data = json.loads(first_content.text)
            
            # NEW: Integrate additional table data from multi-table generation
            additional_table_data = result.get("additional_table_data", [])
            if additional_table_data:
                print(f"🔍 INTEGRATION DEBUG: Integrating {len(additional_table_data)} additional tables")
                
                all_modules = formatted_data.get("modules", [])
                
                # Insert additional tables before the source table
                source_table_index = None
                for i, module in enumerate(all_modules):
                    if module.get("type") == "numbered_citation_table":
                        source_table_index = i
                        break
                
                if source_table_index is not None:
                    # Insert before source table
                    all_modules = all_modules[:source_table_index] + additional_table_data + all_modules[source_table_index:]
                    print(f"🔍 INTEGRATION DEBUG: Inserted tables before source table at index {source_table_index}")
                else:
                    # Append at end
                    all_modules.extend(additional_table_data)
                    print(f"🔍 INTEGRATION DEBUG: Appended tables at end (no source table found)")
                
                formatted_data["modules"] = all_modules
                print(f"🔍 INTEGRATION DEBUG: Final module count: {len(all_modules)}")
            
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
                        
                        for facility in facilities:
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
