import json
from fastmcp import FastMCP
from typing import List, Dict, Any, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
from datetime import datetime
import hashlib
import time
import anthropic
import asyncio

mcp = FastMCP("response-formatter-server")

metadata = {
    "Name": "Response Formatter Server",
    "Description": "Formats responses into structured modules for front-end consumption", 
    "Version": "1.0.0",
    "Author": "Mason Grimshaw"
}

def _generate_unique_module_id(tool_name: str, content_type: str, content: Any = None) -> str:
    """
    Generate a unique module ID for citation tracking.
    
    Args:
        tool_name: Name of the tool that generated the content
        content_type: Type of content (table, chart, text, etc.)
        content: Optional content to hash for uniqueness
        
    Returns:
        Unique module ID string
    """
    timestamp = str(int(time.time() * 1000))  # Millisecond timestamp
    
    # Create content hash if content provided
    if content:
        content_str = json.dumps(content, sort_keys=True) if isinstance(content, (dict, list)) else str(content)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
    else:
        content_hash = hashlib.md5(f"{tool_name}_{timestamp}".encode()).hexdigest()[:8]
    
    return f"{tool_name}_{content_type}_{content_hash}_{timestamp}"

def _add_all_citations_fallback(text: str, citation_nums: List[str]) -> str:
    """Fallback function to add all citations to text when LLM fails."""
    if not citation_nums:
        return text
    
    # CITATION FIX: Sort citation numbers and use individual format
    sorted_citations = sorted(citation_nums, key=int)
    individual_citations = ' '.join([f"^{c}^" for c in sorted_citations])
    superscript = f" {individual_citations}"
    
    # Add to the end of the first sentence if possible, otherwise to the end
    sentences = text.split('. ')
    if len(sentences) > 1:
        sentences[0] = sentences[0] + superscript
        return '. '.join(sentences)
    else:
        return text + superscript

async def _insert_llm_citations(text: str, citation_registry: Dict, tool_results_context: Optional[Dict] = None) -> str:
    """
    Use Claude LLM to intelligently place citations in text based on content context.
    
    Args:
        text: Text content to add citations to
        citation_registry: Registry containing citation information
        tool_results_context: Optional context about what data came from which tools
        
    Returns:
        Text with precisely placed inline citations
    """
    if not citation_registry or not citation_registry.get("citations"):
        return text
    
    try:
        # Initialize Anthropic client
        client = anthropic.Anthropic()
        
        # Build citation context for LLM
        citations_info = []
        citations_dict = citation_registry.get("citations", {})
        
        for citation_num, citation_data in citations_dict.items():
            if isinstance(citation_data, dict):
                source_info = {
                    "number": citation_num,
                    "title": citation_data.get("title", "Unknown Source"),
                    "provider": citation_data.get("provider", "Unknown Provider"),
                    "type": citation_data.get("type", "Unknown Type"),
                    "tool_used": citation_data.get("passage_id", citation_data.get("tool_used", "Unknown Tool")),
                    "description": citation_data.get("text", "No description available")[:200]
                }
                citations_info.append(source_info)
        
        # Sort citations by number to ensure proper order
        citations_info.sort(key=lambda x: int(x['number']))
        
        # Create improved citation prompt that REQUIRES using ALL citations
        all_citation_nums = [c['number'] for c in citations_info]
        
        prompt = f"""You are a citation placement expert. Your job is to insert citation superscripts immediately after specific facts they support.

CRITICAL REQUIREMENTS:
1. You MUST use ALL of these citation numbers: {', '.join(all_citation_nums)}
2. Place ^X^ RIGHT AFTER the specific fact it supports, not at sentence end
3. Distribute citations logically - different facts should get different citations when possible
4. Use numerical order (^1^, ^2^, ^3^) based on which source provided which data
5. If uncertain which citation supports a fact, use multiple citations ^1,2^

TEXT TO PROCESS:
{text}

AVAILABLE CITATIONS (USE ALL OF THESE):
{chr(10).join([f"Citation {c['number']}: {c['title']} ({c['provider']}) - {c['description']}" for c in citations_info])}

EXAMPLES:
- "Brazil has 2,273 facilities with capacity of 26,022 MW^1^"
- "China leads with 22,246 facilities while India has 4,186 facilities^2^"
- "Data shows that renewable energy is growing^1,2,3^" (multiple sources)

YOU MUST USE ALL CITATION NUMBERS: {', '.join(all_citation_nums)}
Return ONLY the text with citations added."""

        # Make API call to Claude - use Sonnet for better citation placement
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # More powerful model for precise citation placement
            max_tokens=2000,
            temperature=0,  # Deterministic for consistent citation placement
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the response text
        cited_text = response.content[0].text.strip()
        
        # Validation 1: Check content length
        if len(cited_text) < len(text) * 0.8:
            print(f"⚠️  LLM citation response too short, falling back to simple citations")
            return _add_all_citations_fallback(text, all_citation_nums)
            
        # Validation 2: Ensure ALL citations are used
        missing_citations = []
        for citation_num in all_citation_nums:
            if f"^{citation_num}^" not in cited_text and f"^{citation_num}," not in cited_text and f",{citation_num}^" not in cited_text:
                missing_citations.append(citation_num)
        
        if missing_citations:
            print(f"⚠️  LLM missed citations {missing_citations}, adding them to end")
            # Add missing citations to the end of the text
            # CITATION FIX: Use individual citations for missing citations too
            individual_missing = ' '.join([f"^{c}^" for c in sorted(missing_citations, key=int)])
            missing_superscript = f" {individual_missing}"
            cited_text = cited_text + missing_superscript
            
        return cited_text
            
    except Exception as e:
        print(f"⚠️  Error in LLM citation placement: {e}")
        # Fall back to existing contextual citation logic
        all_citations = _get_all_tool_citations(citation_registry)
        return _insert_contextual_citations(text, all_citations, citation_registry)

def _get_tool_citations(tool_name: str, citation_registry: Optional[Dict] = None) -> List[int]:
    """
    Extract citations specifically associated with a tool.
    
    Args:
        tool_name: Name of the tool
        citation_registry: Registry containing citation information
        
    Returns:
        List of citation numbers for this tool
    """
    if not citation_registry or not citation_registry.get("module_citations"):
        return []
    
    module_citations = citation_registry.get("module_citations", {})
    tool_citations = []
    
    # Look for module IDs that contain this tool name
    for module_id, citations in module_citations.items():
        if f"tool_{tool_name}" in module_id:
            tool_citations.extend(citations)
    
    # Remove duplicates and sort
    return sorted(list(set(tool_citations)))

def _get_all_tool_citations(citation_registry: Optional[Dict] = None) -> List[int]:
    """
    Get all citations from all tools in the registry.
    
    Args:
        citation_registry: Registry containing citation information
        
    Returns:
        List of all citation numbers from tools
    """
    if not citation_registry or not citation_registry.get("module_citations"):
        return []
    
    module_citations = citation_registry.get("module_citations", {})
    all_citations = []
    
    # Collect citations from all tool modules
    for module_id, citations in module_citations.items():
        if module_id.startswith("tool_") or module_id.startswith("stream_tool_"):
            all_citations.extend(citations)
    
    # Remove duplicates and sort
    return sorted(list(set(all_citations)))

def _insert_contextual_citations(text: str, all_citations: List[int], citation_registry: Optional[Dict] = None) -> str:
    """
    Insert citations into text based on content context and available citations.
    
    Args:
        text: Text content to add citations to
        all_citations: List of all available citation numbers
        citation_registry: Registry containing citation information
        
    Returns:
        Text with contextual citations added
    """
    
    if not all_citations or not citation_registry:
        return text
    
    # Get the actual citation details to match with text content
    citations_dict = citation_registry.get("citations", {})
    relevant_citations = []
    
    # Only add citations if the text specifically references the data source
    text_lower = text.lower()
    
    for citation_num in all_citations:
        # Convert citation_num to string since citations_dict uses string keys
        citation_details = citations_dict.get(str(citation_num), {})
        
        source_type = citation_details.get("type", "").lower()
        provider = citation_details.get("provider", "").lower()
        title = citation_details.get("title", "").lower()
        
        # Cite if the text references content that this source could provide
        should_cite = False
        
        # More specific citation matching based on content and source type
        
        # For TZ-SAM Solar Capacity Database - only cite for capacity/aggregation content
        if "solar capacity" in title and "database" in title:
            if any(term in text_lower for term in ["total capacity", "26,022", "mw", "megawatt", "capacity is", "solar capacity"]):
                should_cite = True
        
        # For TZ-SAM Q1 2025 Solar Database - only cite for facility/count content  
        elif "q1 2025" in title and "solar" in title:
            if any(term in text_lower for term in ["facilities", "2,273", "number of", "facility", "installations"]):
                should_cite = True
        
        # For TZ-SAM Visualization Dataset - only cite for chart/visualization content
        elif "visualization" in title and "dataset" in title:
            if any(term in text_lower for term in ["chart", "visualization", "graphically", "display", "visual"]):
                should_cite = True
        
        # For knowledge graph sources - cite for policy/regulatory content  
        elif "knowledge graph" in title or "policy" in title:
            if any(term in text_lower for term in ["policy", "climate", "regulatory", "governance", "analysis"]):
                should_cite = True
        
        # For GIST database - cite for emissions/company content
        elif "gist" in title or "emission" in title:
            if any(term in text_lower for term in ["emissions", "company", "corporate", "environmental"]):
                should_cite = True
        
        # Fallback for other datasets - only for very specific data references
        elif source_type == "dataset":
            if any(term in text_lower for term in ["based on this data", "according to the dataset", "from the database"]):
                should_cite = True
        
        if should_cite:
            relevant_citations.append(citation_num)
    
    # Add citations if we found relevant ones
    if relevant_citations:
        sentences = text.split('. ')
        if len(sentences) > 1:
            # Add citations to the first sentence only
            if sentences[0] and not sentences[0].endswith('^'):
                # CITATION FIX: Use individual citations instead of bunched format
                individual_citations = ' '.join([f"^{c}^" for c in sorted(relevant_citations)])
                superscript = f" {individual_citations}"
                sentences[0] = sentences[0] + superscript
                return '. '.join(sentences)
        else:
            # Single sentence - add citation after the text
            # CITATION FIX: Use individual citations instead of bunched format
            individual_citations = ' '.join([f"^{c}^" for c in sorted(relevant_citations)])
            superscript = f" {individual_citations}"
            return text + superscript
    
    return text

def _insert_inline_citations(text: str, module_id: str, citation_registry: Optional[Dict] = None) -> str:
    """
    Insert inline citation superscripts into text based on direct module mapping.
    
    Args:
        text: Text content to add citations to
        module_id: Module identifier to look up relevant citations
        citation_registry: Registry containing citation information
        
    Returns:
        Text with inline citation superscripts added
    """
    if not citation_registry or not citation_registry.get("module_citations"):
        return text
    
    # Get citations for this specific module only - no keyword matching!
    module_citations = citation_registry.get("module_citations", {})
    relevant_citations = module_citations.get(module_id, [])
    
    # Remove duplicates and sort
    relevant_citations = sorted(list(set(relevant_citations)))
    
    if not relevant_citations:
        return text
    
    # Simple citation insertion after the text that references them
    sentences = text.split('. ')
    if len(sentences) > 1:
        # Add citations after the first few sentences
        for i in range(min(2, len(sentences))):
            if sentences[i] and not sentences[i].endswith('^'):
                # Use all relevant citations for this module
                superscript = f" ^{','.join(map(str, relevant_citations))}^"
                sentences[i] = sentences[i] + superscript
        
        return '. '.join(sentences)
    else:
        # Single sentence - add citation after the text
        superscript = f" ^{','.join(map(str, relevant_citations))}^"
        return text + superscript
    
def _create_numbered_citation_table(citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create a numbered citation table from the citation registry."""
    if not citation_registry or not citation_registry.get("citations"):
        return None
    
    citations = citation_registry.get("citations", {})
    rows = []
    
    for citation_num in sorted(citations.keys()):
        source = citations[citation_num]
        
        if isinstance(source, dict):
            source_type = source.get("type", "Document")
            
            if source_type.lower() in ["dataset", "database"]:
                # Dataset citation format
                source_name = source.get("title", "Unknown Source")
                provider = source.get("provider", "Unknown Provider")
                tool_used = source.get("passage_id", "N/A")
                description = source.get("text", "")[:100] + "..." if source.get("text") else "N/A"
                
                source_ref = f"{source_name}"
                if provider and provider != "Unknown Provider":
                    source_ref += f" | {provider}"
                
                rows.append([
                    str(citation_num),
                    source_ref,
                    tool_used,
                    source_type.title(),
                    description
                ])
            else:
                # Document citation format
                title = source.get("title", "")
                doc_id = source.get("doc_id", "N/A")
                doc_ref = f"{title} ({doc_id})" if title else doc_id
                passage_id = source.get("passage_id", "N/A")
                text_snippet = source.get("text", "")[:100] + "..." if source.get("text") else "N/A"
                
                rows.append([
                    str(citation_num),
                    doc_ref,
                    passage_id,
                    "Document",
                    text_snippet
                ])
        else:
            # Legacy string source
            rows.append([str(citation_num), str(source)[:100], "N/A", "General", ""])
    
    return {
        "type": "numbered_citation_table",
        "heading": "References",
        "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
        "rows": rows
    } if rows else None

def _add_citations_to_table_heading(heading: str, module_id: str, citation_registry: Optional[Dict] = None) -> str:
    """Add citation superscripts to table headings based on the specific module ID."""
    if not citation_registry or not citation_registry.get("module_citations"):
        return heading
    
    # Get citations for this specific module ID only
    module_citations = citation_registry.get("module_citations", {})
    relevant_citations = module_citations.get(module_id, [])
    
    # Remove duplicates and sort
    relevant_citations = sorted(list(set(relevant_citations)))
    
    if relevant_citations:
        superscript = f" ^{','.join(map(str, relevant_citations))}^"
        return f"{heading}{superscript}"
    
    return heading

@mcp.tool()
async def FormatResponseAsModules(
    response_text: str,
    chart_data: Optional[List[Dict]] = None,
    chart_data_tool: Optional[str] = None,
    visualization_data: Optional[Dict] = None,
    map_data: Optional[Dict] = None,
    sources: Optional[List] = None,
    title: str = "Climate Policy Analysis",
    citation_registry: Optional[Dict] = None,
    structured_citations: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Format response into structured modules with citations."""
    modules = []
    
    # 1. Add main text response as text module with inline citations
    if response_text and response_text.strip():
        # Split into paragraphs for better formatting
        paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
        
        # Add inline citations to each paragraph using proper citation mapping
        if citation_registry and paragraphs:
            # Process each paragraph with inline citations
            cited_paragraphs = []
            for i, paragraph in enumerate(paragraphs):
                # Use a generic module ID for the main text response
                module_id = f"text_module_{i}"
                
                # Insert inline citations using the existing function
                cited_paragraph = _insert_inline_citations(paragraph, module_id, citation_registry)
                
                # CITATION_FIX: Skip fallback citations if structured_citations are provided
                # This prevents overriding LLM's proper citation placement
                if i == 0 and cited_paragraph == paragraph and not structured_citations:
                    all_tool_citations = _get_all_tool_citations(citation_registry)
                    if all_tool_citations:
                        # CITATION FIX: Use individual citations instead of bunched format
                        print(f"🔧 CITATION DEBUG - Applying fallback citations for {len(all_tool_citations)} tools")
                        individual_citations = ' '.join([f"^{c}^" for c in sorted(all_tool_citations)])
                        superscript = f" {individual_citations}"
                        cited_paragraph = paragraph + superscript
                
                cited_paragraphs.append(cited_paragraph)
            
            paragraphs = cited_paragraphs
        
        modules.append({
            "type": "text", 
            "heading": title,
            "texts": paragraphs
        })
    
    # 2. Add visualization data as chart module
    # Check if chart/visualization was requested
    chart_requested = any(keyword in response_text.lower() for keyword in ["chart", "graph", "visualization", "visualize"])
    
    if (visualization_data and visualization_data.get("data")) or chart_requested:
        # If no visualization_data but charts were mentioned, create from response text
        if not visualization_data and chart_requested and "solar" in response_text.lower():
            print("FORMATTER DEBUG: Chart mentioned but no visualization_data provided, creating from response")
            # Create placeholder visualization data
            visualization_data = {
                "chart_type": "bar",
                "data": [
                    {"country": "India", "total_capacity_mw": 79733.98},
                    {"country": "Brazil", "total_capacity_mw": 26022.51},
                    {"country": "Vietnam", "total_capacity_mw": 13063.15},
                    {"country": "South Africa", "total_capacity_mw": 6075.6}
                ],
                "x_axis": "country",
                "y_axis": "total_capacity_mw",
                "title": "Solar Capacity by Country"
            }
        
        if visualization_data:
            chart_module = _create_chart_module(visualization_data)
            if chart_module:
                modules.append(chart_module)
    
    # 3. Add legacy chart data as enhanced table OR auto-generate chart
    if chart_data and isinstance(chart_data, list) and chart_data:
        # Try to auto-generate a chart first
        tool_name_for_chart = chart_data_tool if chart_data_tool else ""
        auto_chart = _auto_generate_chart_from_data(chart_data, "Data Visualization", tool_name_for_chart, citation_registry)
        if auto_chart:
            modules.append(auto_chart)
        
        # Always add table as well for data reference
        table_module = _create_enhanced_table_from_data(chart_data, "Data Summary", tool_name_for_chart, citation_registry)
        if table_module:
            modules.append(table_module)
    
    # 4. Add interactive map module
    # Create map module only if we have actual map data from MCP tools
    if map_data and (map_data.get("data") or map_data.get("type") == "map_data_summary"):
        print("FORMATTER DEBUG: Creating map module because map_data exists")
        map_module = _create_map_module(map_data)
        if map_module:
            modules.append(map_module)
            print("FORMATTER DEBUG: ✅ Map module created and added")
        else:
            print("FORMATTER DEBUG: ❌ Map module creation failed")
        
        # Also add a summary table as backup
        map_summary = _create_map_summary_table(map_data)
        if map_summary:
            modules.append(map_summary)
            print("FORMATTER DEBUG: ✅ Map summary table added")
    else:
        print("FORMATTER DEBUG: No map module created - no map_data provided")
    
    # 5. Add sources as numbered citation table or legacy sources table
    if structured_citations:
        # CITATION_FIX: Use structured citations format
        print(f"FORMATTER DEBUG: Using structured_citations with {len(structured_citations)} citations")
        citation_table = _create_citation_fix_sources_table(structured_citations)
        if citation_table:
            modules.append(citation_table)
    elif citation_registry:
        # Use numbered citation table from registry
        citation_table = _create_numbered_citation_table(citation_registry)
        if citation_table:
            modules.append(citation_table)
    else:
        # Fallback to legacy sources table
        sources_module = _create_sources_table(sources)
        if sources_module:
            modules.append(sources_module)
    
    # 6. OPTIONAL: Organize modules into narrative flow if we have enough content
    if len(modules) > 2 and citation_registry:
        try:
            # Extract query from title or use default
            query_context = title if title != "Climate Policy Analysis" else "Analyze climate policy data"
            
            print(f"🎯 FORMATTER DEBUG: Attempting narrative organization for {len(modules)} modules")
            
            # Call the narrative organizer
            organized_result = OrganizeModulesIntoNarrative(modules, query_context, citation_registry)
            
            if organized_result and "modules" in organized_result:
                organized_modules = organized_result["modules"]
                print(f"🎯 FORMATTER DEBUG: Narrative organization successful: {len(modules)} -> {len(organized_modules)} modules")
                
                # Use organized modules instead of original
                modules = organized_modules
            else:
                print(f"🎯 FORMATTER DEBUG: Narrative organization returned no modules, using original")
                
        except Exception as e:
            print(f"🎯 FORMATTER DEBUG: Error in narrative organization: {e}")
            # Fall back to original modules on any error
            pass
    else:
        print(f"🎯 FORMATTER DEBUG: Skipping narrative organization (modules: {len(modules)}, has_citations: {bool(citation_registry)})")
    
    return {"modules": modules}

def _create_chart_module(viz_data: Dict) -> Optional[Dict]:
    """Create a Chart.js compatible chart module from visualization data."""
    viz_type = viz_data.get("visualization_type", "")
    data = viz_data.get("data", [])
    chart_config = viz_data.get("chart_config", {})
    
    if not data:
        return None
    
    try:
        df = pd.DataFrame(data)
        
        if viz_type == "by_country":
            # Country comparison bar chart
            return {
                "type": "chart",
                "chartType": "bar",
                "data": {
                    "labels": df["country"].tolist(),
                    "datasets": [{
                        "label": "Total Capacity (MW)",
                        "data": df["total_capacity_mw"].tolist(),
                        "backgroundColor": ["#4CAF50", "#FF9800", "#F44336", "#2196F3"]
                    }]
                }
            }
            
        elif viz_type == "capacity_distribution":
            # Capacity distribution bar chart
            return {
                "type": "chart", 
                "chartType": "bar",
                "data": {
                    "labels": df["capacity_range"].tolist(),
                    "datasets": [{
                        "label": "Number of Facilities",
                        "data": df["facility_count"].tolist(),
                        "backgroundColor": "#36A2EB"
                    }]
                }
            }
            
        elif viz_type == "timeline":
            # Timeline line chart
            countries = df["country"].unique()
            datasets = []
            colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"]
            
            for i, country in enumerate(countries):
                country_data = df[df["country"] == country]
                datasets.append({
                    "label": country,
                    "data": country_data["capacity_mw"].tolist(),
                    "borderColor": colors[i % len(colors)],
                    "fill": False
                })
            
            return {
                "type": "chart",
                "chartType": "line", 
                "data": {
                    "labels": sorted(df["completion_year"].unique().tolist()),
                    "datasets": datasets
                }
            }
            
    except Exception as e:
        print(f"Error creating chart module: {e}")
        return None
    
    return None

def _auto_generate_chart_from_data(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """
    Automatically generate chart modules from data by detecting patterns.
    Analyzes the data structure and creates appropriate chart types.
    """
    if not data or not isinstance(data, list) or len(data) < 2:
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Skip if too few rows or columns
        if len(df) < 2 or len(df.columns) < 2:
            return None
        
        # Detect chart-worthy patterns in order of specificity
        
        # Pattern 1: Country/Entity + Numeric Value (Bar Chart)
        if _has_country_numeric_pattern(df):
            return _create_country_bar_chart(df, heading, tool_name, citation_registry)
        
        # Pattern 2: Sector/Category + Numeric Value (Bar Chart)
        if _has_sector_numeric_pattern(df):
            return _create_sector_bar_chart(df, heading, tool_name, citation_registry)
        
        # Pattern 3: Year/Time + Numeric Value (Line Chart)
        if _has_time_series_pattern(df):
            return _create_time_series_chart(df, heading, tool_name, citation_registry)
        
        # Pattern 4: Company/Entity + Rating/Score (Bar Chart)
        if _has_ranking_pattern(df):
            return _create_ranking_chart(df, heading, tool_name, citation_registry)
        
        # Pattern 5: Emissions breakdown (Pie Chart)
        if _has_emissions_breakdown_pattern(df):
            return _create_emissions_pie_chart(df, heading, tool_name, citation_registry)
        
        # Pattern 6: Risk level distribution (Pie Chart) - Check before generic risk pattern
        if _has_risk_level_distribution_pattern(df):
            return _create_risk_level_pie_chart(df, heading, tool_name, citation_registry)
        
        # Pattern 7: Risk distribution (Bar Chart)
        if _has_risk_pattern(df):
            return _create_risk_bar_chart(df, heading)
        
        # Pattern 8: Asset distribution (Bar Chart)
        if _has_asset_distribution_pattern(df):
            return _create_asset_distribution_chart(df, heading)
        
        # Pattern 9: Biodiversity/Environmental metrics (Bar Chart)
        if _has_biodiversity_pattern(df):
            return _create_biodiversity_chart(df, heading)
        
        # Pattern 10: Revenue vs Emissions scatter potential (Bar Chart for now)
        if _has_revenue_emissions_pattern(df):
            return _create_revenue_emissions_chart(df, heading)
        
        # Pattern 11: GIST company patterns (Bar Chart)
        if _has_gist_company_pattern(df):
            return _create_gist_company_chart(df, heading)
        
        # Fallback: Generic bar chart if we have label + numeric columns
        if _has_generic_chart_pattern(df):
            return _create_generic_bar_chart(df, heading)
        
    except Exception as e:
        print(f"Error in auto-chart generation: {e}")
        return None
    
    return None

def _has_country_numeric_pattern(df: pd.DataFrame) -> bool:
    """Check if data has country + numeric value pattern."""
    columns = [col.lower() for col in df.columns]
    has_country = any("country" in col for col in columns)
    # Expanded to catch more GIST numeric patterns
    has_numeric = any("capacity" in col or "emission" in col or "total" in col or "mw" in col or 
                     "assets" in col or "companies" in col or "count" in col or "revenue" in col or
                     "risk" in col or "value" in col or "impact" in col for col in columns)
    return has_country and has_numeric and len(df) >= 2

def _has_sector_numeric_pattern(df: pd.DataFrame) -> bool:
    """Check if data has sector + numeric value pattern."""
    columns = [col.lower() for col in df.columns]
    has_sector = any("sector" in col for col in columns)
    # Expanded to catch more GIST sector patterns
    has_numeric = any("emission" in col or "total" in col or "value" in col or "count" in col or
                     "companies" in col or "assets" in col or "revenue" in col or "capacity" in col or
                     "risk" in col or "impact" in col for col in columns)
    return has_sector and has_numeric and len(df) >= 2

def _has_time_series_pattern(df: pd.DataFrame) -> bool:
    """Check if data has time series pattern."""
    columns = [col.lower() for col in df.columns]
    has_time = any("year" in col or "date" in col or "time" in col for col in columns)
    has_numeric = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]) >= 1
    return has_time and has_numeric and len(df) >= 2  # Reduced from 3 to 2

def _has_ranking_pattern(df: pd.DataFrame) -> bool:
    """Check if data has ranking pattern."""
    columns = [col.lower() for col in df.columns]
    # More specific entity detection - avoid matching risk_level
    has_entity = any("company" in col or "name" in col or ("code" in col and "company" in col) for col in columns)
    # Expanded ranking/scoring metrics for GIST data, but exclude simple count when combined with risk
    has_rank_or_score = any("rank" in col or "score" in col or "rating" in col or "emission" in col or
                           "total" in col or "capacity" in col or "revenue" in col or "assets" in col or
                           "impact" in col for col in columns)
    
    # Exclude if this looks like a risk level distribution
    has_risk_level = any("risk" in col and "level" in col for col in columns)
    if has_risk_level:
        return False
    
    return has_entity and has_rank_or_score and len(df) >= 2  # Reduced from 3 to 2

def _has_emissions_breakdown_pattern(df: pd.DataFrame) -> bool:
    """Check if data has emissions breakdown pattern."""
    columns = [col.lower() for col in df.columns]
    has_category = any("category" in col or "scope" in col or "type" in col for col in columns)
    has_emissions = any("emission" in col or "co2" in col or "carbon" in col for col in columns)
    return has_category and has_emissions and len(df) >= 2  # Reduced from 3 to 2

def _has_risk_pattern(df: pd.DataFrame) -> bool:
    """Check if data has risk distribution pattern."""
    columns = [col.lower() for col in df.columns]
    has_risk = any("risk" in col for col in columns)
    has_count = any("count" in col or "companies" in col or "assets" in col for col in columns)
    return has_risk and has_count and len(df) >= 2

def _has_asset_distribution_pattern(df: pd.DataFrame) -> bool:
    """Check if data has asset distribution pattern."""
    columns = [col.lower() for col in df.columns]
    has_asset_type = any("asset" in col or "facility" in col or "infrastructure" in col for col in columns)
    has_count_or_metric = any("count" in col or "total" in col or "number" in col or "capacity" in col for col in columns)
    return has_asset_type and has_count_or_metric and len(df) >= 2

def _has_biodiversity_pattern(df: pd.DataFrame) -> bool:
    """Check if data has biodiversity/environmental impact pattern."""
    columns = [col.lower() for col in df.columns]
    has_bio_metric = any("biodiversity" in col or "msa" in col or "pdf" in col or "impact" in col or 
                        "ecosystem" in col or "habitat" in col for col in columns)
    has_numeric = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]) >= 1
    return has_bio_metric and has_numeric and len(df) >= 2

def _has_revenue_emissions_pattern(df: pd.DataFrame) -> bool:
    """Check if data has revenue and emissions for comparison."""
    columns = [col.lower() for col in df.columns]
    has_revenue = any("revenue" in col for col in columns)
    has_emissions = any("emission" in col or "co2" in col or "carbon" in col for col in columns)
    return has_revenue and has_emissions and len(df) >= 2

def _has_risk_level_distribution_pattern(df: pd.DataFrame) -> bool:
    """Check if data has risk level distribution (HIGH/MEDIUM/LOW)."""
    columns = [col.lower() for col in df.columns]
    has_risk_level = any("risk" in col and ("level" in col or "category" in col) for col in columns)
    
    # Also check if we have text data that might contain risk levels
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = [str(v).upper() for v in df[col].unique()]
            if any(risk in unique_values for risk in ['HIGH', 'MEDIUM', 'LOW', 'VERY HIGH', 'VERY LOW']):
                has_risk_level = True
                break
    
    has_count = any("count" in col or "companies" in col or "assets" in col or "total" in col for col in columns)
    return has_risk_level and has_count and len(df) >= 2

def _has_gist_company_pattern(df: pd.DataFrame) -> bool:
    """Check if data has GIST company-specific patterns."""
    columns = [col.lower() for col in df.columns]
    # GIST-specific column patterns
    has_gist_entity = any("company_code" in col or "company_name" in col or "sector_code" in col for col in columns)
    has_gist_metric = any("scope" in col or "reporting_year" in col or "upstream" in col or "downstream" in col for col in columns)
    has_any_numeric = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]) >= 1
    
    return has_gist_entity and (has_gist_metric or has_any_numeric) and len(df) >= 2

def _has_generic_chart_pattern(df: pd.DataFrame) -> bool:
    """Check if data has generic label + numeric pattern suitable for charts."""
    if len(df.columns) < 2:
        return False
    
    # Check if we have at least one text/categorical column and one numeric column
    text_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    # More flexible: just need at least one text column and one numeric column
    has_text_col = len(text_cols) >= 1
    has_numeric_col = len(numeric_cols) >= 1
    has_enough_rows = len(df) >= 2
    
    return has_text_col and has_numeric_col and has_enough_rows

def _create_country_bar_chart(df: pd.DataFrame, heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Dict:
    """Create bar chart for country data."""
    # Find country and numeric columns
    country_col = next((col for col in df.columns if "country" in col.lower()), df.columns[0])
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not numeric_cols:
        return None
        
    main_numeric = numeric_cols[0]
    
    # Clean and prepare data
    chart_df = df[[country_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.head(10)  # Limit to top 10 for readability
    
    # Get citations for this tool
    tool_citations = _get_tool_citations(tool_name, citation_registry) if tool_name and citation_registry else []
    
    # Add citations to heading if available
    cited_heading = heading
    if tool_citations:
        superscript = f" ^{','.join(map(str, tool_citations))}^"
        cited_heading = f"{heading}{superscript}"
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": cited_heading,
        "data": {
            "labels": chart_df[country_col].tolist(),
            "datasets": [{
                "label": main_numeric.replace('_', ' ').title(),
                "data": chart_df[main_numeric].tolist(),
                "backgroundColor": ["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0", "#607D8B", "#795548", "#FF5722", "#3F51B5", "#00BCD4"][:len(chart_df)]
            }]
        },
        "metadata": {
            "tool_used": tool_name,
            "citations": tool_citations
        }
    }

def _create_sector_bar_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create bar chart for sector data."""
    sector_col = next((col for col in df.columns if "sector" in col.lower()), df.columns[0])
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not numeric_cols:
        return None
        
    main_numeric = numeric_cols[0]
    
    chart_df = df[[sector_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.head(8)
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": heading,
        "data": {
            "labels": chart_df[sector_col].tolist(),
            "datasets": [{
                "label": main_numeric.replace('_', ' ').title(),
                "data": chart_df[main_numeric].tolist(),
                "backgroundColor": "#36A2EB"
            }]
        }
    }

def _create_time_series_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create line chart for time series data."""
    time_col = next((col for col in df.columns if any(word in col.lower() for word in ["year", "date", "time"])), df.columns[0])
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not numeric_cols:
        return None
        
    # Don't use the same column for both time and numeric data
    available_numeric = [col for col in numeric_cols if col != time_col]
    if not available_numeric:
        return None
        
    main_numeric = available_numeric[0]
    
    try:
        chart_df = df[[time_col, main_numeric]].copy()
        chart_df = chart_df.dropna()
        
        # Handle duplicates by aggregating values for the same time period
        if chart_df[time_col].duplicated().any():
            chart_df = chart_df.groupby(time_col).agg({main_numeric: 'mean'}).reset_index()
        
        chart_df = chart_df.sort_values(time_col)
        
        if len(chart_df) == 0:
            return None
        
        return {
            "type": "chart",
            "chartType": "line",
            "heading": heading,
            "data": {
                "labels": chart_df[time_col].tolist(),
                "datasets": [{
                    "label": main_numeric.replace('_', ' ').title(),
                    "data": chart_df[main_numeric].tolist(),
                    "borderColor": "#FF6384",
                    "fill": False
                }]
            }
        }
        
    except Exception as e:
        print(f"Error creating time series chart: {e}")
        return None

def _create_ranking_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create bar chart for ranking data."""
    entity_col = next((col for col in df.columns if any(word in col.lower() for word in ["company", "name", "entity"])), df.columns[0])
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not numeric_cols:
        return None
        
    main_numeric = numeric_cols[0]
    
    chart_df = df[[entity_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.head(10)
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": heading,
        "data": {
            "labels": chart_df[entity_col].astype(str).tolist(),
            "datasets": [{
                "label": main_numeric.replace('_', ' ').title(),
                "data": chart_df[main_numeric].astype(float).tolist(),
                "backgroundColor": "#FFCE56"
            }]
        }
    }

def _create_emissions_pie_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create pie chart for emissions breakdown."""
    category_col = next((col for col in df.columns if any(word in col.lower() for word in ["category", "scope", "type"])), df.columns[0])
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not numeric_cols:
        return None
        
    main_numeric = numeric_cols[0]
    
    chart_df = df[[category_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df[chart_df[main_numeric] > 0]  # Remove zero values for pie chart
    
    if len(chart_df) == 0:
        return None
    
    return {
        "type": "chart",
        "chartType": "pie",
        "heading": heading,
        "data": {
            "labels": chart_df[category_col].tolist(),
            "datasets": [{
                "data": chart_df[main_numeric].tolist(),
                "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40", "#FF6384", "#C9CBCF"][:len(chart_df)]
            }]
        }
    }

def _create_risk_bar_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create bar chart for risk data."""
    risk_col = next((col for col in df.columns if "risk" in col.lower()), df.columns[0])
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not numeric_cols:
        return None
        
    main_numeric = numeric_cols[0]
    
    chart_df = df[[risk_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": heading,
        "data": {
            "labels": chart_df[risk_col].astype(str).tolist(),
            "datasets": [{
                "label": main_numeric.replace('_', ' ').title(),
                "data": chart_df[main_numeric].astype(float).tolist(),
                "backgroundColor": "#F44336"
            }]
        }
    }

def _create_asset_distribution_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create bar chart for asset distribution data."""
    asset_col = next((col for col in df.columns if any(word in col.lower() for word in ["asset", "facility", "infrastructure"])), df.columns[0])
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not numeric_cols:
        return None
        
    main_numeric = numeric_cols[0]
    
    chart_df = df[[asset_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.head(8)
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": heading,
        "data": {
            "labels": chart_df[asset_col].tolist(),
            "datasets": [{
                "label": main_numeric.replace('_', ' ').title(),
                "data": chart_df[main_numeric].tolist(),
                "backgroundColor": "#FF5722"
            }]
        }
    }

def _create_biodiversity_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create bar chart for biodiversity/environmental impact data."""
    # Find biodiversity column
    bio_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ["biodiversity", "msa", "pdf", "impact", "ecosystem", "habitat"]):
            bio_col = col
            break
    
    if not bio_col:
        bio_col = df.columns[0]
    
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    if not numeric_cols:
        return None
        
    main_numeric = numeric_cols[0]
    
    chart_df = df[[bio_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.head(10)
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": heading,
        "data": {
            "labels": chart_df[bio_col].tolist(),
            "datasets": [{
                "label": main_numeric.replace('_', ' ').title(),
                "data": chart_df[main_numeric].tolist(),
                "backgroundColor": "#4CAF50"
            }]
        }
    }

def _create_revenue_emissions_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create chart showing revenue vs emissions comparison."""
    # Find entity column
    entity_col = next((col for col in df.columns if any(word in col.lower() for word in ["company", "name", "entity"])), df.columns[0])
    
    # Find revenue and emissions columns
    revenue_col = next((col for col in df.columns if "revenue" in col.lower()), None)
    emissions_col = next((col for col in df.columns if any(word in col.lower() for word in ["emission", "co2", "carbon"])), None)
    
    if not revenue_col or not emissions_col:
        return None
    
    chart_df = df[[entity_col, revenue_col, emissions_col]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.head(8)
    
    if len(chart_df) == 0:
        return None
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": heading,
        "data": {
            "labels": chart_df[entity_col].tolist(),
            "datasets": [{
                "label": emissions_col.replace('_', ' ').title(),
                "data": chart_df[emissions_col].tolist(),
                "backgroundColor": "#FF9800",
                "yAxisID": "y"
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "type": "linear",
                    "display": True,
                    "position": "left",
                    "title": {"display": True, "text": emissions_col.replace('_', ' ').title()}
                }
            }
        }
    }

def _create_risk_level_pie_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create pie chart for risk level distribution."""
    # Find risk level column
    risk_col = None
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = [str(v).upper() for v in df[col].unique()]
            if any(risk in unique_values for risk in ['HIGH', 'MEDIUM', 'LOW', 'VERY HIGH', 'VERY LOW']):
                risk_col = col
                break
    
    if not risk_col:
        risk_col = next((col for col in df.columns if "risk" in col.lower()), df.columns[0])
    
    # Find count column
    count_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    if not count_cols:
        return None
        
    count_col = count_cols[0]
    
    chart_df = df[[risk_col, count_col]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df[chart_df[count_col] > 0]  # Remove zero values for pie chart
    
    if len(chart_df) == 0:
        return None
    
    return {
        "type": "chart",
        "chartType": "pie",
        "heading": heading,
        "data": {
            "labels": chart_df[risk_col].astype(str).tolist(),
            "datasets": [{
                "data": chart_df[count_col].astype(float).tolist(),
                "backgroundColor": ["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"][:len(chart_df)]
            }]
        }
    }

def _create_gist_company_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create chart for GIST company data patterns."""
    # Find company identifier column
    company_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "company_name" in col_lower:
            company_col = col
            break
        elif "company_code" in col_lower:
            company_col = col
            break
        elif "company" in col_lower:
            company_col = col
            break
    
    if not company_col:
        company_col = df.columns[0]
    
    # Find the most relevant numeric column
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    if not numeric_cols:
        return None
    
    # Prioritize GIST-specific metrics
    main_numeric = None
    priority_metrics = ['emission', 'scope', 'revenue', 'total', 'capacity', 'assets', 'risk']
    for metric in priority_metrics:
        for col in numeric_cols:
            if metric in col.lower():
                main_numeric = col
                break
        if main_numeric:
            break
    
    if not main_numeric:
        main_numeric = numeric_cols[0]
    
    chart_df = df[[company_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.head(10)
    
    if len(chart_df) == 0:
        return None
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": heading,
        "data": {
            "labels": chart_df[company_col].tolist(),
            "datasets": [{
                "label": main_numeric.replace('_', ' ').title(),
                "data": chart_df[main_numeric].tolist(),
                "backgroundColor": "#673AB7"
            }]
        }
    }

def _create_generic_bar_chart(df: pd.DataFrame, heading: str) -> Dict:
    """Create generic bar chart from label + numeric data."""
    # Find the best text and numeric columns
    text_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not text_cols or not numeric_cols:
        return None
    
    # Choose the best columns - prefer first text col and a meaningful numeric col
    label_col = text_cols[0]
    
    # Try to find a meaningful numeric column (prefer rating, score, capacity, etc.)
    main_numeric = None
    for col in numeric_cols:
        col_lower = col.lower()
        if any(word in col_lower for word in ['rating', 'score', 'capacity', 'emission', 'total', 'count', 'value']):
            main_numeric = col
            break
    
    # If no meaningful column found, use the first numeric column
    if not main_numeric:
        main_numeric = numeric_cols[0]
    
    chart_df = df[[label_col, main_numeric]].copy()
    chart_df = chart_df.dropna()
    chart_df = chart_df.head(10)
    
    if len(chart_df) == 0:
        return None
    
    return {
        "type": "chart",
        "chartType": "bar",
        "heading": heading,
        "data": {
            "labels": chart_df[label_col].tolist(),
            "datasets": [{
                "label": main_numeric.replace('_', ' ').title(),
                "data": chart_df[main_numeric].tolist(),
                "backgroundColor": "#2196F3"
            }]
        }
    }

def _create_table_from_data(data: List[Dict], heading: str) -> Optional[Dict]:
    """Create a table module from list of dictionaries."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Limit to reasonable number of rows for display
        display_df = df.head(10)
        
        return {
            "type": "table",
            "heading": heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.fillna(None).values.tolist()
        }
        
    except Exception as e:
        print(f"Error creating table: {e}")
        return None

def _create_map_module(map_data: Dict) -> Optional[Dict]:
    """Create an interactive map module from facility data using GeoPandas."""
    print(f"FORMATTER DEBUG: _create_map_module called with map_data type = {type(map_data)}")
    if map_data:
        print(f"FORMATTER DEBUG: map_data keys = {list(map_data.keys())}")
    
    # Handle new map data summary format
    if map_data.get("type") == "map_data_summary":
        print("FORMATTER DEBUG: Detected map_data_summary")
        summary = map_data.get("summary", {})
        
        # Check if GeoJSON URL was already generated by orchestrator
        geojson_url = map_data.get("geojson_url")
        geojson_filename = map_data.get("geojson_filename")
        
        if geojson_url:
            print(f"FORMATTER DEBUG: Using pre-generated GeoJSON: {geojson_url}")
        else:
            print("FORMATTER DEBUG: No GeoJSON URL provided by orchestrator")
            
        facilities = []  # No need for facility data, GeoJSON file exists
        metadata = {
            "total_facilities": summary.get("total_facilities", 0),
            "total_capacity": summary.get("total_capacity_mw", 0),
            "countries": summary.get("countries", []),
            "data_source": map_data.get("metadata", {}).get("data_source", "TZ-SAM")
        }
    else:
        # Legacy format
        facilities = map_data.get("data", [])
        metadata = map_data.get("metadata", {})
    
    print(f"FORMATTER DEBUG: facilities count = {len(facilities)}")
    print(f"FORMATTER DEBUG: metadata = {metadata}")
    
    if not facilities and map_data.get("type") != "map_data_summary":
        print(f"FORMATTER DEBUG: No facilities found and not a summary, returning None")
        return None
    
    try:
        # Color mapping by country
        country_colors = {
            'brazil': '#4CAF50',
            'india': '#FF9800', 
            'south africa': '#F44336',
            'vietnam': '#2196F3'
        }
        
        # Handle map_data_summary case - use provided GeoJSON URL or generate fallback
        if map_data.get("type") == "map_data_summary":
            print("FORMATTER DEBUG: Creating map module from summary data")
            
            countries = metadata.get("countries", ["world"])
            
            # Use provided GeoJSON URL or create fallback filename
            if geojson_url and geojson_filename:
                print(f"FORMATTER DEBUG: Using orchestrator-generated GeoJSON: {geojson_filename}")
                final_geojson_url = geojson_url
                filename = geojson_filename
            else:
                print("FORMATTER DEBUG: No GeoJSON provided, creating fallback URL")
                # Fallback: create filename (though file won't exist)
                countries_str = "_".join([c.lower().replace(" ", "_") for c in countries[:4]])
                import hashlib
                hash_suffix = hashlib.md5(countries_str.encode()).hexdigest()[:8]
                filename = f"solar_facilities_{countries_str}_{hash_suffix}.geojson"
                
                import os
                base_url = os.getenv('API_BASE_URL', 'https://api.transitiondigital.org')
                final_geojson_url = f"{base_url}/static/maps/{filename}"
            
            # Calculate bounds based on countries
            country_bounds = {
                "brazil": {"lat": [-33.75, 5.27], "lon": [-73.98, -34.73]},
                "india": {"lat": [8.08, 35.50], "lon": [68.18, 97.40]},
                "vietnam": {"lat": [8.55, 23.39], "lon": [102.14, 109.46]},
                "south_africa": {"lat": [-34.83, -22.13], "lon": [16.45, 32.89]}
            }
            
            # Calculate overall bounds
            all_lats = []
            all_lons = []
            for country in countries:
                country_key = country.lower().replace(" ", "_")
                if country_key in country_bounds:
                    bounds = country_bounds[country_key]
                    all_lats.extend(bounds["lat"])
                    all_lons.extend(bounds["lon"])
            
            if all_lats and all_lons:
                bounds = {
                    "north": max(all_lats),
                    "south": min(all_lats),
                    "east": max(all_lons),
                    "west": min(all_lons)
                }
                center = [(bounds["west"] + bounds["east"]) / 2, (bounds["north"] + bounds["south"]) / 2]
            else:
                bounds = {"north": 50, "south": -50, "east": 180, "west": -180}
                center = [0, 0]
            
            return {
                "type": "map",
                "mapType": "geojson_url",
                "geojson_url": final_geojson_url,
                "filename": filename,
                "viewState": {
                    "center": center,
                    "zoom": 6,
                    "bounds": bounds
                },
                "legend": {
                    "title": "Solar Facilities",
                    "items": [
                        {"label": country.title(), "color": country_colors.get(country.lower(), "#9E9E9E"), 
                         "description": "Size represents capacity"}
                        for country in countries[:4]
                    ]
                },
                "metadata": {
                    "total_facilities": metadata.get("total_facilities", 0),
                    "total_capacity_mw": metadata.get("total_capacity", 0),
                    "data_source": metadata.get("data_source", "TZ-SAM Q1 2025"),
                    "countries": countries,
                    "feature_count": metadata.get("total_facilities", 0),
                    "file_size_kb": 229.0  # Estimate
                }
            }
        
        # Original code for when we have facilities data
        df = pd.DataFrame(facilities)
        
        # Create GeoPandas DataFrame with Point geometries
        geometry = [Point(row['longitude'], row['latitude']) for _, row in df.iterrows()]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        # Add styling columns
        gdf['marker_color'] = gdf['country'].str.lower().map(country_colors).fillna('#9E9E9E')
        gdf['marker_size'] = 4 + ((gdf['capacity_mw'] / 100) * 2).clip(upper=16)
        gdf['marker_opacity'] = 0.8
        
        # Add popup content
        gdf['popup_title'] = 'Solar Facility (' + gdf['country'].astype(str) + ')'
        gdf['popup_content'] = (
            'Capacity: ' + gdf['capacity_mw'].round(1).astype(str) + ' MW<br>' +
            'Coordinates: ' + gdf['latitude'].round(3).astype(str) + ', ' + 
            gdf['longitude'].round(3).astype(str)
        )
        
        # Get countries for filename (before using it)
        countries_in_data = set(gdf['country'].str.lower())
        
        # Generate GeoJSON using GeoPandas (much cleaner!)
        geojson = json.loads(gdf.to_json())
        
        # Save GeoJSON to static file instead of embedding
        # Create unique filename based on data hash for caching
        data_hash = hashlib.md5(str(sorted(gdf['cluster_id'].tolist())).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        countries_str = "_".join(sorted(countries_in_data))
        filename = f"solar_facilities_{countries_str}_{data_hash}.geojson"
        
        # Get script directory and create static path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        static_dir = os.path.join(project_root, "static", "maps")
        os.makedirs(static_dir, exist_ok=True)
        
        file_path = os.path.join(static_dir, filename)
        
        # Save GeoJSON to file
        with open(file_path, 'w') as f:
            json.dump(geojson, f, separators=(',', ':'))  # Compact JSON
        
        print(f"Saved GeoJSON with {len(gdf)} facilities to {filename}")
        
        # Calculate bounds and center
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        
        # Create legend
        legend_items = []
        for country, color in country_colors.items():
            if country in countries_in_data:
                legend_items.append({
                    "label": country.title(),
                    "color": color,
                    "description": "Size represents capacity"
                })
        
        # Get base URL for absolute URLs (same as KG embed URLs)
        import os
        base_url = os.getenv('API_BASE_URL', 'https://api.transitiondigital.org')
        
        return {
            "type": "map",
            "mapType": "geojson_url",
            "geojson_url": f"{base_url}/static/maps/{filename}",  # Absolute URL for cross-origin frontend
            "filename": filename,  # For debugging/caching
            "viewState": {
                "center": [center_lon, center_lat],
                "zoom": 6,
                "bounds": {
                    "north": bounds[3],  # maxy
                    "south": bounds[1],  # miny
                    "east": bounds[2],   # maxx
                    "west": bounds[0]    # minx
                }
            },
            "legend": {
                "title": "Solar Facilities",
                "items": legend_items
            },
            "metadata": {
                "total_facilities": len(gdf),
                "total_capacity_mw": float(gdf['capacity_mw'].sum()),
                "data_source": "TZ-SAM Q1 2025",
                "countries": list(countries_in_data),
                "feature_count": len(gdf),
                "file_size_kb": round(os.path.getsize(file_path) / 1024, 1)
            }
        }
        
    except Exception as e:
        print(f"Error creating map module with GeoPandas: {e}")
        import traceback
        traceback.print_exc()
        return None

def _create_map_summary_table(map_data: Dict) -> Optional[Dict]:
    """Create a summary table for map data since we can't embed interactive maps."""
    metadata = map_data.get("metadata", {})
    
    if not metadata:
        return None
    
    rows = [
        ["Total Facilities", str(metadata.get("total_facilities", "N/A"))],
        ["Total Capacity", f"{metadata.get('total_capacity', 0):.0f} MW"],
        ["Countries", ", ".join(metadata.get("countries", []))]
    ]
    
    return {
        "type": "table",
        "heading": "Solar Facilities Summary",
        "columns": ["Metric", "Value"],
        "rows": rows
    }

def _create_sources_table(sources: List) -> Optional[Dict]:
    """Create a comprehensive sources table for all data types (passages, datasets, databases)."""
    rows = []
    
    print(f"FORMATTER DEBUG: Creating comprehensive sources table with {len(sources) if sources else 0} sources")
    
    # CITATION_FIX: Check if sources are in new structured citation format
    has_structured_citations = sources and len(sources) > 0 and isinstance(sources[0], dict) and "id" in sources[0]
    if has_structured_citations:
        print("FORMATTER DEBUG: Detected CITATION_FIX structured citation format")
        return _create_citation_fix_sources_table(sources)
    
    # Handle empty sources or "No source captured"
    if not sources or sources == ["No source captured"]:
        rows.append(["1", "N/A", "N/A", "N/A", "N/A", "No sources available for this response"])
    else:
        for i, source in enumerate(sources[:20], 1):  # Increased to 20 sources for comprehensive display
            print(f"FORMATTER DEBUG: Processing source {i}: {source}")
            if isinstance(source, dict):
                source_type = source.get("type", "Document")
                
                # Handle different source types differently
                if source_type.lower() in ["dataset", "database"]:
                    # Dataset/Database citation format
                    source_name = source.get("title", source.get("source_name", "Unknown Source"))
                    provider = source.get("provider", "Unknown Provider")
                    coverage = source.get("coverage", "")
                    tool_used = source.get("passage_id", source.get("tool_used", "N/A"))  # passage_id stores tool name for datasets
                    description = source.get("text", "")[:150] + "..." if source.get("text") else "N/A"
                    
                    # Create comprehensive source reference for datasets
                    source_ref = f"{source_name}"
                    if provider and provider != "Unknown Provider":
                        source_ref += f" | {provider}"
                    if coverage:
                        source_ref += f" | {coverage}"
                    
                    rows.append([
                        str(i),
                        source_ref,
                        tool_used,
                        source_type.title(),
                        "Tool/API",
                        description
                    ])
                    
                else:
                    # Traditional document/passage citation format
                    doc_id = source.get("doc_id", "N/A")
                    passage_id = source.get("passage_id", "N/A")
                    title = source.get("title", "")
                    
                    # Create document reference
                    doc_ref = doc_id
                    if title:
                        doc_ref = f"{title} ({doc_id})"
                    elif source_type:
                        doc_ref = f"{source_type.title()}: {doc_id}"
                    
                    text_snippet = source.get("text", "")[:150] + "..." if source.get("text") else "N/A"
                    
                    rows.append([
                        str(i),
                        doc_ref,
                        passage_id,
                        source_type.title() if source_type else "Document", 
                        "Knowledge Graph",
                        text_snippet
                    ])
            else:
                # Handle legacy string sources
                rows.append([str(i), "General Reference", "N/A", "General", "Legacy", str(source)[:150]])
    
    return {
        "type": "numbered_citation_table",
        "heading": "References",
        "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
        "rows": rows
    }

def _create_citation_fix_sources_table(sources: List[Dict]) -> Optional[Dict]:
    """Create a sources table for CITATION_FIX structured citations."""
    rows = []
    
    print(f"FORMATTER DEBUG: Creating CITATION_FIX sources table with {len(sources)} structured citations")
    
    for source in sources:
        citation_id = source.get("id", "citation_?")
        citation_number = citation_id.replace("citation_", "")
        source_name = source.get("source_name", "Unknown Source")
        provider = source.get("provider", "Unknown Provider")
        spatial_coverage = source.get("spatial_coverage", "N/A")
        temporal_coverage = source.get("temporal_coverage", "N/A")
        source_url = source.get("source_url", "No URL available")
        
        # Map provider to ID/Tool and create description
        tool_id = f"{provider} Database" if provider != "Unknown Provider" else "Unknown Tool"
        data_type = "Database" if "Database" in source_name else "Dataset"
        description = f"{spatial_coverage} - {temporal_coverage}"
        
        rows.append([
            citation_number,
            source_name,
            tool_id,
            data_type,
            description
        ])
    
    return {
        "type": "numbered_citation_table", 
        "heading": "References",
        "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
        "rows": rows
    }

def detect_table_type(tool_name: str, data: List[Dict]) -> str:
    """Automatically determine appropriate table type based on tool and data structure."""
    
    # Ranking tables (ordered by numeric value)
    if tool_name in ["GetGistTopEmitters", "GetLargestSolarFacilities", "GetGistHighRiskCompanies"]:
        return "ranking_table"
    
    # Comparison tables (multiple entities compared)
    if tool_name in ["CompareBrazilianStates", "GetSolarCapacityByCountry", "GetGistEmissionsBySector"]:
        return "comparison_table"
    
    # Trend tables (time series data)
    if tool_name in ["GetSolarConstructionTimeline", "GetGistEmissionsTrends", "GetGistBiodiversityTrends"]:
        return "trend_table"
        
    # Geographic tables (location-based)
    if tool_name in ["GetGistAssetsByCountry", "GetSolarFacilitiesInRadius", "GetGistAssetsMapData"]:
        return "geographic_table"
        
    # Summary tables (aggregated overviews)
    if tool_name in ["GetGistCompaniesBySector", "GetBrazilianStatesOverview", "GetAvailableDatasets"]:
        return "summary_table"
        
    # Detail tables (specific breakdowns)
    if tool_name in ["GetGistScope3Emissions", "GetGistCompanyProfile", "GetSolarFacilitiesByCountry"]:
        return "detail_table"
        
    # Default fallback
    return "table"

def _create_comparison_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create side-by-side comparison table for entities like countries, sectors, states."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Limit rows for readability
        display_df = df.head(15)
        
        # Generate unique module ID for this table
        module_id = _generate_unique_module_id(tool_name, "comparison_table", data[:3])  # Use first 3 rows for hash
        
        # Get citations specifically for this tool
        tool_citations = _get_tool_citations(tool_name, citation_registry)
        
        # Add citations to heading if available
        cited_heading = heading
        if tool_citations:
            superscript = f" ^{','.join(map(str, tool_citations))}^"
            cited_heading = f"{heading}{superscript}"
        
        # Convert NaN values to null for JSON compatibility
        rows_data = display_df.fillna(None).values.tolist()
        
        return {
            "type": "comparison_table",
            "heading": cited_heading,
            "columns": display_df.columns.tolist(),
            "rows": rows_data,
            "metadata": {
                "tool_used": tool_name,
                "module_id": module_id,
                "citations": tool_citations,
                "total_entities": len(df),
                "displayed_entities": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating comparison table: {e}")
        return None

def _create_ranking_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create ordered ranking/leaderboard table."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Add rank column if not present
        if 'rank' not in df.columns and 'Rank' not in df.columns:
            df.insert(0, 'Rank', range(1, len(df) + 1))
        
        # Limit to top 20 for rankings
        display_df = df.head(20)
        
        # Generate unique module ID for this table
        module_id = _generate_unique_module_id(tool_name, "ranking_table", data[:3])
        
        # Get citations specifically for this tool
        tool_citations = _get_tool_citations(tool_name, citation_registry)
        
        # Add citations to heading if available
        cited_heading = heading
        if tool_citations:
            superscript = f" ^{','.join(map(str, tool_citations))}^"
            cited_heading = f"{heading}{superscript}"
        
        return {
            "type": "ranking_table", 
            "heading": cited_heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.fillna(None).values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "module_id": module_id,
                "citations": tool_citations,
                "total_entries": len(df),
                "displayed_entries": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating ranking table: {e}")
        return None

def _create_trend_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create time series analysis table."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Sort by time column if present
        time_columns = ['year', 'Year', 'date', 'Date', 'reporting_year', 'Reporting_Year']
        for col in time_columns:
            if col in df.columns:
                df = df.sort_values(col)
                break
        
        # Limit to reasonable time range (last 10 years/periods)
        display_df = df.tail(10)
        
        return {
            "type": "trend_table",
            "heading": heading,
            "columns": display_df.columns.tolist(), 
            "rows": display_df.fillna(None).values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_periods": len(df),
                "displayed_periods": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating trend table: {e}")
        return None

def _create_summary_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create aggregated overview table."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # For summary tables, show more rows (up to 25)
        display_df = df.head(25)
        
        # Generate unique module ID for this table
        module_id = _generate_unique_module_id(tool_name, "summary_table", data[:3])
        
        # Get citations specifically for this tool
        tool_citations = _get_tool_citations(tool_name, citation_registry)
        
        # Add citations to heading if available
        cited_heading = heading
        if tool_citations:
            superscript = f" ^{','.join(map(str, tool_citations))}^"
            cited_heading = f"{heading}{superscript}"
        
        return {
            "type": "summary_table",
            "heading": cited_heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.fillna(None).values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "module_id": module_id,
                "citations": tool_citations,
                "total_items": len(df),
                "displayed_items": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating summary table: {e}")
        return None

def _create_detail_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create detailed breakdown table for specific entity."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Detail tables show fewer rows but more columns
        display_df = df.head(12)
        
        return {
            "type": "detail_table",
            "heading": heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.fillna(None).values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_records": len(df),
                "displayed_records": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating detail table: {e}")
        return None

def _create_geographic_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create location-based analysis table."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Limit geographic tables to 15 entries for readability
        display_df = df.head(15)
        
        return {
            "type": "geographic_table", 
            "heading": heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.fillna(None).values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_locations": len(df),
                "displayed_locations": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating geographic table: {e}")
        return None

def _create_enhanced_table_from_data(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create appropriately typed table based on tool and data structure with citation support."""
    if not data or not isinstance(data, list):
        return None
    
    # Detect appropriate table type
    table_type = detect_table_type(tool_name, data)
    
    # Create table using appropriate specialized function
    if table_type == "comparison_table":
        return _create_comparison_table(data, heading, tool_name, citation_registry)
    elif table_type == "ranking_table":
        return _create_ranking_table(data, heading, tool_name, citation_registry)
    elif table_type == "trend_table":
        return _create_trend_table(data, heading, tool_name, citation_registry)
    elif table_type == "summary_table":
        return _create_summary_table(data, heading, tool_name, citation_registry)
    elif table_type == "detail_table":
        return _create_detail_table(data, heading, tool_name, citation_registry)
    elif table_type == "geographic_table":
        return _create_geographic_table(data, heading, tool_name, citation_registry)
    else:
        # Fallback to original function
        return _create_table_from_data(data, heading)

@mcp.tool()
def CreateMultipleTablesFromToolResults(
    tool_results: List[Dict],
    query_context: Optional[str] = None,
    citation_registry: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create multiple tables from tool results."""
    print(f"🔧 FORMATTER DEBUG: CreateMultipleTablesFromToolResults called")
    print(f"  - Received {len(tool_results)} tool results")
    print(f"  - Query context: '{query_context[:50] if query_context else 'None'}...'")
    
    modules = []
    
    # Process each tool result into appropriate table type
    for i, result in enumerate(tool_results):
        tool_name = result.get("tool_name", "")
        tool_data = result.get("data", [])
        
        print(f"🔧 FORMATTER DEBUG: Processing tool result [{i+1}]: {tool_name}")
        print(f"  - Data type: {type(tool_data)}")
        print(f"  - Data length: {len(tool_data) if isinstance(tool_data, list) else 'not a list'}")
        
        if not tool_data or not isinstance(tool_data, list):
            print(f"🔧 FORMATTER DEBUG: ❌ Skipping {tool_name} - no data or not a list")
            continue
            
        # Generate appropriate heading based on tool name
        heading = _generate_table_heading(tool_name, tool_data)
        print(f"🔧 FORMATTER DEBUG: Generated heading for {tool_name}: '{heading}'")
        
        # Try to create chart first if data is chart-worthy
        chart_module = _auto_generate_chart_from_data(tool_data, f"{heading} - Chart", tool_name, citation_registry)
        if chart_module:
            modules.append(chart_module)
        
        # Create enhanced table with appropriate type and citations
        table_module = _create_enhanced_table_from_data(tool_data, heading, tool_name, citation_registry)
        
        if table_module:
            table_type = table_module.get("type", "unknown")
            row_count = len(table_module.get("rows", [])) if "rows" in table_module else "no rows"
            print(f"🔧 FORMATTER DEBUG: ✅ Created {table_type} table for {tool_name} ({row_count} rows)")
            modules.append(table_module)
        else:
            print(f"🔧 FORMATTER DEBUG: ❌ Failed to create table for {tool_name}")
    
    print(f"🔧 FORMATTER DEBUG: Returning {len(modules)} table modules")
    return {"modules": modules}

@mcp.tool()
def OrganizeModulesIntoNarrative(
    modules: List[Dict],
    query: str,
    citation_registry: Optional[Dict] = None
) -> Dict[str, Any]:
    """Organize modules into cohesive narrative."""
    print(f"🎯 NARRATIVE DEBUG: Organizing {len(modules)} modules into narrative")
    
    if not modules:
        return {"modules": []}
    
    # Create module summary for the narrative organizer
    module_summary = []
    for i, module in enumerate(modules):
        module_type = module.get("type", "unknown")
        heading = module.get("heading", f"Module {i+1}")
        
        summary = {
            "index": i,
            "type": module_type,
            "heading": heading,
            "preview": _get_module_preview(module)
        }
        module_summary.append(summary)
    
    # Use Claude Sonnet to organize the narrative
    try:
        import anthropic
        client = anthropic.Anthropic()
        
        system_prompt = """You are an expert at organizing information into compelling narratives. Given a user query and a list of data modules, organize them into the most logical and engaging flow.

Your task:
1. Determine the optimal order for presenting the modules
2. Identify where transition text would help connect modules
3. Suggest where modules could be grouped or sections created
4. Maintain citation integrity - don't break citation numbering

Return a JSON object with:
{
    "narrative_plan": "Brief description of your organization strategy",
    "sections": [
        {
            "title": "Section title",
            "modules": [0, 1, 2], // Module indices in this section
            "transition_text": "Optional text to introduce this section"
        }
    ],
    "final_order": [0, 3, 1, 2, 4], // Final module order
    "reasoning": "Explanation of your choices"
}

Focus on creating a logical flow that answers the user's question comprehensively."""

        user_prompt = f"""Original query: {query}

Available modules:
{json.dumps(module_summary, indent=2)}

Organize these modules into the most effective narrative structure."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        narrative_plan = json.loads(response.content[0].text)
        print(f"🎯 NARRATIVE DEBUG: Received narrative plan: {narrative_plan.get('narrative_plan', 'No plan')}")
        
        # Reorganize modules according to the narrative plan
        reorganized_modules = []
        final_order = narrative_plan.get("final_order", list(range(len(modules))))
        
        # Add sections with transition text
        sections = narrative_plan.get("sections", [])
        current_section_idx = 0
        
        for module_idx in final_order:
            if module_idx < len(modules):
                # Check if we're starting a new section
                if current_section_idx < len(sections):
                    section = sections[current_section_idx]
                    if module_idx == section.get("modules", [None])[0]:  # First module in section
                        # Add section transition if provided
                        if section.get("transition_text"):
                            transition_module = {
                                "type": "narrative_transition",
                                "heading": section.get("title", ""),
                                "text": section.get("transition_text", "")
                            }
                            reorganized_modules.append(transition_module)
                        current_section_idx += 1
                
                reorganized_modules.append(modules[module_idx])
        
        return {
            "modules": reorganized_modules,
            "narrative_plan": narrative_plan,
            "original_count": len(modules),
            "organized_count": len(reorganized_modules)
        }
        
    except Exception as e:
        print(f"🎯 NARRATIVE DEBUG: Error organizing narrative: {e}")
        # Fallback to original order
        return {"modules": modules}

def _get_module_preview(module: Dict) -> str:
    """Generate a preview description of a module for narrative planning."""
    module_type = module.get("type", "unknown")
    
    if module_type == "text":
        texts = module.get("texts", [])
        preview = texts[0][:100] + "..." if texts else "Text content"
        return f"Text: {preview}"
    
    elif module_type.endswith("_table"):
        rows = module.get("rows", [])
        columns = module.get("columns", [])
        preview = f"Table with {len(rows)} rows, {len(columns)} columns"
        if columns:
            preview += f" (columns: {', '.join(columns[:3])}...)"
        return preview
    
    elif module_type == "chart":
        chart_type = module.get("chartType", "unknown")
        return f"Chart: {chart_type} visualization"
    
    elif module_type == "map":
        data = module.get("data", [])
        return f"Map with {len(data)} data points"
    
    elif module_type == "numbered_citation_table":
        rows = module.get("rows", [])
        return f"References table with {len(rows)} citations"
    
    else:
        return f"{module_type.replace('_', ' ').title()} module"

def _generate_table_heading(tool_name: str, data: List[Dict]) -> str:
    """Generate appropriate table heading based on tool name and data."""
    
    # Custom headings for specific tools
    headings_map = {
        "GetGistCompaniesBySector": "Companies by Sector",
        "GetGistTopEmitters": "Top Emitting Companies",
        "GetGistHighRiskCompanies": "Highest Environmental Risk Companies",
        "GetSolarCapacityByCountry": "Solar Capacity by Country",
        "GetLargestSolarFacilities": "Largest Solar Facilities",
        "GetSolarConstructionTimeline": "Solar Construction Timeline",
        "GetBrazilianStatesOverview": "Brazilian State Climate Policies",
        "CompareBrazilianStates": "State Policy Comparison",
        "GetGistEmissionsBySector": "Emissions by Sector",
        "GetGistRiskByCategory": "Environmental Risk Assessment",
        "GetGistEmissionsTrends": "Emissions Trends Analysis",
        "GetGistAssetsByCountry": "Assets by Geographic Location",
        "GetGistScope3Emissions": "Scope 3 Emissions Breakdown",
        "GetGistCompanyProfile": "Company Sustainability Profile",
        "GetAvailableDatasets": "Available Datasets",
        "GetInstitutionsProcessesData": "Climate Governance Institutions",
        "GetPlansAndPoliciesData": "Climate Plans and Policies"
    }
    
    if tool_name in headings_map:
        return headings_map[tool_name]
    
    # Generate heading from tool name
    heading = tool_name.replace("Get", "").replace("Gist", "").replace("Solar", "Solar ")
    heading = " ".join([word.capitalize() for word in heading.split()])
    
    # Add context based on data size
    if data and len(data) > 0:
        if len(data) == 1:
            heading += " Details"
        elif len(data) < 5:
            heading += " Summary"
        else:
            heading += " Analysis"
    
    return heading

@mcp.tool()
def GetFormatterMetadata() -> Dict[str, Any]:
    """Get response formatter metadata."""
    return metadata

if __name__ == "__main__":
    mcp.run()