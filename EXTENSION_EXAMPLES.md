# Extension Examples

This document provides practical examples of how to extend the Climate Policy API system with new data sources, servers, and capabilities.

## Example 1: Adding Climate Finance Dataset

Let's add a comprehensive climate finance dataset with project funding information.

### Step 1: Prepare the Data

```python
# data/climate_finance/process_data.py
import pandas as pd
import json

# Sample climate finance data structure
finance_data = [
    {
        "project_id": "CF001",
        "project_name": "Solar Power Initiative Brazil",
        "country": "Brazil",
        "sector": "renewable_energy",
        "funding_amount_usd": 50000000,
        "funding_source": "Green Climate Fund",
        "project_status": "ongoing",
        "start_date": "2023-01-15",
        "end_date": "2026-12-31",
        "description": "Large-scale solar installation project in Northeast Brazil",
        "expected_co2_reduction_tons": 125000,
        "beneficiaries": 75000
    },
    {
        "project_id": "CF002", 
        "project_name": "Coastal Resilience Vietnam",
        "country": "Vietnam",
        "sector": "adaptation",
        "funding_amount_usd": 25000000,
        "funding_source": "World Bank",
        "project_status": "completed",
        "start_date": "2020-03-01",
        "end_date": "2023-11-30",
        "description": "Mangrove restoration and coastal protection infrastructure",
        "expected_co2_reduction_tons": 0,
        "beneficiaries": 150000
    }
    # ... more projects
]

# Save as structured dataset
df = pd.DataFrame(finance_data)
df.to_csv('data/climate_finance/projects.csv', index=False)

# Create summary statistics
summary = {
    "total_projects": len(df),
    "total_funding_usd": df['funding_amount_usd'].sum(),
    "countries": df['country'].unique().tolist(),
    "sectors": df['sector'].unique().tolist(),
    "funding_sources": df['funding_source'].unique().tolist()
}

with open('data/climate_finance/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

### Step 2: Create the MCP Server

```python
# mcp/climate_finance_server.py
import asyncio
import pandas as pd
import json
import os
from mcp import Application, Tool
from mcp.types import TextContent
from typing import Optional

app = Application()

# Load data on startup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
finance_data = pd.read_csv(os.path.join(project_root, "data/climate_finance/projects.csv"))

with open(os.path.join(project_root, "data/climate_finance/summary.json"), 'r') as f:
    finance_summary = json.load(f)

@app.tool()
def get_climate_finance_projects(
    country: Optional[str] = None,
    sector: Optional[str] = None,
    funding_source: Optional[str] = None,
    min_amount: Optional[float] = None
) -> list[TextContent]:
    """
    Get climate finance projects with optional filtering.
    
    Args:
        country: Filter by country name
        sector: Filter by sector (renewable_energy, adaptation, mitigation, etc.)
        funding_source: Filter by funding source
        min_amount: Minimum funding amount in USD
    """
    filtered_data = finance_data.copy()
    
    if country:
        filtered_data = filtered_data[filtered_data['country'].str.contains(country, case=False, na=False)]
    
    if sector:
        filtered_data = filtered_data[filtered_data['sector'].str.contains(sector, case=False, na=False)]
    
    if funding_source:
        filtered_data = filtered_data[filtered_data['funding_source'].str.contains(funding_source, case=False, na=False)]
    
    if min_amount:
        filtered_data = filtered_data[filtered_data['funding_amount_usd'] >= min_amount]
    
    result = filtered_data.to_dict('records')
    
    return [TextContent(
        type="text",
        text=json.dumps(result, default=str)
    )]

@app.tool()
def get_climate_finance_summary(country: Optional[str] = None) -> list[TextContent]:
    """
    Get summary statistics for climate finance data.
    
    Args:
        country: Optional country filter for summary stats
    """
    if country:
        country_data = finance_data[finance_data['country'].str.contains(country, case=False, na=False)]
        summary = {
            "country": country,
            "total_projects": len(country_data),
            "total_funding_usd": float(country_data['funding_amount_usd'].sum()),
            "avg_project_size_usd": float(country_data['funding_amount_usd'].mean()),
            "sectors": country_data['sector'].unique().tolist(),
            "funding_sources": country_data['funding_source'].unique().tolist(),
            "total_beneficiaries": int(country_data['beneficiaries'].sum()),
            "total_co2_reduction_tons": float(country_data['expected_co2_reduction_tons'].sum())
        }
    else:
        summary = finance_summary.copy()
        summary.update({
            "avg_project_size_usd": float(finance_data['funding_amount_usd'].mean()),
            "total_beneficiaries": int(finance_data['beneficiaries'].sum()),
            "total_co2_reduction_tons": float(finance_data['expected_co2_reduction_tons'].sum())
        })
    
    return [TextContent(
        type="text",
        text=json.dumps(summary, default=str)
    )]

@app.tool()
def get_finance_visualization_data(chart_type: str = "funding_by_country") -> list[TextContent]:
    """
    Generate data for climate finance visualizations.
    
    Args:
        chart_type: Type of chart (funding_by_country, funding_by_sector, timeline, etc.)
    """
    if chart_type == "funding_by_country":
        country_funding = finance_data.groupby('country')['funding_amount_usd'].sum().sort_values(ascending=False)
        chart_data = {
            "type": "bar",
            "title": "Climate Finance by Country",
            "data": {
                "labels": country_funding.index.tolist(),
                "datasets": [{
                    "label": "Funding (USD)",
                    "data": country_funding.values.tolist(),
                    "backgroundColor": ["#2E8B57", "#4682B4", "#CD853F", "#8B4513", "#556B2F"]
                }]
            }
        }
    
    elif chart_type == "funding_by_sector":
        sector_funding = finance_data.groupby('sector')['funding_amount_usd'].sum()
        chart_data = {
            "type": "pie",
            "title": "Climate Finance by Sector", 
            "data": {
                "labels": sector_funding.index.tolist(),
                "datasets": [{
                    "label": "Funding (USD)",
                    "data": sector_funding.values.tolist(),
                    "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
                }]
            }
        }
    
    elif chart_type == "project_timeline":
        # Convert dates and create timeline data
        finance_data['start_year'] = pd.to_datetime(finance_data['start_date']).dt.year
        timeline_data = finance_data.groupby('start_year').agg({
            'project_id': 'count',
            'funding_amount_usd': 'sum'
        }).reset_index()
        
        chart_data = {
            "type": "line",
            "title": "Climate Finance Timeline",
            "data": {
                "labels": timeline_data['start_year'].tolist(),
                "datasets": [
                    {
                        "label": "Number of Projects",
                        "data": timeline_data['project_id'].tolist(),
                        "borderColor": "#FF6384",
                        "yAxisID": "y"
                    },
                    {
                        "label": "Total Funding (USD)",
                        "data": timeline_data['funding_amount_usd'].tolist(),
                        "borderColor": "#36A2EB", 
                        "yAxisID": "y1"
                    }
                ]
            },
            "options": {
                "scales": {
                    "y": {"type": "linear", "display": True, "position": "left"},
                    "y1": {"type": "linear", "display": True, "position": "right"}
                }
            }
        }
    
    else:
        chart_data = {"error": f"Unknown chart type: {chart_type}"}
    
    return [TextContent(
        type="text", 
        text=json.dumps(chart_data)
    )]

if __name__ == "__main__":
    app.run()
```

### Step 3: Update Knowledge Graph Connections

```python
# Add to mcp/cpr_kg_server.py in add_dataset_connections()

def add_dataset_connections(G):
    # ... existing code ...
    
    # Add climate finance dataset
    finance_dataset_node_id = "CLIMATE_FINANCE_PROJECTS"
    finance_dataset_label = "Climate Finance Projects Dataset"
    
    if not G.has_node(finance_dataset_node_id):
        G.add_node(
            finance_dataset_node_id,
            kind="Dataset",
            label=finance_dataset_label,
            description="Comprehensive database of climate finance projects including funding amounts, sectors, and impact metrics",
            server_name="finance",
            total_projects=len(finance_data),
            countries=["Brazil", "Vietnam", "India", "South Africa"],
            total_funding_usd=finance_summary["total_funding_usd"]
        )

    # Link to relevant concepts
    finance_concepts = [
        "green finance", "climate finance", "adaptation finance", 
        "mitigation finance", "renewable energy finance", "climate investment"
    ]
    
    for concept_label in finance_concepts:
        concept_id = _concept_id(concept_label)
        if concept_id and G.has_node(concept_id):
            if not G.has_edge(concept_id, finance_dataset_node_id):
                G.add_edge(concept_id, finance_dataset_node_id, type="HAS_DATASET_ABOUT")
            if not G.has_edge(finance_dataset_node_id, concept_id):
                G.add_edge(finance_dataset_node_id, concept_id, type="DATASET_ON_TOPIC")
    
    # Link to country concepts
    country_concepts = ["Brazil", "Vietnam", "India", "South Africa"]
    for country in country_concepts:
        country_concept_id = _concept_id(country)
        if country_concept_id and G.has_node(country_concept_id):
            if not G.has_edge(country_concept_id, finance_dataset_node_id):
                G.add_edge(country_concept_id, finance_dataset_node_id, type="HAS_DATASET_ABOUT")
```

### Step 4: Register in Orchestration Layer

```python
# Update mcp/mcp_chat.py in run_query()

async def run_query(q: str):
    async with MultiServerClient() as client:
        # Connect to all servers
        await client.connect_to_server("kg", os.path.join(mcp_dir, "cpr_kg_server.py"))
        await client.connect_to_server("solar", os.path.join(mcp_dir, "solar_facilities_server.py"))
        await client.connect_to_server("formatter", os.path.join(mcp_dir, "response_formatter_server.py"))
        
        # Add new climate finance server
        await client.connect_to_server("finance", os.path.join(mcp_dir, "climate_finance_server.py"))
        
        # ... rest of the function

# Update system prompt to include new tools
system_prompt += """

Climate Finance Server Tools:
- `get_climate_finance_projects`: Get climate finance projects with filtering options
- `get_climate_finance_summary`: Get summary statistics for climate finance
- `get_finance_visualization_data`: Generate charts for climate finance data

When users ask about climate finance, green finance, adaptation funding, or project financing:
1. Use get_climate_finance_projects to get relevant project data
2. Use get_climate_finance_summary for overview statistics  
3. Use get_finance_visualization_data to create charts
4. Combine with knowledge graph passages for comprehensive analysis
"""
```

### Step 5: Test the New Functionality

```bash
# Test the new climate finance capabilities
curl -X POST http://localhost:8099/query \
  -H "Content-Type: application/json" \
  -d '{"query": "climate finance projects in Brazil"}'

curl -X POST http://localhost:8099/thorough-response \
  -H "Content-Type: application/json" \
  -d '{"query": "show me green finance data and charts"}'
```

Expected response will now include:
- Text analysis of climate finance concepts
- Table with Brazilian climate finance projects  
- Charts showing funding distribution
- Summary statistics

## Example 2: Adding Real-Time Weather Data

### Step 1: Create Weather API Integration

```python
# mcp/weather_data_server.py
import asyncio
import aiohttp
import json
import os
from datetime import datetime, timedelta
from mcp import Application
from mcp.types import TextContent
from typing import Optional

app = Application()

# Configuration
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

@app.tool()
async def get_current_weather(city: str, country: Optional[str] = None) -> list[TextContent]:
    """
    Get current weather conditions for a city.
    
    Args:
        city: City name
        country: Optional country code for disambiguation
    """
    location = f"{city},{country}" if country else city
    
    async with aiohttp.ClientSession() as session:
        url = f"{WEATHER_BASE_URL}/weather"
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                weather_info = {
                    "city": data["name"],
                    "country": data["sys"]["country"],
                    "temperature_c": data["main"]["temp"],
                    "feels_like_c": data["main"]["feels_like"],
                    "humidity_percent": data["main"]["humidity"],
                    "pressure_hpa": data["main"]["pressure"],
                    "description": data["weather"][0]["description"],
                    "wind_speed_ms": data["wind"]["speed"],
                    "visibility_m": data.get("visibility", None),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                return [TextContent(type="text", text=json.dumps(weather_info))]
            else:
                error = {"error": f"Weather API error: {response.status}"}
                return [TextContent(type="text", text=json.dumps(error))]

@app.tool()
async def get_weather_forecast(city: str, country: Optional[str] = None, days: int = 5) -> list[TextContent]:
    """
    Get weather forecast for a city.
    
    Args:
        city: City name
        country: Optional country code
        days: Number of days to forecast (1-5)
    """
    location = f"{city},{country}" if country else city
    
    async with aiohttp.ClientSession() as session:
        url = f"{WEATHER_BASE_URL}/forecast"
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "cnt": min(days * 8, 40)  # API returns 3-hour intervals
        }
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                forecast = []
                for item in data["list"]:
                    forecast.append({
                        "datetime": item["dt_txt"],
                        "temperature_c": item["main"]["temp"],
                        "description": item["weather"][0]["description"],
                        "humidity_percent": item["main"]["humidity"],
                        "wind_speed_ms": item["wind"]["speed"]
                    })
                
                result = {
                    "city": data["city"]["name"],
                    "country": data["city"]["country"],
                    "forecast": forecast
                }
                
                return [TextContent(type="text", text=json.dumps(result))]
            else:
                error = {"error": f"Forecast API error: {response.status}"}
                return [TextContent(type="text", text=json.dumps(error))]

@app.tool()
async def get_extreme_weather_alerts(lat: float, lon: float) -> list[TextContent]:
    """
    Get weather alerts for a geographic location.
    
    Args:
        lat: Latitude
        lon: Longitude  
    """
    async with aiohttp.ClientSession() as session:
        url = f"{WEATHER_BASE_URL}/onecall"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "exclude": "minutely,daily"
        }
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                alerts = data.get("alerts", [])
                processed_alerts = []
                
                for alert in alerts:
                    processed_alerts.append({
                        "event": alert["event"],
                        "severity": alert.get("severity", "unknown"),
                        "description": alert["description"],
                        "start": datetime.fromtimestamp(alert["start"]).isoformat(),
                        "end": datetime.fromtimestamp(alert["end"]).isoformat(),
                        "sender": alert["sender_name"]
                    })
                
                result = {
                    "location": {"lat": lat, "lon": lon},
                    "alerts": processed_alerts,
                    "alert_count": len(processed_alerts)
                }
                
                return [TextContent(type="text", text=json.dumps(result))]
            else:
                error = {"error": f"Alerts API error: {response.status}"}
                return [TextContent(type="text", text=json.dumps(error))]

if __name__ == "__main__":
    app.run()
```

### Step 2: Update Knowledge Graph

```python
# Add weather dataset connection in cpr_kg_server.py

def add_dataset_connections(G):
    # ... existing code ...
    
    # Add real-time weather dataset
    weather_dataset_node_id = "REAL_TIME_WEATHER_DATA"
    weather_dataset_label = "Real-Time Weather Data"
    
    if not G.has_node(weather_dataset_node_id):
        G.add_node(
            weather_dataset_node_id,
            kind="Dataset",
            label=weather_dataset_label,
            description="Real-time weather conditions, forecasts, and extreme weather alerts from OpenWeatherMap API",
            server_name="weather",
            data_source="OpenWeatherMap API",
            update_frequency="real-time"
        )

    # Link to weather and extreme weather concepts
    weather_concepts = [
        "extreme weather", "weather patterns", "temperature", "precipitation", 
        "storms", "climate conditions", "weather monitoring"
    ]
    
    for concept_label in weather_concepts:
        concept_id = _concept_id(concept_label)
        if concept_id and G.has_node(concept_id):
            if not G.has_edge(concept_id, weather_dataset_node_id):
                G.add_edge(concept_id, weather_dataset_node_id, type="HAS_DATASET_ABOUT")
```

### Step 3: Integrate with Orchestration

```python
# Update system prompt in mcp_chat.py

system_prompt += """

Real-Time Weather Server Tools:
- `get_current_weather`: Get current weather conditions for any city
- `get_weather_forecast`: Get weather forecast up to 5 days  
- `get_extreme_weather_alerts`: Get weather alerts for geographic coordinates

When users ask about current weather, weather conditions, or extreme weather events:
1. Use weather tools to get real-time data
2. Combine with historical extreme weather data from knowledge graph
3. Provide current conditions + historical context + trends
"""

# Register server in run_query()
await client.connect_to_server("weather", os.path.join(mcp_dir, "weather_data_server.py"))
```

## Example 3: Adding Economic Impact Analysis

### Step 1: Create Economic Analysis Server

```python
# mcp/economic_analysis_server.py
import pandas as pd
import numpy as np
import json
from mcp import Application
from mcp.types import TextContent
from typing import Optional, List

app = Application()

# Sample economic impact calculation functions
def calculate_carbon_pricing_impact(emissions_tons_co2: float, carbon_price_usd: float) -> dict:
    """Calculate economic impact of carbon pricing."""
    total_cost = emissions_tons_co2 * carbon_price_usd
    return {
        "emissions_tons_co2": emissions_tons_co2,
        "carbon_price_usd_per_ton": carbon_price_usd,
        "total_carbon_cost_usd": total_cost,
        "annual_cost_per_capita": total_cost / 1000000,  # Assuming 1M population
        "impact_category": "high" if total_cost > 1000000 else "moderate" if total_cost > 100000 else "low"
    }

def calculate_renewable_energy_savings(capacity_mw: float, country: str) -> dict:
    """Calculate economic savings from renewable energy deployment."""
    
    # Country-specific factors (simplified)
    country_factors = {
        "Brazil": {"grid_emission_factor": 0.074, "electricity_price": 0.17},
        "India": {"grid_emission_factor": 0.708, "electricity_price": 0.08},
        "South Africa": {"grid_emission_factor": 0.928, "electricity_price": 0.07},
        "Vietnam": {"grid_emission_factor": 0.514, "electricity_price": 0.08}
    }
    
    factors = country_factors.get(country, {"grid_emission_factor": 0.5, "electricity_price": 0.10})
    
    # Calculations
    annual_generation_mwh = capacity_mw * 8760 * 0.25  # 25% capacity factor
    emissions_avoided_tons = annual_generation_mwh * factors["grid_emission_factor"]
    cost_savings_usd = annual_generation_mwh * factors["electricity_price"] * 1000
    
    return {
        "capacity_mw": capacity_mw,
        "country": country,
        "annual_generation_mwh": annual_generation_mwh,
        "emissions_avoided_tons_co2": emissions_avoided_tons,
        "annual_cost_savings_usd": cost_savings_usd,
        "20_year_savings_usd": cost_savings_usd * 20,
        "cost_per_ton_co2_avoided": cost_savings_usd / emissions_avoided_tons if emissions_avoided_tons > 0 else 0
    }

@app.tool()
def analyze_carbon_pricing_impact(
    country: str,
    emissions_tons_co2: float,
    carbon_price_scenarios: Optional[List[float]] = None
) -> list[TextContent]:
    """
    Analyze economic impact of carbon pricing scenarios.
    
    Args:
        country: Country name
        emissions_tons_co2: Annual CO2 emissions in tons
        carbon_price_scenarios: List of carbon price scenarios in USD/ton
    """
    if carbon_price_scenarios is None:
        carbon_price_scenarios = [10, 25, 50, 100, 200]  # USD per ton CO2
    
    results = []
    for price in carbon_price_scenarios:
        impact = calculate_carbon_pricing_impact(emissions_tons_co2, price)
        impact["country"] = country
        results.append(impact)
    
    analysis = {
        "country": country,
        "scenarios": results,
        "summary": {
            "low_price_impact": results[0]["total_carbon_cost_usd"],
            "high_price_impact": results[-1]["total_carbon_cost_usd"],
            "price_range_usd": f"${results[0]['total_carbon_cost_usd']:,.0f} - ${results[-1]['total_carbon_cost_usd']:,.0f}"
        }
    }
    
    return [TextContent(type="text", text=json.dumps(analysis))]

@app.tool()
def analyze_renewable_energy_economics(
    projects_data: str,
    country: Optional[str] = None
) -> list[TextContent]:
    """
    Analyze economic benefits of renewable energy projects.
    
    Args:
        projects_data: JSON string of project data with capacity information
        country: Optional country filter
    """
    try:
        projects = json.loads(projects_data)
        if not isinstance(projects, list):
            projects = [projects]
        
        results = []
        total_capacity = 0
        total_savings = 0
        total_emissions_avoided = 0
        
        for project in projects:
            if country and project.get("country", "").lower() != country.lower():
                continue
                
            capacity = project.get("capacity_mw", 0)
            project_country = project.get("country", "Unknown")
            
            if capacity > 0:
                economics = calculate_renewable_energy_savings(capacity, project_country)
                economics["project_id"] = project.get("project_id", "Unknown")
                economics["project_name"] = project.get("project_name", "Unknown")
                
                results.append(economics)
                total_capacity += capacity
                total_savings += economics["annual_cost_savings_usd"]
                total_emissions_avoided += economics["emissions_avoided_tons_co2"]
        
        analysis = {
            "projects_analyzed": len(results),
            "total_capacity_mw": total_capacity,
            "total_annual_savings_usd": total_savings,
            "total_emissions_avoided_tons_co2": total_emissions_avoided,
            "average_cost_per_mw": total_savings / total_capacity if total_capacity > 0 else 0,
            "projects": results
        }
        
        return [TextContent(type="text", text=json.dumps(analysis))]
        
    except json.JSONDecodeError:
        error = {"error": "Invalid JSON format in projects_data"}
        return [TextContent(type="text", text=json.dumps(error))]

@app.tool()
def create_economic_visualization(
    analysis_data: str,
    chart_type: str = "cost_comparison"
) -> list[TextContent]:
    """
    Create visualizations for economic analysis.
    
    Args:
        analysis_data: JSON string of analysis results
        chart_type: Type of chart to create
    """
    try:
        data = json.loads(analysis_data)
        
        if chart_type == "cost_comparison" and "scenarios" in data:
            # Carbon pricing scenarios chart
            scenarios = data["scenarios"]
            chart_data = {
                "type": "bar",
                "title": f"Carbon Pricing Impact - {data.get('country', 'Unknown')}",
                "data": {
                    "labels": [f"${s['carbon_price_usd_per_ton']}/ton" for s in scenarios],
                    "datasets": [{
                        "label": "Total Carbon Cost (USD)",
                        "data": [s["total_carbon_cost_usd"] for s in scenarios],
                        "backgroundColor": ["#FF6B6B", "#FFA726", "#FFEE58", "#66BB6A", "#42A5F5"]
                    }]
                }
            }
        
        elif chart_type == "savings_breakdown" and "projects" in data:
            # Renewable energy savings chart
            projects = data["projects"][:10]  # Top 10 projects
            chart_data = {
                "type": "horizontalBar",
                "title": "Annual Cost Savings by Project",
                "data": {
                    "labels": [p.get("project_name", p.get("project_id", "Unknown"))[:30] for p in projects],
                    "datasets": [{
                        "label": "Annual Savings (USD)",
                        "data": [p["annual_cost_savings_usd"] for p in projects],
                        "backgroundColor": "#4CAF50"
                    }]
                }
            }
        
        elif chart_type == "emissions_vs_cost":
            # Emissions avoided vs cost chart
            projects = data.get("projects", [])
            chart_data = {
                "type": "scatter",
                "title": "Cost Savings vs Emissions Avoided",
                "data": {
                    "datasets": [{
                        "label": "Projects",
                        "data": [
                            {
                                "x": p["emissions_avoided_tons_co2"],
                                "y": p["annual_cost_savings_usd"]
                            }
                            for p in projects
                        ],
                        "backgroundColor": "#36A2EB"
                    }]
                },
                "options": {
                    "scales": {
                        "x": {"title": {"display": True, "text": "Emissions Avoided (tons CO2)"}},
                        "y": {"title": {"display": True, "text": "Annual Savings (USD)"}}
                    }
                }
            }
        
        else:
            chart_data = {"error": f"Unsupported chart type: {chart_type}"}
        
        return [TextContent(type="text", text=json.dumps(chart_data))]
        
    except json.JSONDecodeError:
        error = {"error": "Invalid JSON format in analysis_data"}
        return [TextContent(type="text", text=json.dumps(error))]

if __name__ == "__main__":
    app.run()
```

### Step 2: Integration and Testing

```python
# Test the complete integration
curl -X POST http://localhost:8099/thorough-response \
  -H "Content-Type: application/json" \
  -d '{
    "query": "economic impact of carbon pricing in Brazil with renewable energy savings analysis"
  }'
```

This would trigger:
1. Knowledge graph search for carbon pricing and renewable energy concepts
2. Climate finance data retrieval for Brazilian projects
3. Solar facilities data for capacity information
4. Economic analysis calculations
5. Visualization generation
6. Comprehensive synthesis

## Best Practices for Extensions

### 1. Modular Design
- Keep each MCP server focused on one domain
- Use clear, descriptive tool names
- Include comprehensive docstrings

### 2. Error Handling
- Always return valid JSON from tools
- Include error objects for API failures
- Log errors for debugging

### 3. Data Validation
- Validate input parameters
- Handle missing or malformed data gracefully
- Provide meaningful error messages

### 4. Performance Optimization  
- Cache expensive computations
- Use async operations for I/O
- Limit data size in responses

### 5. Documentation
- Document all new concepts and relationships
- Provide usage examples
- Update API documentation

### 6. Testing
- Write unit tests for new tools
- Test integration with existing servers
- Validate response formats

These examples demonstrate how the system's modular architecture enables rapid extension with new data sources, analysis capabilities, and visualizations while maintaining consistency and reliability.