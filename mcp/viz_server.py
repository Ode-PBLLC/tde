#!/usr/bin/env python3
"""
Intelligent Data Visualization Server

FastMCP server that provides smart chart generation tools.
Analyzes data characteristics to select optimal chart types and
returns Chart.js v3+ compatible configurations.
"""

import json
import colorsys
from typing import Dict, List, Any, Optional
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("viz-server")

# Color schemes for different contexts
COLOR_SCHEMES = {
    'environmental': ['#4CAF50', '#8BC34A', '#CDDC39', '#FDD835'],  # Greens/yellows for positive environmental
    'risks': ['#F44336', '#FF5722', '#FF9800', '#FFC107'],         # Reds/oranges for risks/warnings  
    'neutral': ['#2196F3', '#03A9F4', '#00BCD4', '#009688'],       # Blues for neutral data
    'comparison': ['#3F51B5', '#9C27B0', '#E91E63', '#009688', '#FF5722', '#795548']  # Diverse palette
}

def analyze_data_characteristics(data: List[Dict]) -> Dict[str, Any]:
    """
    Analyze data to determine its characteristics for visualization selection.
    
    Args:
        data: List of data points
        
    Returns:
        Dictionary with data characteristics
    """
    if not data:
        return {'type': 'empty', 'series_count': 0}
    
    characteristics = {
        'type': 'unknown',
        'series_count': len(data),
        'has_time': False,
        'has_categories': False,
        'has_numeric': False,
        'numeric_fields': [],
        'categorical_fields': [],
        'time_fields': []
    }
    
    # Analyze first few data points to understand structure
    sample = data[:5] if len(data) > 5 else data
    
    for item in sample:
        if isinstance(item, dict):
            for key, value in item.items():
                key_lower = key.lower()
                
                # Check for time indicators
                if any(time_word in key_lower for time_word in ['year', 'date', 'time', 'month', 'day']):
                    characteristics['has_time'] = True
                    if key not in characteristics['time_fields']:
                        characteristics['time_fields'].append(key)
                
                # Check for numeric data
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    characteristics['has_numeric'] = True
                    if key not in characteristics['numeric_fields']:
                        characteristics['numeric_fields'].append(key)
                
                # Check for categorical data
                elif isinstance(value, str) and not any(time_word in key_lower for time_word in ['year', 'date', 'time']):
                    characteristics['has_categories'] = True
                    if key not in characteristics['categorical_fields']:
                        characteristics['categorical_fields'].append(key)
    
    # Determine data type
    if characteristics['has_time'] and characteristics['has_numeric']:
        characteristics['type'] = 'time_series'
    elif characteristics['has_categories'] and characteristics['has_numeric']:
        characteristics['type'] = 'categorical'
    elif characteristics['has_numeric'] and len(characteristics['numeric_fields']) > 1:
        characteristics['type'] = 'multi_metric'
    else:
        characteristics['type'] = 'simple'
    
    return characteristics

def select_optimal_chart_type(data: List[Dict], context: str, characteristics: Dict) -> str:
    """
    Select the best chart type based on data characteristics and context.
    
    Args:
        data: Raw data points
        context: Description of what the data represents
        characteristics: Data characteristics from analysis
        
    Returns:
        Chart.js chart type string
    """
    context_lower = context.lower()
    data_type = characteristics['type']
    series_count = characteristics['series_count']
    
    # Time series → line chart
    if data_type == 'time_series':
        return 'line'
    
    # Context-based decisions
    if any(word in context_lower for word in ['trend', 'over time', 'growth', 'decline', 'change']):
        return 'line'
    
    if any(word in context_lower for word in ['proportion', 'percentage', 'share', 'part of']):
        return 'doughnut' if series_count > 6 else 'pie'
    
    if any(word in context_lower for word in ['distribution', 'spread', 'variance']):
        return 'bar'
    
    # Data-based decisions
    if data_type == 'categorical':
        if series_count > 10:
            return 'bar'  # Better for many categories
        elif series_count <= 5:
            return 'pie'   # Good for few categories
        else:
            return 'bar'   # Default for moderate categories
    
    # Default fallback
    return 'bar'

def select_color_scheme(context: str, chart_type: str, data_count: int) -> List[str]:
    """
    Select appropriate colors based on context and chart type.
    
    Args:
        context: Description of data context
        chart_type: Type of chart being created
        data_count: Number of data points
        
    Returns:
        List of color strings
    """
    context_lower = context.lower()
    
    # Context-based color selection
    if any(word in context_lower for word in ['emission', 'positive', 'green', 'renewable', 'sustainable']):
        colors = COLOR_SCHEMES['environmental']
    elif any(word in context_lower for word in ['risk', 'danger', 'warning', 'problem', 'negative']):
        colors = COLOR_SCHEMES['risks']
    else:
        colors = COLOR_SCHEMES['neutral']
    
    # Extend colors if needed
    if data_count > len(colors):
        # Generate additional colors
        extended_colors = colors[:]
        for i in range(data_count - len(colors)):
            # Generate colors by varying hue
            hue = (i * 0.618033988749) % 1  # Golden ratio for nice distribution
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            extended_colors.append(hex_color)
        return extended_colors[:data_count]
    
    return colors[:data_count]

def extract_chart_data(data: List[Dict], chart_type: str) -> Dict[str, Any]:
    """
    Extract and format data for Chart.js configuration.
    
    Args:
        data: Raw data points
        chart_type: Selected chart type
        
    Returns:
        Chart.js data object
    """
    if not data:
        return {"labels": [], "datasets": []}
    
    # Simple case: data is already in good format
    if len(data) > 0 and isinstance(data[0], dict):
        # Try to find suitable label and value fields
        first_item = data[0]
        
        # Look for common label fields
        label_field = None
        for field in ['label', 'name', 'category', 'year', 'date', 'month']:
            if field in first_item:
                label_field = field
                break
        
        # Look for numeric value fields
        value_fields = []
        for field, value in first_item.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value_fields.append(field)
        
        if label_field and value_fields:
            labels = [str(item.get(label_field, '')) for item in data]
            
            datasets = []
            for i, field in enumerate(value_fields[:3]):  # Limit to 3 series for readability
                dataset_data = [item.get(field, 0) for item in data]
                datasets.append({
                    "label": field.replace('_', ' ').title(),
                    "data": dataset_data
                })
            
            return {
                "labels": labels,
                "datasets": datasets
            }
    
    # Fallback: try to extract any numeric data
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], (int, float)):
            # Simple numeric array
            return {
                "labels": [f"Item {i+1}" for i in range(len(data))],
                "datasets": [{
                    "label": "Value",
                    "data": data
                }]
            }
    
    # Final fallback
    return {
        "labels": ["No Data"],
        "datasets": [{
            "label": "No Data",
            "data": [0]
        }]
    }

def build_chart_options(chart_type: str, title: str) -> Dict[str, Any]:
    """
    Build Chart.js options object with sensible defaults.
    
    Args:
        chart_type: Type of chart
        title: Chart title
        
    Returns:
        Chart.js options object
    """
    base_options = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "title": {
                "display": bool(title),
                "text": title,
                "font": {"size": 16}
            },
            "legend": {
                "display": True,
                "position": "top"
            }
        }
    }
    
    # Chart-specific options
    if chart_type in ['line', 'bar']:
        base_options["scales"] = {
            "y": {
                "beginAtZero": True,
                "grid": {"display": True},
                "title": {"display": False}
            },
            "x": {
                "grid": {"display": False},
                "title": {"display": False}
            }
        }
    
    if chart_type == 'line':
        base_options["elements"] = {
            "line": {
                "tension": 0.1  # Slight curve
            },
            "point": {
                "radius": 4,
                "hoverRadius": 6
            }
        }
    
    return base_options

@mcp.tool()
def create_smart_chart(
    data: List[Dict], 
    context: str, 
    title: str,
    preferred_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an intelligent Chart.js visualization from data.
    
    Analyzes data characteristics and context to select optimal chart type,
    colors, and formatting. Returns a complete Chart.js configuration.
    
    Args:
        data: List of data points to visualize
        context: Description of what the data represents
        title: Chart title
        preferred_type: Optional override for chart type selection
        
    Returns:
        Complete Chart.js configuration object
    """
    try:
        # Analyze data characteristics
        characteristics = analyze_data_characteristics(data)
        
        # Select chart type
        chart_type = preferred_type or select_optimal_chart_type(data, context, characteristics)
        
        # Extract and format data
        chart_data = extract_chart_data(data, chart_type)
        
        # Select colors
        colors = select_color_scheme(context, chart_type, len(chart_data["datasets"]))
        
        # Apply colors to datasets
        for i, dataset in enumerate(chart_data["datasets"]):
            color = colors[i % len(colors)]
            
            if chart_type in ['pie', 'doughnut']:
                # For pie charts, each slice gets different color
                if len(chart_data["labels"]) > 1:
                    pie_colors = select_color_scheme(context, chart_type, len(chart_data["labels"]))
                    dataset["backgroundColor"] = pie_colors
                    dataset["borderColor"] = ["#ffffff"] * len(chart_data["labels"])
                else:
                    dataset["backgroundColor"] = color
            else:
                # For other charts, consistent color per dataset
                dataset["borderColor"] = color
                dataset["backgroundColor"] = color + "20"  # Add transparency
                if chart_type == 'line':
                    dataset["backgroundColor"] = color + "10"  # Less fill for lines
        
        # Build options
        options = build_chart_options(chart_type, title)
        
        return {
            "type": chart_type,
            "data": chart_data,
            "options": options,
            "metadata": {
                "data_characteristics": characteristics,
                "color_scheme_used": "auto-selected",
                "data_points": len(data)
            }
        }
        
    except Exception as e:
        # Return error chart
        return {
            "type": "bar",
            "data": {
                "labels": ["Error"],
                "datasets": [{
                    "label": "Chart Generation Error",
                    "data": [1],
                    "backgroundColor": "#F44336"
                }]
            },
            "options": build_chart_options("bar", f"Error: {str(e)}")
        }

@mcp.tool()
def create_time_series(
    data: List[Dict], 
    title: str,
    time_field: str = "year",
    value_field: str = "value"
) -> Dict[str, Any]:
    """
    Create a specialized time series line chart.
    
    Args:
        data: Time series data points
        title: Chart title
        time_field: Field name containing time values
        value_field: Field name containing numeric values
        
    Returns:
        Chart.js line chart configuration
    """
    try:
        # Extract time series data
        labels = [str(item.get(time_field, '')) for item in data]
        values = [item.get(value_field, 0) for item in data]
        
        chart_data = {
            "labels": labels,
            "datasets": [{
                "label": value_field.replace('_', ' ').title(),
                "data": values,
                "borderColor": "#2196F3",
                "backgroundColor": "rgba(33, 150, 243, 0.1)",
                "tension": 0.1,
                "pointBackgroundColor": "#2196F3",
                "pointBorderColor": "#ffffff",
                "pointRadius": 4
            }]
        }
        
        options = build_chart_options("line", title)
        options["scales"]["x"]["title"] = {
            "display": True,
            "text": time_field.replace('_', ' ').title()
        }
        options["scales"]["y"]["title"] = {
            "display": True,
            "text": value_field.replace('_', ' ').title()
        }
        
        return {
            "type": "line",
            "data": chart_data,
            "options": options
        }
        
    except Exception as e:
        return create_smart_chart(data, f"time series: {title}", title)

@mcp.tool()
def create_comparison(
    data: List[Dict], 
    title: str,
    category_field: str = "category",
    value_field: str = "value",
    chart_type: str = "bar"
) -> Dict[str, Any]:
    """
    Create a comparison chart (bar or pie).
    
    Args:
        data: Comparison data points
        title: Chart title
        category_field: Field containing categories
        value_field: Field containing values to compare
        chart_type: 'bar' or 'pie'
        
    Returns:
        Chart.js configuration for comparison chart
    """
    try:
        labels = [str(item.get(category_field, '')) for item in data]
        values = [item.get(value_field, 0) for item in data]
        
        colors = select_color_scheme("comparison", chart_type, len(data))
        
        dataset = {
            "label": value_field.replace('_', ' ').title(),
            "data": values
        }
        
        if chart_type in ['pie', 'doughnut']:
            dataset["backgroundColor"] = colors
            dataset["borderColor"] = ["#ffffff"] * len(data)
        else:
            dataset["backgroundColor"] = colors
            dataset["borderColor"] = colors
        
        chart_data = {
            "labels": labels,
            "datasets": [dataset]
        }
        
        return {
            "type": chart_type,
            "data": chart_data,
            "options": build_chart_options(chart_type, title)
        }
        
    except Exception as e:
        return create_smart_chart(data, f"comparison: {title}", title)

if __name__ == "__main__":
    mcp.run()