"""
ODE MCP Generic - Visualization Server
Generic chart primitives that can be orchestrated by LLM for any domain.
"""
import json
from fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import time

mcp = FastMCP("visualization-server")

metadata = {
    "Name": "Generic Visualization Server",
    "Description": "Provides generic chart primitives for data visualization across all domains",
    "Version": "1.0.0",
    "Author": "ODE Framework"
}

def _generate_chart_id(chart_type: str, data_hash: str = None) -> str:
    """Generate unique chart ID"""
    timestamp = str(int(time.time() * 1000))
    if data_hash:
        return f"{chart_type}_{data_hash}_{timestamp}"
    else:
        return f"{chart_type}_{timestamp}"

def _validate_data_structure(data: List[Dict], required_fields: List[str]) -> bool:
    """Validate that data has required fields"""
    if not data:
        return False
    
    first_item = data[0] if isinstance(data, list) else data
    return all(field in first_item for field in required_fields)

def _create_chart_response(chart_type: str, fig: go.Figure, title: str, data_summary: Dict) -> Dict[str, Any]:
    """Create standardized chart response"""
    chart_json = json.loads(fig.to_json())
    chart_id = _generate_chart_id(chart_type)
    
    return {
        "type": "chart",
        "chart_type": chart_type,
        "chart_id": chart_id,
        "title": title,
        "chart_data": chart_json,
        "data_summary": data_summary,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "chart_library": "plotly",
            "interactive": True
        }
    }

@mcp.tool()
def CreateLineChart(
    x_data: List[Union[str, float, int]],
    y_data: List[Union[float, int]], 
    title: str = "Line Chart",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    line_color: str = "#1f77b4"
) -> Dict[str, Any]:
    """
    Create a line chart from x and y data points.
    
    Args:
        x_data: X-axis values (can be strings, numbers, or dates)
        y_data: Y-axis numeric values
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label  
        line_color: Line color (hex code or name)
    
    Returns:
        Chart data structure for frontend rendering
    """
    try:
        if len(x_data) != len(y_data):
            return {"error": "X and Y data must have the same length"}
        
        if len(x_data) == 0:
            return {"error": "Data cannot be empty"}
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name='Data',
            line=dict(color=line_color, width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            hovermode="x unified"
        )
        
        data_summary = {
            "data_points": len(x_data),
            "y_range": [min(y_data), max(y_data)],
            "x_type": "categorical" if isinstance(x_data[0], str) else "numeric"
        }
        
        return _create_chart_response("line", fig, title, data_summary)
        
    except Exception as e:
        return {"error": f"Line chart creation failed: {str(e)}"}

@mcp.tool()
def CreateBarChart(
    categories: List[str],
    values: List[Union[float, int]],
    title: str = "Bar Chart", 
    x_label: str = "Categories",
    y_label: str = "Values",
    bar_color: str = "#1f77b4",
    horizontal: bool = False
) -> Dict[str, Any]:
    """
    Create a bar chart from categories and values.
    
    Args:
        categories: Category names for bars
        values: Numeric values for each category
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        bar_color: Bar color (hex code or name)
        horizontal: Whether to create horizontal bars
    
    Returns:
        Chart data structure for frontend rendering
    """
    try:
        if len(categories) != len(values):
            return {"error": "Categories and values must have the same length"}
        
        if len(categories) == 0:
            return {"error": "Data cannot be empty"}
        
        if horizontal:
            fig = go.Figure(go.Bar(
                x=values,
                y=categories,
                orientation='h',
                marker_color=bar_color,
                text=values,
                textposition='auto'
            ))
            fig.update_layout(xaxis_title=y_label, yaxis_title=x_label)
        else:
            fig = go.Figure(go.Bar(
                x=categories,
                y=values,
                marker_color=bar_color,
                text=values,
                textposition='auto'
            ))
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=False
        )
        
        data_summary = {
            "categories_count": len(categories),
            "value_range": [min(values), max(values)],
            "total_value": sum(values),
            "orientation": "horizontal" if horizontal else "vertical"
        }
        
        return _create_chart_response("bar", fig, title, data_summary)
        
    except Exception as e:
        return {"error": f"Bar chart creation failed: {str(e)}"}

@mcp.tool()
def CreatePieChart(
    labels: List[str],
    values: List[Union[float, int]],
    title: str = "Pie Chart",
    colors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a pie chart from labels and values.
    
    Args:
        labels: Labels for pie slices
        values: Values for each slice
        title: Chart title
        colors: Optional list of colors for slices
    
    Returns:
        Chart data structure for frontend rendering
    """
    try:
        if len(labels) != len(values):
            return {"error": "Labels and values must have the same length"}
        
        if len(labels) == 0:
            return {"error": "Data cannot be empty"}
        
        # Calculate percentages
        total = sum(values)
        percentages = [v/total * 100 for v in values]
        
        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            textposition='auto',
            marker=dict(colors=colors) if colors else None
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=True
        )
        
        data_summary = {
            "slice_count": len(labels),
            "total_value": total,
            "percentages": dict(zip(labels, percentages)),
            "largest_slice": labels[values.index(max(values))]
        }
        
        return _create_chart_response("pie", fig, title, data_summary)
        
    except Exception as e:
        return {"error": f"Pie chart creation failed: {str(e)}"}

@mcp.tool()
def CreateScatterPlot(
    x_data: List[Union[float, int]],
    y_data: List[Union[float, int]],
    title: str = "Scatter Plot",
    x_label: str = "X Axis", 
    y_label: str = "Y Axis",
    point_color: str = "#1f77b4",
    point_size: int = 8
) -> Dict[str, Any]:
    """
    Create a scatter plot from x and y data points.
    
    Args:
        x_data: X-axis numeric values
        y_data: Y-axis numeric values
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        point_color: Point color (hex code or name)
        point_size: Size of points
    
    Returns:
        Chart data structure for frontend rendering
    """
    try:
        if len(x_data) != len(y_data):
            return {"error": "X and Y data must have the same length"}
        
        if len(x_data) == 0:
            return {"error": "Data cannot be empty"}
        
        fig = go.Figure(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                size=point_size,
                color=point_color,
                opacity=0.7
            ),
            name='Data Points'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            hovermode="closest"
        )
        
        data_summary = {
            "data_points": len(x_data),
            "x_range": [min(x_data), max(x_data)],
            "y_range": [min(y_data), max(y_data)],
            "correlation": np.corrcoef(x_data, y_data)[0, 1] if len(x_data) > 1 else 0
        }
        
        return _create_chart_response("scatter", fig, title, data_summary)
        
    except Exception as e:
        return {"error": f"Scatter plot creation failed: {str(e)}"}

@mcp.tool()
def CreateHistogram(
    data: List[Union[float, int]],
    bins: int = 10,
    title: str = "Histogram",
    x_label: str = "Values",
    y_label: str = "Frequency",
    bar_color: str = "#1f77b4"
) -> Dict[str, Any]:
    """
    Create a histogram from numeric data.
    
    Args:
        data: Numeric values to create histogram from
        bins: Number of bins for the histogram
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label  
        bar_color: Bar color (hex code or name)
    
    Returns:
        Chart data structure for frontend rendering
    """
    try:
        if len(data) == 0:
            return {"error": "Data cannot be empty"}
        
        fig = go.Figure(go.Histogram(
            x=data,
            nbinsx=bins,
            marker_color=bar_color,
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            bargap=0.1
        )
        
        data_summary = {
            "data_points": len(data),
            "value_range": [min(data), max(data)],
            "mean": np.mean(data),
            "std_dev": np.std(data),
            "bins": bins
        }
        
        return _create_chart_response("histogram", fig, title, data_summary)
        
    except Exception as e:
        return {"error": f"Histogram creation failed: {str(e)}"}

@mcp.tool()
def CreateTimeSeriesChart(
    dates: List[str],
    values: List[Union[float, int]],
    title: str = "Time Series Chart",
    y_label: str = "Values",
    line_color: str = "#1f77b4",
    show_trend: bool = False
) -> Dict[str, Any]:
    """
    Create a time series chart from dates and values.
    
    Args:
        dates: Date strings (ISO format recommended)
        values: Numeric values corresponding to dates
        title: Chart title
        y_label: Y-axis label
        line_color: Line color (hex code or name)
        show_trend: Whether to show trend line
    
    Returns:
        Chart data structure for frontend rendering
    """
    try:
        if len(dates) != len(values):
            return {"error": "Dates and values must have the same length"}
        
        if len(dates) == 0:
            return {"error": "Data cannot be empty"}
        
        # Convert dates to datetime if they're strings
        try:
            parsed_dates = pd.to_datetime(dates)
        except:
            return {"error": "Invalid date format. Use ISO format (YYYY-MM-DD) or datetime strings"}
        
        fig = go.Figure()
        
        # Main time series line
        fig.add_trace(go.Scatter(
            x=parsed_dates,
            y=values,
            mode='lines+markers',
            name='Time Series',
            line=dict(color=line_color, width=2),
            marker=dict(size=4)
        ))
        
        # Add trend line if requested
        if show_trend and len(values) > 2:
            z = np.polyfit(range(len(values)), values, 1)
            trend_line = np.poly1d(z)(range(len(values)))
            
            fig.add_trace(go.Scatter(
                x=parsed_dates,
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label,
            template="plotly_white",
            hovermode="x unified"
        )
        
        data_summary = {
            "data_points": len(dates),
            "date_range": [str(min(parsed_dates)), str(max(parsed_dates))],
            "value_range": [min(values), max(values)],
            "trend_included": show_trend
        }
        
        return _create_chart_response("timeseries", fig, title, data_summary)
        
    except Exception as e:
        return {"error": f"Time series chart creation failed: {str(e)}"}

@mcp.tool()
def CreateHeatmap(
    data: List[List[Union[float, int]]],
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Heatmap",
    colorscale: str = "Blues"
) -> Dict[str, Any]:
    """
    Create a heatmap from 2D data array.
    
    Args:
        data: 2D array of numeric values
        x_labels: Optional labels for x-axis
        y_labels: Optional labels for y-axis
        title: Chart title
        colorscale: Plotly colorscale name
    
    Returns:
        Chart data structure for frontend rendering
    """
    try:
        if not data or not data[0]:
            return {"error": "Data cannot be empty"}
        
        # Validate 2D structure
        row_lengths = [len(row) for row in data]
        if len(set(row_lengths)) > 1:
            return {"error": "All rows must have the same length"}
        
        fig = go.Figure(go.Heatmap(
            z=data,
            x=x_labels if x_labels else list(range(len(data[0]))),
            y=y_labels if y_labels else list(range(len(data))),
            colorscale=colorscale,
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_white"
        )
        
        # Flatten data for statistics
        flat_data = [val for row in data for val in row]
        
        data_summary = {
            "dimensions": [len(data), len(data[0])],
            "value_range": [min(flat_data), max(flat_data)],
            "total_cells": len(flat_data),
            "colorscale": colorscale
        }
        
        return _create_chart_response("heatmap", fig, title, data_summary)
        
    except Exception as e:
        return {"error": f"Heatmap creation failed: {str(e)}"}

@mcp.tool()
def CreateTableVisualization(
    data: List[Dict[str, Any]],
    title: str = "Data Table",
    max_rows: int = 100
) -> Dict[str, Any]:
    """
    Create a formatted table from structured data.
    
    Args:
        data: List of dictionaries with consistent keys
        title: Table title
        max_rows: Maximum number of rows to display
    
    Returns:
        Table data structure for frontend rendering
    """
    try:
        if not data:
            return {"error": "Data cannot be empty"}
        
        # Limit rows
        display_data = data[:max_rows]
        
        # Get consistent columns from first row
        columns = list(display_data[0].keys()) if display_data else []
        
        # Convert to format suitable for table display
        table_data = {
            "type": "table",
            "title": title,
            "columns": columns,
            "rows": [[row.get(col, "") for col in columns] for row in display_data],
            "metadata": {
                "total_rows_available": len(data),
                "rows_displayed": len(display_data),
                "columns_count": len(columns),
                "truncated": len(data) > max_rows
            }
        }
        
        return table_data
        
    except Exception as e:
        return {"error": f"Table creation failed: {str(e)}"}

@mcp.tool()
def GetVisualizationCapabilities() -> Dict[str, Any]:
    """
    Get information about available visualization capabilities.
    
    Returns:
        Dictionary describing available chart types and their parameters
    """
    return {
        "available_charts": {
            "line": {
                "description": "Line charts for continuous data and trends",
                "required_params": ["x_data", "y_data"],
                "optional_params": ["title", "x_label", "y_label", "line_color"]
            },
            "bar": {
                "description": "Bar charts for categorical data comparison",
                "required_params": ["categories", "values"],
                "optional_params": ["title", "x_label", "y_label", "bar_color", "horizontal"]
            },
            "pie": {
                "description": "Pie charts for part-to-whole relationships",
                "required_params": ["labels", "values"],
                "optional_params": ["title", "colors"]
            },
            "scatter": {
                "description": "Scatter plots for correlation analysis",
                "required_params": ["x_data", "y_data"],
                "optional_params": ["title", "x_label", "y_label", "point_color", "point_size"]
            },
            "histogram": {
                "description": "Histograms for data distribution analysis",
                "required_params": ["data"],
                "optional_params": ["bins", "title", "x_label", "y_label", "bar_color"]
            },
            "timeseries": {
                "description": "Time series charts for temporal data",
                "required_params": ["dates", "values"],
                "optional_params": ["title", "y_label", "line_color", "show_trend"]
            },
            "heatmap": {
                "description": "Heatmaps for 2D data visualization",
                "required_params": ["data"],
                "optional_params": ["x_labels", "y_labels", "title", "colorscale"]
            },
            "table": {
                "description": "Formatted tables for structured data",
                "required_params": ["data"],
                "optional_params": ["title", "max_rows"]
            }
        },
        "supported_formats": ["plotly_json", "html"],
        "interactive_features": ["zoom", "pan", "hover", "selection"],
        "export_formats": ["png", "svg", "pdf", "html"],
        "server_info": metadata
    }

if __name__ == "__main__":
    print(f"🎨 {metadata['Name']} v{metadata['Version']}")
    print(f"📈 Generic chart primitives for data visualization")
    print("Available tools:")
    print("  - CreateLineChart: Line charts for trends")
    print("  - CreateBarChart: Bar charts for categories") 
    print("  - CreatePieChart: Pie charts for proportions")
    print("  - CreateScatterPlot: Scatter plots for correlation")
    print("  - CreateHistogram: Histograms for distributions")
    print("  - CreateTimeSeriesChart: Time series analysis")
    print("  - CreateHeatmap: 2D data visualization")
    print("  - CreateTableVisualization: Formatted data tables")
    print("  - GetVisualizationCapabilities: Server info")
    print("\\nReady for MCP connections!")