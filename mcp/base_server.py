"""
ODE MCP Generic - Abstract Base Server Class
Provides consistent patterns for implementing domain-specific MCP servers.
"""
from abc import ABC, abstractmethod
from fastmcp import FastMCP
from typing import Dict, Any, List, Optional
import json
import time

class BaseMCPServer(ABC):
    """
    Abstract base class for all MCP servers in the ODE framework.
    
    This class provides a consistent structure for implementing domain-specific
    MCP servers with proper citation handling and tool registration.
    
    Example implementation for a Financial Data Server:
    
    ```python
    from base_server import BaseMCPServer
    from base_data_manager import BaseDataManager
    
    class FinanceDataManager(BaseDataManager):
        def load_data(self) -> bool:
            # Load financial data from CSV/API/database
            self.stocks_data = pd.read_csv("data/stocks.csv")
            return True
        
        def search(self, query: str, filters: Dict = None) -> List[Dict]:
            # Search through financial data
            matches = self.stocks_data[
                self.stocks_data['company_name'].str.contains(query, case=False)
            ]
            return matches.head(100).to_dict('records')
        
        def get_statistics(self) -> Dict[str, Any]:
            return {
                "total_stocks": len(self.stocks_data),
                "sectors": list(self.stocks_data['sector'].unique()),
                "date_range": {
                    "start": self.stocks_data['date'].min(),
                    "end": self.stocks_data['date'].max()
                }
            }
    
    class FinanceServer(BaseMCPServer):
        def __init__(self):
            super().__init__("finance-server")
            self.data_manager = FinanceDataManager("data/finance/")
            
        def get_metadata(self) -> Dict[str, str]:
            return {
                "Name": "Financial Data Server",
                "Description": "Provides stock market and company financial data",
                "Version": "1.0.0", 
                "Author": "Your Organization"
            }
        
        def setup_tools(self):
            @self.mcp.tool()
            def GetStockPrice(symbol: str) -> Dict[str, Any]:
                '''Get current stock price and basic metrics for a symbol'''
                try:
                    # Your implementation here
                    stock_data = self.data_manager.get_stock_data(symbol)
                    return {
                        "symbol": symbol,
                        "price": stock_data.get("current_price", 0),
                        "change": stock_data.get("change_percent", 0),
                        "volume": stock_data.get("volume", 0),
                        "market_cap": stock_data.get("market_cap", 0)
                    }
                except Exception as e:
                    return {"error": f"Failed to get stock price: {str(e)}"}
            
            @self.mcp.tool()
            def SearchCompanies(query: str, sector: str = None) -> Dict[str, Any]:
                '''Search for companies by name with optional sector filter'''
                try:
                    filters = {"sector": sector} if sector else None
                    companies = self.data_manager.search(query, filters)
                    return {
                        "query": query,
                        "results": companies,
                        "count": len(companies)
                    }
                except Exception as e:
                    return {"error": f"Company search failed: {str(e)}"}
        
        def get_citation_info(self, tool_name: str, tool_args: Dict = None) -> Dict:
            # Map your tools to their data sources
            mapping = {
                "GetStockPrice": {
                    "source_name": "Yahoo Finance API",
                    "provider": "Yahoo Finance",
                    "spatial_coverage": "Global markets",
                    "temporal_coverage": "Real-time",
                    "source_url": "https://finance.yahoo.com"
                },
                "SearchCompanies": {
                    "source_name": "SEC EDGAR Company Database",
                    "provider": "U.S. Securities and Exchange Commission",
                    "spatial_coverage": "United States public companies",
                    "temporal_coverage": "Current active companies",
                    "source_url": "https://sec.gov/edgar"
                }
            }
            return mapping.get(tool_name, self._default_citation())
    
    # Usage:
    if __name__ == "__main__":
        server = FinanceServer()
        server.run()
    ```
    """
    
    def __init__(self, server_name: str):
        """
        Initialize the MCP server.
        
        Args:
            server_name: Unique name for this server (e.g., "finance-server", "weather-server")
        """
        self.mcp = FastMCP(server_name)
        self.server_name = server_name
        self._setup_metadata()
        self.setup_tools()
        
    def _setup_metadata(self):
        """Internal method to register server metadata"""
        metadata = self.get_metadata()
        # Set metadata on the MCP server if the library supports it
        # This is for future compatibility with FastMCP metadata features
        
    @abstractmethod
    def get_metadata(self) -> Dict[str, str]:
        """
        Return server metadata that appears in MCP tool registry.
        
        MUST return dict with these exact keys:
        - "Name": Human-readable server name
        - "Description": What this server does (1-2 sentences)
        - "Version": Semantic version (e.g., "1.0.0") 
        - "Author": Your name or organization
        
        Example:
        ```python
        def get_metadata(self) -> Dict[str, str]:
            return {
                "Name": "Weather Data Server",
                "Description": "Provides current and historical weather data for global locations",
                "Version": "1.2.0",
                "Author": "Weather Analytics Corp"
            }
        ```
        
        Returns:
            Dictionary with server metadata
        """
        pass
    
    @abstractmethod  
    def setup_tools(self):
        """
        Register all MCP tools this server provides.
        
        Use @self.mcp.tool() decorator to register each function as an MCP tool.
        Each tool should return Dict[str, Any] for consistent processing.
        
        Guidelines:
        - Use descriptive function names (GetWeatherData, not get_data)
        - Include type hints for all parameters
        - Add docstrings describing what each tool does
        - Handle errors gracefully and return error dict instead of raising
        - Return consistent data structures
        
        Example:
        ```python
        def setup_tools(self):
            @self.mcp.tool()
            def GetWeatherData(city: str, days: int = 7) -> Dict[str, Any]:
                '''Get weather forecast for a city over specified days'''
                try:
                    weather_data = self.data_manager.get_weather(city, days)
                    return {
                        "city": city,
                        "forecast": weather_data,
                        "forecast_days": days,
                        "data_source": "national_weather_service"
                    }
                except Exception as e:
                    return {
                        "error": f"Weather data retrieval failed: {str(e)}",
                        "city": city,
                        "requested_days": days
                    }
        ```
        """
        pass
    
    @abstractmethod
    def get_citation_info(self, tool_name: str, tool_args: Dict = None) -> Dict:
        """
        Return citation metadata for each tool to enable proper attribution.
        
        This method is called by the citation system to understand where
        your data comes from, enabling automatic citation generation.
        
        Args:
            tool_name: Name of the tool being called (e.g., "GetWeatherData")
            tool_args: Arguments passed to the tool (optional, for dynamic citations)
        
        MUST return dict with these exact keys:
        - "source_name": Name of the data source
        - "provider": Organization providing the data  
        - "spatial_coverage": Geographic coverage (e.g., "Global", "USA", "Europe")
        - "temporal_coverage": Time coverage (e.g., "2020-2024", "Real-time")
        - "source_url": URL to data source (empty string if none)
        
        You can use tool_args for dynamic citations:
        ```python
        def get_citation_info(self, tool_name: str, tool_args: Dict = None) -> Dict:
            if tool_name == "GetWeatherData":
                city = tool_args.get("city", "Unknown") if tool_args else "Unknown"
                return {
                    "source_name": f"Weather Data for {city}",
                    "provider": "National Weather Service",
                    "spatial_coverage": city if city != "Unknown" else "Global",
                    "temporal_coverage": "Real-time + 7-day forecast", 
                    "source_url": "https://weather.gov"
                }
            elif tool_name == "GetHistoricalWeather":
                return {
                    "source_name": "Historical Weather Database",
                    "provider": "NOAA Climate Data",
                    "spatial_coverage": "Global",
                    "temporal_coverage": "1880-present",
                    "source_url": "https://noaa.gov/climate"
                }
            
            return self._default_citation()
        ```
        
        Returns:
            Dictionary with citation metadata
        """
        pass
    
    def _default_citation(self) -> Dict:
        """
        Fallback citation when tool not found in mapping.
        Override if you want custom fallback behavior.
        """
        return {
            "source_name": f"{self.server_name} Dataset",
            "provider": "Unknown Provider",
            "spatial_coverage": "Unknown Coverage",
            "temporal_coverage": "Unknown Period", 
            "source_url": ""
        }
    
    def run(self, host: str = "localhost", port: int = None):
        """
        Run the MCP server (for standalone testing).
        
        Args:
            host: Host to bind to
            port: Port to use (auto-assigned if None)
        """
        print(f"🚀 Starting {self.server_name}")
        metadata = self.get_metadata()
        print(f"📋 {metadata['Name']} v{metadata['Version']}")
        print(f"📝 {metadata['Description']}")
        print(f"👤 By {metadata['Author']}")
        
        # List available tools
        tools = [name for name in dir(self) if hasattr(getattr(self, name), '__name__')]
        print(f"🔧 Available tools: {len([t for t in tools if not t.startswith('_')])}")
        
        try:
            # Start the FastMCP server
            self.mcp.run(host=host, port=port)
        except KeyboardInterrupt:
            print(f"\\n🛑 {self.server_name} stopped")

class HealthCheckMixin:
    """
    Optional mixin to add health check capabilities to MCP servers.
    
    Usage:
    ```python
    class MyServer(BaseMCPServer, HealthCheckMixin):
        # Your implementation
    ```
    """
    
    def add_health_check_tool(self):
        """Add health check tool to the server"""
        @self.mcp.tool()
        def HealthCheck() -> Dict[str, Any]:
            '''Check server health and data availability'''
            try:
                # Check data manager if available
                if hasattr(self, 'data_manager'):
                    stats = self.data_manager.get_statistics()
                    return {
                        "status": "healthy",
                        "timestamp": time.time(),
                        "data_stats": stats,
                        "server": self.server_name
                    }
                else:
                    return {
                        "status": "healthy",
                        "timestamp": time.time(),
                        "server": self.server_name,
                        "message": "Server running without data manager"
                    }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "timestamp": time.time(),
                    "error": str(e),
                    "server": self.server_name
                }

# Example minimal implementation for testing
class ExampleServer(BaseMCPServer):
    """
    Minimal example server for testing and demonstration.
    Copy and modify this for your domain.
    """
    
    def __init__(self):
        super().__init__("example-server")
        
    def get_metadata(self) -> Dict[str, str]:
        return {
            "Name": "Example MCP Server",
            "Description": "Minimal example server for testing the ODE MCP framework",
            "Version": "1.0.0",
            "Author": "ODE Framework"
        }
    
    def setup_tools(self):
        @self.mcp.tool()
        def GetExampleData() -> Dict[str, Any]:
            '''Return example data for testing'''
            return {
                "message": "Hello from Example Server!",
                "timestamp": time.time(),
                "data": [1, 2, 3, 4, 5]
            }
        
        @self.mcp.tool()  
        def EchoMessage(message: str) -> Dict[str, Any]:
            '''Echo back a message with metadata'''
            return {
                "original_message": message,
                "echoed_at": time.time(),
                "server": self.server_name
            }
    
    def get_citation_info(self, tool_name: str, tool_args: Dict = None) -> Dict:
        return {
            "source_name": "Example Dataset",
            "provider": "ODE Framework",
            "spatial_coverage": "Global",
            "temporal_coverage": "Current",
            "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
        }

if __name__ == "__main__":
    # Run example server for testing
    server = ExampleServer()
    server.run()