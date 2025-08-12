"""
Finance Domain Server Example
Complete implementation showing how to use the ODE MCP Generic framework for financial data.
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys

# Add the parent mcp directory to path to import base classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'mcp'))

from base_server import BaseMCPServer, HealthCheckMixin
from base_data_manager import BaseDataManager

class FinanceDataManager(BaseDataManager):
    """
    Example data manager for financial data.
    
    In a real implementation, this would connect to:
    - Financial APIs (Yahoo Finance, Alpha Vantage, etc.)
    - Databases (PostgreSQL, MongoDB, etc.)
    - Data vendors (Bloomberg, Reuters, etc.)
    """
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.stocks_data = pd.DataFrame()
        self.companies_data = pd.DataFrame()
        self.market_data = pd.DataFrame()
        
    def load_data(self) -> bool:
        """Load financial datasets"""
        try:
            # Try to load existing data files
            stocks_path = os.path.join(self.data_path, "stocks.csv")
            companies_path = os.path.join(self.data_path, "companies.csv")
            market_path = os.path.join(self.data_path, "market_data.csv")
            
            if os.path.exists(stocks_path):
                self.stocks_data = pd.read_csv(stocks_path)
                print(f"✅ Loaded {len(self.stocks_data)} stock records")
            else:
                print("📝 Creating sample stock data")
                self.stocks_data = self._create_sample_stocks_data()
                os.makedirs(self.data_path, exist_ok=True)
                self.stocks_data.to_csv(stocks_path, index=False)
            
            if os.path.exists(companies_path):
                self.companies_data = pd.read_csv(companies_path)
                print(f"✅ Loaded {len(self.companies_data)} company records")
            else:
                print("📝 Creating sample company data")
                self.companies_data = self._create_sample_companies_data()
                self.companies_data.to_csv(companies_path, index=False)
                
            if os.path.exists(market_path):
                self.market_data = pd.read_csv(market_path)
                print(f"✅ Loaded {len(self.market_data)} market records")
            else:
                print("📝 Creating sample market data")
                self.market_data = self._create_sample_market_data()
                self.market_data.to_csv(market_path, index=False)
            
            # Validate data
            if self.stocks_data.empty or self.companies_data.empty:
                print("❌ Error: Critical datasets are empty")
                return False
            
            print("🎉 Financial data loading completed")
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Financial data loading failed: {e}")
            return False
    
    def _create_sample_stocks_data(self) -> pd.DataFrame:
        """Create sample stock price data"""
        import random
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ORCL', 'CRM']
        
        data = []
        base_date = datetime.now() - timedelta(days=365)
        
        for symbol in symbols:
            base_price = random.uniform(50, 500)
            
            for i in range(365):
                date = base_date + timedelta(days=i)
                
                # Simple random walk for price
                change = random.uniform(-0.05, 0.05)
                base_price = max(base_price * (1 + change), 1.0)
                
                data.append({
                    'symbol': symbol,
                    'date': date.strftime('%Y-%m-%d'),
                    'price': round(base_price, 2),
                    'volume': random.randint(1000000, 100000000),
                    'market_cap': round(base_price * random.randint(1000000, 5000000000), 0),
                    'pe_ratio': round(random.uniform(10, 50), 2) if random.random() > 0.1 else None
                })
        
        return pd.DataFrame(data)
    
    def _create_sample_companies_data(self) -> pd.DataFrame:
        """Create sample company information data"""
        companies = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet Software'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': 'Technology', 'industry': 'Semiconductors'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'sector': 'Technology', 'industry': 'Internet Software'},
            {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'sector': 'Communication Services', 'industry': 'Entertainment'},
            {'symbol': 'ORCL', 'name': 'Oracle Corporation', 'sector': 'Technology', 'industry': 'Software'},
            {'symbol': 'CRM', 'name': 'Salesforce Inc.', 'sector': 'Technology', 'industry': 'Software'}
        ]
        
        import random
        for company in companies:
            company.update({
                'employees': random.randint(10000, 200000),
                'founded': random.randint(1970, 2010),
                'country': 'United States',
                'exchange': 'NASDAQ',
                'description': f'{company["name"]} operates in the {company["industry"]} industry.'
            })
        
        return pd.DataFrame(companies)
    
    def _create_sample_market_data(self) -> pd.DataFrame:
        """Create sample market index data"""
        import random
        
        indices = ['S&P500', 'NASDAQ', 'DOW', 'RUSSELL2000']
        data = []
        base_date = datetime.now() - timedelta(days=365)
        
        for index in indices:
            base_value = random.uniform(3000, 15000)
            
            for i in range(365):
                date = base_date + timedelta(days=i)
                change = random.uniform(-0.03, 0.03)
                base_value = max(base_value * (1 + change), 1000)
                
                data.append({
                    'index': index,
                    'date': date.strftime('%Y-%m-%d'),
                    'value': round(base_value, 2),
                    'change_percent': round(change * 100, 2)
                })
        
        return pd.DataFrame(data)
    
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search across financial data"""
        if not self.is_loaded:
            return []
        
        try:
            results = []
            query_lower = query.lower()
            
            # Search companies
            company_matches = self.companies_data[
                (self.companies_data['name'].str.contains(query, case=False, na=False)) |
                (self.companies_data['symbol'].str.contains(query, case=False, na=False)) |
                (self.companies_data['sector'].str.contains(query, case=False, na=False))
            ]
            
            for _, company in company_matches.iterrows():
                # Get latest stock price
                latest_price = self.stocks_data[
                    self.stocks_data['symbol'] == company['symbol']
                ].sort_values('date').tail(1)
                
                result = company.to_dict()
                result['type'] = 'company'
                
                if not latest_price.empty:
                    result.update({
                        'current_price': float(latest_price.iloc[0]['price']),
                        'market_cap': float(latest_price.iloc[0]['market_cap']),
                        'last_updated': latest_price.iloc[0]['date']
                    })
                
                results.append(result)
            
            # Apply filters
            if filters:
                if 'sector' in filters:
                    results = [r for r in results if r.get('sector') == filters['sector']]
                if 'min_market_cap' in filters:
                    results = [r for r in results if r.get('market_cap', 0) >= filters['min_market_cap']]
                if 'max_price' in filters:
                    results = [r for r in results if r.get('current_price', float('inf')) <= filters['max_price']]
            
            return results[:50]
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive financial data statistics"""
        if not self.is_loaded:
            return {"error": "Data not loaded"}
        
        try:
            # Get latest prices for each stock
            latest_prices = self.stocks_data.groupby('symbol').apply(
                lambda x: x.sort_values('date').tail(1)
            ).reset_index(drop=True)
            
            stats = {
                "companies_count": len(self.companies_data),
                "stocks_tracked": len(latest_prices),
                "sectors": list(self.companies_data['sector'].unique()),
                "exchanges": list(self.companies_data['exchange'].unique()),
                "price_range": {
                    "min": float(latest_prices['price'].min()),
                    "max": float(latest_prices['price'].max()),
                    "average": float(latest_prices['price'].mean())
                },
                "market_cap_stats": {
                    "total": float(latest_prices['market_cap'].sum()),
                    "average": float(latest_prices['market_cap'].mean()),
                    "median": float(latest_prices['market_cap'].median())
                },
                "date_range": {
                    "start": self.stocks_data['date'].min(),
                    "end": self.stocks_data['date'].max()
                },
                "sector_breakdown": self.companies_data['sector'].value_counts().to_dict(),
                "data_quality": {
                    "complete_price_records": len(self.stocks_data.dropna()),
                    "companies_with_pe_ratio": len(latest_prices[latest_prices['pe_ratio'].notna()])
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Statistics calculation failed: {e}")
            return {"error": f"Statistics calculation failed: {str(e)}"}
    
    def get_stock_price_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get price history for a specific stock"""
        if not self.is_loaded:
            return []
        
        try:
            stock_data = self.stocks_data[
                self.stocks_data['symbol'].str.upper() == symbol.upper()
            ].sort_values('date').tail(days)
            
            return stock_data.to_dict('records')
            
        except Exception as e:
            print(f"❌ Price history retrieval failed: {e}")
            return []
    
    def get_sector_analysis(self, sector: str) -> Dict[str, Any]:
        """Get analysis for a specific sector"""
        if not self.is_loaded:
            return {}
        
        try:
            sector_companies = self.companies_data[
                self.companies_data['sector'].str.lower() == sector.lower()
            ]
            
            if sector_companies.empty:
                return {"error": f"Sector '{sector}' not found"}
            
            # Get latest prices for sector companies
            sector_symbols = sector_companies['symbol'].tolist()
            sector_prices = self.stocks_data[
                self.stocks_data['symbol'].isin(sector_symbols)
            ].groupby('symbol').apply(
                lambda x: x.sort_values('date').tail(1)
            ).reset_index(drop=True)
            
            analysis = {
                "sector": sector,
                "companies_count": len(sector_companies),
                "total_market_cap": float(sector_prices['market_cap'].sum()),
                "average_price": float(sector_prices['price'].mean()),
                "price_range": {
                    "min": float(sector_prices['price'].min()),
                    "max": float(sector_prices['price'].max())
                },
                "companies": sector_companies.to_dict('records')
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Sector analysis failed: {e}")
            return {"error": f"Sector analysis failed: {str(e)}"}

class FinanceServer(BaseMCPServer, HealthCheckMixin):
    """
    Complete financial data MCP server implementation.
    
    This demonstrates a real-world server with:
    - Stock price data
    - Company information
    - Market analysis tools
    - Sector analysis
    - Historical data access
    """
    
    def __init__(self):
        super().__init__("finance-server")
        
        # Initialize data manager
        data_path = os.getenv("FINANCE_DATA_PATH", "examples/finance_example/data/")
        self.data_manager = FinanceDataManager(data_path)
        
        # Load financial data
        if not self.data_manager.load_data():
            print("⚠️ Warning: Financial data loading failed, server will have limited functionality")
        
        # Add health check capability
        self.add_health_check_tool()
        
    def get_metadata(self) -> Dict[str, str]:
        return {
            "Name": "Finance Data Server",
            "Description": "Provides stock market data, company information, and financial analysis tools",
            "Version": "1.0.0",
            "Author": "ODE Framework - Finance Example"
        }
    
    def setup_tools(self):
        """Register all financial analysis tools"""
        
        @self.mcp.tool()
        def GetStockPrice(symbol: str, days: int = 1) -> Dict[str, Any]:
            """Get current or recent stock price data for a symbol"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Financial data not loaded"}
                
                price_data = self.data_manager.get_stock_price_history(symbol, days)
                
                if not price_data:
                    return {"error": f"No data found for symbol {symbol}"}
                
                latest = price_data[-1] if price_data else None
                
                return {
                    "symbol": symbol.upper(),
                    "current_price": latest['price'] if latest else None,
                    "volume": latest['volume'] if latest else None,
                    "market_cap": latest['market_cap'] if latest else None,
                    "pe_ratio": latest['pe_ratio'] if latest else None,
                    "last_updated": latest['date'] if latest else None,
                    "price_history": price_data if days > 1 else None,
                    "data_source": "finance_demo"
                }
                
            except Exception as e:
                return {"error": f"Stock price retrieval failed: {str(e)}"}
        
        @self.mcp.tool()
        def SearchCompanies(query: str, sector: Optional[str] = None, 
                          min_market_cap: Optional[float] = None,
                          max_price: Optional[float] = None) -> Dict[str, Any]:
            """Search for companies by name, symbol, or sector"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Financial data not loaded"}
                
                filters = {}
                if sector:
                    filters['sector'] = sector
                if min_market_cap:
                    filters['min_market_cap'] = min_market_cap
                if max_price:
                    filters['max_price'] = max_price
                
                results = self.data_manager.search(query, filters)
                
                return {
                    "query": query,
                    "filters": filters,
                    "results": results,
                    "result_count": len(results),
                    "data_source": "finance_demo"
                }
                
            except Exception as e:
                return {"error": f"Company search failed: {str(e)}"}
        
        @self.mcp.tool()
        def GetSectorAnalysis(sector: str) -> Dict[str, Any]:
            """Get comprehensive analysis for a market sector"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Financial data not loaded"}
                
                analysis = self.data_manager.get_sector_analysis(sector)
                
                if "error" in analysis:
                    return analysis
                
                analysis["data_source"] = "finance_demo"
                return analysis
                
            except Exception as e:
                return {"error": f"Sector analysis failed: {str(e)}"}
        
        @self.mcp.tool()
        def GetMarketOverview() -> Dict[str, Any]:
            """Get overall market statistics and overview"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Financial data not loaded"}
                
                stats = self.data_manager.get_statistics()
                
                return {
                    "market_overview": stats,
                    "data_source": "finance_demo",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {"error": f"Market overview failed: {str(e)}"}
        
        @self.mcp.tool()
        def GetAvailableSectors() -> Dict[str, Any]:
            """Get list of all available market sectors"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Financial data not loaded"}
                
                sectors = list(self.data_manager.companies_data['sector'].unique())
                sector_counts = self.data_manager.companies_data['sector'].value_counts().to_dict()
                
                return {
                    "sectors": sectors,
                    "sector_counts": sector_counts,
                    "total_sectors": len(sectors),
                    "data_source": "finance_demo"
                }
                
            except Exception as e:
                return {"error": f"Failed to get sectors: {str(e)}"}
        
        @self.mcp.tool()
        def GetPriceHistory(symbol: str, days: int = 30) -> Dict[str, Any]:
            """Get historical price data for analysis and charting"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Financial data not loaded"}
                
                history = self.data_manager.get_stock_price_history(symbol, days)
                
                if not history:
                    return {"error": f"No price history found for {symbol}"}
                
                # Calculate some basic metrics
                prices = [record['price'] for record in history]
                
                return {
                    "symbol": symbol.upper(),
                    "days_requested": days,
                    "records_returned": len(history),
                    "price_history": history,
                    "price_summary": {
                        "min": min(prices),
                        "max": max(prices),
                        "start": prices[0] if prices else None,
                        "end": prices[-1] if prices else None,
                        "change_percent": ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) >= 2 else 0
                    },
                    "data_source": "finance_demo"
                }
                
            except Exception as e:
                return {"error": f"Price history retrieval failed: {str(e)}"}
    
    def get_citation_info(self, tool_name: str, tool_args: Dict = None) -> Dict:
        """Provide citation information for financial data tools"""
        citation_mapping = {
            "GetStockPrice": {
                "source_name": "Demo Financial Database - Stock Prices",
                "provider": "ODE Framework Finance Example",
                "spatial_coverage": "US Stock Markets (Demo Data)",
                "temporal_coverage": "Past 12 months (simulated)",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "SearchCompanies": {
                "source_name": "Demo Company Information Database",
                "provider": "ODE Framework Finance Example",
                "spatial_coverage": "US Public Companies (Demo Data)",
                "temporal_coverage": "Current company information (simulated)",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "GetSectorAnalysis": {
                "source_name": "Demo Sector Analysis Engine",
                "provider": "ODE Framework Finance Example",
                "spatial_coverage": "US Market Sectors (Demo Data)",
                "temporal_coverage": "Current market data (simulated)",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "GetMarketOverview": {
                "source_name": "Demo Market Statistics Database",
                "provider": "ODE Framework Finance Example",
                "spatial_coverage": "US Financial Markets (Demo Data)",
                "temporal_coverage": "Current market state (simulated)",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "GetPriceHistory": {
                "source_name": "Demo Historical Price Database",
                "provider": "ODE Framework Finance Example",
                "spatial_coverage": "US Stock Markets (Demo Data)",
                "temporal_coverage": "Past 12 months (simulated)",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            }
        }
        
        return citation_mapping.get(tool_name, self._default_citation())

# Example usage and testing
if __name__ == "__main__":
    print("💰 Starting Finance Server Example")
    print("")
    print("This demonstrates a complete domain implementation:")
    print("  📈 Stock price data and analysis")
    print("  🏢 Company information and search")
    print("  📊 Sector analysis and market overview")
    print("  📉 Historical price data for charting")
    print("  💡 Proper citation handling")
    print("")
    
    server = FinanceServer()
    
    print("💰 Finance Server Status:")
    print(f"  Data loaded: {server.data_manager.is_data_loaded()}")
    if server.data_manager.is_data_loaded():
        stats = server.data_manager.get_statistics()
        print(f"  Companies tracked: {stats.get('companies_count', 'unknown')}")
        print(f"  Sectors available: {len(stats.get('sectors', []))}")
        print(f"  Price data points: {len(server.data_manager.stocks_data)}")
    
    print("")
    print("🔧 Ready for MCP connections!")
    print("Use this as a template for your own domain servers.")
    
    # Start the server
    server.run()