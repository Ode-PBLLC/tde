"""
ODE MCP Generic - Template Server
Complete reference implementation showing how to use the abstract base classes.
Copy this file and modify it for your specific domain.
"""
import json
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from base_server import BaseMCPServer, HealthCheckMixin
from base_data_manager import BaseDataManager

class TemplateDataManager(BaseDataManager):
    """
    Example data manager implementation for template/demo purposes.
    
    This shows how to implement the BaseDataManager interface for your domain.
    Replace this with your actual data management logic.
    """
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.main_data = pd.DataFrame()
        self.metadata = {}
        
    def load_data(self) -> bool:
        """
        Load template/demo data from files.
        
        In a real implementation, this would load your domain-specific data
        from CSV files, databases, APIs, etc.
        """
        try:
            # Try to load main dataset
            main_csv_path = os.path.join(self.data_path, "main_data.csv")
            
            if os.path.exists(main_csv_path):
                self.main_data = pd.read_csv(main_csv_path)
                print(f"✅ Loaded {len(self.main_data)} records from main_data.csv")
            else:
                # Create sample data if no file exists
                print("📝 No data file found, creating sample dataset")
                self.main_data = self._create_sample_data()
                
                # Save sample data for future use
                os.makedirs(self.data_path, exist_ok=True)
                self.main_data.to_csv(main_csv_path, index=False)
                print(f"💾 Saved sample data to {main_csv_path}")
            
            # Load metadata if available
            metadata_path = os.path.join(self.data_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print("✅ Loaded metadata")
            else:
                self.metadata = {
                    "data_source": "Template Demo Data",
                    "last_updated": "2024-01-15",
                    "version": "1.0.0"
                }
                
                # Save metadata
                with open(metadata_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                print("💾 Created default metadata")
            
            # Validate data
            if self.main_data.empty:
                print("⚠️ Warning: Dataset is empty")
                return False
            
            required_columns = ['id', 'name', 'category', 'value']
            missing_columns = [col for col in required_columns if col not in self.main_data.columns]
            if missing_columns:
                print(f"❌ Error: Missing required columns: {missing_columns}")
                return False
            
            self.is_loaded = True
            print(f"🎉 Data loading completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration"""
        import random
        from datetime import datetime, timedelta
        
        categories = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Education']
        
        data = []
        for i in range(100):
            data.append({
                'id': f"ITEM_{i:03d}",
                'name': f"Sample Item {i+1}",
                'category': random.choice(categories),
                'value': random.randint(10, 1000),
                'description': f"This is a sample description for item {i+1}",
                'status': random.choice(['active', 'inactive', 'pending']),
                'created_date': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
            })
        
        return pd.DataFrame(data)
    
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """
        Search through the template data.
        
        In a real implementation, customize this for your domain's search needs.
        """
        if not self.is_loaded:
            return []
        
        try:
            # Start with all data
            results = self.main_data.copy()
            
            # Apply text search
            if query.strip():
                mask = (
                    results['name'].str.contains(query, case=False, na=False) |
                    results['description'].str.contains(query, case=False, na=False) |
                    results['category'].str.contains(query, case=False, na=False)
                )
                results = results[mask]
            
            # Apply filters
            if filters:
                cleaned_filters = self.validate_filters(filters)
                
                for key, value in cleaned_filters.items():
                    if key in results.columns:
                        if key == 'min_value':
                            results = results[results['value'] >= value]
                        elif key == 'max_value':
                            results = results[results['value'] <= value]
                        elif key == 'category':
                            results = results[results['category'] == value]
                        elif key == 'status':
                            results = results[results['status'] == value]
                        else:
                            # Exact match for other fields
                            results = results[results[key] == value]
            
            # Limit results and convert to list of dictionaries
            limited_results = results.head(50).to_dict('records')
            
            print(f"🔍 Search '{query}' with filters {filters}: {len(limited_results)} results")
            return limited_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return comprehensive statistics about the template dataset.
        
        Customize this for your domain's metrics.
        """
        if not self.is_loaded:
            return {"error": "Data not loaded"}
        
        try:
            stats = {
                "total_records": len(self.main_data),
                "categories": {
                    "unique_count": self.main_data['category'].nunique(),
                    "list": list(self.main_data['category'].unique())
                },
                "status_distribution": self.main_data['status'].value_counts().to_dict(),
                "value_stats": {
                    "min": float(self.main_data['value'].min()),
                    "max": float(self.main_data['value'].max()),
                    "mean": float(self.main_data['value'].mean()),
                    "median": float(self.main_data['value'].median())
                },
                "data_quality": {
                    "complete_records": len(self.main_data.dropna()),
                    "missing_descriptions": len(self.main_data[self.main_data['description'].isna()]),
                    "completeness_percent": (len(self.main_data.dropna()) / len(self.main_data)) * 100
                },
                "metadata": self.metadata
            }
            
            # Add date range if created_date column exists
            if 'created_date' in self.main_data.columns:
                try:
                    dates = pd.to_datetime(self.main_data['created_date'])
                    stats["date_range"] = {
                        "earliest": dates.min().isoformat(),
                        "latest": dates.max().isoformat(),
                        "span_days": (dates.max() - dates.min()).days
                    }
                except:
                    stats["date_range"] = {"error": "Invalid date format"}
            
            return stats
            
        except Exception as e:
            print(f"❌ Statistics calculation failed: {e}")
            return {"error": f"Statistics calculation failed: {str(e)}"}
    
    def get_category_breakdown(self, category: str) -> List[Dict]:
        """
        Custom method specific to template data.
        Add methods like this for domain-specific functionality.
        """
        if not self.is_loaded:
            return []
        
        try:
            category_data = self.main_data[
                self.main_data['category'].str.lower() == category.lower()
            ]
            return category_data.to_dict('records')
        except Exception as e:
            print(f"❌ Category breakdown failed: {e}")
            return []

class TemplateServer(BaseMCPServer, HealthCheckMixin):
    """
    Complete example MCP server implementation using the abstract base classes.
    
    This demonstrates:
    - How to use BaseMCPServer and BaseDataManager
    - How to implement domain-specific tools
    - How to provide proper citation information
    - How to add health checking capabilities
    
    To create your own server:
    1. Copy this file to your_domain_server.py
    2. Rename TemplateServer to YourDomainServer
    3. Update metadata with your domain info
    4. Replace TemplateDataManager with your data manager
    5. Replace tools with your domain-specific tools
    6. Update citation mappings for your data sources
    """
    
    def __init__(self):
        super().__init__("template-server")
        
        # Initialize data manager
        data_path = os.getenv("TEMPLATE_DATA_PATH", "data/template/")
        self.data_manager = TemplateDataManager(data_path)
        
        # Load data on startup
        if not self.data_manager.load_data():
            print("⚠️ Warning: Data loading failed, server will have limited functionality")
        
        # Add health check tool from mixin
        self.add_health_check_tool()
        
    def get_metadata(self) -> Dict[str, str]:
        return {
            "Name": "Template MCP Server",
            "Description": "Example server implementation for the ODE MCP framework",
            "Version": "1.0.0",
            "Author": "ODE Framework Team"
        }
    
    def setup_tools(self):
        """
        Register all MCP tools for this server.
        
        This demonstrates various patterns:
        - Simple data retrieval tools
        - Search tools with filters
        - Statistical analysis tools
        - Custom domain-specific tools
        """
        
        @self.mcp.tool()
        def GetTemplateData(limit: int = 10) -> Dict[str, Any]:
            """Get a sample of template data records"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Data not loaded"}
                
                # Get sample data
                sample_data = self.data_manager.main_data.head(limit).to_dict('records')
                
                return {
                    "data": sample_data,
                    "total_available": len(self.data_manager.main_data),
                    "returned_count": len(sample_data),
                    "data_source": "template_demo"
                }
                
            except Exception as e:
                return {"error": f"Failed to get template data: {str(e)}"}
        
        @self.mcp.tool()
        def SearchTemplateData(query: str, category: Optional[str] = None, 
                             min_value: Optional[int] = None, max_value: Optional[int] = None,
                             status: Optional[str] = None) -> Dict[str, Any]:
            """Search template data with optional filters"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Data not loaded"}
                
                # Build filters dictionary
                filters = {}
                if category:
                    filters['category'] = category
                if min_value is not None:
                    filters['min_value'] = min_value
                if max_value is not None:
                    filters['max_value'] = max_value
                if status:
                    filters['status'] = status
                
                # Perform search
                results = self.data_manager.search(query, filters)
                
                return {
                    "query": query,
                    "filters": filters,
                    "results": results,
                    "result_count": len(results),
                    "data_source": "template_demo"
                }
                
            except Exception as e:
                return {"error": f"Search failed: {str(e)}"}
        
        @self.mcp.tool()
        def GetTemplateStatistics() -> Dict[str, Any]:
            """Get comprehensive statistics about the template dataset"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Data not loaded"}
                
                stats = self.data_manager.get_statistics()
                
                return {
                    "statistics": stats,
                    "data_source": "template_demo"
                }
                
            except Exception as e:
                return {"error": f"Statistics calculation failed: {str(e)}"}
        
        @self.mcp.tool()
        def GetCategoryBreakdown(category: str) -> Dict[str, Any]:
            """Get detailed breakdown for a specific category"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Data not loaded"}
                
                breakdown = self.data_manager.get_category_breakdown(category)
                
                return {
                    "category": category,
                    "items": breakdown,
                    "item_count": len(breakdown),
                    "data_source": "template_demo"
                }
                
            except Exception as e:
                return {"error": f"Category breakdown failed: {str(e)}"}
        
        @self.mcp.tool()
        def GetAvailableCategories() -> Dict[str, Any]:
            """Get list of all available categories in the dataset"""
            try:
                if not self.data_manager.is_data_loaded():
                    return {"error": "Data not loaded"}
                
                categories = list(self.data_manager.main_data['category'].unique())
                category_counts = self.data_manager.main_data['category'].value_counts().to_dict()
                
                return {
                    "categories": categories,
                    "category_counts": category_counts,
                    "total_categories": len(categories),
                    "data_source": "template_demo"
                }
                
            except Exception as e:
                return {"error": f"Failed to get categories: {str(e)}"}
    
    def get_citation_info(self, tool_name: str, tool_args: Dict = None) -> Dict:
        """
        Provide citation information for each tool.
        
        This maps each tool to its data source for proper attribution.
        Update this with your actual data sources.
        """
        citation_mapping = {
            "GetTemplateData": {
                "source_name": "Template Demo Dataset v1.0",
                "provider": "ODE Framework",
                "spatial_coverage": "Synthetic/Demo data",
                "temporal_coverage": "2024 (generated)",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "SearchTemplateData": {
                "source_name": "Template Demo Dataset v1.0",
                "provider": "ODE Framework", 
                "spatial_coverage": "Synthetic/Demo data",
                "temporal_coverage": "2024 (generated)",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "GetTemplateStatistics": {
                "source_name": "Template Demo Dataset Statistics",
                "provider": "ODE Framework",
                "spatial_coverage": "Aggregate statistics",
                "temporal_coverage": "Current dataset state",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "GetCategoryBreakdown": {
                "source_name": "Template Demo Dataset v1.0",
                "provider": "ODE Framework",
                "spatial_coverage": "Category-filtered data",
                "temporal_coverage": "2024 (generated)",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "GetAvailableCategories": {
                "source_name": "Template Demo Dataset Metadata",
                "provider": "ODE Framework",
                "spatial_coverage": "Dataset schema information",
                "temporal_coverage": "Current dataset state",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            },
            "HealthCheck": {
                "source_name": "Template Server Health Monitor",
                "provider": "ODE Framework",
                "spatial_coverage": "Server status information",
                "temporal_coverage": "Real-time",
                "source_url": "https://github.com/ode-pbllc/ode-mcp-generic"
            }
        }
        
        return citation_mapping.get(tool_name, self._default_citation())

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Starting Template Server")
    print("")
    print("This is a complete example implementation showing:")
    print("  ✅ Abstract base class usage")
    print("  ✅ Data management patterns") 
    print("  ✅ Tool registration")
    print("  ✅ Citation handling")
    print("  ✅ Error handling")
    print("  ✅ Health checking")
    print("")
    print("To create your own server:")
    print("  1. Copy this file to your_domain_server.py")
    print("  2. Replace TemplateDataManager with your data manager")
    print("  3. Update tools for your domain")
    print("  4. Update citation mappings")
    print("  5. Test and deploy!")
    print("")
    
    # Run the server
    server = TemplateServer()
    
    print("📊 Template Server Status:")
    print(f"  Data loaded: {server.data_manager.is_data_loaded()}")
    if server.data_manager.is_data_loaded():
        stats = server.data_manager.get_statistics()
        print(f"  Total records: {stats.get('total_records', 'unknown')}")
        print(f"  Categories: {stats.get('categories', {}).get('unique_count', 'unknown')}")
    
    print("")
    print("🔧 Ready to accept MCP connections!")
    
    # Start the server
    server.run()