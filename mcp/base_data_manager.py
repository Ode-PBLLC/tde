"""
ODE MCP Generic - Abstract Base Data Manager Class
Provides consistent patterns for managing domain-specific data.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import json
import os

class BaseDataManager(ABC):
    """
    Abstract base class for managing domain-specific data.
    
    This class provides a consistent interface for data loading, searching,
    and statistics across different domains.
    
    Example implementation for Environmental Data:
    
    ```python
    import pandas as pd
    from base_data_manager import BaseDataManager
    
    class EnvironmentalDataManager(BaseDataManager):
        def __init__(self, data_path: str):
            super().__init__(data_path)
            self.air_quality_data = pd.DataFrame()
            self.water_quality_data = pd.DataFrame()
        
        def load_data(self) -> bool:
            try:
                # Load multiple datasets
                self.air_quality_data = pd.read_csv(f"{self.data_path}/air_quality.csv")
                self.water_quality_data = pd.read_csv(f"{self.data_path}/water_quality.csv")
                
                # Validate data
                if self.air_quality_data.empty or self.water_quality_data.empty:
                    print("Warning: Some datasets are empty")
                    return False
                
                print(f"Loaded {len(self.air_quality_data)} air quality records")
                print(f"Loaded {len(self.water_quality_data)} water quality records")
                return True
                
            except Exception as e:
                print(f"Failed to load environmental data: {e}")
                return False
        
        def search(self, query: str, filters: Dict = None) -> List[Dict]:
            results = []
            
            # Search air quality data
            if not filters or filters.get("data_type") in [None, "air"]:
                air_matches = self.air_quality_data[
                    self.air_quality_data['location'].str.contains(query, case=False, na=False)
                ]
                for _, row in air_matches.iterrows():
                    results.append({
                        "type": "air_quality",
                        "location": row['location'],
                        "pollutant": row.get('pollutant', 'Unknown'),
                        "value": row.get('value', 0),
                        "date": row.get('date', 'Unknown')
                    })
            
            # Search water quality data
            if not filters or filters.get("data_type") in [None, "water"]:
                water_matches = self.water_quality_data[
                    self.water_quality_data['location'].str.contains(query, case=False, na=False)
                ]
                for _, row in water_matches.iterrows():
                    results.append({
                        "type": "water_quality",
                        "location": row['location'],
                        "parameter": row.get('parameter', 'Unknown'),
                        "value": row.get('value', 0),
                        "date": row.get('date', 'Unknown')
                    })
            
            # Apply additional filters
            if filters:
                if 'min_value' in filters:
                    results = [r for r in results if r.get('value', 0) >= filters['min_value']]
                if 'max_value' in filters:
                    results = [r for r in results if r.get('value', 0) <= filters['max_value']]
                if 'date_from' in filters:
                    # Date filtering logic here
                    pass
            
            return results[:100]  # Limit results
        
        def get_statistics(self) -> Dict[str, Any]:
            return {
                "air_quality_records": len(self.air_quality_data),
                "water_quality_records": len(self.water_quality_data),
                "total_records": len(self.air_quality_data) + len(self.water_quality_data),
                "date_range": {
                    "air_quality_start": self.air_quality_data['date'].min() if not self.air_quality_data.empty else None,
                    "air_quality_end": self.air_quality_data['date'].max() if not self.air_quality_data.empty else None,
                    "water_quality_start": self.water_quality_data['date'].min() if not self.water_quality_data.empty else None,
                    "water_quality_end": self.water_quality_data['date'].max() if not self.water_quality_data.empty else None
                },
                "locations_covered": len(set(
                    list(self.air_quality_data['location'].unique()) + 
                    list(self.water_quality_data['location'].unique())
                )),
                "data_types": ["air_quality", "water_quality"]
            }
        
        def get_air_quality_by_location(self, location: str) -> List[Dict]:
            '''Custom method specific to environmental data'''
            matches = self.air_quality_data[
                self.air_quality_data['location'].str.contains(location, case=False, na=False)
            ]
            return matches.to_dict('records')
        
        def get_water_quality_by_location(self, location: str) -> List[Dict]:
            '''Custom method specific to environmental data'''
            matches = self.water_quality_data[
                self.water_quality_data['location'].str.contains(location, case=False, na=False)
            ]
            return matches.to_dict('records')
    
    # Usage:
    data_manager = EnvironmentalDataManager("data/environmental/")
    if data_manager.load_data():
        results = data_manager.search("Los Angeles", {"data_type": "air"})
        stats = data_manager.get_statistics()
    ```
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data manager.
        
        Args:
            data_path: Path to the directory containing domain data files
        """
        self.data_path = data_path
        self.is_loaded = False
        
        # Ensure data directory exists
        if not os.path.exists(data_path):
            print(f"Warning: Data path {data_path} does not exist")
            
    @abstractmethod
    def load_data(self) -> bool:
        """
        Load domain-specific data from files/database/API.
        
        Called once during server startup to initialize all data sources.
        Should set self.is_loaded = True on success.
        
        Implementation should:
        1. Load all necessary data files (CSV, JSON, database, etc.)
        2. Perform basic validation
        3. Handle missing files gracefully
        4. Return True if successful, False if failed
        5. Print status messages for debugging
        
        Example:
        ```python
        def load_data(self) -> bool:
            try:
                # Load main dataset
                self.companies_data = pd.read_csv(f"{self.data_path}/companies.csv")
                
                # Load supplementary data
                if os.path.exists(f"{self.data_path}/stock_prices.csv"):
                    self.stock_prices = pd.read_csv(f"{self.data_path}/stock_prices.csv")
                else:
                    print("Warning: Stock prices file not found")
                    self.stock_prices = pd.DataFrame()
                
                # Validate required columns
                required_cols = ['company_name', 'sector', 'market_cap']
                if not all(col in self.companies_data.columns for col in required_cols):
                    print("Error: Missing required columns in companies data")
                    return False
                
                print(f"Successfully loaded {len(self.companies_data)} companies")
                self.is_loaded = True
                return True
                
            except Exception as e:
                print(f"Data loading failed: {e}")
                return False
        ```
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """
        Search across your data with optional filters.
        
        This is the primary interface for finding relevant data based on
        user queries. Should be fast and return consistent results.
        
        Args:
            query: Text to search for (company names, locations, keywords, etc.)
            filters: Optional dict of filters to narrow results:
                    - {"sector": "technology", "min_employees": 1000}
                    - {"country": "USA", "year": 2024}
                    - {"category": "renewable", "min_capacity": 100}
        
        Implementation guidelines:
        1. Search across primary text fields (names, descriptions)
        2. Use case-insensitive matching
        3. Handle empty/null values gracefully
        4. Apply filters if provided
        5. Limit results to reasonable number (50-100)
        6. Return consistent dictionary format
        
        Example:
        ```python
        def search(self, query: str, filters: Dict = None) -> List[Dict]:
            if not self.is_loaded:
                return []
            
            # Search company names and descriptions
            results = self.companies_data[
                (self.companies_data['company_name'].str.contains(query, case=False, na=False)) |
                (self.companies_data['description'].str.contains(query, case=False, na=False))
            ]
            
            # Apply filters
            if filters:
                if 'sector' in filters:
                    results = results[results['sector'] == filters['sector']]
                if 'min_market_cap' in filters:
                    results = results[results['market_cap'] >= filters['min_market_cap']]
                if 'country' in filters:
                    results = results[results['country'] == filters['country']]
            
            # Convert to list of dictionaries and limit
            return results.head(50).to_dict('records')
        ```
        
        Returns:
            List of dictionaries containing matching records
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return summary statistics about your dataset.
        
        Used for health checks, debugging, and providing users with
        data coverage information. Should be fast to compute.
        
        Should include:
        - Record counts
        - Date ranges (if temporal data)
        - Geographic coverage (if spatial data)
        - Categories/types available
        - Data quality metrics
        
        Example:
        ```python
        def get_statistics(self) -> Dict[str, Any]:
            if not self.is_loaded:
                return {"error": "Data not loaded"}
            
            return {
                "total_companies": len(self.companies_data),
                "sectors": list(self.companies_data['sector'].unique()),
                "countries": list(self.companies_data['country'].unique()),
                "market_cap_range": {
                    "min": self.companies_data['market_cap'].min(),
                    "max": self.companies_data['market_cap'].max(),
                    "median": self.companies_data['market_cap'].median()
                },
                "date_range": {
                    "start": self.stock_prices['date'].min() if not self.stock_prices.empty else None,
                    "end": self.stock_prices['date'].max() if not self.stock_prices.empty else None
                },
                "data_quality": {
                    "companies_with_missing_sector": len(self.companies_data[self.companies_data['sector'].isna()]),
                    "companies_with_missing_market_cap": len(self.companies_data[self.companies_data['market_cap'].isna()])
                },
                "last_updated": "2024-01-15"  # When data was last refreshed
            }
        ```
        
        Returns:
            Dictionary with comprehensive dataset statistics
        """
        pass
    
    def is_data_loaded(self) -> bool:
        """Check if data has been successfully loaded"""
        return self.is_loaded
    
    def get_data_path(self) -> str:
        """Get the current data path"""
        return self.data_path
    
    def validate_filters(self, filters: Dict) -> Dict:
        """
        Validate and clean filter parameters.
        Override this method to add domain-specific filter validation.
        
        Args:
            filters: Raw filters from user input
            
        Returns:
            Cleaned and validated filters
        """
        if not filters:
            return {}
        
        # Basic validation - override for domain-specific logic
        cleaned = {}
        for key, value in filters.items():
            if value is not None and value != '':
                cleaned[key] = value
                
        return cleaned

class CSVDataManager(BaseDataManager):
    """
    Concrete implementation for simple CSV-based data.
    Use this for basic datasets or as a starting point.
    """
    
    def __init__(self, data_path: str, main_csv_file: str):
        super().__init__(data_path)
        self.main_csv_file = main_csv_file
        self.data = pd.DataFrame()
        
    def load_data(self) -> bool:
        try:
            csv_path = os.path.join(self.data_path, self.main_csv_file)
            if not os.path.exists(csv_path):
                print(f"Error: CSV file not found: {csv_path}")
                return False
                
            self.data = pd.read_csv(csv_path)
            
            if self.data.empty:
                print(f"Warning: CSV file is empty: {csv_path}")
                return False
                
            print(f"Loaded {len(self.data)} records from {self.main_csv_file}")
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to load CSV data: {e}")
            return False
    
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        if not self.is_loaded:
            return []
        
        # Search all string columns
        mask = False
        for col in self.data.select_dtypes(include=['object']).columns:
            mask = mask | self.data[col].str.contains(query, case=False, na=False)
        
        results = self.data[mask]
        
        # Apply basic filters
        if filters:
            for key, value in filters.items():
                if key in results.columns:
                    results = results[results[key] == value]
        
        return results.head(100).to_dict('records')
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"error": "Data not loaded"}
        
        stats = {
            "total_records": len(self.data),
            "columns": list(self.data.columns),
            "file_name": self.main_csv_file
        }
        
        # Add column type information
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                unique_values = self.data[col].nunique()
                stats[f"{col}_unique_values"] = unique_values
        
        return stats

class JSONDataManager(BaseDataManager):
    """
    Concrete implementation for JSON-based data.
    Useful for APIs or structured document data.
    """
    
    def __init__(self, data_path: str, json_files: List[str]):
        super().__init__(data_path)
        self.json_files = json_files
        self.data = []
        
    def load_data(self) -> bool:
        try:
            self.data = []
            
            for json_file in self.json_files:
                json_path = os.path.join(self.data_path, json_file)
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            self.data.extend(file_data)
                        else:
                            self.data.append(file_data)
                    print(f"Loaded {json_file}")
                else:
                    print(f"Warning: JSON file not found: {json_path}")
            
            if not self.data:
                print("Error: No data loaded from JSON files")
                return False
                
            print(f"Total records loaded: {len(self.data)}")
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to load JSON data: {e}")
            return False
    
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        if not self.is_loaded:
            return []
        
        results = []
        query_lower = query.lower()
        
        for record in self.data:
            # Search in all string values
            match = False
            for key, value in record.items():
                if isinstance(value, str) and query_lower in value.lower():
                    match = True
                    break
            
            if match:
                # Apply filters
                if filters:
                    filter_match = True
                    for filter_key, filter_value in filters.items():
                        if record.get(filter_key) != filter_value:
                            filter_match = False
                            break
                    if filter_match:
                        results.append(record)
                else:
                    results.append(record)
        
        return results[:100]
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"error": "Data not loaded"}
        
        # Analyze structure
        all_keys = set()
        for record in self.data:
            all_keys.update(record.keys())
        
        return {
            "total_records": len(self.data),
            "files_loaded": self.json_files,
            "unique_fields": list(all_keys),
            "average_fields_per_record": sum(len(record) for record in self.data) / len(self.data) if self.data else 0
        }

# Example usage and testing
if __name__ == "__main__":
    # Example CSV manager
    csv_manager = CSVDataManager("data/", "sample_data.csv")
    
    # Example JSON manager  
    json_manager = JSONDataManager("data/", ["config.json", "metadata.json"])
    
    print("Data manager classes ready for implementation")