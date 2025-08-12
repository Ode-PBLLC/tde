# Data Directory

This directory is for domain-specific data files used by your MCP servers.

## Important Notes

⚠️ **This directory is typically not included in the generic repository.**

For security and size reasons, actual data files should be added when you fork this repository for your specific domain.

## Recommended Structure

```
data/
├── your_domain/
│   ├── main_dataset.csv
│   ├── metadata.json
│   ├── supplementary_data.csv
│   └── raw/
│       ├── source_file_1.csv
│       ├── source_file_2.json
│       └── external_api_dump.json
├── processed/
│   ├── cleaned_data.csv
│   ├── aggregated_stats.json
│   └── indexed_data.db
└── temp/
    ├── processing_cache.json
    └── temp_analysis_results.csv
```

## Data Management Best Practices

### 1. Organization
- Use descriptive directory names for your domain
- Separate raw data from processed data  
- Keep original source files in `raw/` subdirectories
- Use consistent naming conventions

### 2. File Formats
- **CSV**: For tabular data (most common)
- **JSON**: For structured/nested data
- **SQLite**: For large datasets requiring queries
- **Excel**: If necessary, but CSV preferred for performance

### 3. Metadata
Always include metadata files describing:
```json
{
  "dataset_name": "Your Dataset Name",
  "version": "1.0.0",
  "last_updated": "2024-01-15",
  "source": "Original data source URL",
  "description": "Brief description of the data",
  "columns": {
    "column_name": "Description of what this column contains"
  },
  "record_count": 10000,
  "data_quality": {
    "completeness": 95.5,
    "notes": ["Some records missing geographic coordinates"]
  }
}
```

### 4. Data Loading Examples

#### CSV Data Loading
```python
class YourDataManager(BaseDataManager):
    def load_data(self) -> bool:
        try:
            # Load main dataset
            self.main_data = pd.read_csv(f"{self.data_path}/main_dataset.csv")
            
            # Load supplementary data
            self.supplementary = pd.read_csv(f"{self.data_path}/supplementary_data.csv")
            
            # Validate required columns
            required_cols = ['id', 'name', 'value']
            if not all(col in self.main_data.columns for col in required_cols):
                return False
            
            return True
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False
```

#### JSON Data Loading
```python
def load_json_data(self) -> bool:
    try:
        with open(f"{self.data_path}/structured_data.json", 'r') as f:
            self.json_data = json.load(f)
        return True
    except Exception as e:
        print(f"JSON loading failed: {e}")
        return False
```

## Security Considerations

### ✅ Safe to Include
- Sample/demo data (small files)
- Test datasets
- Configuration files
- Schema definitions

### ❌ Never Include  
- Large datasets (>10MB)
- Sensitive personal information
- Proprietary data
- API keys or credentials
- Temporary/cache files

## Environment Variables

Configure data paths using environment variables:

```bash
# .env file
YOUR_DOMAIN_DATA_PATH=data/your_domain/
DATABASE_PATH=data/processed/your_data.db
CACHE_PATH=data/temp/
```

```python
# In your data manager
data_path = os.getenv("YOUR_DOMAIN_DATA_PATH", "data/your_domain/")
```

## Data Sources Examples

### Finance Domain
```
data/finance/
├── stocks.csv              # Stock price data
├── companies.csv           # Company information  
├── market_indices.csv      # Market index data
├── earnings_reports.json   # Quarterly earnings
└── metadata.json           # Dataset documentation
```

### Environmental Domain
```
data/environmental/
├── air_quality.csv         # Air quality measurements
├── water_quality.csv       # Water quality data
├── monitoring_stations.csv # Sensor locations
├── weather_data.csv        # Historical weather
└── metadata.json           # Dataset documentation
```

### Healthcare Domain  
```
data/healthcare/
├── clinical_trials.csv     # Clinical trial data
├── drug_database.csv       # FDA approved drugs
├── medical_devices.json    # Device information
├── research_publications.csv # Research papers
└── metadata.json           # Dataset documentation
```

## Getting Data Into Your Fork

### Option 1: Manual Addition
1. Fork the generic repository
2. Add your data files to appropriate directories
3. Commit and push your data

### Option 2: Download Scripts
Create scripts to download data during setup:

```python
# scripts/download_domain_data.py
import requests
import pandas as pd

def download_stock_data():
    # Download from API
    data = requests.get("https://api.example.com/stocks").json()
    df = pd.DataFrame(data)
    df.to_csv("data/finance/stocks.csv", index=False)

if __name__ == "__main__":
    download_stock_data()
```

### Option 3: Environment-Based Loading
Load data from external sources based on environment configuration:

```python
def load_data(self):
    data_source = os.getenv("DATA_SOURCE", "local")
    
    if data_source == "s3":
        return self._load_from_s3()
    elif data_source == "database":
        return self._load_from_database()
    else:
        return self._load_from_local_files()
```

## Testing Data

Always include small test datasets for development and testing:

```
data/test/
├── sample_data.csv         # Small sample (10-100 records)
├── test_metadata.json      # Test configuration
└── invalid_data.csv        # For testing error handling
```

## Setup Checklist

- [ ] Create domain-specific directory structure
- [ ] Add metadata.json files for all datasets
- [ ] Set up data loading in your DataManager class
- [ ] Configure environment variables for data paths
- [ ] Add small test datasets
- [ ] Update .gitignore for sensitive/large files
- [ ] Document data sources and update procedures