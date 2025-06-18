# GIST MCP Server Implementation Plan

## Overview
Create a comprehensive MCP server `gist_server.py` that provides access to the complete GIST (Global Infrastructure Sustainability Toolkit) dataset with 6 sheets containing both schema and extensive environmental/sustainability data for 100 companies.

## Data Structure Understanding

### 6 Excel Sheets:
1. **Data Dictionary** (168 rows, 4 cols) - Schema definitions for all datasets
2. **EXSITU** (68 rows, 75 cols) - Company-level aggregated environmental risk data  
3. **EXSITU_ASSET_DATA** (40,978 rows, 36 cols) - Individual asset-level environmental risk data with coordinates
4. **DEFORESTATION** (64 rows, 5 cols) - Company deforestation risk indicators
5. **SCOPE_3_DATA** (779 rows, 23 cols) - Multi-year Scope 3 emissions data (2016-2024)
6. **BIODIVERSITY_PDF_DATA** (779 rows, 37 cols) - Multi-year biodiversity impact data (PDF, CO2E, LCE metrics)

### Key Data Characteristics:
- **100 unique companies** across all datasets (64 companies appear in ALL datasets)
- **40,978 individual assets** with geographic coordinates (lat/lng) 
- **9 years** of time series data (2016-2024) for emissions and biodiversity
- **Geographic coverage**: Primarily Brazil (37K+ assets), US, GB, CA, AR
- **Sector coverage**: OGES, FINS, WHRE, MOMI, REEN (Oil/Gas, Finance, Wholesale/Retail, Mining, Renewable Energy)

## MCP Server Tools Implementation

### 1. Schema/Dictionary Tools
- [ ] **`GetGistDataDictionary(dataset_name=None)`** - Get field definitions from Data Dictionary sheet
- [ ] **`SearchGistFields(search_term)`** - Search across all field definitions
- [ ] **`GetGistDatasetSchemas()`** - List all available datasets with field counts

### 2. Company Discovery & Analysis Tools  
- [ ] **`GetGistCompanies(sector=None, country=None)`** - List companies with filtering options
- [ ] **`GetGistCompanyProfile(company_code)`** - Complete profile across all datasets for a company
- [ ] **`GetGistCompaniesBySector()`** - Companies grouped by sector with counts
- [ ] **`GetGistCompanyDataAvailability(company_code)`** - Show which datasets contain data for a company

### 3. Environmental Risk Analysis Tools
- [ ] **`GetGistCompanyRisks(company_code)`** - Environmental risk summary for a company (EXSITU data)
- [ ] **`GetGistRiskByCategory(risk_type)`** - Companies by risk level (MSA, water stress, floods, etc.)
- [ ] **`GetGistHighRiskCompanies(risk_threshold=0.7)`** - Companies with highest environmental risks
- [ ] **`GetGistAssetRisks(company_code, asset_limit=100)`** - Asset-level risks for a company

### 4. Geographic/Asset Analysis Tools
- [ ] **`GetGistAssetsMapData(company_code=None, country=None, limit=1000)`** - Asset coordinates for mapping
- [ ] **`GetGistAssetsInRadius(latitude, longitude, radius_km)`** - Find assets near coordinates  
- [ ] **`GetGistAssetsByCountry()`** - Asset distribution by country
- [ ] **`GetGistAssetDetails(asset_id)`** - Detailed info for specific asset

### 5. Emissions & Carbon Analysis Tools
- [ ] **`GetGistScope3Emissions(company_code, year=None)`** - Scope 3 emissions data
- [ ] **`GetGistEmissionsTrends(company_code)`** - Multi-year emissions trends
- [ ] **`GetGistEmissionsBySector(year=None)`** - Sector-level emissions comparison
- [ ] **`GetGistTopEmitters(limit=20, year=None)`** - Highest emitting companies

### 6. Biodiversity Impact Tools
- [ ] **`GetGistBiodiversityImpacts(company_code, year=None)`** - Biodiversity footprint data (PDF/CO2E/LCE)
- [ ] **`GetGistBiodiversityTrends(company_code)`** - Multi-year biodiversity impact trends
- [ ] **`GetGistBiodiversityBySector(year=None)`** - Sector biodiversity comparison
- [ ] **`GetGistBiodiversityWorstPerformers(metric='PDF', limit=20)`** - Highest impact companies

### 7. Deforestation Analysis Tools
- [ ] **`GetGistDeforestationRisks(company_code=None)`** - Deforestation proximity indicators
- [ ] **`GetGistDeforestationExposed()`** - Companies with high deforestation exposure
- [ ] **`GetGistForestChangeProximity()`** - Analysis of forest change proximity across companies

### 8. Time Series & Trend Tools
- [ ] **`GetGistYearlyTrends(metric, company_code=None)`** - Multi-year trends for key metrics
- [ ] **`GetGistCompanyEvolution(company_code)`** - Company's sustainability evolution over time
- [ ] **`GetGistSectorTrends(sector_code, metric)`** - Sector-level trends over time

### 9. Comparative Analysis Tools
- [ ] **`CompareGistCompanies(company_codes)`** - Side-by-side comparison of companies
- [ ] **`GetGistSectorBenchmarks(sector_code)`** - Sector averages and benchmarks
- [ ] **`GetGistPerformanceRankings(metric, sector=None)`** - Ranked performance lists

### 10. Visualization Data Tools
- [ ] **`GetGistVisualizationData(viz_type, filters={})`** - Structured data for charts
  - viz_types: "emissions_by_sector", "risk_distribution", "asset_map", "biodiversity_trends", "scope3_breakdown"
- [ ] **`GetGistDashboardData(company_code)`** - Complete dashboard data for a company
- [ ] **`GetGistSectorDashboard(sector_code)`** - Sector-level dashboard data

## Technical Implementation Strategy

### Core Data Manager Class:
```python
class GistDataManager:
    def __init__(self):
        self.excel_file = pd.ExcelFile(GIST_FILE_PATH)
        self.schemas = self._load_schemas()
        self.companies_cache = self._build_company_index()
        self.assets_cache = self._build_asset_index()
        
    def _load_all_sheets(self):
        return {name: pd.read_excel(self.excel_file, sheet_name=name) 
                for name in self.excel_file.sheet_names}
```

### Key Implementation Features:
- [ ] Multi-sheet data integration across all 6 datasets
- [ ] Time series analysis for 2016-2024 data
- [ ] Geographic search capabilities using 40K+ asset coordinates  
- [ ] Cross-dataset company linking using COMPANY_CODE
- [ ] Risk assessment across multiple environmental dimensions
- [ ] Sector benchmarking and comparative analysis
- [ ] Sustainability trend analysis over 9-year period
- [ ] Caching for performance with large datasets

### Integration Requirements:
- [ ] Add to `mcp_chat.py` as "gist" server connection
- [ ] Update system prompts to mention comprehensive environmental and sustainability data
- [ ] Enable cross-referencing with climate policy data from KG server
- [ ] Support visualization through response formatter for maps, charts, trends

## Implementation Priority

### Phase 1: Core Infrastructure (High Priority)
1. [ ] Set up basic server structure with FastMCP
2. [ ] Implement GistDataManager class with data loading
3. [ ] Create core company discovery tools
4. [ ] Implement schema/dictionary tools

### Phase 2: Environmental Analysis (High Priority)  
5. [ ] Build environmental risk analysis tools
6. [ ] Implement geographic/asset analysis tools
7. [ ] Create emissions analysis tools
8. [ ] Add biodiversity impact tools

### Phase 3: Advanced Analytics (Medium Priority)
9. [ ] Implement time series and trend tools
10. [ ] Build comparative analysis tools
11. [ ] Create visualization data tools
12. [ ] Add deforestation analysis tools

### Phase 4: Integration & Testing (Medium Priority)
13. [ ] Integrate with existing MCP infrastructure
14. [ ] Update system prompts and chat integration
15. [ ] Test cross-dataset functionality
16. [ ] Performance optimization for large datasets

## Value Proposition
This comprehensive server provides:
1. **Complete corporate sustainability profiles** across environmental, emissions, and biodiversity dimensions
2. **Asset-level risk analysis** with geographic mapping capabilities  
3. **Time series tracking** of sustainability performance over 9 years
4. **Sector benchmarking** and comparative analysis
5. **Deforestation and biodiversity risk assessment**
6. **Comprehensive Scope 3 emissions analysis**
7. **Multi-dimensional environmental risk evaluation**

The server enables deep sustainability analysis, ESG reporting support, and environmental risk assessment for financial and policy decision-making.