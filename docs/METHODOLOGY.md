# Methodology

## System Architecture

Transition Digital AI combines multiple data sources through a Model Context Protocol (MCP) server architecture. The system uses AI-powered Retrieval Augmented Generation (RAG) to intelligently query and synthesize information from diverse climate-related datasets. We maintain consistent references and 

### How It Works

1. **Query Processing**: User queries are processed by an AI model that determines which data sources and tools are most relevant.

2. **Dynamic Tool Selection**: Our MCP server provides access to specialized tools for each dataset, including:
   - Knowledge graph queries for policy documents
   - Geospatial queries for facility locations
   - Statistical analysis for trends and comparisons

3. **RAG Integration**: The system uses semantic search and knowledge graph traversal to find relevant information, then augments responses with real-time data from multiple sources.

4. **Response Synthesis**: Results from different tools are combined into a coherent response with appropriate visualizations (charts, maps, tables) and inline citations.

## Data Sources

### Climate Policy Radar Knowledge Graph
A comprehensive knowledge graph containing climate policies, laws, and regulations from around the world. This dataset enables semantic search across policy documents, identification of related concepts, and extraction of specific passages. The knowledge graph structure allows for complex queries about policy relationships and evolution over time.

### GIST Environmental Risk Database
The GIST (Green Investment Screening Tool) database provides environmental risk assessments for companies and financial institutions. It includes data on water stress exposure, biodiversity risks, and climate-related financial impacts. This dataset is crucial for analyzing how environmental factors affect business operations and investment decisions.

### TransitionZero Solar Asset Mapper (TZ-SAM)
A global inventory of solar energy facilities updated quarterly. The Q1 2025 demo subset includes 8,319 facilities across Brazil, India, South Africa, and Vietnam, totaling 124.9 GW of capacity. Each facility record includes location coordinates, capacity, technology type, and construction timeline, enabling detailed geographic and temporal analysis of solar energy deployment.

### Climate Watch Data
Historical and projected emissions data from Climate Watch, covering greenhouse gas emissions by country, sector, and gas type. This dataset supports trend analysis, international comparisons, and progress tracking toward climate commitments.

All responses include source attribution and numbered citations, ensuring transparency and traceability of information.