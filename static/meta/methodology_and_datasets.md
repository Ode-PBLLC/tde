# System Architecture
Transition Digital AI combines multiple data sources through a Model Context Protocol (MCP) server architecture. The system uses AI-powered Retrieval Augmented Generation (RAG) to intelligently query and synthesize information from diverse climate-related datasets. We maintain consistent references and citations.

How It Works
1. Query Processing: User queries are processed by an AI model that determines which data sources and tools are most relevant.
2. Dynamic Tool Selection: Our MCP server provides access to specialized tools for each dataset, including:
   * Knowledge graph queries for policy documents
   * Geospatial queries for facility locations
   * Statistical analysis for trends and comparisons
   * RAG Integration: The system uses semantic search and knowledge graph traversal to find relevant information, then augments responses with data from multiple sources.
3. Response Synthesis: Results from different tools are combined into a coherent response with appropriate visualizations (charts, maps, tables) and inline citations.

# Data Sources
<b>Climate Policy Radar Concept Store, Knowledge Graph, and Passage Library</b>

A comprehensive passage library containing climate policies, laws, and regulations from around the world. This dataset enables semantic search across policy documents, identification of related concepts, and extraction of specific passages. The knowledge graph structure allows for complex queries about policy and concept relationships. For this demo, documents are limited to those pertaining to or within Brazil.

<b>GIST Environmental Impact</b>

This dataset contains comprehensive biodiversity and environmental impact metrics of companies and their assets. It includes the counts of assets in proximity to various stressors including deforestation, urbanization, and climate change impacts. The data also includes risk levels for many of these features, company-wide impacts on biodiversity, and summaries of Scope 3 Emissions from each company.

<b>TransitionZero Solar Asset Mapper (TZ-SAM)</b>

A global inventory of solar energy facilities updated quarterly. The Q1 2025 demo subset includes 2273 facilities across Brazil, totaling 26,022MW of capacity. Each facility record includes location coordinates, capacity, and rough construction timeline, enabling detailed geographic and temporal analysis of solar energy deployment.

<b>Science Panel of the Amazon</b>

The Science Panel for the Amazon released the Amazon Assessment Report 2021 at COP26, which has been called an “encyclopedia” of the Amazon region. This landmark report is unprecedented for its scientific and geographic scope, the inclusion of Indigenous scientists, and its transparency, having undergone peer review and public consultation.

<b>PRODES Deforestation</b>

The PRODES project carries out satellite monitoring of clear-cut deforestation in the Legal Amazon and has produced, since 1988, the annual deforestation rates in the region, which are used by the Brazilian government to establish public policies. Annual rates are estimated from the deforestation increments identified in each satellite image covering the Legal Amazon. The first presentation of the data is carried out by December of each year, in the form of an estimate, when approximately 50% of the images covering the Legal Amazon are normally processed. The consolidated data are presented in the first half of the following year.

<b>Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis</b>

Extreme Heat indices derived from ERA5-Land measurements. This data represents quintiles of the multi-year mean of the daily land surface temperature temperature over the last five years. This data was prepared by PlanetSapling.

<b>IPCC Reports on Extreme Heat and Climate</b>

Combined corpus of IPCC AR6 WG1 Chapter 11 and WG2 Chapter 12 passages. The passages outline climate change within Latin American and Brazil and the scientific markers that allow us to track these changes.

<b>World Meteorological Organization: State of the Climate in Latin America and the Caribbean 2024</b>

The WMO State of the Climate in Latin America and the Caribbean 2024, is the fifth edition of climate reports published annually for this region and has involved National Meteorological and Hydrological Services (NMHSs), WMO Regional Climate Centres (RCCs), and several research institutions, as well as United Nations agencies, international and regional organizations. The report provides the status of key climate indicators and information on climate-related impacts and risks. It addresses specific physical science, socio-economic and policy-related aspects that are relevant to LAC and responds to Members needs in the fields of climate monitoring, climate change and climate services.

<b>NDC Align</b>

A joint initiative of the Grantham Research Institute on Climate Change & the Environment at The London School of Economics and Political Science (LSE), in partnership with JUMA (PUC-Rio), LACLIMA, Climate Policy Radar, and Ode, with support from the ClimateWorks Foundation. This dataset is designed to help users understand gaps between ambition in a country’s NDC and the country’s domestic governance landscape.

<b>MapBiomas Annual Report on Deforestation 2024</b>

The sixth Annual Report on Deforestation in Brazil, prepared by MapBiomas, presents a comprehensive overview of deforestation across all Brazilian biomes from 2019 to 2024, focusing on the most recent year and considering different territorial and land ownership categories. The report consolidates and analyzes validated and refined deforestation alerts using high-resolution images by MapBiomas Alert (https://alerta.mapbiomas.org/) from multiple deforestation detection systems, assesses indications of irregularity or illegality, and examines actions to combat deforestation by government agencies and financial institutions.