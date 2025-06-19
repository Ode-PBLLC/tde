# Streaming API Frontend Integration Guide

## Overview
This guide demonstrates how to integrate with the `/query/stream` endpoint that provides real-time environmental risk analysis data through Server-Sent Events (SSE).

## API Endpoint
```
POST http://localhost:8099/query/stream
Content-Type: application/json
```

## Request Format
```json
{
  "query": "Your analysis query here"
}
```

## Response Stream Format

The API returns data as Server-Sent Events with the following event types:

### 1. Thinking Events
Progress indicators showing the AI's processing steps:
```javascript
data: {"type": "thinking", "data": {"message": "🚀 Initializing search across all databases...", "category": "initialization"}}
data: {"type": "thinking_complete", "data": {"message": "✅ Completed ALWAYSRUN", "category": "initialization"}}
```

### 2. Tool Execution Events
Shows which data sources are being queried:
```javascript
data: {"type": "tool_call", "data": {"tool": "GetGistCompanies", "args": {"sector": "OGES", "country": "Brazil"}}}
data: {"type": "tool_result", "data": {"tool": "GetGistCompanies", "result": {...}}}
```

### 3. Final Response Event
Complete analysis with structured data:
```javascript
data: {"type": "complete", "data": {"query": "...", "modules": [...], "metadata": {...}}}
```

## Frontend Implementation

### Basic JavaScript Implementation
```javascript
async function queryEnvironmentalData(query) {
  const response = await fetch('http://localhost:8099/query/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          handleStreamEvent(data);
        } catch (e) {
          console.warn('Failed to parse SSE data:', line);
        }
      }
    }
  }
}

function handleStreamEvent(event) {
  switch (event.type) {
    case 'thinking':
      updateProgressIndicator(event.data.message, event.data.category);
      break;
    case 'thinking_complete':
      markStepComplete(event.data.message);
      break;
    case 'tool_call':
      showDataSourceQuery(event.data.tool, event.data.args);
      break;
    case 'tool_result':
      processToolResult(event.data.tool, event.data.result);
      break;
    case 'complete':
      displayFinalResults(event.data);
      break;
  }
}

function displayFinalResults(data) {
  data.modules.forEach(module => {
    switch (module.type) {
      case 'text':
        renderTextModule(module);
        break;
      case 'table':
        renderTableModule(module);
        break;
      case 'chart':
        renderChartModule(module);
        break;
      case 'numbered_citation_table':
        renderCitationModule(module);
        break;
    }
  });
}

function renderChartModule(module) {
  // Direct Chart.js v3+ integration - no transformation needed
  const ctx = document.getElementById('chart-container');
  new Chart(ctx, {
    type: module.chartType, // 'bar', 'line', or 'pie'
    data: module.data, // Direct use - already Chart.js compatible
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: !!module.heading,
          text: module.heading
        },
        legend: {
          display: module.chartType !== 'bar' || module.data.datasets.length > 1
        }
      },
      scales: module.chartType !== 'pie' ? {
        y: {
          beginAtZero: true,
          ticks: {
            callback: function(value) {
              return value.toLocaleString(); // Format large numbers
            }
          }
        }
      } : {}
    }
  });
}

// Alternative rendering for React with react-chartjs-2
function ChartComponent({ module }) {
  return (
    <div className="chart-container">
      <h3>{module.heading}</h3>
      <Chart
        type={module.chartType}
        data={module.data}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: module.chartType !== 'bar' || module.data.datasets.length > 1
            }
          }
        }}
      />
    </div>
  );
}
```

### React Hook Example
```jsx
import { useState, useEffect } from 'react';

export function useEnvironmentalQuery(query) {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState([]);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const executeQuery = async (queryText) => {
    setLoading(true);
    setProgress([]);
    setResults(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:8099/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'thinking') {
              setProgress(prev => [...prev, data.data]);
            } else if (data.type === 'complete') {
              setResults(data.data);
              setLoading(false);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return { loading, progress, results, error, executeQuery };
}
```

## Data Structure Reference

### Chart Module Structure
The API generates charts in Chart.js v3+ compatible format:
```javascript
{
  "type": "chart",
  "chartType": "bar|line|pie",
  "heading": "Chart Title",
  "data": {
    "labels": ["Label1", "Label2", "Label3"],
    "datasets": [{
      "label": "Dataset Name",
      "data": [value1, value2, value3],
      "backgroundColor": "#4CAF50",
      "borderColor": "#4CAF50",
      "fill": false
    }]
  }
}
```

### Chart Types Generated
- **Bar Charts**: Country comparisons, sector analysis, company rankings, asset distributions
- **Line Charts**: Time series data, emissions trends over time
- **Pie Charts**: Risk level distributions, emissions breakdowns by category

### Companies Data
```javascript
{
  "total_companies": 100,
  "filtered_companies": 4,
  "companies": [
    {
      "company_code": "PETROB00002",
      "company_name": "Vibra Energia SA",
      "datasets": ["EXSITU", "SCOPE_3_DATA", "BIODIVERSITY_PDF_DATA"],
      "sector_code": "OGES",
      "country": "Brazil"
    }
  ]
}
```

### Risk Assessment Data
```javascript
{
  "risk_type": "WATER_STRESS",
  "risk_level": "HIGH",
  "companies_found": 64,
  "companies": [
    {
      "company_code": "BANCOB00001",
      "company_name": "Banco Bradesco SA",
      "total_assets": 6552,
      "at_risk_assets": 999,
      "risk_percentage": 15.25
    }
  ]
}
```

### Emissions Data
```javascript
{
  "company_code": "PETROB00002",
  "company_name": "Vibra Energia SA",
  "years_available": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
  "emissions_by_year": [
    {
      "reporting_year": 2023,
      "total_scope3_emissions": 83624652.86,
      "upstream_emissions": 183666.86,
      "downstream_emissions": 83440986.0,
      "emissions_intensity": 3540.8441616907285
    }
  ]
}
```

### Citation System

The API implements a comprehensive citation system that tracks data sources throughout the analysis:

#### How Citations Work
1. **Inline Citations**: Text modules contain numbered citations in superscript format `^1^`, `^2^`, etc. that appear after the referenced text
2. **Citation Table**: A dedicated module at the end lists all sources
3. **Source Tracking**: Every data point is attributed to its original source

#### Citation Table Structure
```javascript
{
  "type": "numbered_citation_table",
  "heading": "References",
  "columns": ["#", "Source", "Type", "Description"],
  "rows": [
    ["1", "GIST Corporate Directory", "Database", "Company sector and location data"],
    ["2", "GIST Water Risk", "Database", "Water stress exposure for 64 companies"],
    ["3", "GIST Scope 3 Emissions", "Time Series", "2020-2023 emissions data"],
    ["4", "GIST Climate Policy", "Assessment", "Policy effectiveness scores"]
  ]
}
```

#### Inline Citations in Text
Text modules include citations that appear after the referenced content:
```javascript
{
  "type": "text",
  "texts": [
    "According to GIST data, Brazil has 45 oil & gas companies ^1^.",
    "Water stress affects 64 companies, with 15.25% of assets at risk ^2^.",
    "Emissions increased by 23% from 2020-2023 ^3^."
  ]
}
```

#### Citation Implementation Example
```javascript
function renderTextWithCitations(textModule, citationTable) {
  const citationMap = new Map();
  
  // Build citation lookup from table
  citationTable.rows.forEach(row => {
    citationMap.set(row[0], {
      source: row[1],
      type: row[2],
      description: row[3]
    });
  });
  
  // Process text with clickable citations (superscript format)
  const processedTexts = textModule.texts.map(text => {
    return text.replace(/\^(\d+(?:,\d+)*)\^/g, (match, nums) => {
      const citationNums = nums.split(',');
      const links = citationNums.map(num => {
        const citation = citationMap.get(num);
        if (citation) {
          return `<a href="#citation-${num}" 
                     class="citation-link" 
                     title="${citation.source}: ${citation.description}">
                     ${num}
                  </a>`;
        }
        return num;
      }).join(',');
      return `<sup>[${links}]</sup>`;
    });
  });
  
  return processedTexts;
}

// React component for citations
function CitationTable({ module }) {
  return (
    <div className="citations-section">
      <h3>{module.heading}</h3>
      <table className="citation-table">
        <thead>
          <tr>
            {module.columns.map(col => <th key={col}>{col}</th>)}
          </tr>
        </thead>
        <tbody>
          {module.rows.map(row => (
            <tr key={row[0]} id={`citation-${row[0]}`}>
              <td className="citation-number">{row[0]}</td>
              <td className="citation-source">{row[1]}</td>
              <td className="citation-type">{row[2]}</td>
              <td className="citation-description">{row[3]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Interactive citation tooltip
function CitationTooltip({ citationNumber, citations }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const citation = citations.find(c => c[0] === citationNumber);
  
  return (
    <span 
      className="citation"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <sup>[{citationNumber}]</sup>
      {showTooltip && citation && (
        <div className="citation-tooltip">
          <strong>{citation[1]}</strong>
          <p>{citation[3]}</p>
          <span className="citation-type">{citation[2]}</span>
        </div>
      )}
    </span>
  );
}
```

#### Citation Source Types
Common source types you'll encounter:
- **Database**: Core data from GIST databases
- **Time Series**: Historical data over multiple years
- **Assessment**: Analytical evaluations and scores
- **Report**: Detailed reports and documents
- **Calculation**: Derived metrics and computations

#### Key Implementation Notes
1. **Citation Format**: Uses superscript notation `^1^`, `^2,3^` that appears after the referenced text
2. **Citation Numbers**: Always sequential starting from 1
3. **Cross-referencing**: Citation numbers in text match row numbers in the citation table
4. **Source Attribution**: Every data point traces back to its source
5. **Citation Persistence**: Numbers remain consistent throughout the response
6. **Table Location**: Citation table always appears as the last module

### Final Response Structure
```javascript
{
  "query": "Original query text",
  "modules": [
    {
      "type": "text",
      "heading": "Climate Policy Analysis",
      "texts": [
        "Brazil's oil sector shows significant emissions growth ^1,3^.",
        "Water stress impacts 15.25% of banking assets ^2^."
      ]
    },
    {
      "type": "table",
      "heading": "Companies Summary",
      "columns": ["company_code", "company_name", "sector_code"],
      "rows": [["PETROB00002", "Vibra Energia SA", "OGES"]]
    },
    {
      "type": "chart",
      "heading": "Scope 3 Emissions Trends",
      "chart_type": "line",
      "data": {
        "x_axis": "Year",
        "y_axis": "Scope 3 Emissions (tCO2e)",
        "series": [
          {
            "name": "Petróleo Brasileiro SA",
            "data": [427000000, 435000000, 442000000, 441000000]
          },
          {
            "name": "Vibra Energia SA", 
            "data": [75396469, 81003955, 95998153, 92454153]
          }
        ]
      }
    },
    {
      "type": "numbered_citation_table",
      "heading": "References",
      "columns": ["#", "Source", "Type", "Description"],
      "rows": [
        ["1", "GIST Corporate Directory", "Database", "Company sector and location data"],
        ["2", "GIST Water Risk", "Database", "Water stress exposure analysis"],
        ["3", "GIST Scope 3 Emissions", "Time Series", "2020-2023 emissions data"]
      ]
    }
  ],
  "metadata": {
    "modules_count": 5,
    "has_tables": true,
    "has_charts": true,
    "table_types": ["numbered_citation_table"],
    "total_citations": 3
  }
}
```

## UI Components Recommendations

### Progress Indicator
Show real-time progress using the thinking events:
- "🚀 Initializing search across all databases..."
- "⚙️ Running GistCompanies..."
- "🧠 Analyzing and summarizing..."
- "✅ Analysis complete"

### Data Visualization
Render different module types:
- **Text modules**: Standard formatted text with headings
- **Table modules**: Structured data tables
- **Chart modules**: Interactive charts (line, bar, pie, etc.) with series data
- **Citation tables**: Reference lists with numbered citations

### Chart Types Supported
The API automatically generates charts based on data patterns:

**Bar Charts** - Generated for:
- Country comparisons (`"Brazil": 1500, "India": 2000`)
- Sector analysis (`"Oil & Gas": 500, "Utilities": 300`)
- Company rankings (`"Company A": 100, "Company B": 80`)
- Asset distributions (`"Refineries": 25, "Pipelines": 15`)
- Risk assessments (`"High Risk": 12, "Medium Risk": 8`)

**Line Charts** - Generated for:
- Time series data (`2020: 100, 2021: 150, 2022: 200`)
- Emissions trends over time
- Multi-year company performance

**Pie Charts** - Generated for:
- Risk level distributions (`"HIGH": 40%, "MEDIUM": 35%, "LOW": 25%`)
- Emissions breakdowns by category
- Proportional data with 3-8 categories

### Chart Data Format
All charts use Chart.js v3+ compatible format with:
- `labels`: Array of x-axis labels
- `datasets`: Array with data, colors, and styling
- Automatic color assignment from professional palette
- Responsive design optimized for mobile and desktop

### Error Handling
- Network timeouts (120s max)
- JSON parsing errors
- API unavailability
- Invalid query responses

## Performance Considerations

1. **Streaming**: Process data incrementally as it arrives
2. **Memory**: Clear previous results before new queries
3. **Caching**: Consider caching complete responses
4. **Timeouts**: Implement proper timeout handling (120s default)

## Real-World Examples

### Chart Query Examples
```javascript
// Time series chart query
const timeSeriesQuery = "Show Scope 3 emissions trends for Brazilian oil companies from 2020-2023";

// Bar chart query  
const comparisonQuery = "Compare renewable energy capacity by country in Latin America";

// Pie chart query
const distributionQuery = "Show risk level distribution for companies with water stress exposure";

// Execute and handle charts
const { loading, progress, results, executeQuery } = useEnvironmentalQuery();
await executeQuery(timeSeriesQuery);

// Process results
results.modules.forEach(module => {
  if (module.type === 'chart') {
    console.log(`Chart Type: ${module.chartType}`);
    console.log(`Data Ready for Chart.js:`, module.data);
    renderChart(module);
  }
});
```

### Complete Integration Example
```javascript
import { Chart } from 'chart.js/auto';

function EnvironmentalDashboard() {
  const [modules, setModules] = useState([]);
  
  const handleQueryResults = (results) => {
    setModules(results.modules);
    
    // Automatically render charts
    results.modules
      .filter(m => m.type === 'chart')
      .forEach((chartModule, index) => {
        const canvas = document.getElementById(`chart-${index}`);
        new Chart(canvas, {
          type: chartModule.chartType,
          data: chartModule.data, // Direct use - no transformation
          options: { responsive: true }
        });
      });
  };

  return (
    <div>
      {modules.map((module, index) => (
        module.type === 'chart' ? (
          <div key={index} className="chart-container">
            <h3>{module.heading}</h3>
            <canvas id={`chart-${index}`} />
          </div>
        ) : (
          <div key={index}>{/* Render other module types */}</div>
        )
      ))}
    </div>
  );
}
```