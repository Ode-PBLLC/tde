# TDE System Improvement Plan

## Overview
This document outlines specific issues identified during website testing on 2025-01-07 and provides concrete fixes with examples and implementation leads.

**Last Updated:** September 7, 2025

---

## ✅ COMPLETED IMPROVEMENTS (September 7, 2025)

### Successfully Fixed Oversharing Issues
**Status:** ✅ COMPLETED

**What Was Fixed:**
1. **Better Tool Selection** - Queries now use 3 sources instead of 5 for NDC questions
   - Added negative examples to Phase 0 prefilter (`mcp/mcp_chat_redo.py` lines 1280-1317)
   - NDC queries no longer pull irrelevant solar facility data

2. **Enhanced Fact Extraction** - Units now properly preserved
   - Modified fact extraction to include complete sentences with units (`mcp/mcp_chat_redo.py` lines 1658-1743)
   - Facts now include context like "45% renewable energy target by 2030"

3. **Smart Visualization Filtering** - Maps only appear when relevant
   - Added LLM-based relevance checking (`mcp/mcp_chat_redo.py` lines 2703-2759)
   - No more solar facility maps for policy questions

4. **Fixed Table Formatting** - No more meaningless percentage totals
   - Added new `CreateDataTable` tool (`mcp/viz_server.py` lines 702-897)
   - Tables no longer sum percentages (was showing 45% + 50% + 33% = 128%)
   - Proper handling of mixed data types

**Deployment Status:** ✅ Live on production (https://transitiondigital-ai.ode.sunship.dev/)

---

## 🟡 REMAINING TABLE & DATA DISPLAY ISSUES (September 7, 2025)

### 1. Percentage Values Stored as Decimals
**Status:** 🔧 IN PROGRESS
**Priority:** 🔴 CRITICAL

**Problem Example:**
```
Table shows: "Renewable Energy Target: 0.45%"
Should show: "Renewable Energy Target: 45%"
```

**Root Cause:** Data is being stored as decimals (0.45) instead of percentages (45) in the data extraction phase

**Fix Implementation:**
```python
# In viz_server.py CreateDataTable
if format_type == "percentage":
    if isinstance(value, (int, float)) and value < 1:
        value = value * 100  # Convert 0.45 to 45
    row.append(f"{value}%")
```

**Files to Check:**
- `mcp/viz_server.py` - CreateDataTable function
- Phase 1/2 fact extraction in `mcp/mcp_chat_redo.py`
- Look for any `/100` operations

---

### 2. Year Formatting with Commas
**Status:** ❌ NOT FIXED
**Priority:** 🟡 MEDIUM

**Problem Example:**
```
Table shows: "Target Year: 2,030"
Should show: "Target Year: 2030"
```

**Fix Implementation:**
```python
# In CreateDataTable
elif format_type == "number":
    if isinstance(value, (int, float)) and 1900 <= value <= 2100:
        row.append(str(int(value)))  # No commas for years
    else:
        row.append(f"{int(value):,}")
```

---

### 3. Mixed Unit Display Without Context
**Status:** ❌ NOT FIXED
**Priority:** 🟡 MEDIUM

**Problem:** Tables show numbers without units or context
**Example:** "45" instead of "45% of energy mix"

**Fix Implementation:**
- Auto-add units to column headers based on format type
- Include baseline/context in additional column
```python
# Enhance column headers
if col["format"] == "percentage":
    col["label"] = f"{col['label']} (%)"
elif "capacity" in col["key"].lower():
    col["label"] = f"{col['label']} (MW)"
```

---

### 4. Missing Data Source Attribution & Citations
**Status:** ❌ NOT FIXED
**Priority:** 🟡 MEDIUM

**Problem:** Tables don't indicate where data came from or link to sources

**Fix Implementation:**
```python
# Add citation references directly to table module
table_module = {
    "type": "table",
    "heading": "Renewable Energy Targets",
    "citation_ids": [1, 2],  # Link to citation registry
    "caption": "Percentage of total energy mix by 2030",
    "columns": [...],
    "rows": [...],
    "metadata": {
        "source": "Brazil NDC 2023 Update",
        "data_date": "2023-11"
    }
}

# Display as: "Table 1: Renewable Energy Targets¹²"
# Where ¹² are clickable citations linking to source documents
```

**Additional Changes:**
- Modify frontend to render citation superscripts on table headings
- Ensure citations link to actual source documents in citation registry

---

### 5. Data Validation at Extraction
**Status:** ❌ NOT FIXED
**Priority:** 🟡 MEDIUM

**Problem:** Bad data enters system without validation

**Fix Implementation:**
- Validate percentages during fact extraction
- Log warnings for suspicious values
- Auto-correct obvious errors (0.45 → 45)
```python
# In fact extraction
if "percent" in fact_text and value < 1:
    log.warning(f"Suspicious percentage: {value}")
    value = value * 100  # Auto-correct
```

---

## Implementation Timeline

### Day 1 (Monday) - Critical Fixes
- [ ] Fix percentage decimal issue (#1) - 2 hours
- [ ] Fix year comma formatting (#2) - 1 hour
- [ ] Test both fixes - 1 hour

### Day 2 (Tuesday) - Context & Citations
- [ ] Add unit context to columns (#3) - 2 hours
- [ ] Add citation references to tables (#4) - 2 hours
- [ ] Implement validation at extraction (#5) - 2 hours

### Day 3 (Wednesday) - Testing & Deployment
- [ ] Comprehensive testing of all fixes
- [ ] Deploy to production
- [ ] Verify on live site

---

## 🔴 CRITICAL FIXES (Must Fix Immediately)

### 1. Fix Data Access for NDC/Policy Content
**Status:** ❌ NOT STARTED

**Problem Example:**
```
User Query: "Help me understand Brazil's NDC commitments related to energy"
System Response: "Brazil has 2,273 facilities..." (talks about solar facilities instead of NDC commitments)
Actual Data Available: LSE Excel files contain NDC data but system doesn't access it
```

**What Should Happen:**
- System should provide Brazil's actual NDC targets: 43% emissions reduction by 2030, net-zero by 2050
- Should mention renewable energy targets, biofuels strategy, etc.

**Implementation Leads:**
- File to check: `data/lse/1 NDC Overview and Domestic Policy Comparison Content.xlsx`
- The data EXISTS (confirmed Brazil is mentioned in Sheet1)
- Need to properly parse Excel sheets in MCP server
- Consider converting Excel to indexed SQLite database for faster access
- Update `mcp/mcp_chat_redo.py` to properly query LSE data

---

### 2. Stop Hallucinating Statistics
**Status:** ❌ NOT STARTED

**Problem Example:**
```
System Claims: "The analysis framework includes 13 different sectors"
Reality: No source for "13 sectors" found in any data file

System Claims: "Energy commitments analysis shows: Max: 83.0, Min: 2.0, Median: 45.0"
Reality: These numbers don't exist in the data
```

**Implementation Leads:**
- Add validation in `mcp/response_formatter.py`
- Only allow facts that come directly from tool responses
- Implement fact-checking layer that traces each claim back to source
- Consider adding a `fact_source` field to track origin of each statement

---

### 3. Fix Query Hanging/Timeout Issues
**Status:** ❌ NOT STARTED

**Problem Example:**
```
User Query: "Show me the largest solar facilities in Brazil"
System: Gets stuck at "🔄 Running deeper analysis on 3 servers..."
Never completes after 20+ seconds
```

**Implementation Leads:**
- Add timeout to Phase 2 operations in `mcp/mcp_chat_redo.py`
- Implement async timeout: `asyncio.wait_for(phase2_analysis(), timeout=10.0)`
- Add progress streaming so user knows system is working
- Consider skipping Phase 2 when Phase 1 has sufficient data

---

## 🟡 HIGH PRIORITY FIXES

### 4. Implement Map Visualization for Solar Queries
**Status:** ❌ NOT STARTED

**Problem Example:**
```
User Query: "Show me the largest solar facilities in Brazil"
Data Available: 
  - Facility 14079: 996.1 MW at (-15.97, -43.51)
  - Facility 29667: 839.6 MW at (-15.73, -45.96)
  - [8 more with coordinates]
System Response: No map generated despite having all coordinate data
```

**Implementation Leads:**
- Generate GeoJSON from query results
- Use existing map generation code (files already exist in `static/maps/`)
- Pattern to follow: `static/maps/solar_facilities_brazil_*.geojson`
- Update visualization server to auto-generate maps for coordinate data

---

### 5. Improve Query Relevance Matching
**Status:** ❌ NOT STARTED

**Problem Example:**
```
User: "Help me understand Brazil's NDC commitments related to energy"
System: Talks about solar facility counts instead of NDC policy
Should: Recognize "NDC commitments" → query policy data, not facility data
```

**Implementation Leads:**
- Improve Phase 1 prompt in `mcp/mcp_chat_redo.py`
- Add keyword matching: "NDC" → LSE data, "facilities" → solar data
- Implement intent classification before tool selection
- Consider adding a routing layer that maps query types to appropriate tools

---

### 6. Add Facility Details Beyond Just Numbers
**Status:** ❌ NOT STARTED

**Problem Example:**
```
Current: "Brazil has 2,273 facilities"
Missing: Facility names, operators, technology types, operational status
Available in data: All these fields exist in the database
```

**Implementation Leads:**
- Query additional columns from `solar_facilities` table
- Check schema: `sqlite3 data/solar_facilities.db .schema solar_facilities`
- Include facility metadata in responses
- Format as table or structured list

---

## 🟢 MEDIUM PRIORITY IMPROVEMENTS

### 7. Better "No Data" Handling
**Status:** ❌ NOT STARTED

**Problem Example:**
```
System: "specific detailed results about Brazil's energy commitments in its NDC are not readily available"
Reality: The data IS available in LSE Excel files
```

**Implementation Leads:**
- Implement data discovery check before claiming absence
- Add fallback queries to find related data
- Better error messages that are accurate about what's actually missing

---

### 8. Context-Aware Citations
**Status:** ❌ NOT STARTED

**Problem Example:**
```
Current: [Citation 02] [Citation 03] (unclear what these reference)
Should: [Solar Database: 2,273 facilities] [Date range: 2017-2025]
```

**Implementation Leads:**
- Modify citation format in `mcp/response_formatter.py`
- Include source type in citation
- Make citations clickable with source preview on hover

---

### 9. Optimize Phase 2 Triggers
**Status:** ❌ NOT STARTED

**Problem Example:**
```
System: "🔬 The current data lacks specific NDC commitment targets..."
Then: "🔄 Running deeper analysis on 3 servers..."
Issue: Phase 2 runs even when it won't find NDC data in solar database
```

**Implementation Leads:**
- Add logic to check if Phase 2 can actually help
- Skip Phase 2 if query type doesn't match available data
- Implement smarter trigger conditions

---

## ⚡ QUICK WINS

### 10. Fix Misleading Language
**Status:** ❌ NOT STARTED

**Problem Example:**
```
System: "current facility-level data between 2017-2025 appears to be limited"
Reality: Full data exists from 2017-04-19 to 2025-03-22
```

**Implementation Leads:**
- Update response templates
- Change "limited" to "available" when data exists
- Be precise about date ranges

---

### 11. Add Response Validation
**Status:** ❌ NOT STARTED

**Problem Example:**
```
User asks about: NDC commitments
System provides: Solar facility statistics
Missing: Check if response matches query intent
```

**Implementation Leads:**
- Add validation step before sending response
- Compare key terms: did we answer about what was asked?
- Implement relevance scoring

---

### 12. Include Actual Values in Responses
**Status:** ❌ NOT STARTED

**Problem Example:**
```
Current: "Brazil has large solar facilities"
Better: "Brazil's largest: 996 MW facility at São Francisco, 840 MW at Janaúba..."
```

**Implementation Leads:**
- Always include top 5-10 specific examples
- Add actual numbers, not just generalizations
- Query and display real data points

---

## 📊 Testing Queries for Validation

After implementing fixes, test with these queries:

1. **NDC Test:** "What are Brazil's NDC energy commitments?"
   - Should return: 43% reduction target, renewable energy goals, net-zero timeline

2. **Solar Facility Test:** "Show me the largest solar facilities in Brazil"
   - Should return: Map with top 10 facilities, capacities, locations

3. **Mixed Query Test:** "How do Brazil's solar facilities contribute to NDC goals?"
   - Should return: Both policy context AND facility data

4. **Performance Test:** All queries should complete within 10 seconds

---

## 🚀 Implementation Priority

### Week 1 Sprint
- [ ] Fix Data Access for NDC/Policy Content (#1)
- [ ] Stop Hallucinating Statistics (#2)  
- [ ] Fix Query Hanging/Timeout Issues (#3)

### Week 2 Sprint
- [ ] Implement Map Visualization (#4)
- [ ] Improve Query Relevance Matching (#5)
- [ ] Add Facility Details (#6)

### Week 3 Sprint
- [ ] Better "No Data" Handling (#7)
- [ ] Fix Misleading Language (#10)
- [ ] Add Response Validation (#11)
- [ ] Include Actual Values (#12)

### Backlog
- [ ] Context-Aware Citations (#8)
- [ ] Optimize Phase 2 Triggers (#9)

---

## 📝 Notes

- Each fix should include unit tests
- Document changes in CLAUDE.md for future reference
- Consider adding monitoring to track which fixes have most impact
- Regular testing with the queries above to ensure no regressions