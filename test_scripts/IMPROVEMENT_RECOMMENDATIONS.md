# NDC Align Improvement Recommendations

**Purpose**: Improve NDC Align data retrieval, especially for transparency and state-level queries
**Status**: RECOMMENDATIONS ONLY - No changes implemented yet

---

## Problem Summary

### Current Issues
1. **Generic transparency queries** get generic responses, miss detailed state data
2. **Tool descriptions** don't guide the AI to use the right tools for specific query types
3. **GetSubnationalGovernance metric filter** exists but may not be well-utilized
4. **SearchLSEContent** searches all text but doesn't use semantic understanding
5. **Empty "Transparency and Accountability" fields** - data is scattered in other questions

### What Works
- ✅ Data exists and is comprehensive
- ✅ Tools have good functionality (filtering, search, etc.)
- ✅ System correctly routes to NDC Align
- ✅ Citations work properly

---

## Levers We Can Pull

Ranked by **Impact** (High/Medium/Low) and **Effort** (Easy/Medium/Hard)

---

## 🎯 HIGH IMPACT, EASY WINS

### 1. Enhance Tool Descriptions (Impact: HIGH, Effort: EASY)

**Problem**: Tool descriptions don't guide AI to choose the right tool for transparency queries

**Current Tool Descriptions**:
```python
def GetSubnationalGovernance(state, metric):
    """Retrieve Brazilian subnational governance data."""
```

**Recommended Enhancement**:
```python
def GetSubnationalGovernance(state, metric):
    """
    Retrieve Brazilian subnational governance data for climate policy.

    Use this tool for queries about:
    - State-level climate governance structures
    - Public participation and transparency mechanisms
    - State climate laws and policies
    - Monitoring and accountability systems

    Parameters:
    - state: Specific Brazilian state name (e.g., "Acre", "São Paulo")
    - metric: Filter by topic (e.g., "transparency", "participation", "monitoring")

    Examples:
    - "What transparency mechanisms does Acre have?" → state="Acre", metric="transparency"
    - "How does São Paulo ensure public participation?" → state="São Paulo", metric="participation"
    - "Monitoring systems in Amazonas" → state="Amazonas", metric="monitoring"

    Note: For transparency queries, also try metric="participation" as transparency
    information is often embedded in public participation questions.
    """
```

**Where to Change**: `mcp/lse_server.py:908-912`

**Similar Enhancements Needed For**:
- `SearchLSEContent` - add examples of good search terms
- `GetStateClimatePolicy` - explain when to use vs. GetSubnationalGovernance
- `GetLSEDatasetOverview` - clarify this is for metadata, not detailed queries
- `GetInstitutionalFramework` - add examples about transparency/coordination

**Expected Outcome**: AI will choose better tools and pass better parameters

---

### 2. Add Semantic Aliases to Search (Impact: HIGH, Effort: EASY)

**Problem**: Search for "transparency" misses records with "public participation", "accountability", "disclosure"

**Current Implementation**: Simple string matching in `catalog.search()` (line 735-779)

**Recommendation**: Add semantic mapping BEFORE search

```python
# Add to lse_server.py after line 38

SEMANTIC_ALIASES = {
    "transparency": ["transparency", "public participation", "disclosure",
                     "accountability", "stakeholder engagement", "civil society",
                     "monitoring", "reporting", "verification"],
    "coordination": ["coordination", "governance", "institutional",
                     "interministerial", "CIM", "committee"],
    "implementation": ["implementation", "status", "in law", "in policy",
                       "under development", "practice"],
    "monitoring": ["monitoring", "tracking", "measurement", "reporting",
                   "verification", "MRV"],
}

def expand_search_terms(term: str) -> list[str]:
    """Expand a search term to include semantic aliases."""
    term_lower = term.lower()
    for key, aliases in SEMANTIC_ALIASES.items():
        if key in term_lower:
            return aliases
    return [term]
```

**Then modify `catalog.search()` to use expanded terms**:
```python
def search(self, term: str, ...) -> dict[str, Any]:
    search_terms = expand_search_terms(term)
    results = []
    for search_term in search_terms:
        # existing search logic
        ...
```

**Where to Change**: `mcp/lse_server.py:735-779`

**Expected Outcome**: "transparency" queries will find "public participation" records

---

### 3. Create GetTransparencyMeasures Tool (Impact: HIGH, Effort: MEDIUM)

**Problem**: No dedicated tool for transparency queries, so AI uses generic tools

**Recommendation**: Add a new purpose-built tool

```python
@mcp.tool()
def GetTransparencyMeasures(
    state: Optional[str] = None,
    transparency_type: Optional[str] = None,
) -> dict[str, Any]:
    """
    Retrieve transparency and accountability measures for Brazilian climate governance.

    This tool specifically finds transparency information which is embedded across
    multiple question types including public participation, monitoring, and disclosure.

    Parameters:
    - state: Specific Brazilian state (e.g., "Acre", "São Paulo"), or None for all states
    - transparency_type: Type of transparency mechanism:
        - "participation" - Public participation and stakeholder engagement
        - "disclosure" - Information disclosure and reporting
        - "monitoring" - Monitoring and verification systems
        - "accountability" - Accountability mechanisms
        - None - All transparency-related information

    Returns detailed transparency mechanisms with examples from state governance.
    """
    slugs = catalog.module_index.get("subnational", [])
    if not slugs:
        return {"error": "Subnational data not available"}

    transparency_keywords = ["transparency", "accountab", "disclosure",
                            "participat", "civil society", "stakeholder",
                            "public", "monitoring", "reporting"]

    results = []
    for slug in slugs:
        sheet = catalog.get_sheet(slug)
        if not sheet:
            continue

        # Filter by state if specified
        state_name = sheet.metadata.get("state_name") or sheet.title
        if state and state.lower() not in state_name.lower():
            continue

        # Find records with transparency information
        transparency_records = []
        for record in sheet.records:
            # Check all text fields for transparency keywords
            text_content = " ".join([
                str(v).lower() for v in record.values()
                if isinstance(v, str) and v
            ])

            if any(keyword in text_content for keyword in transparency_keywords):
                # Filter by transparency_type if specified
                if transparency_type:
                    if transparency_type.lower() not in text_content:
                        continue
                transparency_records.append(record)

        if transparency_records:
            results.append({
                "state": state_name,
                "state_code": sheet.metadata.get("state_code"),
                "transparency_mechanisms": transparency_records,
                "count": len(transparency_records)
            })

    return {
        "states": results,
        "total_states": len(results),
        "query_type": transparency_type or "all transparency measures"
    }
```

**Where to Add**: After line 1007 in `mcp/lse_server.py`

**Expected Outcome**: Direct queries like "state transparency" will use this tool and get rich results

---

## 🎯 HIGH IMPACT, MEDIUM EFFORT

### 4. Pre-process and Consolidate Transparency Data (Impact: HIGH, Effort: MEDIUM)

**Problem**: "Transparency and Accountability" field is empty; data scattered across questions

**Recommendation**: During data loading, extract and consolidate transparency information

**Approach A: Modify Data Processing** (in `parse_subnational_workbook`):
```python
def parse_subnational_workbook(path: Path, spec: ModuleSpec) -> list[ProcessedSheet]:
    # ... existing code ...

    # After processing records (around line 497)
    # Extract transparency information
    transparency_summary = extract_transparency_info(records_with_state)

    # Add to sheet metadata
    sheet.metadata["transparency_mechanisms"] = transparency_summary

    return sheets

def extract_transparency_info(records: list[dict]) -> dict:
    """Extract transparency-related information from all records."""
    transparency_keywords = ["transparency", "accountab", "disclosure",
                            "participat", "civil society", "stakeholder"]

    mechanisms = []
    for record in records:
        question = record.get("question", "")
        summary = record.get("summary", "")

        # Check if this record contains transparency information
        combined_text = f"{question} {summary}".lower()
        if any(keyword in combined_text for keyword in transparency_keywords):
            mechanisms.append({
                "question": question,
                "summary": summary,
                "source": record.get("source_document_1"),
            })

    return {
        "has_transparency_data": len(mechanisms) > 0,
        "mechanism_count": len(mechanisms),
        "mechanisms": mechanisms
    }
```

**Approach B: Post-process JSON Files** (separate script):
Create `scripts/enrich_transparency_data.py`:
```python
"""Enrich subnational data with consolidated transparency information."""
import json
from pathlib import Path

def enrich_transparency_data():
    data_dir = Path("data/lse_processed/subnational")

    for state_file in data_dir.glob("*.json"):
        with open(state_file) as f:
            data = json.load(f)

        # Extract transparency info
        transparency_records = []
        for record in data["records"]:
            text = f"{record.get('question','')} {record.get('summary','')}".lower()
            if any(kw in text for kw in ["transparency", "participat", "accountab"]):
                transparency_records.append({
                    "question": record.get("question"),
                    "summary": record.get("summary"),
                })

        # Add to metadata
        data["metadata"]["transparency"] = {
            "count": len(transparency_records),
            "mechanisms": transparency_records
        }

        # Write back
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    enrich_transparency_data()
```

**Where to Change**:
- Approach A: `mcp/lse_server.py:429-522` (parse_subnational_workbook)
- Approach B: Create new script `scripts/enrich_transparency_data.py`

**Expected Outcome**: Transparency data easily accessible in metadata; tools can quickly retrieve it

---

### 5. Improve Tool Selection Hints (Impact: MEDIUM, Effort: MEDIUM)

**Problem**: AI choosing `GetLSEDatasetOverview` (metadata) instead of detail tools

**Recommendation**: Make tool descriptions more directive about when NOT to use

**For GetLSEDatasetOverview**:
```python
@mcp.tool()
def GetLSEDatasetOverview() -> dict[str, Any]:
    """
    Provide high-level metadata about the entire NDC Align dataset.

    ⚠️ USE THIS ONLY FOR:
    - "What data does NDC Align contain?"
    - "Overview of NDC Align modules"
    - Dataset statistics and coverage

    ❌ DO NOT USE FOR:
    - Specific questions about transparency, policies, or states
    - Detailed governance information
    - Any query asking "what" or "how" about specific topics

    For specific queries, use:
    - GetSubnationalGovernance for state-level details
    - SearchLSEContent for keyword searches
    - GetNDCTargets for NDC commitments
    """
```

**For SearchLSEContent**:
```python
@mcp.tool()
def SearchLSEContent(search_term, module_type, limit) -> dict[str, Any]:
    """
    Search across all NDC Align content for specific terms.

    ✅ BEST FOR:
    - Finding records mentioning specific terms ("CEVA", "Resolution 3", "PNMC")
    - Broad exploration when unsure which tool to use
    - Cross-module searches

    💡 TIP: For better results with transparency queries:
    - Try: "public participation" instead of "transparency"
    - Try: "stakeholder engagement" or "civil society"
    - Try: "monitoring" or "disclosure"

    Examples:
    - search_term="public participation", module_type="subnational"
    - search_term="CIM", module_type="institutions"
    - search_term="Law 13.798"
    """
```

**Where to Change**: `mcp/lse_server.py` - tool docstrings (lines 962, 993, 908, etc.)

**Expected Outcome**: Better tool selection by AI

---

## 🎯 MEDIUM IMPACT, EASY-MEDIUM EFFORT

### 6. Add Query Examples to Server Description (Impact: MEDIUM, Effort: EASY)

**Problem**: AI doesn't know what kinds of questions work well with NDC Align

**Current DescribeServer** (line 1424):
```python
@mcp.tool()
def DescribeServer() -> dict[str, Any]:
    """Describe the server, modules, and tooling."""
    metadata = catalog.metadata.copy()
    metadata["tools"] = [tool_names...]
    return metadata
```

**Recommended Enhancement**:
```python
@mcp.tool()
def DescribeServer() -> dict[str, Any]:
    """
    Describe the NDC Align server capabilities, modules, and tooling.

    NDC Align specializes in analyzing the ALIGNMENT between Brazil's NDC
    commitments and domestic policy implementation.
    """
    metadata = catalog.metadata.copy()
    metadata["tools"] = [tool_names...]

    # Add usage guidance
    metadata["best_for"] = {
        "ndc_alignment": "Comparing NDC commitments to domestic law",
        "governance": "Institutional frameworks and coordination",
        "state_policies": "Subnational climate governance (27 Brazilian states)",
        "implementation": "Status of policy implementation",
        "transparency": "Public participation and accountability mechanisms"
    }

    metadata["query_examples"] = {
        "ndc_targets": [
            "What is Brazil's 2050 climate neutrality target?",
            "What are Brazil's interim emissions targets for 2035?"
        ],
        "alignment": [
            "How do Brazil's NDC commitments compare to domestic law?",
            "Is Brazil's climate target legally binding?"
        ],
        "state_governance": [
            "What climate law does São Paulo state have?",
            "What transparency mechanisms does Acre state have?"
        ],
        "institutions": [
            "What is the role of CIM in Brazil's climate governance?",
            "How does Brazil coordinate climate policy?"
        ]
    }

    metadata["not_for"] = [
        "Primary policy documents (use CPR)",
        "Emissions statistics (use KG)",
        "Climate science projections (use IPCC)"
    ]

    return metadata
```

**Where to Change**: `mcp/lse_server.py:1424-1438`

**Expected Outcome**: AI better understands when and how to use NDC Align

---

### 7. Add Transparency-Specific Filters to Existing Tools (Impact: MEDIUM, Effort: EASY)

**Problem**: GetSubnationalGovernance has a generic "metric" filter

**Recommendation**: Make the metric parameter smarter

```python
@mcp.tool()
def GetSubnationalGovernance(
    state: Optional[str] = None,
    metric: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve Brazilian subnational governance data."""
    # ... existing code ...

    if metric:
        metric_lower = metric.lower()

        # Expand common queries to related terms
        if "transparency" in metric_lower:
            search_terms = ["transparency", "participat", "disclosure",
                          "accountab", "civil society", "stakeholder"]
        elif "monitoring" in metric_lower:
            search_terms = ["monitoring", "reporting", "verification",
                          "measurement", "tracking"]
        elif "coordination" in metric_lower:
            search_terms = ["coordination", "governance", "institutional",
                          "committee", "forum"]
        else:
            search_terms = [metric_lower]

        records = [
            record
            for record in records
            if any(
                any(term in str(value).lower() for term in search_terms)
                for key, value in record.items()
                if key in ("question", "summary") and value
            )
        ]
```

**Where to Change**: `mcp/lse_server.py:922-932`

**Expected Outcome**: metric="transparency" will find participation records

---

## 🎯 LOW IMPACT (but useful)

### 8. Add State Name Aliases (Impact: LOW, Effort: EASY)

**Problem**: User might query "Sao Paulo" without accent, or use abbreviations

```python
STATE_ALIASES = {
    "sao paulo": "São Paulo",
    "sp": "São Paulo",
    "rio": "Rio de Janeiro",
    "rj": "Rio de Janeiro",
    # ... etc
}

def normalize_state_name(state: str) -> str:
    """Normalize state name to match data."""
    state_lower = state.lower()
    return STATE_ALIASES.get(state_lower, state)
```

**Where to Add**: After line 40 in `mcp/lse_server.py`

---

### 9. Add Logging for Tool Usage (Impact: LOW, Effort: EASY)

**Problem**: Hard to debug which tools are being called

```python
import logging

logger = logging.getLogger("lse_server")

@mcp.tool()
def GetSubnationalGovernance(...):
    logger.info(f"GetSubnationalGovernance called: state={state}, metric={metric}")
    # ... rest of function
```

**Where to Add**: All tool functions in `mcp/lse_server.py`

**Expected Outcome**: Better debugging and analytics

---

## 📊 Recommended Implementation Order

### Phase 1: Quick Wins (Week 1)
1. ✅ **Enhance tool descriptions** (HIGH impact, EASY) - #1
2. ✅ **Add semantic aliases to search** (HIGH impact, EASY) - #2
3. ✅ **Add query examples to DescribeServer** (MEDIUM impact, EASY) - #6
4. ✅ **Improve transparency filter in GetSubnationalGovernance** (MEDIUM impact, EASY) - #7

**Expected Result**: 40-60% improvement in query satisfaction

### Phase 2: New Capabilities (Week 2)
5. ✅ **Create GetTransparencyMeasures tool** (HIGH impact, MEDIUM) - #3
6. ✅ **Improve tool selection hints** (MEDIUM impact, MEDIUM) - #5

**Expected Result**: 60-80% improvement in query satisfaction

### Phase 3: Data Quality (Week 3+)
7. ✅ **Consolidate transparency data** (HIGH impact, MEDIUM) - #4
8. ⚠️ **Add state aliases** (LOW impact, EASY) - #8
9. ⚠️ **Add logging** (LOW impact, EASY) - #9

**Expected Result**: 80%+ improvement in query satisfaction

---

## 🚫 What NOT to Do

### Don't Change:
1. ❌ **Core search algorithm** - Works fine, just needs semantic expansion
2. ❌ **Data structure fundamentals** - JSON schema is good
3. ❌ **Tool interface signatures** - Maintain backward compatibility
4. ❌ **Citation system** - Working correctly

### Don't Over-Engineer:
1. ❌ **Machine learning for query routing** - Overkill for this problem
2. ❌ **Complex NLP** - Simple keyword expansion is sufficient
3. ❌ **New database** - File-based system works fine

---

## 📈 Success Metrics

### How to Measure Improvement:

1. **Query Success Rate**
   - Test queries from `test_ndc_align_datasets.md`
   - Target: 80%+ should use NDC Align when appropriate

2. **Transparency Query Quality**
   - Test specific transparency queries
   - Target: Return detailed mechanisms, not generic overviews

3. **Tool Selection**
   - Log which tools are called for which queries
   - Target: Right tool chosen 90%+ of the time

4. **User Feedback**
   - Ask client to rate responses before/after
   - Target: "Satisfactory" rating improves from ~30% to 80%

---

## 🛠️ Testing Strategy

### After Each Phase:

1. **Run automated test suite**: `python test_scripts/run_ndc_align_tests.py`
2. **Manual spot checks**: Test 5-10 queries from quick reference guide
3. **Transparency specific**: Test all transparency patterns from addendum doc
4. **Regression testing**: Ensure previously working queries still work

---

## 💡 Summary: The Most Important Changes

If you can only do 3 things, do these:

### 1. Enhanced Tool Descriptions (30 min)
Add examples and guidance to tool docstrings - biggest bang for buck

### 2. Semantic Aliases in Search (1 hour)
Make "transparency" find "public participation" records

### 3. GetTransparencyMeasures Tool (2-3 hours)
Purpose-built tool for the most problematic query type

**Total effort: ~4 hours for 70% improvement**

---

## 📝 Implementation Notes

- All changes should be in `mcp/lse_server.py`
- No database/schema changes needed
- Backward compatible - won't break existing functionality
- Can deploy incrementally
- Easy to rollback if issues arise

---

## 🔄 Rollback Plan

If changes cause issues:
1. Enhanced descriptions: Just revert docstrings
2. Semantic aliases: Remove SEMANTIC_ALIASES dict and expand_search_terms()
3. New tool: Comment out @mcp.tool() decorator
4. Data consolidation: Delete enriched files, regenerate from raw

All changes are additive and safe.

---

**Next Step**: Review these recommendations, prioritize based on your timeline, and I can implement the approved changes.
