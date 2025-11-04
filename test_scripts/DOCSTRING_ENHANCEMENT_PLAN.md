# Tool Docstring Enhancement Plan

**Purpose**: Show exactly what enhanced tool docstrings should look like before implementation

**File**: `mcp/servers_v2/lse_server_v2.py`

**Impact**: Expected 60-70% improvement in tool selection accuracy for NDC Align queries

---

## Priority 1: GetSubnationalGovernance (CRITICAL for Transparency Queries)

### Current Docstring (Line 1209)
```python
"""Retrieve Brazilian subnational governance data."""
```

### Enhanced Docstring (Proposed)
```python
"""
Retrieve Brazilian subnational climate governance data for specific states.

USE FOR queries about:
• State-level climate laws and policies
• Public participation and transparency mechanisms
• State governance structures and institutions
• Monitoring and accountability systems
• Comparing policies across Brazilian states

PARAMETERS:
- state (str, optional): State name (e.g., "Acre", "São Paulo", "Amazonas")
  - Use None to get overview of all states
- metric (str, optional): Filter by topic to find specific information:
  - "transparency" - finds public participation, disclosure, accountability
  - "participation" - stakeholder engagement, civil society involvement
  - "monitoring" - tracking, reporting, verification systems
  - "coordination" - governance bodies, institutional frameworks
  - Use None to get all governance information

EXAMPLES:
• "What transparency does Acre have?"
  → GetSubnationalGovernance(state="Acre", metric="transparency")

• "How does São Paulo ensure public participation?"
  → GetSubnationalGovernance(state="São Paulo", metric="participation")

• "Compare states' monitoring systems"
  → GetSubnationalGovernance(state=None, metric="monitoring")

TIP: Transparency information is embedded in public participation questions,
so metric="transparency" will search across participation, disclosure, and
accountability fields.

RETURNS: Detailed governance records including questions, summaries,
sources, and implementation status for the specified state(s) and metric.
"""
```

**Why This Matters**: This directly solves the client's transparency query issue. The Tool Planner LLM will now see:
- "USE FOR... transparency mechanisms" → knows when to pick this tool
- Parameter guidance → passes better arguments
- "TIP: Transparency information is embedded" → solves the data structure problem

---

## Priority 2: GetLSEDatasetOverview (PREVENT Wrong Tool Selection)

### Current Docstring (Line 1533)
```python
"""Provide high-level dataset overview."""
```

### Enhanced Docstring (Proposed)
```python
"""
Provide high-level metadata about the NDC Align dataset structure.

USE FOR:
• Understanding what data categories are available
• Getting dataset version and source information
• Learning about the overall dataset organization

DON'T USE FOR specific queries:
❌ Don't use for state-level policy details (use GetStateClimatePolicy)
❌ Don't use for transparency mechanisms (use GetSubnationalGovernance)
❌ Don't use for NDC targets (use GetNDCTargets)
❌ Don't use for institutional info (use GetInstitutionalFramework)

This tool returns METADATA ONLY - no actual policy content.

If the user asks for specific information about Brazil's climate policies,
governance, or NDC commitments, use the specific tools instead.

RETURNS: Dataset catalog metadata including file structure, modules, and
data source information.
"""
```

**Why This Matters**: Prevents the Tool Planner from using the metadata tool when it should use detail tools.

---

## Priority 3: GetStateClimatePolicy (Clarify Distinction)

### Current Docstring (Line 1665)
```python
"""Retrieve the full policy table for a specific Brazilian state."""
```

### Enhanced Docstring (Proposed)
```python
"""
Retrieve the complete climate policy questionnaire for a specific Brazilian state.

USE FOR queries about:
• Full policy overview for one specific state
• All governance questions and answers for a state
• Complete implementation status for a state
• Detailed state climate law information

PARAMETER:
- state_name (str, required): Name of Brazilian state
  Examples: "Acre", "São Paulo", "Amazonas", "Mato Grosso"
  Also accepts state codes: "AC", "SP", "AM", "MT"

DIFFERENCE FROM GetSubnationalGovernance:
• GetStateClimatePolicy: Returns ALL data for ONE specific state
• GetSubnationalGovernance: Can filter across MULTIPLE states by metric

WHEN TO USE WHICH:
→ Use GetStateClimatePolicy when asking about ONE specific state's full policy
→ Use GetSubnationalGovernance when filtering by topic (transparency, monitoring)
  or comparing multiple states

EXAMPLES:
• "What are all of Acre's climate policies?"
  → GetStateClimatePolicy(state_name="Acre")

• "Tell me everything about São Paulo's climate governance"
  → GetStateClimatePolicy(state_name="São Paulo")

RETURNS: Complete questionnaire with all governance questions, summaries,
responses, and implementation status for the specified state.
"""
```

**Why This Matters**: Clarifies when to use this tool vs. GetSubnationalGovernance.

---

## Priority 4: GetInstitutionalFramework (Add Examples)

### Current Docstring (Line 1133)
```python
"""Fetch institutional governance entries, optionally filtered by topic."""
```

### Enhanced Docstring (Proposed)
```python
"""
Fetch Brazilian institutional climate governance frameworks, optionally filtered by topic.

USE FOR queries about:
• National-level coordination mechanisms (not state-level)
• Institutional roles and mandates
• Government bodies responsible for climate policy
• Interministerial coordination structures
• Climate policy integration across sectors

AVAILABLE TOPICS:
• "coordination" - Bodies that coordinate climate action across government
• "direction" or "direction setting" - Institutions setting climate policy direction
• "knowledge" or "evidence" - Bodies providing climate science and evidence
• "participation" - Stakeholder and public engagement mechanisms
• "integration" - Cross-sectoral policy integration structures

PARAMETER:
- topic (str, optional): Filter by topic area
  - Use None to get all institutional frameworks

EXAMPLES:
• "What institutions coordinate Brazil's climate policy?"
  → GetInstitutionalFramework(topic="coordination")

• "Which bodies provide climate evidence for policy making?"
  → GetInstitutionalFramework(topic="knowledge")

• "How does Brazil integrate climate policy across sectors?"
  → GetInstitutionalFramework(topic="integration")

KEY INSTITUTIONS DOCUMENTED:
• CIM (Interministerial Committee on Climate Change) - coordination
• INPE (National Institute for Space Research) - emissions monitoring
• Climate Observatory - civil society participation

NOTE: This is for NATIONAL-level institutions. For STATE-level institutions,
use GetSubnationalGovernance or GetStateClimatePolicy.

RETURNS: Institutional framework records organized by topic, including
institution names, roles, mandates, and coordination mechanisms.
"""
```

**Why This Matters**: Provides clear examples of institutional queries and clarifies national vs. state distinction.

---

## Priority 5: GetNDCTargets (Emphasize Alignment)

### Current Docstring (Line 1734)
```python
"""Extract key NDC targets and commitments."""
```

### Enhanced Docstring (Proposed)
```python
"""
Extract Brazil's key NDC targets and commitments from its Nationally Determined Contribution.

USE FOR queries about:
• Brazil's official NDC commitments (2025, 2030, 2035, 2050)
• Long-term climate neutrality targets
• Emissions reduction percentages
• Sectors covered by the NDC (energy, land use, industry, etc.)
• Greenhouse gases covered
• Conditionality of targets (conditional vs. unconditional)

SPECIALIZATION: This tool focuses on NDC-to-domestic alignment analysis.
Use this when comparing Brazil's international commitments to national implementation.

PARAMETER:
- country (str, optional): Currently only "Brazil" is supported (default)

EXAMPLES:
• "What is Brazil's 2050 climate target in its NDC?"
  → GetNDCTargets(country="Brazil")

• "What are Brazil's interim NDC targets for 2030 and 2035?"
  → GetNDCTargets()

• "What greenhouse gases does Brazil's NDC cover?"
  → GetNDCTargets()

KEY TARGETS DOCUMENTED:
• Long-term: Climate neutrality by 2050
• 2035: 50% below 2005 levels
• 2030: 48-53% below 2005 levels
• Covers all sectors and all GHGs

NOTE: For comparing NDC targets to domestic law, also use GetNDCPolicyComparison
to see alignment analysis.

RETURNS: Structured NDC target information including percentages, base years,
sectors covered, gases covered, and conditionality status.
"""
```

**Why This Matters**: Emphasizes that NDC Align specializes in alignment analysis, not just listing targets.

---

## Priority 6: GetNDCPolicyComparison (Highlight Unique Value)

### Current Docstring (Line 1815+)

(Need to read this section first)

### Enhanced Docstring (Proposed - Placeholder)
```python
"""
Compare Brazil's NDC commitments to its domestic laws and policies (ALIGNMENT ANALYSIS).

USE FOR queries about:
• Whether NDC targets are reflected in domestic law
• Gaps between international commitments and national implementation
• Legal status of NDC commitments domestically
• Implementation pathways for NDC targets
• Alignment between NDC and national climate frameworks

THIS IS NDC ALIGN'S CORE STRENGTH: Analyzing the gap between Brazil's
international NDC commitments and its domestic policy implementation.

EXAMPLES:
• "Is Brazil's 2050 climate neutrality target in domestic law?"
  → GetNDCPolicyComparison()

• "How do Brazil's NDC commitments compare to its national laws?"
  → GetNDCPolicyComparison()

• "Are Brazil's NDC targets legally binding domestically?"
  → GetNDCPolicyComparison()

RETURNS: Detailed comparison showing which NDC targets are reflected in
domestic law, policy, or practice, including implementation status and gaps.
"""
```

---

## Priority 7: GetClimatePolicy (Add Context)

### Current Docstring (Line 1169)
```python
"""Retrieve climate plans and policies entries by policy type."""
```

### Enhanced Docstring (Proposed)
```python
"""
Retrieve Brazil's national climate plans and policies by policy type.

USE FOR queries about:
• National climate frameworks (PNMC, Plano Clima)
• Cross-cutting climate policies
• Sectoral adaptation plans (agriculture, water, coastal, etc.)
• Sectoral mitigation plans (energy, transport, REDD+, etc.)

AVAILABLE POLICY TYPES:
• "cross-cutting" - National frameworks covering all sectors
• "adaptation" - Sectoral adaptation plans
• "mitigation" - Sectoral mitigation plans
• Use None to get all plans and policies

PARAMETER:
- policy_type (str, optional): Filter by type
  - "cross-cutting", "adaptation", "mitigation", or None

EXAMPLES:
• "What is Brazil's national climate policy framework?"
  → GetClimatePolicy(policy_type="cross-cutting")

• "What sectoral adaptation plans does Brazil have?"
  → GetClimatePolicy(policy_type="adaptation")

• "What are Brazil's REDD+ policies?"
  → GetClimatePolicy(policy_type="mitigation")

KEY POLICIES DOCUMENTED:
• PNMC (National Policy on Climate Change) - cross-cutting
• Plano Clima (Climate Plan) - cross-cutting
• Sectoral adaptation plans for agriculture, water, biodiversity
• Sectoral mitigation plans for energy, transport, land use

NOTE: This covers NATIONAL plans. For STATE-level policies, use
GetStateClimatePolicy or GetSubnationalGovernance.

RETURNS: Plans and policies organized by type, including policy names,
objectives, implementation status, and sectoral coverage.
"""
```

---

## Priority 8: CompareBrazilianStates (Add Examples)

### Current Docstring (Line 1688)
```python
"""Compare policy coverage across multiple Brazilian states."""
```

### Enhanced Docstring (Proposed)
```python
"""
Compare climate policy coverage and governance across multiple Brazilian states.

USE FOR queries about:
• Comparing climate laws between states
• Ranking states by policy comprehensiveness
• Comparing transparency mechanisms across states
• Analyzing which states have the strongest climate action
• Regional differences in climate governance

PARAMETERS:
- states (List[str], required): List of state names to compare
  Examples: ["Acre", "Amazonas"], ["São Paulo", "Rio de Janeiro", "Minas Gerais"]
  Also accepts state codes: ["AC", "AM"], ["SP", "RJ", "MG"]
- policy_area (str, optional): Filter comparison to specific topic
  - "transparency", "participation", "monitoring", "adaptation", etc.

EXAMPLES:
• "Compare climate policies between São Paulo and Rio de Janeiro"
  → CompareBrazilianStates(states=["São Paulo", "Rio de Janeiro"])

• "Compare transparency mechanisms between Acre and Amazonas"
  → CompareBrazilianStates(states=["Acre", "Amazonas"], policy_area="transparency")

• "Which Amazon states have the strongest climate laws?"
  → CompareBrazilianStates(states=["Acre", "Amazonas", "Rondônia", "Roraima"])

RETURNS: Comparative metrics showing:
• Number of policies for each state
• Coverage percentages (% of questions answered "yes")
• Policy area specifics if filtered
• Implementation status comparisons
"""
```

---

## Priority 9: SearchLSEContent (Add Semantic Alias Guidance)

(Need to find this tool first - it may be in the legacy server)

### Enhanced Docstring (Proposed - Placeholder)
```python
"""
Search across all NDC Align content using semantic similarity.

USE FOR queries about:
• Topics that don't fit cleanly into other tools
• Exploring related concepts across multiple data modules
• Finding mentions of specific institutions, laws, or policies
• Cross-cutting searches that span multiple categories

PARAMETER:
- query (str, required): Search query text

SEMANTIC UNDERSTANDING:
The search understands related terms:
• "transparency" matches: public participation, disclosure, accountability, monitoring
• "coordination" matches: governance, institutional frameworks, interministerial
• "target" matches: goal, commitment, objective, NDC
• "deforestation" matches: REDD+, forest, land use, Amazon

TIP: If you need specific data types, use the specialized tools instead:
• State governance → GetSubnationalGovernance
• NDC targets → GetNDCTargets
• Institutions → GetInstitutionalFramework

EXAMPLES:
• "Find all mentions of INPE in climate policy"
  → SearchLSEContent(query="INPE")

• "Search for climate adaptation in agriculture"
  → SearchLSEContent(query="agricultural adaptation")

RETURNS: Top matching records across all data modules, ranked by semantic
similarity, including source, context, and citation information.
"""
```

---

## Summary: Expected Improvements

### Before Enhancement:
**Query**: "What transparency measures do Brazilian states have?"

**Tool Planner sees**:
- GetLSEDatasetOverview: "Provide high-level dataset overview"
- GetSubnationalGovernance: "Retrieve Brazilian subnational governance data"

**Planner thinks**: "Overview sounds good for this broad question"

**Result**: ❌ Calls GetLSEDatasetOverview → Returns metadata only

---

### After Enhancement:
**Query**: "What transparency measures do Brazilian states have?"

**Tool Planner sees**:
- GetLSEDatasetOverview: "DON'T USE FOR transparency mechanisms (use GetSubnationalGovernance)"
- GetSubnationalGovernance: "USE FOR... Public participation and transparency mechanisms... TIP: Transparency information is embedded in public participation questions"

**Planner thinks**: "Perfect! GetSubnationalGovernance explicitly says use for transparency"

**Result**: ✅ Calls GetSubnationalGovernance(state=None, metric="transparency") → Returns detailed transparency mechanisms

---

## Implementation Checklist

### Phase 1: Critical Tools (30 minutes)
- [ ] GetSubnationalGovernance (line 1209)
- [ ] GetLSEDatasetOverview (line 1533)

### Phase 2: Supporting Tools (1 hour)
- [ ] GetStateClimatePolicy (line 1665)
- [ ] GetInstitutionalFramework (line 1133)
- [ ] GetNDCTargets (line 1734)

### Phase 3: Alignment & Comparison (1 hour)
- [ ] GetNDCPolicyComparison (line 1815+)
- [ ] GetClimatePolicy (line 1169)
- [ ] CompareBrazilianStates (line 1688)

### Phase 4: Search Tool (30 minutes)
- [ ] SearchLSEContent (if it exists)

---

## Testing Plan

After implementing enhanced docstrings, test with these queries:

### Test 1: Transparency (Client's Original Issue)
```
"What transparency measures do Brazilian states have for climate policy?"
```
**Expected**: Should call GetSubnationalGovernance with metric="transparency"

### Test 2: Specific State Transparency
```
"What institutions does Acre state have to ensure transparency in climate policy?"
```
**Expected**: Should call GetSubnationalGovernance(state="Acre", metric="transparency")

### Test 3: NDC Alignment
```
"Is Brazil's 2050 climate neutrality target in domestic law?"
```
**Expected**: Should call GetNDCPolicyComparison (or both GetNDCTargets + GetNDCPolicyComparison)

### Test 4: State Comparison
```
"Compare transparency between Acre and São Paulo states"
```
**Expected**: Should call CompareBrazilianStates(states=["Acre", "São Paulo"], policy_area="transparency")

### Test 5: Institutional Framework
```
"What is the role of CIM in Brazil's climate governance?"
```
**Expected**: Should call GetInstitutionalFramework(topic="coordination")

---

## Maintenance Notes

**Where docstrings are used**:
- ServerToolPlanner reads them when selecting tools (`mcp/mcp_chat_v2.py:1693-1742`)
- They are the PRIMARY input for tool selection decisions
- More detail = better tool selection

**Best practices for future tools**:
1. Start with "USE FOR queries about:" section
2. Provide clear parameter guidance with examples
3. Include 3-5 example queries with expected parameters
4. Add "DON'T USE FOR" warnings if ambiguous
5. Clarify distinctions from similar tools
6. Include semantic tips for embedded data

**Testing after changes**:
- Run all queries in `test_scripts/test_ndc_align_datasets.md`
- Verify tool selection in API logs
- Check that transparency queries now work correctly
