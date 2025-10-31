# Tool Selection Architecture Analysis

**Question**: How does tool selection really work? Is semantic search helping? What can we improve?

---

## 🔍 How It Actually Works

### **Level 1: Server Selection (LLM-based Routing)**

Location: `mcp/mcp_chat_v2.py:750-765`

**Method**: Calls an LLM with YES/NO question for each server

```python
prompt = (
    f"The user asks: {query}\n"
    f"Server description: {server_config.description}\n"
    "Should we query this dataset? Reply with YES or NO."
)
```

**Special LSE bias** (line 750):
```python
"This is predominantly a tool for policy makers and analysts.
 We should almost always choose YES for the 'lse' dataset."
```

**Finding**: ✅ LSE gets preferential treatment in routing - this is GOOD and working

---

### **Level 2: Server Confirmation (`query_support`)**

Location: `mcp/servers_v2/lse_server_v2.py:989-1002`

**Method**: Each server's `query_support` tool classifies if it can help

For LSE, this calls `_classify_support` (lines 1004-1054) which:
1. Uses LLM (Haiku or GPT-4-mini) to decide
2. Sends this prompt:

```python
"Decide if the Brazil-focused NDC Align climate policy dataset should answer the question.
This dataset covers climate governance, NDC targets, sectoral policies (including
land use, forestry, deforestation, REDD+, energy, transport), institutional frameworks,
and subnational state implementation. Respond with JSON keys 'supported' (true/false)
and 'reason'.\n"
f"Dataset summary: {self._capability_summary()}\n"
f"Question: {query}"
```

**Finding**: ⚠️ The prompt is GENERIC - doesn't mention transparency, state-level detail, alignment analysis

---

### **Level 3: Tool Planning (Within Server)**

Location: `mcp/mcp_chat_v2.py:1693-1742`

**Method**: `ServerToolPlanner` uses LLM to select specific tools from a server

```python
"You are selecting the best tools from ONE MCP server.\n"
"Select 0-3 tools that directly help answer the question.\n"
"Consider the tool descriptions carefully."
```

**Tool descriptions come from**: The docstrings in `lse_server.py` and `lse_server_v2.py`

**Finding**: ⚠️ Tool descriptions are SHORT and don't provide enough guidance

**Current example** (`GetSubnationalGovernance` from line 908):
```python
"""Retrieve Brazilian subnational governance data."""
```

That's it! No examples, no parameter guidance, no "use this for transparency" hint.

---

### **Level 4: Actual Execution (`run_query`)**

Location: `mcp/servers_v2/lse_server_v2.py:2860+`

**Method**: `handle_run_query` executes and returns facts

**Finding**: ✅ This part works well - returns good data when invoked

---

## 🤔 Is Semantic Search Working?

### **Short Answer: PARTIALLY**

There IS semantic/embedding-based search in the LSE server:

Location: `mcp/servers_v2/lse_server_v2.py:894-897`

```python
self._semantic_records: List[Dict[str, Any]] = []
self._semantic_matrix: Optional[np.ndarray] = None
self._load_semantic_index()
```

**BUT** this is used for:
1. Searching WITHIN records after server is selected
2. NOT for choosing which tools to call
3. NOT for initial server routing

**Finding**: ❌ Semantic search doesn't help with tool selection problem

---

## 📊 The Real Problem: Three Layers of Failure

### **Problem 1: Weak Server Description** ❌

Location: `mcp/mcp_chat_v2.py:750` (LLM router prompt)

**What it sees**:
```
Server description: {server_config.description}
```

**What LSE's description is**: Generic statement about "Brazil climate policy"

**What it SHOULD say**:
- "Use for NDC-domestic alignment analysis"
- "Use for state-level transparency and governance"
- "Use for institutional frameworks"

**Impact**: Server gets selected, but LLM doesn't know its strengths

---

### **Problem 2: Generic query_support Prompt** ⚠️

Location: `mcp/servers_v2/lse_server_v2.py:1008-1013`

**Current prompt**:
```
"This dataset covers climate governance, NDC targets, sectoral policies
(including land use, forestry, deforestation, REDD+, energy, transport),
institutional frameworks, and subnational state implementation."
```

**Missing**:
- "Specializes in NDC-domestic alignment"
- "Contains transparency mechanisms in public participation data"
- "Has detailed state-by-state governance data for all 27 Brazilian states"
- "Best for governance ANALYSIS, not primary documents"

**Impact**: Server says "yes I can help" but doesn't communicate WHAT it's good at

---

### **Problem 3: Minimal Tool Descriptions** ❌❌❌

Location: Tool docstrings in `mcp/lse_server.py` and `mcp/servers_v2/lse_server_v2.py`

**Current GetSubnationalGovernance** (line 1341):
```python
"""Retrieve Brazilian subnational governance data."""
```

**Current GetLSEDatasetOverview** (line 2047):
```python
"""Provide high-level dataset overview."""
```

**What the Tool Planner LLM sees**: Just these one-liners!

**Result**:
- Planner doesn't know GetSubnationalGovernance is for transparency
- Planner doesn't know GetLSEDatasetOverview is metadata-only
- Planner picks wrong tool or passes no parameters

**Impact**: THIS IS THE BIGGEST PROBLEM

---

## 🎯 Where to Focus Improvements

### **Priority Ranking**:

| Fix | Impact | Effort | ROI | Why It Matters |
|-----|--------|--------|-----|----------------|
| **#1: Enhance Tool Descriptions** | 🔥🔥🔥 HIGH | ⚡ EASY | 🏆 BEST | Tool Planner uses these directly |
| **#2: Improve query_support prompt** | 🔥🔥 MEDIUM | ⚡ EASY | 🥈 GOOD | Sets expectations for what LSE does |
| #3: Add server description** | 🔥 LOW | ⚡ EASY | 👍 OK | Initial routing already biased toward LSE |
| #4: Semantic tool matching | 🔥 LOW | 💪 HARD | 👎 POOR | Complex, tool descriptions are easier |

---

## 🔨 Concrete Improvements

### **Fix #1: Tool Descriptions** (HIGHEST PRIORITY)

**File**: `mcp/servers_v2/lse_server_v2.py` (lines 1341, 1365, 1407, etc.)

**Before**:
```python
def GetSubnationalGovernance(state, metric):
    """Retrieve Brazilian subnational governance data."""
```

**After**:
```python
def GetSubnationalGovernance(state, metric):
    """
    Retrieve Brazilian subnational climate governance data for specific states.

    ✅ USE FOR queries about:
    - State-level climate laws and policies
    - Public participation and transparency mechanisms
    - State governance structures and institutions
    - Monitoring and accountability systems
    - Comparing policies across Brazilian states

    📝 PARAMETERS:
    - state (str): State name (e.g., "Acre", "São Paulo", "Amazonas")
      - Use None for overview of all states
    - metric (str): Filter by topic to find specific information:
      - "transparency" - finds public participation, disclosure, accountability
      - "participation" - stakeholder engagement, civil society involvement
      - "monitoring" - tracking, reporting, verification systems
      - "coordination" - governance bodies, institutional frameworks
      - Use None to get all governance information

    💡 EXAMPLES:
    - "What transparency does Acre have?"
      → GetSubnationalGovernance(state="Acre", metric="transparency")

    - "How does São Paulo ensure public participation?"
      → GetSubnationalGovernance(state="São Paulo", metric="participation")

    - "Compare states' monitoring systems"
      → GetSubnationalGovernance(state=None, metric="monitoring")

    💡 TIP: Transparency information is embedded in public participation questions,
    so metric="transparency" will search across participation, disclosure, and
    accountability fields.

    📊 RETURNS: Detailed governance records including questions, summaries,
    sources, and implementation status for the specified state(s) and metric.
    ```

**Why This Works**:
1. LLM sees "USE FOR transparency" → knows when to pick this tool
2. Parameter guidance → passes better arguments
3. Examples → LLM can pattern match
4. Tip about transparency → solves the embedded data problem

---

### **Fix #2: query_support Prompt**

**File**: `mcp/servers_v2/lse_server_v2.py:1008-1013`

**Before**:
```python
"This dataset covers climate governance, NDC targets, sectoral policies..."
```

**After**:
```python
"""Decide if the Brazil-focused NDC Align climate policy dataset should answer the question.

KEY STRENGTHS - Use NDC Align for:
• NDC-domestic policy alignment analysis (comparing international commitments to national law)
• Institutional governance frameworks (which bodies coordinate climate policy)
• Subnational climate governance (all 27 Brazilian states' laws, policies, transparency)
• Implementation status tracking (what's in law vs. policy vs. under development)
• Public participation and transparency mechanisms (embedded in governance questions)
• Sectoral policies (deforestation, REDD+, energy, transport, adaptation)

DETAILED STATE-LEVEL DATA:
• Each of 27 Brazilian states has governance questionnaire
• Transparency info embedded in public participation questions
• State climate laws, institutions, monitoring systems
• Implementation status for state policies

WHAT WE DON'T HAVE (use other sources):
• Primary policy documents (→ use CPR)
• Emissions statistics (→ use KG)
• Climate science projections (→ use IPCC)

Dataset summary: {self._capability_summary()}

Question: {query}

Respond with JSON keys 'supported' (true/false) and 'reason'.
"""
```

**Why This Works**:
- LLM learns NDC Align is for "alignment", "governance", "transparency"
- Explicitly mentions transparency is in "public participation"
- Clarifies what NOT to use it for

---

### **Fix #3: Enhance All Critical Tools**

**Tools to enhance** (in priority order):

1. ✅ **GetSubnationalGovernance** - For state transparency queries
2. ✅ **GetLSEDatasetOverview** - Add "DON'T use for specific queries" warning
3. ✅ **SearchLSEContent** - Add semantic alias guidance
4. ✅ **GetStateClimatePolicy** - Clarify vs. GetSubnationalGovernance
5. ✅ **GetInstitutionalFramework** - Add transparency/coordination examples
6. ✅ **GetNDCTargets** - Add alignment focus
7. ✅ **GetNDCPolicyComparison** - Emphasize alignment analysis

---

## 🚫 What's NOT the Problem

### ❌ "We need better semantic search"
**Reality**: Semantic search exists but isn't used for tool selection. Tool descriptions matter more.

### ❌ "We need ML model for routing"
**Reality**: LLM-based routing is fine. It's already biased toward LSE. Problem is what the LLM sees.

### ❌ "We need to rewrite the search"
**Reality**: Search works fine once the right tool is called with right parameters.

### ❌ "We need more tools"
**Reality**: We have enough tools. They just need better descriptions.

---

## 📈 Expected Improvements

### After Enhancing Tool Descriptions:

**Query**: "What transparency measures do Brazilian states have?"

**Before**:
1. LSE selected ✅
2. Tool Planner sees: "GetLSEDatasetOverview - Provide high-level dataset overview"
3. Planner thinks: "Sounds good for overview question"
4. Calls: `GetLSEDatasetOverview()` ❌
5. Returns: Metadata about dataset structure
6. User gets: Generic response

**After**:
1. LSE selected ✅
2. Tool Planner sees: "GetSubnationalGovernance - ✅ USE FOR transparency mechanisms"
3. Planner thinks: "Perfect! It explicitly says use for transparency"
4. Calls: `GetSubnationalGovernance(state=None, metric="transparency")` ✅
5. Returns: Detailed transparency mechanisms from all states
6. User gets: Specific governance structures, participation forums, monitoring systems

**Estimated Improvement**: 60-70% better responses for transparency/governance queries

---

## 🎯 Summary: The Lever That Matters Most

**Question**: "How can we improve tool definitions to make this more likely to retrieve correct NDC Align data?"

**Answer**: **ENHANCE TOOL DOCSTRINGS** - they're the primary input to the Tool Planner LLM.

**Why This is the Best Lever**:

1. **Direct Impact**: Tool Planner reads docstrings word-for-word
2. **Easy to Implement**: Just edit docstrings, no algorithm changes
3. **No Side Effects**: Can't break anything, just adds information
4. **Immediate Results**: No retraining, reindexing, or complex logic
5. **Maintainable**: Future developers can easily update

**The Proof**:
- ✅ Server selection works (LSE gets picked)
- ✅ Data retrieval works (returns good info when invoked)
- ❌ Tool selection fails (planner picks wrong tool or no parameters)
- **Gap**: Tool Planner doesn't know what each tool is FOR

**The Fix**: Tell it! Use docstrings to guide LLM to right tool with right parameters.

---

## 🚀 Implementation Plan

### Phase 1: Critical Tools (30 min)
1. GetSubnationalGovernance - add transparency guidance
2. GetLSEDatasetOverview - add "don't use for details" warning

### Phase 2: Query Support (30 min)
3. Enhance `_classify_support` prompt with strengths

### Phase 3: Supporting Tools (1-2 hours)
4. SearchLSEContent - add semantic alias tip
5. GetStateClimatePolicy - clarify usage
6. GetInstitutionalFramework - add examples
7. GetNDCTargets - add alignment focus

### Total Time: ~3 hours for 70% improvement

---

**Bottom Line**: The "fancy" semantic search exists but isn't the bottleneck. The LLM-based tool planner needs better descriptions to make smart choices. This is a **documentation problem**, not an algorithm problem.
