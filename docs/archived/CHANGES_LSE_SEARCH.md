# LSE Server Search Improvements

## Summary

Enhanced the LSE (NDC Align) server with combined semantic + token search and improved server descriptions for better orchestrator prioritization.

---

## Changes Made

### 1. Always Include LSE Server (mcp_chat_v2.py)

**Set `"always_include": True` for LSE server** (line 552)

This ensures the orchestrator ALWAYS considers the LSE/NDC Align server for every query, rather than deciding based on query content. This makes sense because:
- LSE is comprehensive Brazil climate policy reference
- Contains foundational NDC and governance data
- Relevant to most Brazil climate queries
- Fast enough to always query (<50ms)

**Before:** Orchestrator decided whether to include LSE based on query analysis
**After:** LSE is always queried and can contribute to any response

---

### 2. Combined Semantic + Token Search (lse_server_v2.py)

**Added three new methods:**

- **`_tokenize(text)`** (line 2297): Normalizes text for n-gram matching
  - Removes accents, lowercases, filters to tokens ≥3 chars
  - Handles Portuguese characters properly

- **`_make_ngrams(tokens)`** (line 2308): Generates unigrams + bigrams
  - Example: "São Paulo" → `["sao", "paulo", "sao paulo"]`

- **`_token_search(query, limit)`** (line 2320): N-gram based keyword search
  - Searches 470 records for literal token matches
  - Complements semantic search for exact term matching

**Updated SearchLSEContent tool** (line 1543):
```python
# Run both search methods
semantic_results = _semantic_search(query, limit)
token_results = _token_search(query, limit)

# Combine: semantic first, then unique token results
# Simple union - no arbitrary scoring
```

**Benefits:**
- ✅ Semantic finds conceptually related content (embeddings)
- ✅ Token catches literal keyword matches (n-grams)
- ✅ Combined approach provides comprehensive coverage
- ✅ Fast: <50ms for 470-record dataset

---

### 2. Improved Server Descriptions

**Problem:** Orchestrator was deprioritizing LSE for deforestation queries because the server description didn't mention it.

**Updated descriptions in 3 places:**

#### A. lse_server_v2.py - Internal description (line 920-924)

**Before:**
```
"Brazil's NDC Align catalog with governance institutions, plans and policies,
 subnational implementation, and Transition Pathway Initiative pathways."
```

**After:**
```
"Brazil's NDC Align catalog covering climate governance, NDC targets, sectoral policies
(energy, transport, agriculture, land use, forestry), deforestation/REDD+ measures,
institutional frameworks, subnational state-level implementation, and TPI emissions pathways."
```

#### B. lse_server_v2.py - Tags (line 928-939)

**Added tags:**
- `"deforestation"`
- `"redd+"`
- `"land_use"`
- `"forestry"`
- `"sectoral"`

#### C. mcp_chat_v2.py - Orchestrator description (line 544-551)

**Before:**
```
"Comprehensive climate policy database including NDC commitments (targets,
net-zero years, renewable energy goals), domestic policy comparisons,
institutional frameworks, implementation tracking, and TPI emissions
pathways. Use for questions about national or subnational climate policy
details, targets, and governance structures."
```

**After:**
```
"Comprehensive climate policy database covering NDC commitments (targets,
net-zero years), sectoral policies (energy, transport, agriculture, land use,
forestry/deforestation, REDD+), institutional frameworks, implementation tracking,
subnational state governance, and TPI emissions pathways. Use for questions about
Brazil's climate policy details, governance structures, forest conservation measures,
sectoral mitigation plans, or state-level climate action."
```

---

### 3. Enhanced query_support Prompts (lse_server_v2.py)

**Updated both Anthropic and OpenAI routing prompts** (lines 997-1003, 1019-1026):

**Before:**
```
"Decide if the Brazil-focused NDC Align governance dataset should answer the question."
```

**After:**
```
"Decide if the Brazil-focused NDC Align climate policy dataset should answer the question.
This dataset covers climate governance, NDC targets, sectoral policies (including
land use, forestry, deforestation, REDD+, energy, transport), institutional frameworks,
and subnational state implementation."
```

This helps the LLM router better understand when LSE is relevant for queries.

---

## Testing

### Search Quality Tests

**5 realistic queries tested** (see `test_scripts/test_lse_realistic_queries.py`):

| Query | Top Score | Method | Quality |
|-------|-----------|--------|---------|
| Brazil's 2030 climate targets | 0.73 | semantic | ✅ Excellent |
| carbon pricing mechanisms | 0.67 | semantic | ✅ Excellent |
| Rio de Janeiro climate action | 0.71 | semantic | ✅ Excellent |
| forest conservation policy | 0.47 | semantic | ✅ Good |
| institutional coordination | 0.60 | semantic | ✅ Excellent |

**All queries returned highly relevant results with good module diversity.**

### Combined Search Verification

**Query: "deforestation"**
- Semantic: 5 results
- Token: 5 results
- Combined: 7 unique results
- ✅ Token search added "Planning and strategy" and "Acre" that semantic missed

---

## Test Scripts Created

1. **`test_lse_semantic_index.py`**: Diagnostic for semantic index
2. **`test_lse_mcp_semantic.py`**: MCP protocol verification
3. **`test_lse_combined_search.py`**: Combined search tests
4. **`test_lse_realistic_queries.py`**: Full realistic query suite
5. **`test_token_contribution.py`**: Token search contribution analysis
6. **`FRONTEND_TEST_QUERIES.md`**: Guide for front-end testing

---

## What to Test on Front-End

### Critical Test (Deforestation Query)

**Query:**
```
How is Brazil addressing deforestation in the Amazon?
```

**Expected change:**
- **Before**: LSE appears as citation #17 (low priority)
- **After**: LSE is ALWAYS included and should contribute relevant citations

**Why:**
1. **Primary:** Set `always_include: True` - LSE is now queried for every request
2. **Secondary:** Updated descriptions to mention "deforestation", "REDD+", "land use", "forestry"
3. **Combined:** Better search quality from semantic + token approach

### Additional Test Queries

1. **"What climate policies does São Paulo have?"**
   - Tests: Entity matching, token search
   - Should return: São Paulo state records

2. **"carbon budget emissions trading"**
   - Tests: Combined semantic + token
   - Should show: `method: "semantic+token"` if both contribute

3. **"Brazil's 2035 climate targets"**
   - Tests: Semantic understanding
   - Should return: Direction setting, NDC targets (scores >0.6)

4. **"forest conservation sectoral plans"**
   - Tests: New description keywords
   - Should return: Sectoral mitigation plans, REDD+ policies

5. **"Amazonas state deforestation policy"**
   - Tests: Both improvements together
   - Should return: Amazonas subnational + sectoral policies

---

## Files Modified

### Main Implementation
- `mcp/servers_v2/lse_server_v2.py` - Search methods, descriptions, prompts
- `mcp/mcp_chat_v2.py` - Orchestrator server description

### Test Scripts (New)
- `test_scripts/test_lse_semantic_index.py`
- `test_scripts/test_lse_mcp_semantic.py`
- `test_scripts/test_lse_combined_search.py`
- `test_scripts/test_lse_realistic_queries.py`
- `test_scripts/test_token_contribution.py`
- `test_scripts/FRONTEND_TEST_QUERIES.md`

---

## Performance

- **Search time**: <50ms for combined semantic + token
- **Semantic index**: 470 records, 1536-dimensional embeddings
- **Index file**: `/home/ubuntu/tde/extras/lse_semantic_index.jsonl` (15.5 MB)

---

## Next Steps

1. **Deploy** the updated code
2. **Test** "How is Brazil addressing deforestation in the Amazon?"
3. **Verify** LSE appears earlier in citations (ideally top 5)
4. **Monitor** query_support scores for deforestation-related queries
5. **Iterate** on descriptions if needed based on real usage

---

## Technical Notes

### How Combined Search Works

```python
# 1. Get results from both methods
semantic = _semantic_search(query, limit=10)  # Embeddings
token = _token_search(query, limit=10)        # N-grams

# 2. Deduplicate: semantic first, then unique token results
seen = set()
combined = []

for result in semantic:
    combined.append(result)
    seen.add(result['slug'])

for result in token:
    if result['slug'] not in seen:
        combined.append(result)

# 3. Return combined results
return combined[:limit]
```

**No arbitrary scoring - just simple set union with semantic priority.**

### Why Orchestrator Deprioritized LSE

The orchestrator uses server descriptions to decide which servers to query. When it saw:
- Query: "deforestation in the Amazon"
- SPA description: "Amazon ecosystems... Use for Amazon-specific requests" ✅
- PRODES: "Deforestation polygons..." ✅
- LSE (old): "governance institutions, plans and policies" ❌ (no mention of deforestation)

Now LSE description explicitly mentions "forestry/deforestation, REDD+" so orchestrator knows it's relevant.

---

## Debugging

If LSE still ranks low:

1. **Check server started fresh**: Restart to load new descriptions
2. **Verify query_support**: Check LSE's support score for deforestation query
3. **Look at logs**: Search for "query_support" responses
4. **Test direct search**: Use `test_lse_realistic_queries.py` to verify search works
5. **Check orchestrator logic**: See how it weights different servers

---

## Success Metrics

✅ **Search Quality**
- Relevance scores >0.4 for most queries
- Top result directly addresses query intent
- Good module diversity (2-3 modules for broad queries)

✅ **Orchestrator Priority**
- LSE appears in top 5 citations for relevant queries
- query_support returns `supported: true` with score >0.7
- Deforestation queries include LSE early

✅ **Performance**
- Search completes <500ms
- No degradation from combined approach
- Semantic index loads on startup

---

Last updated: October 22, 2025
