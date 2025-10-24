# LSE Server Front-End Test Queries

Use these queries to test the combined semantic + token search on your deployed front end.

## 🎯 Purpose

These queries are designed to:
- Test both semantic (conceptual) and token (keyword) search
- Cover different data modules (NDC, states, policies, institutions)
- Show real-world use cases
- Verify search quality and relevance

---

## ✅ Recommended Test Queries

### 1. **Specific State Query** (Token-heavy)
```
What climate policies does São Paulo have?
```
**Expected behavior:**
- Should find São Paulo state records
- Token search helps with exact entity name matching
- Results: Subnational governance data for São Paulo

**What to check:**
- Top results are all São Paulo-specific
- Covers climate plans, emissions inventory, governance

---

### 2. **Conceptual Policy Query** (Semantic-heavy)
```
How is Brazil addressing deforestation in the Amazon?
```
**Expected behavior:**
- Semantic search finds conceptually related content
- Should surface sectoral mitigation, REDD+, land use policies
- Results: Mix of national policies and Amazonas state data

**What to check:**
- Results mention forest conservation, land use, REDD+
- Mix of policy frameworks and subnational data
- Relevance scores >0.4

---

### 3. **Technical/Specific Term** (Combined)
```
carbon budget emissions trading
```
**Expected behavior:**
- Both methods contribute
- Token finds literal "carbon" and "emissions" mentions
- Semantic finds related carbon pricing mechanisms

**What to check:**
- Results include carbon pricing policies
- Cross-cutting policies and institutional framework
- Method indicator shows "semantic+token" if both used

---

### 4. **Subnational Comparison** (Multi-entity)
```
Compare climate action in Minas Gerais and Rio de Janeiro
```
**Expected behavior:**
- Should find both states
- Enables comparison of subnational policies
- Token search helps match state names exactly

**What to check:**
- Results include both Minas Gerais and Rio de Janeiro
- Can use CompareBrazilianStates tool for detailed comparison
- Relevant policy areas covered

---

### 5. **NDC Implementation** (Conceptual)
```
What is Brazil's progress on meeting its 2035 climate commitments?
```
**Expected behavior:**
- Semantic search understands "progress", "meeting", "commitments"
- Finds NDC targets, implementation status, direction setting
- Results: National commitments and institutions modules

**What to check:**
- Top results discuss 2035 targets (50% reduction)
- Implementation evidence and monitoring
- Direction setting and governance

---

### 6. **Sector-Specific** (Domain knowledge)
```
renewable energy transition plans
```
**Expected behavior:**
- Semantic finds energy-related mitigation plans
- Cross-cutting policies and sectoral plans
- May include state-level energy initiatives

**What to check:**
- Sectoral mitigation plans appear
- Energy sector coverage
- Both national and subnational results

---

### 7. **Institutional/Governance** (Abstract)
```
Who coordinates climate policy between federal and state governments?
```
**Expected behavior:**
- Semantic handles abstract "who coordinates" question well
- Finds institutional coordination, integration, governance
- Results: Institutions & processes module

**What to check:**
- Coordination and integration records
- Inter-ministerial and federal-state governance
- Clear governance structure information

---

### 8. **Edge Case: Portuguese Term**
```
Amazônia desmatamento
```
**Expected behavior:**
- Token search handles accented characters
- Should normalize "Amazônia" → "amazonia"
- Finds deforestation-related content

**What to check:**
- Tokenization properly handles Portuguese
- Amazonas state results
- Forest/deforestation policies

---

### 9. **Acronym/Short Query**
```
NDC
```
**Expected behavior:**
- Token search excellent for exact matches
- Should find all NDC-related content
- Results: NDC overview module heavily featured

**What to check:**
- High number of results (NDC is central)
- NDC overview, targets, implementation
- All NDC-related modules represented

---

### 10. **Multi-faceted Complex Query**
```
How do Brazilian states monitor and report their greenhouse gas emissions inventories?
```
**Expected behavior:**
- Long conceptual query - semantic excels here
- Finds subnational monitoring, reporting, inventories
- Results: State governance + institutional processes

**What to check:**
- Subnational results discussing emissions inventory
- Reporting requirements and monitoring frameworks
- Multiple states represented (shows pattern, not single state)

---

## 🔍 What to Look For

### Search Quality Indicators

✅ **Good Results:**
- Relevance scores >0.4 for top results
- Top 3 results directly address the query
- Mix of modules when appropriate (diversity)
- Snippets show relevant context

⚠️ **Needs Attention:**
- All scores <0.3 (weak matches)
- Results don't match query intent
- Too much duplication
- Missing obvious matches

### Method Verification

Check the `method` field in responses:
- `"semantic"` - Embeddings found good matches
- `"token"` - N-gram keyword matching used
- `"semantic+token"` - Both contributed results
- `"catalog"` - Fallback (both returned empty - rare)

---

## 💡 Pro Tips for Testing

1. **Try variations:** If "São Paulo emissions" works, try "Sao Paulo emissions" (no tilde)
2. **Test synonyms:** "climate action" vs "climate policy" vs "climate governance"
3. **Mix languages:** Some records may have Portuguese content
4. **Short vs long:** Try both "deforestation" and the complex query above
5. **Watch response time:** All should be <500ms

---

## 📊 Expected Performance Benchmarks

| Metric | Target |
|--------|--------|
| Top result relevance | >0.5 for most queries |
| Results returned | 5-10 per query |
| Response time | <500ms |
| Module diversity | 2-3 modules for broad queries |
| Exact match accuracy | 100% for state names |

---

## 🐛 Debugging Tips

If results seem off:

1. **Check method used:** Look at the `method` field
2. **Verify semantic index:** Should have 470 records loaded
3. **Test simple query first:** Try "São Paulo" - should be obvious match
4. **Compare with test script:** Run `test_lse_realistic_queries.py` locally
5. **Check logs:** Look for semantic index loading messages

---

## Quick Reference: What Each Query Tests

| Query # | Primary Test | Search Method Expected |
|---------|-------------|------------------------|
| 1 | Entity name matching | Semantic (high score) |
| 2 | Conceptual understanding | Semantic |
| 3 | Technical terminology | Semantic+Token |
| 4 | Multi-entity handling | Semantic+Token |
| 5 | Temporal/implementation | Semantic |
| 6 | Sector knowledge | Semantic |
| 7 | Abstract concepts | Semantic |
| 8 | Portuguese/accents | Token |
| 9 | Acronyms | Token |
| 10 | Complex multi-clause | Semantic |

Happy testing! 🚀
