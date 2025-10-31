# NDC Align (LSE Server) Dataset Test Questions

This document contains test questions for all datasets in the NDC Align LSE server, with expected answers to validate API responses.

## Dataset 1: NDC Overview & Domestic Comparison

**Description**: Compares Brazil's NDC commitments with domestic policy implementation
**Module**: `ndc_overview`
**Group**: `national_commitments`
**File**: `1 NDC Overview and Domestic Policy Comparison Content.xlsx`

### Test Question 1.1: Long-term Climate Neutrality Target
**Query**: "What is Brazil's long-term climate neutrality target according to its NDC?"

**Expected Answer Components**:
- Brazil has committed to achieving climate neutrality by 2050
- Covers all greenhouse gases
- This is a political commitment (not yet a formal quantitative target)
- Reinforced by Resolution 3 of September 14, 2023 from CIM
- Status: In existing law/policy and under further development

**Data Source**: NDC Overview question 1a

---

### Test Question 1.2: 2035 Emissions Reduction Target
**Query**: "What are Brazil's interim emissions reduction targets for 2035?"

**Expected Answer Components**:
- Brazil's NDC aims for 59-67% reduction below 2005 levels by 2035
- Covers CO2, CH4, N2O, SF6, PFCs, and HFCs
- Aligned with domestic policy through Resolution CIM No. 3/2023
- National Mitigation Strategy must define targets for 2030 and 2035

**Data Source**: NDC Overview question 2a

---

### Test Question 1.3: 2030 Emissions Target Status
**Query**: "Does Brazil have a formal emissions reduction target for 2030?"

**Expected Answer Components**:
- Brazil's NDC does NOT establish a formal quantitative target for 2030
- Resolution CIM No. 3/2023 indicates intention to define a 2030 target
- Status: Under development
- The Plano Clima will likely include 2030 targets in future regulations

**Data Source**: NDC Overview question 2b

---

## Dataset 2: Institutions & Processes - Coordination

**Description**: Information about institutional coordination for climate governance
**Module**: `institutions`
**Group**: `governance_processes`
**File**: `2 Institutions and Processes Module Content.xlsx`
**Sheet**: `Coordination`

### Test Question 2.1: Climate Coordination Body
**Query**: "What is Brazil's main institutional body for coordinating climate policy?"

**Expected Answer Components**:
- Interministerial Committee on Climate Change (CIM)
- Coordinates climate policy across government ministries
- Issues resolutions to guide climate action implementation
- Key role in updating the National Plan on Climate Change (Plano Clima)

**Data Source**: Institutions - Coordination records

---

### Test Question 2.2: Ministerial Coordination
**Query**: "How does Brazil coordinate climate policy across different government ministries?"

**Expected Answer Components**:
- Through the Interministerial Committee on Climate Change (CIM)
- Establishes Technical Groups on Mitigation (GTT - Mitigation) and Adaptation (GTT - Adaptation)
- Coordinates National Mitigation and Adaptation Strategies
- Involves multiple ministries in climate governance

**Data Source**: Institutions - Coordination records

---

### Test Question 2.3: Sub-national Coordination
**Query**: "How does Brazil coordinate climate policy between federal and state governments?"

**Expected Answer Components**:
- Should reference mechanisms for federal-state coordination
- May mention state-level climate committees
- Could reference the National Climate Change Policy (PNMC)

**Data Source**: Institutions - Coordination records

---

## Dataset 3: Plans & Policies - Cross-Cutting Policies

**Description**: Economy-wide and cross-sectoral climate policies
**Module**: `plans_policies`
**Group**: `policy_frameworks`
**File**: `3 Plans and Policies Module Content.xlsx`
**Sheet**: `Cross Cutting Policies`

### Test Question 3.1: National Climate Change Policy
**Query**: "What is Brazil's main cross-cutting climate policy framework?"

**Expected Answer Components**:
- National Policy on Climate Change (PNMC - Law No. 12.187/2009)
- Economy-wide climate action plan
- Being updated to align with Paris Agreement
- Proposed Bill PL 6,539/2019 seeks to modernize the framework

**Data Source**: Plans & Policies - Cross Cutting Policies

---

### Test Question 3.2: National Climate Change Plan
**Query**: "What is the Plano Clima and what does it include?"

**Expected Answer Components**:
- National Plan on Climate Change (Plano Clima)
- Being updated through Resolution CIM No. 3/2023
- Will include National Mitigation and Adaptation Strategies
- Defines targets for 2030 and 2035
- Aligned with 2050 climate neutrality goal

**Data Source**: Plans & Policies - Cross Cutting Policies

---

### Test Question 3.3: Sectoral Climate Plans
**Query**: "Does Brazil have sectoral climate action plans across different economic sectors?"

**Expected Answer Components**:
- Yes, Brazil develops sectoral mitigation and adaptation plans
- Plans cover different economic sectors
- Expected to be integrated into the National Mitigation Strategy
- Plans extend to 2030

**Data Source**: Plans & Policies - Cross Cutting Policies and Sectoral plans

---

## Dataset 4: Plans & Policies - Sectoral Adaptation Plans

**Description**: Sector-specific climate adaptation strategies
**Module**: `plans_policies`
**Group**: `policy_frameworks`
**File**: `3 Plans and Policies Module Content.xlsx`
**Sheet**: `Sectoral Adaptation Plans`

### Test Question 4.1: Agriculture Adaptation
**Query**: "What climate adaptation measures does Brazil have for the agriculture sector?"

**Expected Answer Components**:
- Should reference agricultural adaptation strategies
- May mention climate-resilient farming practices
- Could include water management for agriculture
- Part of the National Adaptation Strategy

**Data Source**: Plans & Policies - Sectoral Adaptation Plans

---

### Test Question 4.2: Water Resources Adaptation
**Query**: "What are Brazil's adaptation plans for water resources management?"

**Expected Answer Components**:
- Water security measures
- Drought and flood management
- Integration with agricultural needs
- Part of broader adaptation framework

**Data Source**: Plans & Policies - Sectoral Adaptation Plans

---

### Test Question 4.3: Health Sector Adaptation
**Query**: "Does Brazil have climate adaptation plans for the health sector?"

**Expected Answer Components**:
- Should reference health sector adaptation measures
- May mention climate-related health risks
- Could include disease surveillance related to climate change
- Part of National Adaptation Strategy

**Data Source**: Plans & Policies - Sectoral Adaptation Plans

---

## Dataset 5: Plans & Policies - Sectoral Mitigation Plans

**Description**: Sector-specific greenhouse gas mitigation strategies
**Module**: `plans_policies`
**Group**: `policy_frameworks`
**File**: `3 Plans and Policies Module Content.xlsx`
**Sheet**: `Sectoral Mitigation Plans`

### Test Question 5.1: Deforestation Mitigation
**Query**: "What are Brazil's policies for reducing emissions from deforestation?"

**Expected Answer Components**:
- Deforestation is a key emission source for Brazil
- Should reference forest protection policies
- May mention REDD+ or similar programs
- Part of NDC mitigation commitments
- Amazon and other biome protection

**Data Source**: Plans & Policies - Sectoral Mitigation Plans

---

### Test Question 5.2: Energy Sector Mitigation
**Query**: "What are Brazil's mitigation plans for the energy sector?"

**Expected Answer Components**:
- Renewable energy targets
- Brazil's hydroelectric capacity
- Solar and wind energy expansion
- Energy efficiency measures
- Part of National Mitigation Strategy

**Data Source**: Plans & Policies - Sectoral Mitigation Plans

---

### Test Question 5.3: Transport Sector Mitigation
**Query**: "What mitigation measures does Brazil have for the transport sector?"

**Expected Answer Components**:
- Transport emissions reduction strategies
- May mention biofuels (ethanol)
- Electric vehicle policies
- Public transport improvements
- Part of sectoral mitigation framework

**Data Source**: Plans & Policies - Sectoral Mitigation Plans

---

## Dataset 6: Subnational Governance - São Paulo

**Description**: State-level climate governance for São Paulo
**Module**: `subnational`
**Group**: `brazilian_states`
**File**: `4 Subnational Module Content.xlsx`
**Sheet**: `São Paulo (SP)`

### Test Question 6.1: São Paulo Climate Law
**Query**: "Does São Paulo state have its own climate change law?"

**Expected Answer Components**:
- Yes, São Paulo has the State Policy for Climate Change (PEMC)
- Established by State Law 13.798/2009
- Guides state's actions on climate change
- Status: In law/policy/practice

**Data Source**: Subnational - São Paulo, question 1a

---

### Test Question 6.2: São Paulo Emissions Targets
**Query**: "What are São Paulo state's greenhouse gas emissions reduction targets?"

**Expected Answer Components**:
- Should reference specific state-level targets
- May be aligned with national targets
- Likely tied to State Law 13.798/2009
- May include interim and long-term targets

**Data Source**: Subnational - São Paulo records

---

### Test Question 6.3: São Paulo Climate Governance
**Query**: "How does São Paulo state coordinate its climate policies?"

**Expected Answer Components**:
- State-level climate governance institutions
- May mention state climate committees
- Coordination with municipal governments
- Implementation mechanisms for State Law 13.798/2009

**Data Source**: Subnational - São Paulo records

---

## Dataset 7: Subnational Governance - Amazonas

**Description**: State-level climate governance for Amazonas (Amazon region)
**Module**: `subnational`
**Group**: `brazilian_states`
**File**: `4 Subnational Module Content.xlsx`
**Sheet**: `Amazonas (AM)`

### Test Question 7.1: Amazonas Climate Policy
**Query**: "What climate policies does Amazonas state have?"

**Expected Answer Components**:
- Amazonas is in the Amazon rainforest region
- Should have forest conservation policies
- State-level climate governance framework
- Alignment with federal forest protection goals

**Data Source**: Subnational - Amazonas records

---

### Test Question 7.2: Amazonas Deforestation Policy
**Query**: "What are Amazonas state's policies for preventing deforestation?"

**Expected Answer Components**:
- Forest protection and conservation measures
- Indigenous land protection
- Monitoring and enforcement mechanisms
- Aligned with federal Amazon protection policies

**Data Source**: Subnational - Amazonas records

---

### Test Question 7.3: Amazonas Climate Adaptation
**Query**: "What climate adaptation measures does Amazonas state have?"

**Expected Answer Components**:
- Adaptation strategies for the Amazon region
- May mention biodiversity protection
- Water resource management
- Community resilience measures

**Data Source**: Subnational - Amazonas records

---

## Dataset 8: Institutional Framework - Direction Setting

**Description**: Institutions responsible for setting climate policy direction
**Module**: `institutions`
**Group**: `governance_processes`
**File**: `2 Institutions and Processes Module Content.xlsx`
**Sheet**: `Direction Setting`

### Test Question 8.1: National Climate Direction
**Query**: "Which institutions are responsible for setting Brazil's national climate policy direction?"

**Expected Answer Components**:
- Interministerial Committee on Climate Change (CIM)
- Ministry of Environment and Climate Change
- Presidential/executive leadership
- Role in defining national targets and strategies

**Data Source**: Institutions - Direction Setting records

---

### Test Question 8.2: Legislative Climate Role
**Query**: "What is the role of Brazil's legislature in climate policy direction?"

**Expected Answer Components**:
- Congress (Câmara dos Deputados and Senado)
- Passes climate laws like PNMC
- Reviews and approves climate legislation (e.g., PL 6,539/2019)
- Oversight of climate policy implementation

**Data Source**: Institutions - Direction Setting records

---

### Test Question 8.3: Strategic Planning Bodies
**Query**: "Which bodies are responsible for strategic climate planning in Brazil?"

**Expected Answer Components**:
- CIM (Interministerial Committee on Climate Change)
- Technical Groups on Mitigation and Adaptation
- Ministries involved in climate planning
- Development of Plano Clima and sectoral strategies

**Data Source**: Institutions - Direction Setting records

---

## Dataset 9: Institutional Framework - Knowledge & Evidence

**Description**: Institutions and processes for climate knowledge and evidence
**Module**: `institutions`
**Group**: `governance_processes`
**File**: `2 Institutions and Processes Module Content.xlsx`
**Sheet**: `Knowledge and Evidence`

### Test Question 9.1: Climate Research Institutions
**Query**: "Which institutions provide climate science and research for Brazil's policy making?"

**Expected Answer Components**:
- INPE (National Institute for Space Research)
- Climate monitoring and research bodies
- Universities and research institutions
- Emissions inventory systems

**Data Source**: Institutions - Knowledge and Evidence records

---

### Test Question 9.2: Emissions Monitoring
**Query**: "How does Brazil monitor its greenhouse gas emissions?"

**Expected Answer Components**:
- National Emissions Inventory
- SEEG (System of Greenhouse Gas Emissions Estimates)
- INPE monitoring systems
- Reporting to UNFCCC

**Data Source**: Institutions - Knowledge and Evidence records

---

### Test Question 9.3: Deforestation Monitoring
**Query**: "How does Brazil monitor deforestation?"

**Expected Answer Components**:
- PRODES (Amazon Deforestation Monitoring Program)
- DETER (Real-time Deforestation Detection System)
- INPE satellite monitoring
- Annual deforestation reports

**Data Source**: Institutions - Knowledge and Evidence records

---

## Dataset 10: Institutional Framework - Participation & Stakeholder Engagement

**Description**: Mechanisms for public and stakeholder participation in climate governance
**Module**: `institutions`
**Group**: `governance_processes`
**File**: `2 Institutions and Processes Module Content.xlsx`
**Sheet**: `Participation and Stakeholder Engagement`

### Test Question 10.1: Public Participation Mechanisms
**Query**: "How does Brazil ensure public participation in climate policy making?"

**Expected Answer Components**:
- Public consultation processes
- Stakeholder engagement mechanisms
- Civil society participation
- Multi-stakeholder platforms

**Data Source**: Institutions - Participation and Stakeholder Engagement records

---

### Test Question 10.2: Indigenous Peoples Participation
**Query**: "How are indigenous peoples involved in Brazil's climate governance?"

**Expected Answer Components**:
- Indigenous representation in decision-making
- Traditional knowledge integration
- Land rights and forest protection
- Consultation mechanisms

**Data Source**: Institutions - Participation and Stakeholder Engagement records

---

### Test Question 10.3: Private Sector Engagement
**Query**: "How does Brazil engage the private sector in climate action?"

**Expected Answer Components**:
- Business participation in climate planning
- Public-private partnerships
- Industry commitments
- Economic stakeholder involvement

**Data Source**: Institutions - Participation and Stakeholder Engagement records

---

## Dataset 11: Institutional Framework - Integration

**Description**: Integration of climate policy across government functions
**Module**: `institutions`
**Group**: `governance_processes`
**File**: `2 Institutions and Processes Module Content.xlsx`
**Sheet**: `Integration`

### Test Question 11.1: Cross-sectoral Integration
**Query**: "How does Brazil integrate climate considerations across different policy sectors?"

**Expected Answer Components**:
- Climate mainstreaming in sectoral policies
- Coordination across ministries
- Integration mechanisms through CIM
- Climate considerations in economic planning

**Data Source**: Institutions - Integration records

---

### Test Question 11.2: Budget Integration
**Query**: "How is climate policy integrated into Brazil's budget and fiscal planning?"

**Expected Answer Components**:
- Climate finance mechanisms
- Budget allocations for climate action
- Green budget tagging
- Fiscal integration of climate goals

**Data Source**: Institutions - Integration records

---

### Test Question 11.3: Development Planning Integration
**Query**: "How is climate change integrated into Brazil's development planning?"

**Expected Answer Components**:
- Climate in national development plans
- Sustainable development goals alignment
- Long-term economic planning with climate considerations
- Integration with sectoral development strategies

**Data Source**: Institutions - Integration records

---

## Dataset 12: TPI Transition Pathways

**Description**: Transition Pathway Initiative emissions pathways for Brazil
**Module**: `tpi_graphs`
**Group**: `transition_pathways`
**File**: `1_1 TPI Graph [on NDC Overview].xlsx`

### Test Question 12.1: TPI Pathway Scenarios
**Query**: "What are the emissions pathway scenarios for Brazil according to TPI data?"

**Expected Answer Components**:
- Multiple pathway scenarios (e.g., 1.5°C, 2°C, current policy)
- Historical emissions data
- Projected emissions trajectories
- Comparison of different scenarios

**Data Source**: TPI Graphs - all records

---

### Test Question 12.2: Historical Emissions Trend
**Query**: "What has been Brazil's historical greenhouse gas emissions trend?"

**Expected Answer Components**:
- Historical emissions data points
- Key emission sources (especially deforestation)
- Trends over time
- Baseline years (e.g., 2005 baseline)

**Data Source**: TPI Graphs - historical data records

---

### Test Question 12.3: Paris Agreement Alignment
**Query**: "Are Brazil's emissions targets aligned with Paris Agreement temperature goals according to TPI?"

**Expected Answer Components**:
- Comparison of Brazil's NDC with 1.5°C pathway
- Gap analysis between commitments and Paris goals
- Assessment of ambition level
- Pathway alignment evaluation

**Data Source**: TPI Graphs - pathway comparison records

---

## Testing Methodology

### For Each Test Question:
1. **Submit the query** to the API at `http://localhost:8098/query/stream`
2. **Evaluate the response** against expected answer components
3. **Score the response**:
   - ✅ **PASS**: Response includes all or most expected components with accurate information
   - ⚠️ **PARTIAL**: Response includes some expected components but missing key information
   - ❌ **FAIL**: Response does not include expected information or provides incorrect information
   - 🚫 **NO DATA**: Response indicates LSE/NDC Align data was not accessed

### Success Criteria:
- At least 80% of questions should PASS
- No more than 10% should FAIL
- If responses show 🚫 NO DATA, this indicates the LSE server is not being invoked properly

### Evaluation Notes:
- Responses may include additional context beyond expected components (this is acceptable)
- Exact wording does not need to match, but key facts should be present
- Source citations should reference LSE/NDC Align data when available
- If the API uses other data sources instead of LSE, note this as a potential issue

## Next Steps:
1. Run these queries through the API
2. Document results for each question
3. Identify patterns in failures (specific datasets, question types, etc.)
4. Report findings to address client concerns about NDC Align data access
