import os
import time
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Set ANTHROPIC_API_KEY in your environment or .env")


# Model configuration for prompt testing
TEST_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
TEST_OPENAI_MODEL = "gpt-5.0"
TEST_PROVIDER = "anthropic"  # options: anthropic, openai
TEST_TEMPERATURE = 0.2

system_prompt1 = """You are a helpful, precise analyst who follows instructions exactly.

The current date is 2025-09-22.

You are writing an analyst-grade summary for the given query.
Assume each question is about Brazil. This is a Brazil-focused assistant. We will add support for other countries in the future.
Don't say you're going to answer the question; just answer it.
Use the provided evidence items to respond to the user's query. Each item has an ID like F1.
Present the evidence in this order: F1 -> F2 -> F3 -> F4 -> F5 -> F6 -> F7 -> F8.
Open with a single paragraph that directly answers the question, weaving the key takeaway from the strongest evidence and including at least one citation marker (e.g., [[F1]]).
Follow with additional paragraphs that expand on each major aspect or module, adding engaging but fact-based detail while keeping the flow cohesive.
You must provide the information in an engaging, coherent narrative.
Every sentence that uses evidence must include citation markers using the syntax [[F1]] immediately after the relevant clause.
If multiple evidence items support a sentence, include multiple markers, e.g., [[F1]][[F3]].
Avoid repeating tool scaffolding or status bullet text verbatim—integrate those ideas into polished prose instead of copying headings like 'NDC Overview & Domestic Comparison'.
Do NOT include meta-commentary about data sources, methodologies, or what makes analysis possible. Avoid phrases that describe HOW information was obtained or what tools/datasets enabled the analysis.
Use evidence about datasets and analysis as context for the other evidence ONLY. AVOID discussing these directly unless specifically asked.
Do NOT write information ABOUT your data (NDCAlign has these tables, 'I am going to write about this data', etc). Just write the narrative.
You have a wealth of information present within your general knowledge and through tools. You are in charge of making this information engaging and compelling. Stay humble but state known facts confidently.
Do not start answers with phrases like 'Based on the provided evidence...' OR similar phrasing.
Don't mention the names or ids of documents or data sources (e.g., 'According to document F1...'). Instead, just use the citation markers [[F1]].
Don't include phrases like 'In this document, we will discuss...' or 'This report provides an analysis of...'. Just present the information directly.
"""

user_query1 = """ Query: Which Brazilian states have been experiencing the most extreme heat recently?

Evidence items:
F1: Extracted 308 top-quintile extreme heat polygons for PlanetSapling_Heat_Index_Brazil_Daily_Mean_2020_2025 to characterise persistent heat stress zones in Brazil. (Source: Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis)
F2: The selected polygons cover approximately 3,488,548 km^2 of land. (Source: Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis)
F3: Dataset last updated on 9/15/2025; current export contains top-quintile polygons only. (Source: Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis)
F4: Top extreme-heat exposure by area concentrates in Mato Grosso (2,289,823 km^2), Mato Grosso do Sul (1,573,217 km^2), Bahia (1,464,523 km^2). (Source: Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis)
F5: Recent evidence indicates that Brazil is experiencing significant climate extremes, particularly in the form of heatwaves. (Source: WMO State of the Climate in Latin America and the Caribbean 2024)
F6: Notably, southern Brazil has been affected by record-breaking heatwaves, with a severe event occurring from March 15 to 18, and another impacting central and southern regions from April 27, lasting five days. (Source: IPCC AR6 Central and South America)
F7: During this period, parts of western central Brazil recorded temperatures up to 7°C above normal ([WMO, 2024](data/WMO-Clim-LAC.pdf p.11)). (Source: WMO State of the Climate in Latin America and the Caribbean 2024)
F8: The states most impacted by these extreme heat events include Rio Grande do Sul, where unprecedented flooding occurred in April and May, affecting over 2.3 million people ([WMO, 2024](data/WMO-Clim-LAC.pdf p.11)). (Source: IPCC AR6 Central and South America)

Return the paragraphs as plain text separated by blank lines.
"""

system_prompt2 = """You are a helpful, precise analyst who follows instructions exactly.

The current date is 2025-09-22.

You are writing an analyst-grade summary for the given query.
Assume each question is about Brazil. This is a Brazil-focused assistant. We will add support for other countries in the future.
Don't say you're going to answer the question; just answer it.
Use the provided evidence items to respond to the user's query. Each item has an ID like F1.
Present the evidence in this order: F1 -> F2.
Open with a single paragraph that directly answers the question, weaving the key takeaway from the strongest evidence and including at least one citation marker (e.g., [[F1]]).
Follow with additional paragraphs that expand on each major aspect or module, adding engaging but fact-based detail while keeping the flow cohesive.
You must provide the information in an engaging, coherent narrative.
Every sentence that uses evidence must include citation markers using the syntax [[F1]] immediately after the relevant clause.
If multiple evidence items support a sentence, include multiple markers, e.g., [[F1]][[F3]].
Avoid repeating tool scaffolding or status bullet text verbatim—integrate those ideas into polished prose instead of copying headings like 'NDC Overview & Domestic Comparison'.
Do NOT write about methodologies, datasets, analysis approaches, or what "enables" or "makes possible" any findings. 
If evidence describes datasets or analytical methods, DO NOT include this information in your response, unless you are asked. Focus only on substantive findings and facts.
Do NOT write information ABOUT your data (NDCAlign has these tables, 'I am going to write about this data', etc). Just write the narrative.
You have a wealth of information present within your general knowledge and through tools. You are in charge of making this information engaging and compelling. Stay humble but state known facts confidently.
Do not start answers with phrases like 'Based on the provided evidence...' OR similar phrasing.
Don't mention the names or ids of documents or data sources (e.g., 'According to document F1...'). Instead, just use the citation markers [[F1]].
Don't include phrases like 'In this document, we will discuss...' or 'This report provides an analysis of...'. Just present the information directly."""

user_query2 = """Query: Which companies are most exposed to flood risk in Brazil?

Evidence items:
F1: The GIST Impact Datasets combine multi-sheet Excel sources covering corporate environmental risks, Scope 3 emissions, biodiversity pressure, deforestation proximity, and asset-level geospatial exposure. (Source: GIST Impact Datasets)
F2: Banco Bradesco SA (BANCOB00001) has the greatest share of assets at high flood riverine in Brazil, with 18 assets representing 0.27% of those assessed. (Source: GIST hazard ranking: Flood Riverine)

Return the paragraphs as plain text separated by blank lines."""




user_query3 = """Query: To what extent does the NDC rely on future policies to define its mitigation strategies in high-emitting sectors like land use and energy?

Evidence items:
F1: NDC Overview & Domestic Comparison: 1a) Is there a long term economy-wide long term emissions reduction target?
NDC summary: According to its Nationally Determined Contribution (NDC), Brazil has committed to achieving climate neutrality by 2050, covering all greenhouse gases. While the current NDC does not set an official quantitative target for 2050, it establishes a long-term political commitment to reaching net-zero emissions.
Domestic alignment: Aligned
Domestic summary: Brazil's committment to reaching climate neutrality by 2050 is reinforced in Resolution 3 of September 14, 2023, from the Interministerial Committee on Climate Change (CIM), which guides the update of the National Plan on Climate Change (Plano Clima) and establishes the Temporary Technical Groups on Mitigation (GTT - Mitigation) and Adaptation (GTT - Adaptation). Article 2 of this Resolution states that the National Mitigation Strategy will define the national greenhouse gas (GHG) emissions reduction target for 2030 and an indicative national reduction target for 2035. Additionally, in its sole paragraph, the Resolution specifies that these targets must be aligned with Brazil’s goal of achieving climate neutrality for GHG emissions by 2050.

There is a proposed Bill (PL 6,539/2019) that seeks to amend the National Policy on Climate Change (PNMC - Law No. 12.187/2009) to align it with the Paris Agreement and address new challenges related to climate change. If approved, this Bill will formally establish Brazil's commitment to climate neutrality by 2050 within the PNMC, modifying the wording of Article 12-A as follows:

"The country, with the support of the instruments provided for in Articles 6 and 7, commits to: [...] II – neutralise 100% of its emissions by the year 2050, in accordance with the Long-Term National Strategy."
Status: In existing law/policy and under further development (Source: NDC Overview & Domestic Comparison)
F2: NDC Overview & Domestic Comparison: 2.b) Is there an interim target for 2030
NDC summary: Brazil's NDC does not establish a formal quantitative target for 2030, as it does for 2035. However, there are references to the need to define a target for this year in other documents and regulations that are still under development.

The NDC also mentions that, following the Global Stocktake (GST) conducted at COP28, future emissions trajectories were developed to achieve Brazil’s already established climate targets for 2025, 2030, and 2050, with greater cost-effectiveness for the economy as a whole:

"Following the GST, Brazil recognizes that limiting global warming to 1.5 °C with no or limited overshoot requires deep, rapid and sustained reductions in global greenhouse gas emissions of 43 per cent by 2030 and 60 per cent by 2035 relative to the 2019 level and reaching net zero carbon dioxide emissions by 2050 (decision 1/CMA.5, paragraph 27)."
Domestic alignment: Not yet aligned
Domestic summary: Resolution CIM No. 3/2023, which regulates the update of the National Plan on Climate Change (Plano Clima), states that the National Mitigation and Adaptation Strategies must present national mitigation and adaptation targets for 2030 and indicative targets for 2035, in addition to sectoral mitigation and adaptation plans up to 2030. This suggests that the Brazilian government intends to define a commitment for 2030, but there is still no officially established percentage.
Status: Under development (Source: NDC Overview & Domestic Comparison)
F3: NDC Overview & Domestic Comparison: 2.c) Is there an interim target for 2035
NDC summary: The specific target for 2035 is to reduce net greenhouse gas (GHG) emissions by 59% to 67% compared to 2005 levels. This corresponds to an absolute emissions volume of 1.05 to 0.85 GtCO₂e. This target is formally established in Brazil’s NDC and it represents an interim target toward the country’s long-term commitment to climate neutrality by 2050.
While the 2035 target is clearly defined, additional sectoral policies and regulations may further refine Brazil’s mitigation and adaptation efforts, particularly through the Plano Clima (National Climate Change Plan).
Domestic alignment: Not yet aligned
Domestic summary: The target will be further incorporated into domestic law through the Plano Clima
Status: Under development (Source: NDC Overview & Domestic Comparison)
F4: NDC Overview & Domestic Comparison: 2.d) Is there an interim target for 2040
Status: No evidence found (Source: NDC Align)
F5: NDC Overview & Domestic Comparison: 3.a) Is there a high level goal or set of high level goals or objectives on adaptation? We would expect these goals to apply to the whole economy.
NDC summary: Brazil's NDC outlines the guidelines and objectives of the National Adaptation Strategy, which will be developed through the Climate Plan (Plano Clima), currently under development: “The National Adaptation Strategy contains the guidelines for the design and implementation of adaptation actions at the federal, state and municipal levels: 1. promoting sustainable development in its many dimensions, considering strategic sectors and themes for the country, with a view to reducing inequalities and to a just transition; 2. promoting climate justice based on the dimensions of gender, race, ethnicity, age, social class and other factors that influence vulnerability; 3. promoting environmental protection, conservation and preservation, guided by the principles of precaution and prevention; 4. multilevel and transversal governance, with a view to coherence, synergy and complementarity between strategies, taking into account territorial specificities; 5. promoting transparency and participatory processes with society; 6. mainstreaming adaptation into policies, programs and projects that may be affected by climate change, including through structuring initiatives and a long-term perspective; 7. strengthening institutional capacities at the different levels of government, including those needed to access sources of funding for adaptation and other means of implementation; 8. promoting co-benefits between adaptation and mitigation of greenhouse gas emissions; 9. adoption of the best available knowledge, based on science, good sectoral and societal practices, traditional knowledge and other sources deemed appropriate; 10. integrating incremental and transformational actions, based on an understanding of climate-related risks and their multiple conditioning factors, with different time horizons and scales of execution; 11. promoting public awareness of climate change, its causes, consequences and approaches to risks reduction; 1.2. adopting Ecosystem-Based Adaptation (EbA) approaches, recognizing their potential to reduce climate risks and vulnerabilities in a systemic, flexible, socially just and cost-effective way, with mitigation co-benefits; 13. flexibility and adaptability of strategies, with context changes and revisions to the Plan to incorporate updates to the information and knowledge generated, as well as lessons learned. The National Adaptation Objectives reflect the integration between global commitments and local needs and priorities: 1. increasing the resilience of populations, cities, territories and infrastructures in facing the climate emergency; 2. promoting sustainable and resilient production and regular access to healthy food of adequate quality and quantity; 3. promoting water security, making water available in sufficient quality and quantity for multiple uses, such as supply, production, energy and ecosystems; 4. protect, conserve and strengthen ecosystems and biodiversity and ensure the provision of ecosystem services; 5. safeguarding the health and well-being of populations while respecting the ways of life of traditional peoples and communities; 6. ensuring sustainable and affordable energy security; 7. promoting socio-economic development and reducing inequalities; 8. protecting cultural heritage and preserving cultural practices and heritage sites against climate-related risks; 9. strengthening the vital role of the ocean and coastal zone in tackling climate change. Based on these guidelines, sixteen sectoral and thematic adaptation plans are being developed, incorporating awareness-raising, training and conceptual alignment actions on topics such as EbA, climate emergency, human mobility and climate justice. The plans are as follows: (i) agriculture and livestock; (ii) family farming; (iii) biodiversity; (iv) cities; (v) risk and disaster management; (vi) industry; (vii) energy; (viii) transportation; (ix) racial equality and combating racism; (x) traditional peoples and communities; (xi) indigenous peoples; (xii) water resources; (xiii) health; (xiv) food and nutritional security; (xv) ocean and coastal zone; and (xvi) tourism.”
Domestic alignment: Not yet aligned
Domestic summary: Defined by the Interministerial Committee on Climate Change (CIM), via Resolution 3/2023, the new Climate Plan will include the “National Adaptation Strategy”, with sixteen sectoral adaptation plans. 

It also establishes that the National Adaptation Strategy shall include, among other elements, principles, guidelines, and national adaptation priorities; guidelines for the development of sectoral adaptation plans; guidelines for integrating adaptation actions into climate action plans; and that sectoral adaptation plans must include, among other provisions, sectoral adaptation objectives and priorities. 
Law No. 14.904/2024 establishes guidelines for the development of climate change adaptation plans. 

In addition to the ongoing development of the National Adaptation Strategy under the Climate Plan, Brazil’s 2016 National Adaptation Plan (PNA), established by Ministerial Ordinance No. 150 of May 10, 2016, remains in effect as an important framework for managing climate risks and adaptation actions.
The PNA’s primary objective is to reduce climate vulnerability and enhance resilience by integrating adaptation actions across key sectors, including agriculture, biodiversity, cities, health, water resources, infrastructure, and disaster risk management. The plan follows an iterative and dynamic approach, ensuring that adaptation strategies evolve based on new scientific data, climate projections, and socio-economic developments.
Although the PNA is currently under review, its principles and structure continue to influence Brazil’s adaptation planning, serving as a foundation for the National Adaptation Strategy outlined in the latest NDC. The NDC explicitly commits to revising the PNA and enhancing adaptation efforts through sectoral planning, institutional coordination, and engagement with vulnerable communities.
Status: In existing law/policy and under further development (Source: NDC Overview & Domestic Comparison)
F6: Document UNFCCC.party.166.0 references mitigation: HRISTOV AN; OH J; MEINEN R; MONTES F; OTT T; FIRKINS J; ROTZ A; DELL C; ADESOGAN A; YANG W; TRICARICO J; KEBREAB E; WAGHORN G; DIJKSTRA J; OOSTING S. 2013. Mitigation of greenhouse gas emissions in livestock production - A review of technical options for non-CO2 emissions. In: Gerber P; Henderson B; Makkar H, eds. FAO Animal Production and Health Paper No. Food and Agriculture Organization of the United 177, (Source: CPR passage UNFCCC.party.166.0)
F7: Document UNFCCC.party.166.0 references mitigation: The ABC Platform provides data on the technologies adopted by the ABC Plan and their respective contribution towards Greenhouse Gas (GHG) mitigation for the monitoring of the ABC Plan's pre-established goals. Also, the ABC Platform is responsible for the conceptual validation of the MRV (Monitoring, Reporting and Verification) mechanism, based on expertise and the use of different tools and information technology developed, or under validation stage, by Embrapa and partner institutions. (Source: CPR passage UNFCCC.party.166.0)
F8: Document UNFCCC.party.160.0 references mitigation: (ii) Specific projects, measures and activities to be implemented to contribute to mitigation co-benefits, including information on adaptation plans that also yield mitigation co-benefits, which may cover, but are not limited to, key sectors, such as energy, resources, water resources, coastal resources, human settlements and urban planning, agriculture and forestry; and economic diversification actions, which may cover, but are not limited to, sectors such as manufacturing and industry, energy and mining, transport and communication, construction, tourism, real estate, agriculture and… (Source: CPR passage UNFCCC.party.160.0)

Return the paragraphs as plain text separated by blank lines.
"""

system_prompt3 = """You are a helpful, precise analyst who follows instructions exactly.

The current date is 2025-09-22.

You are writing an analyst-grade summary for the given query.
Assume each question is about Brazil. This is a Brazil-focused assistant. We will add support for other countries in the future.
Don't say you're going to answer the question; just answer it.
Use the provided evidence items to respond to the user's query. Each item has an ID like F1.
Present the evidence in this order: F1 -> F2 -> F3 -> F4 -> F5 -> F6 -> F7 -> F8.
Open with a single paragraph that directly answers the question, weaving the key takeaway from the strongest evidence and including at least one citation marker (e.g., [[F1]]).
Follow with additional paragraphs that expand on each major aspect or module, adding engaging but fact-based detail while keeping the flow cohesive.
You must provide the information in an engaging, coherent narrative.
Every sentence that uses evidence must include citation markers using the syntax [[F1]] immediately after the relevant clause.
If multiple evidence items support a sentence, include multiple markers, e.g., [[F1]][[F3]].
Avoid repeating tool scaffolding or status bullet text verbatim—integrate those ideas into polished prose instead of copying headings like 'NDC Overview & Domestic Comparison'.
Do NOT write about methodologies, datasets, analysis approaches, or what "enables" or "makes possible" any findings, unless specifically asked. If evidence describes datasets or analytical methods, DO NOT include this information in your response. Focus only on substantive findings and facts.
Do NOT write information ABOUT your data (NDCAlign has these tables, 'I am going to write about this data', etc). Just write the narrative.
You have a wealth of information present within your general knowledge and through tools. You are in charge of making this information engaging and compelling. Stay humble but state known facts confidently.
Do not start answers with phrases like 'Based on the provided evidence...' OR similar phrasing.
Don't mention the names or ids of documents or data sources (e.g., 'According to document F1...'). Instead, just use the citation markers [[F1]].
Don't include phrases like 'In this document, we will discuss...' or 'This report provides an analysis of...'. Just present the information directly."""



# system_prompt1 = """You are a precise analyst who follows instructions exactly.

# The current date is 2025-09-22.

# You are writing an analyst-grade summary for the given query.
# Assume each question is about Brazil. This is a Brazil-focused assistant. We will add support for other countries in the future.
# Don't say you're going to answer the question; just answer it.
# Use only the provided evidence items. Each item has an ID like F1.
# Present the evidence in this order: F1 -> F2 -> F3 -> F4 -> F5 -> F6 -> F7 -> F8.
# Open with a single paragraph that directly answers the question, weaving the key takeaway from the strongest evidence and including at least one citation marker (e.g., [[F1]]).
# Follow with additional paragraphs that expand on each major aspect or module, adding engaging but fact-based detail while keeping the flow cohesive.
# You must provide the information in an engaging, coherent narrative.
# Every sentence that uses evidence must include citation markers using the syntax [[F1]] immediately after the relevant clause.
# If multiple evidence items support a sentence, include multiple markers, e.g., [[F1]][[F3]].
# Do not invent new information; common knowledge may be used only to connect cited facts and must never introduce new claims.
# Avoid repeating tool scaffolding or status bullet text verbatim—integrate those ideas into polished prose instead of copying headings like 'NDC Overview & Domestic Comparison'.
# Do NOT write information ABOUT your data (NDCAlign has these tables, 'I am going to write about this data', etc). Just write the narrative.
# You have a wealth of information present within your general knowledge and through tools. You are in charge of making this information engaging and compelling. Stay humble but state known facts confidently.
# Do not start answers with phrases like 'Based on the provided evidence...' OR similar phrasing.
# Don't mention the names or ids of documents or data sources (e.g., 'According to document F1...'). Instead, just use the citation markers [[F1]].
# Don't include phrases like 'In this document, we will discuss...' or 'This report provides an analysis of...'. Just present the information directly.
# """

# user_query1 = """ Query: Which Brazilian states have been experiencing the most extreme heat recently?

# Evidence items:
# F1: Extracted 308 top-quintile extreme heat polygons for PlanetSapling_Heat_Index_Brazil_Daily_Mean_2020_2025 to characterise persistent heat stress zones in Brazil. (Source: Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis)
# F2: The selected polygons cover approximately 3,488,548 km^2 of land. (Source: Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis)
# F3: Dataset last updated on 9/15/2025; current export contains top-quintile polygons only. (Source: Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis)
# F4: Top extreme-heat exposure by area concentrates in Mato Grosso (2,289,823 km^2), Mato Grosso do Sul (1,573,217 km^2), Bahia (1,464,523 km^2). (Source: Extreme Heat Indices Derived from ERA5-Land Daily Aggregated – ECMWF Climate Reanalysis)
# F5: Recent evidence indicates that Brazil is experiencing significant climate extremes, particularly in the form of heatwaves. (Source: WMO State of the Climate in Latin America and the Caribbean 2024)
# F6: Notably, southern Brazil has been affected by record-breaking heatwaves, with a severe event occurring from March 15 to 18, and another impacting central and southern regions from April 27, lasting five days. (Source: IPCC AR6 Central and South America)
# F7: During this period, parts of western central Brazil recorded temperatures up to 7°C above normal ([WMO, 2024](data/WMO-Clim-LAC.pdf p.11)). (Source: WMO State of the Climate in Latin America and the Caribbean 2024)
# F8: The states most impacted by these extreme heat events include Rio Grande do Sul, where unprecedented flooding occurred in April and May, affecting over 2.3 million people ([WMO, 2024](data/WMO-Clim-LAC.pdf p.11)). (Source: IPCC AR6 Central and South America)

# Return the paragraphs as plain text separated by blank lines.
# """

# system_prompt2 = """You are a precise analyst who follows instructions exactly.

# The current date is 2025-09-22.

# You are writing an analyst-grade summary for the given query.
# Assume each question is about Brazil. This is a Brazil-focused assistant. We will add support for other countries in the future.
# Don't say you're going to answer the question; just answer it.
# Use only the provided evidence items. Each item has an ID like F1.
# Present the evidence in this order: F1 -> F2.
# Open with a single paragraph that directly answers the question, weaving the key takeaway from the strongest evidence and including at least one citation marker (e.g., [[F1]]).
# Follow with additional paragraphs that expand on each major aspect or module, adding engaging but fact-based detail while keeping the flow cohesive.
# You must provide the information in an engaging, coherent narrative.
# Every sentence that uses evidence must include citation markers using the syntax [[F1]] immediately after the relevant clause.
# If multiple evidence items support a sentence, include multiple markers, e.g., [[F1]][[F3]].
# Do not invent new information; common knowledge may be used only to connect cited facts and must never introduce new claims.
# Avoid repeating tool scaffolding or status bullet text verbatim—integrate those ideas into polished prose instead of copying headings like 'NDC Overview & Domestic Comparison'.
# Do NOT write information ABOUT your data (NDCAlign has these tables, 'I am going to write about this data', etc). Just write the narrative.
# You have a wealth of information present within your general knowledge and through tools. You are in charge of making this information engaging and compelling. Stay humble but state known facts confidently.
# Do not start answers with phrases like 'Based on the provided evidence...' OR similar phrasing.
# Don't mention the names or ids of documents or data sources (e.g., 'According to document F1...'). Instead, just use the citation markers [[F1]].
# Don't include phrases like 'In this document, we will discuss...' or 'This report provides an analysis of...'. Just present the information directly."""

# user_query2 = """Query: Which companies are most exposed to flood risk in Brazil?

# Evidence items:
# F1: The GIST Impact Datasets combine multi-sheet Excel sources covering corporate environmental risks, Scope 3 emissions, biodiversity pressure, deforestation proximity, and asset-level geospatial exposure. (Source: GIST Impact Datasets)
# F2: Banco Bradesco SA (BANCOB00001) has the greatest share of assets at high flood riverine in Brazil, with 18 assets representing 0.27% of those assessed. (Source: GIST hazard ranking: Flood Riverine)

# Return the paragraphs as plain text separated by blank lines."""




# user_query3 = """Query: To what extent does the NDC rely on future policies to define its mitigation strategies in high-emitting sectors like land use and energy?

# Evidence items:
# F1: NDC Overview & Domestic Comparison: 1a) Is there a long term economy-wide long term emissions reduction target?
# NDC summary: According to its Nationally Determined Contribution (NDC), Brazil has committed to achieving climate neutrality by 2050, covering all greenhouse gases. While the current NDC does not set an official quantitative target for 2050, it establishes a long-term political commitment to reaching net-zero emissions.
# Domestic alignment: Aligned
# Domestic summary: Brazil's committment to reaching climate neutrality by 2050 is reinforced in Resolution 3 of September 14, 2023, from the Interministerial Committee on Climate Change (CIM), which guides the update of the National Plan on Climate Change (Plano Clima) and establishes the Temporary Technical Groups on Mitigation (GTT - Mitigation) and Adaptation (GTT - Adaptation). Article 2 of this Resolution states that the National Mitigation Strategy will define the national greenhouse gas (GHG) emissions reduction target for 2030 and an indicative national reduction target for 2035. Additionally, in its sole paragraph, the Resolution specifies that these targets must be aligned with Brazil’s goal of achieving climate neutrality for GHG emissions by 2050.

# There is a proposed Bill (PL 6,539/2019) that seeks to amend the National Policy on Climate Change (PNMC - Law No. 12.187/2009) to align it with the Paris Agreement and address new challenges related to climate change. If approved, this Bill will formally establish Brazil's commitment to climate neutrality by 2050 within the PNMC, modifying the wording of Article 12-A as follows:

# "The country, with the support of the instruments provided for in Articles 6 and 7, commits to: [...] II – neutralise 100% of its emissions by the year 2050, in accordance with the Long-Term National Strategy."
# Status: In existing law/policy and under further development (Source: NDC Overview & Domestic Comparison)
# F2: NDC Overview & Domestic Comparison: 2.b) Is there an interim target for 2030
# NDC summary: Brazil's NDC does not establish a formal quantitative target for 2030, as it does for 2035. However, there are references to the need to define a target for this year in other documents and regulations that are still under development.

# The NDC also mentions that, following the Global Stocktake (GST) conducted at COP28, future emissions trajectories were developed to achieve Brazil’s already established climate targets for 2025, 2030, and 2050, with greater cost-effectiveness for the economy as a whole:

# "Following the GST, Brazil recognizes that limiting global warming to 1.5 °C with no or limited overshoot requires deep, rapid and sustained reductions in global greenhouse gas emissions of 43 per cent by 2030 and 60 per cent by 2035 relative to the 2019 level and reaching net zero carbon dioxide emissions by 2050 (decision 1/CMA.5, paragraph 27)."
# Domestic alignment: Not yet aligned
# Domestic summary: Resolution CIM No. 3/2023, which regulates the update of the National Plan on Climate Change (Plano Clima), states that the National Mitigation and Adaptation Strategies must present national mitigation and adaptation targets for 2030 and indicative targets for 2035, in addition to sectoral mitigation and adaptation plans up to 2030. This suggests that the Brazilian government intends to define a commitment for 2030, but there is still no officially established percentage.
# Status: Under development (Source: NDC Overview & Domestic Comparison)
# F3: NDC Overview & Domestic Comparison: 2.c) Is there an interim target for 2035
# NDC summary: The specific target for 2035 is to reduce net greenhouse gas (GHG) emissions by 59% to 67% compared to 2005 levels. This corresponds to an absolute emissions volume of 1.05 to 0.85 GtCO₂e. This target is formally established in Brazil’s NDC and it represents an interim target toward the country’s long-term commitment to climate neutrality by 2050.
# While the 2035 target is clearly defined, additional sectoral policies and regulations may further refine Brazil’s mitigation and adaptation efforts, particularly through the Plano Clima (National Climate Change Plan).
# Domestic alignment: Not yet aligned
# Domestic summary: The target will be further incorporated into domestic law through the Plano Clima
# Status: Under development (Source: NDC Overview & Domestic Comparison)
# F4: NDC Overview & Domestic Comparison: 2.d) Is there an interim target for 2040
# Status: No evidence found (Source: NDC Align)
# F5: NDC Overview & Domestic Comparison: 3.a) Is there a high level goal or set of high level goals or objectives on adaptation? We would expect these goals to apply to the whole economy.
# NDC summary: Brazil's NDC outlines the guidelines and objectives of the National Adaptation Strategy, which will be developed through the Climate Plan (Plano Clima), currently under development: “The National Adaptation Strategy contains the guidelines for the design and implementation of adaptation actions at the federal, state and municipal levels: 1. promoting sustainable development in its many dimensions, considering strategic sectors and themes for the country, with a view to reducing inequalities and to a just transition; 2. promoting climate justice based on the dimensions of gender, race, ethnicity, age, social class and other factors that influence vulnerability; 3. promoting environmental protection, conservation and preservation, guided by the principles of precaution and prevention; 4. multilevel and transversal governance, with a view to coherence, synergy and complementarity between strategies, taking into account territorial specificities; 5. promoting transparency and participatory processes with society; 6. mainstreaming adaptation into policies, programs and projects that may be affected by climate change, including through structuring initiatives and a long-term perspective; 7. strengthening institutional capacities at the different levels of government, including those needed to access sources of funding for adaptation and other means of implementation; 8. promoting co-benefits between adaptation and mitigation of greenhouse gas emissions; 9. adoption of the best available knowledge, based on science, good sectoral and societal practices, traditional knowledge and other sources deemed appropriate; 10. integrating incremental and transformational actions, based on an understanding of climate-related risks and their multiple conditioning factors, with different time horizons and scales of execution; 11. promoting public awareness of climate change, its causes, consequences and approaches to risks reduction; 1.2. adopting Ecosystem-Based Adaptation (EbA) approaches, recognizing their potential to reduce climate risks and vulnerabilities in a systemic, flexible, socially just and cost-effective way, with mitigation co-benefits; 13. flexibility and adaptability of strategies, with context changes and revisions to the Plan to incorporate updates to the information and knowledge generated, as well as lessons learned. The National Adaptation Objectives reflect the integration between global commitments and local needs and priorities: 1. increasing the resilience of populations, cities, territories and infrastructures in facing the climate emergency; 2. promoting sustainable and resilient production and regular access to healthy food of adequate quality and quantity; 3. promoting water security, making water available in sufficient quality and quantity for multiple uses, such as supply, production, energy and ecosystems; 4. protect, conserve and strengthen ecosystems and biodiversity and ensure the provision of ecosystem services; 5. safeguarding the health and well-being of populations while respecting the ways of life of traditional peoples and communities; 6. ensuring sustainable and affordable energy security; 7. promoting socio-economic development and reducing inequalities; 8. protecting cultural heritage and preserving cultural practices and heritage sites against climate-related risks; 9. strengthening the vital role of the ocean and coastal zone in tackling climate change. Based on these guidelines, sixteen sectoral and thematic adaptation plans are being developed, incorporating awareness-raising, training and conceptual alignment actions on topics such as EbA, climate emergency, human mobility and climate justice. The plans are as follows: (i) agriculture and livestock; (ii) family farming; (iii) biodiversity; (iv) cities; (v) risk and disaster management; (vi) industry; (vii) energy; (viii) transportation; (ix) racial equality and combating racism; (x) traditional peoples and communities; (xi) indigenous peoples; (xii) water resources; (xiii) health; (xiv) food and nutritional security; (xv) ocean and coastal zone; and (xvi) tourism.”
# Domestic alignment: Not yet aligned
# Domestic summary: Defined by the Interministerial Committee on Climate Change (CIM), via Resolution 3/2023, the new Climate Plan will include the “National Adaptation Strategy”, with sixteen sectoral adaptation plans. 

# It also establishes that the National Adaptation Strategy shall include, among other elements, principles, guidelines, and national adaptation priorities; guidelines for the development of sectoral adaptation plans; guidelines for integrating adaptation actions into climate action plans; and that sectoral adaptation plans must include, among other provisions, sectoral adaptation objectives and priorities. 
# Law No. 14.904/2024 establishes guidelines for the development of climate change adaptation plans. 

# In addition to the ongoing development of the National Adaptation Strategy under the Climate Plan, Brazil’s 2016 National Adaptation Plan (PNA), established by Ministerial Ordinance No. 150 of May 10, 2016, remains in effect as an important framework for managing climate risks and adaptation actions.
# The PNA’s primary objective is to reduce climate vulnerability and enhance resilience by integrating adaptation actions across key sectors, including agriculture, biodiversity, cities, health, water resources, infrastructure, and disaster risk management. The plan follows an iterative and dynamic approach, ensuring that adaptation strategies evolve based on new scientific data, climate projections, and socio-economic developments.
# Although the PNA is currently under review, its principles and structure continue to influence Brazil’s adaptation planning, serving as a foundation for the National Adaptation Strategy outlined in the latest NDC. The NDC explicitly commits to revising the PNA and enhancing adaptation efforts through sectoral planning, institutional coordination, and engagement with vulnerable communities.
# Status: In existing law/policy and under further development (Source: NDC Overview & Domestic Comparison)
# F6: Document UNFCCC.party.166.0 references mitigation: HRISTOV AN; OH J; MEINEN R; MONTES F; OTT T; FIRKINS J; ROTZ A; DELL C; ADESOGAN A; YANG W; TRICARICO J; KEBREAB E; WAGHORN G; DIJKSTRA J; OOSTING S. 2013. Mitigation of greenhouse gas emissions in livestock production - A review of technical options for non-CO2 emissions. In: Gerber P; Henderson B; Makkar H, eds. FAO Animal Production and Health Paper No. Food and Agriculture Organization of the United 177, (Source: CPR passage UNFCCC.party.166.0)
# F7: Document UNFCCC.party.166.0 references mitigation: The ABC Platform provides data on the technologies adopted by the ABC Plan and their respective contribution towards Greenhouse Gas (GHG) mitigation for the monitoring of the ABC Plan's pre-established goals. Also, the ABC Platform is responsible for the conceptual validation of the MRV (Monitoring, Reporting and Verification) mechanism, based on expertise and the use of different tools and information technology developed, or under validation stage, by Embrapa and partner institutions. (Source: CPR passage UNFCCC.party.166.0)
# F8: Document UNFCCC.party.160.0 references mitigation: (ii) Specific projects, measures and activities to be implemented to contribute to mitigation co-benefits, including information on adaptation plans that also yield mitigation co-benefits, which may cover, but are not limited to, key sectors, such as energy, resources, water resources, coastal resources, human settlements and urban planning, agriculture and forestry; and economic diversification actions, which may cover, but are not limited to, sectors such as manufacturing and industry, energy and mining, transport and communication, construction, tourism, real estate, agriculture and… (Source: CPR passage UNFCCC.party.160.0)

# Return the paragraphs as plain text separated by blank lines.
# """

# system_prompt3 = """You are a precise analyst who follows instructions exactly.

# The current date is 2025-09-22.

# You are writing an analyst-grade summary for the given query.
# Assume each question is about Brazil. This is a Brazil-focused assistant. We will add support for other countries in the future.
# Don't say you're going to answer the question; just answer it.
# Use only the provided evidence items. Each item has an ID like F1.
# Present the evidence in this order: F1 -> F2 -> F3 -> F4 -> F5 -> F6 -> F7 -> F8.
# Open with a single paragraph that directly answers the question, weaving the key takeaway from the strongest evidence and including at least one citation marker (e.g., [[F1]]).
# Follow with additional paragraphs that expand on each major aspect or module, adding engaging but fact-based detail while keeping the flow cohesive.
# You must provide the information in an engaging, coherent narrative.
# Every sentence that uses evidence must include citation markers using the syntax [[F1]] immediately after the relevant clause.
# If multiple evidence items support a sentence, include multiple markers, e.g., [[F1]][[F3]].
# Do not invent new information; common knowledge may be used only to connect cited facts and must never introduce new claims.
# Avoid repeating tool scaffolding or status bullet text verbatim—integrate those ideas into polished prose instead of copying headings like 'NDC Overview & Domestic Comparison'.
# Do NOT write information ABOUT your data (NDCAlign has these tables, 'I am going to write about this data', etc). Just write the narrative.
# You have a wealth of information present within your general knowledge and through tools. You are in charge of making this information engaging and compelling. Stay humble but state known facts confidently.
# Do not start answers with phrases like 'Based on the provided evidence...' OR similar phrasing.
# Don't mention the names or ids of documents or data sources (e.g., 'According to document F1...'). Instead, just use the citation markers [[F1]].
# Don't include phrases like 'In this document, we will discuss...' or 'This report provides an analysis of...'. Just present the information directly."""



# def _invoke() -> str:
#             provider = self._choose_provider()
#             print(f"[KGDEBUG] narrative provider={provider}", flush=True)

#             if provider == "anthropic":
#                 try:
#                     print("[KGDEBUG] anthropic call start", flush=True)
#                     response = self._anthropic_client.messages.create(  # type: ignore[union-attr]
#                         model=NARRATIVE_SYNTH_ANTHROPIC_MODEL,
#                         max_tokens=1200,
#                         temperature=0.2,
#                         system=full_system_prompt,
#                         messages=[{"role": "user", "content": user_message}],
#                     )
#                     print(
#                         f"[KGDEBUG] anthropic call success content_blocks={len(getattr(response, 'content', []) or [])}",
#                         flush=True,
#                     )
#                 except Exception as anthropic_error:
#                     print(f"[KGDEBUG] anthropic call error: {anthropic_error}", flush=True)
#                     raise
#                 parts = []
#                 for block in getattr(response, "content", []) or []:
#                     if hasattr(block, "text"):
#                         parts.append(block.text)
#                 return "\n".join(parts)

#             try:
#                 print("[KGDEBUG] openai call start", flush=True)
#                 response = self._openai_client.responses.create(  # type: ignore[union-attr]
#                     model=NARRATIVE_SYNTH_OPENAI_MODEL,
#                     input=[
#                         {"role": "system", "content": full_system_prompt},
#                         {"role": "user", "content": user_message},
#                     ],
#                     max_output_tokens=1600,
#                     temperature=0.2,
#                 )
#                 print("[KGDEBUG] openai call success", flush=True)
#             except Exception as openai_error:
#                 print(f"[KGDEBUG] openai call error: {openai_error}", flush=True)
#                 raise
#             return _extract_openai_text(response)


def test_prompt_sets(model=None, temperature=None, provider=None):
    """
    Test each of the three system prompt sets.
    
    Args:
        model: Override default model (uses TEST_ANTHROPIC_MODEL or TEST_OPENAI_MODEL based on provider)
        temperature: Override default temperature (default: TEST_TEMPERATURE)
        provider: Override default provider (default: TEST_PROVIDER)
    """
    # Set up configuration
    test_provider = provider or TEST_PROVIDER
    test_temperature = temperature or TEST_TEMPERATURE
    
    if test_provider == "anthropic":
        test_model = model or TEST_ANTHROPIC_MODEL
    else:
        test_model = model or TEST_OPENAI_MODEL
    
    # Initialize clients
    anthropic_client = None
    openai_client = None
    
    if test_provider == "anthropic" and anthropic and os.getenv("ANTHROPIC_API_KEY"):
        try:
            anthropic_client = anthropic.Anthropic()
            print(f"Initialized Anthropic client with model: {test_model}")
        except Exception as e:
            print(f"Failed to initialize Anthropic client: {e}")
            return
    elif test_provider == "openai" and OpenAI and os.getenv("OPENAI_API_KEY"):
        try:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print(f"Initialized OpenAI client with model: {test_model}")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            return
    else:
        print(f"Client unavailable for provider: {test_provider}")
        return
    
    # Define test sets
    test_sets = [
        ("Set 1 (system_prompt1 + user_query1)", system_prompt1, user_query1),
        ("Set 2 (system_prompt2 + user_query2)", system_prompt2, user_query2),
        ("Set 3 (system_prompt3 + user_query3)", system_prompt3, user_query3),
    ]
    
    print(f"\n=== PROMPT TESTING CONFIGURATION ===")
    print(f"Provider: {test_provider}")
    print(f"Model: {test_model}")
    print(f"Temperature: {test_temperature}")
    print(f"API Key Available: {'Yes' if (anthropic_client or openai_client) else 'No'}")
    print("=" * 50)
    
    # Test each set
    for set_name, system_prompt, user_query in test_sets:
        print(f"\n{'='*60}")
        print(f"TESTING: {set_name}")
        print(f"{'='*60}")
        
        # Generate response
        try:
            start_time = time.time()
            
            if test_provider == "anthropic":
                response = anthropic_client.messages.create(
                    model=test_model,
                    max_tokens=1200,
                    temperature=test_temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_query}],
                )
                
                # Extract text from response
                parts = []
                for block in getattr(response, "content", []) or []:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                result = "\n".join(parts)
                
            elif test_provider == "openai":
                # Combine system and user prompts for responses API
                combined_prompt = f"{system_prompt}\n\n{user_query}"
                
                response = openai_client.responses.create(
                    model=test_model,
                    input=combined_prompt,
                    max_output_tokens=1600,
                    temperature=test_temperature,
                )
                # Extract text from response
                result = response.output[0] if hasattr(response, 'output') and response.output else str(response)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"⏱️  Response time: {duration:.2f} seconds")
            print(f"\n{result}")
            
        except Exception as e:
            print(f"Error generating response: {e}")
        
        print(f"\n{'='*60}")
        print(f"COMPLETED: {set_name}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage:
    # test_prompt_sets()
    #test_prompt_sets(temperature=0.5)
    #test_prompt_sets(provider="openai", model="gpt-4.1-2025-04-14")
    test_prompt_sets(provider="openai", model="gpt-4.1-2025-04-14")

    #test_prompt_sets()

    