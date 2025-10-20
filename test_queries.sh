#!/bin/bash

# Test script for API queries
API_URL="http://localhost:8098/query"

echo "Testing API Queries"
echo "==================="

# Query 1 - Simple solar facilities query
echo -e "\n[Query 1] Top Brazilian municipalities by solar facilities"
echo "-----------------------------------------------------------"
time curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the top 5 Brazilian municipalities with the most solar facilities?"}' \
  --max-time 60 | jq -r '.response' | head -c 500
echo -e "\n"

# Query 2 - GIST water stress
echo -e "\n[Query 2] Companies with high water stress risk"
echo "-----------------------------------------------------------"
time curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"query": "Which companies have the highest water stress risk in their assets?"}' \
  --max-time 60 | jq -r '.response' | head -c 500
echo -e "\n"

# Query 3 - Climate Policy KG
echo -e "\n[Query 3] Renewable energy concepts in knowledge graph"
echo "-----------------------------------------------------------"
time curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"query": "What climate policy concepts are related to renewable energy?"}' \
  --max-time 60 | jq -r '.response' | head -c 500
echo -e "\n"

# Query 4 - Deforestation
echo -e "\n[Query 4] Deforestation areas in Brazil"
echo "-----------------------------------------------------------"
time curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the largest deforestation areas in Brazil?"}' \
  --max-time 60 | jq -r '.response' | head -c 500
echo -e "\n"

# Query 5 - Cross-dataset query
echo -e "\n[Query 5] Solar facilities near deforestation areas"
echo "-----------------------------------------------------------"
time curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"query": "Are there solar facilities near deforestation areas in Brazil?"}' \
  --max-time 60 | jq -r '.response' | head -c 500
echo -e "\n"

echo "Testing complete!"