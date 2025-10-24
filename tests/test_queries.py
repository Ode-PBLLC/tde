#!/usr/bin/env python3
"""Test all 15 queries against the API and analyze responses."""

import requests
import json
import time
from typing import Dict, List, Any

# List of test queries
QUERIES = [
    "Which Brazilian municipalities have the highest concentration of solar facilities and how does this correlate with recent deforestation patterns in those regions?",
    "What are the top 10 companies with the highest water stress risk exposure across their asset portfolios, and in which Brazilian states are these assets primarily located?",
    "How have Scope 3 emissions trends evolved from 2016-2024 for companies in the energy sector versus the industrial sector?",
    "Which areas in Brazil show extreme heat stress (quintile 5) that overlap with high-density solar facility installations?",
    "What is the relationship between deforestation proximity indicators and biodiversity impact scores (PDF metrics) for agricultural companies operating in Brazil?",
    "Which Brazilian states have municipalities with populations over 100,000 that are within 50km of major deforestation areas detected in the last 5 years?",
    "How do climate policy concepts like 'renewable energy' and 'transportation' connect through the knowledge graph, and what policy passages mention both?",
    "What percentage of GIST-tracked company assets in coastal areas face both high flood risk and extreme heat stress simultaneously?",
    "Which companies show the largest year-over-year increase in biodiversity impacts (CO2E metrics) while also having high forest change proximity indicators?",
    "What Amazon-specific climate adaptation strategies does the Science Panel for the Amazon recommend for regions with both high deforestation rates and water stress?",
    "How many solar facilities are located within Brazilian indigenous territories or protected areas, and what is their combined capacity?",
    "Which sectors have the highest emissions intensity (Scope 3 emissions per revenue) and how does this correlate with their environmental risk scores?",
    "What are the spatial clusters of high biodiversity impact (MSA risk level) assets, and which companies own the majority of assets in these hotspots?",
    "How do land use change patterns (urban area change, agriculture area change, forest area change) differ between companies with strong versus weak climate governance indicators?",
    "Which Brazilian municipalities have experienced both significant solar energy development and reduction in deforestation rates, suggesting successful green transition patterns?"
]

def test_query(query: str, query_num: int) -> Dict[str, Any]:
    """Test a single query against the API."""
    print(f"\n{'='*80}")
    print(f"Query {query_num}: {query[:100]}...")
    print('='*80)

    url = "http://localhost:8098/query/stream"
    headers = {"Content-Type": "application/json"}
    data = {"query": query}

    result = {
        "query_num": query_num,
        "query": query,
        "servers_used": set(),
        "thinking_messages": [],
        "response_chunks": [],
        "citations": [],
        "error": None,
        "timed_out": False
    }

    try:
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=45)

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        event_data = json.loads(line_str[6:])

                        # Track thinking messages
                        if event_data.get('type') == 'thinking':
                            msg = event_data.get('data', {}).get('message', '')
                            result['thinking_messages'].append(msg)

                            # Extract server information
                            if 'Collecting data from' in msg:
                                server = msg.split('Collecting data from ')[-1].replace('...', '').strip()
                                result['servers_used'].add(server)
                            elif 'Selected' in msg and 'data sources:' in msg:
                                sources = msg.split('data sources: ')[-1].strip()
                                for s in sources.split(', '):
                                    result['servers_used'].add(s.strip())

                        # Track response chunks
                        elif event_data.get('type') == 'response':
                            chunk = event_data.get('data', {}).get('chunk', '')
                            result['response_chunks'].append(chunk)

                        # Track citations
                        elif event_data.get('type') == 'citation':
                            result['citations'].append(event_data.get('data', {}))

                    except json.JSONDecodeError:
                        continue

    except requests.Timeout:
        result['timed_out'] = True
        result['error'] = "Request timed out after 45 seconds"
    except Exception as e:
        result['error'] = str(e)

    # Convert sets to lists for JSON serialization
    result['servers_used'] = list(result['servers_used'])

    # Print summary
    print(f"Servers Used: {result['servers_used']}")
    print(f"Thinking Messages: {len(result['thinking_messages'])}")
    print(f"Response Length: {sum(len(c) for c in result['response_chunks'])} chars")
    print(f"Citations: {len(result['citations'])}")
    if result['timed_out']:
        print("⚠️ TIMED OUT")
    if result['error']:
        print(f"❌ ERROR: {result['error']}")

    return result

def main():
    """Test all queries and generate summary."""
    results = []

    for i, query in enumerate(QUERIES[:5], 1):  # Test first 5 queries to avoid timeout
        result = test_query(query, i)
        results.append(result)
        time.sleep(2)  # Be nice to the server

    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)

    # Aggregate server usage
    all_servers = {}
    for r in results:
        for server in r['servers_used']:
            all_servers[server] = all_servers.get(server, 0) + 1

    print("\nServer Usage Frequency:")
    for server, count in sorted(all_servers.items(), key=lambda x: x[1], reverse=True):
        print(f"  {server}: {count} times")

    # Success rate
    successful = sum(1 for r in results if not r['error'] and not r['timed_out'])
    print(f"\nSuccess Rate: {successful}/{len(results)} queries")

    # Citation analysis
    total_citations = sum(len(r['citations']) for r in results)
    print(f"Total Citations: {total_citations}")

    # Average response length
    avg_length = sum(sum(len(c) for c in r['response_chunks']) for r in results) / len(results)
    print(f"Average Response Length: {avg_length:.0f} chars")

    # Save detailed results
    with open('query_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to query_test_results.json")

if __name__ == "__main__":
    main()