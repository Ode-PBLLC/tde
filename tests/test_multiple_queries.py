#!/usr/bin/env python3
import requests
import json
import time

queries = [
    "Which areas in Brazil show extreme heat stress (quintile 5) that overlap with high-density solar facility installations?",
    "How do climate policy concepts like 'renewable energy' and 'transportation' connect through the knowledge graph?",
    "What Amazon-specific climate adaptation strategies does the Science Panel for the Amazon recommend for regions with both high deforestation rates and water stress?"
]

for i, query in enumerate(queries, 1):
    print(f"\n{'='*80}")
    print(f"QUERY {i}: {query}")
    print('='*80)

    start_time = time.time()

    response = requests.post(
        "http://localhost:8098/query/stream",
        json={"query": query},
        timeout=90,
        stream=True
    )

    servers_used = set()
    response_text = []
    facts_collected = 0

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                try:
                    data = json.loads(line_str[6:])

                    # Track servers
                    if data.get('type') == 'thinking':
                        msg = data.get('data', {}).get('message', '')
                        if 'Selected' in msg and 'data sources:' in msg:
                            sources = msg.split('data sources: ')[-1].strip()
                            print(f"Servers: {sources}")

                    # Track facts
                    elif data.get('type') == 'facts_summary':
                        facts_collected = data.get('data', {}).get('total', 0)
                        print(f"Facts collected: {facts_collected}")

                    # Get response
                    elif data.get('type') == 'complete':
                        modules = data['data']['modules']
                        for module in modules:
                            if module['type'] == 'text':
                                response_text = module.get('texts', [])
                        break

                except json.JSONDecodeError:
                    pass

    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f} seconds")
    print("\nRESPONSE:")
    print("-" * 40)
    for text in response_text[:2]:  # First 2 paragraphs
        print(text[:300] + "..." if len(text) > 300 else text)

    time.sleep(2)  # Be nice to the server