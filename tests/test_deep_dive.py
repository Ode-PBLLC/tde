#!/usr/bin/env python3
"""Deep dive into a single complex query to understand the full flow."""

import requests
import json
import time
from datetime import datetime

QUERY = "What is the relationship between deforestation proximity indicators and biodiversity impact scores (PDF metrics) for agricultural companies operating in Brazil?"

def test_streaming_query():
    """Test the streaming endpoint with detailed logging."""

    print(f"{'='*80}")
    print(f"DEEP DIVE QUERY TEST")
    print(f"Time: {datetime.now()}")
    print(f"Query: {QUERY}")
    print(f"{'='*80}\n")

    url = "http://localhost:8098/query/stream"
    headers = {"Content-Type": "application/json"}
    data = {"query": QUERY}

    # Track all events
    events = {
        "conversation_id": None,
        "thinking": [],
        "facts": [],
        "servers_mentioned": set(),
        "tool_calls": [],
        "response_chunks": [],
        "citations": [],
        "errors": [],
        "metadata": {}
    }

    print("STREAMING EVENTS:")
    print("-" * 80)

    try:
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=120)

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        event_data = json.loads(line_str[6:])
                        event_type = event_data.get('type')
                        event_content = event_data.get('data', {})

                        # Print each event
                        timestamp = datetime.now().strftime("%H:%M:%S")

                        if event_type == 'conversation_id':
                            events['conversation_id'] = event_content.get('conversation_id')
                            print(f"[{timestamp}] CONVERSATION: {events['conversation_id']}")

                        elif event_type == 'thinking':
                            msg = event_content.get('message', '')
                            category = event_content.get('category', '')
                            events['thinking'].append({'message': msg, 'category': category})
                            print(f"[{timestamp}] THINKING [{category}]: {msg}")

                            # Extract server names from thinking messages
                            if 'Collecting data from' in msg:
                                server = msg.split('Collecting data from ')[-1].replace('...', '').strip()
                                events['servers_mentioned'].add(server)
                            elif 'Selected' in msg and 'data sources:' in msg:
                                sources = msg.split('data sources: ')[-1].strip()
                                for s in sources.split(', '):
                                    events['servers_mentioned'].add(s.strip())

                        elif event_type == 'facts_summary':
                            events['facts'].append(event_content)
                            print(f"[{timestamp}] FACTS: Phase {event_content.get('phase')}, Total: {event_content.get('total')}")
                            if 'by_server' in event_content:
                                print(f"         By server: {event_content['by_server']}")

                        elif event_type == 'response':
                            chunk = event_content.get('chunk', '')
                            events['response_chunks'].append(chunk)
                            # Print first 100 chars of chunk
                            if len(chunk) > 100:
                                print(f"[{timestamp}] RESPONSE: {chunk[:100]}...")
                            else:
                                print(f"[{timestamp}] RESPONSE: {chunk}")

                        elif event_type == 'citation':
                            events['citations'].append(event_content)
                            print(f"[{timestamp}] CITATION: {event_content.get('source', 'Unknown')}")

                        elif event_type == 'complete':
                            events['metadata'] = event_content.get('metadata', {})
                            print(f"[{timestamp}] COMPLETE: {event_content.get('metadata', {}).get('facts_collected')} facts collected")

                        elif event_type == 'error':
                            events['errors'].append(event_content)
                            print(f"[{timestamp}] ERROR: {event_content}")

                    except json.JSONDecodeError as e:
                        print(f"[{timestamp}] JSON Error: {e}")

    except requests.Timeout:
        print("\n⚠️ Request timed out after 120 seconds")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    # Summary analysis
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY:")
    print(f"{'='*80}")

    print(f"\n1. SERVERS USED:")
    for server in events['servers_mentioned']:
        print(f"   - {server}")

    print(f"\n2. FACTS COLLECTED:")
    for fact in events['facts']:
        print(f"   Phase {fact.get('phase')}: {fact.get('total')} facts")
        if 'by_server' in fact:
            for server, count in fact['by_server'].items():
                print(f"      {server}: {count} facts")

    print(f"\n3. RESPONSE LENGTH: {sum(len(c) for c in events['response_chunks'])} characters")

    print(f"\n4. CITATIONS: {len(events['citations'])} citations")
    for i, citation in enumerate(events['citations'][:5], 1):
        print(f"   {i}. {citation.get('source', 'Unknown')} - {citation.get('description', '')[:50]}")

    print(f"\n5. ERRORS: {len(events['errors'])} errors")
    for error in events['errors']:
        print(f"   - {error}")

    # Full response text
    full_response = ''.join(events['response_chunks'])
    print(f"\n{'='*80}")
    print("FULL RESPONSE (first 1000 chars):")
    print(f"{'='*80}")
    print(full_response[:1000])

    return events

if __name__ == "__main__":
    events = test_streaming_query()