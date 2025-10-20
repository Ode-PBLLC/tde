#!/usr/bin/env python3
import requests
import json

response = requests.post(
    "http://localhost:8098/query/stream",
    json={"query": "What is the relationship between deforestation proximity indicators and biodiversity impact scores (PDF metrics) for agricultural companies operating in Brazil?"},
    timeout=90,
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: ') and '"type": "complete"' in line_str:
            data = json.loads(line_str[6:])
            modules = data['data']['modules']
            for module in modules:
                if module['type'] == 'text':
                    print("RESPONSE TEXT:")
                    print("-" * 80)
                    for text in module.get('texts', []):
                        print(text)
                        print()
            break