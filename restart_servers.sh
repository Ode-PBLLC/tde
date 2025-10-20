#!/bin/bash

cd /home/ubuntu/tde

export API_BASE_URL=https://api.transitiondigital.org


conda activate tde-api

# Kill processes on ports 8098 and 8100
echo "Killing processes on ports 8098 and 8100..."
lsof -ti:8098 | xargs -r kill -9
lsof -ti:8100 | xargs -r kill -9

# Wait a moment for ports to be released
sleep 2

# Start api_server.py
echo "Starting api_server.py..."
nohup python api_server.py > api_server.log 2>&1 &
echo "api_server.py started with PID $!"

# Start kg_visualization_server.py
echo "Starting kg_visualization_server.py..."
nohup python kg_visualization_server.py > kg_visualization.log 2>&1 &
echo "kg_visualization_server.py started with PID $!"

echo "Both servers restarted successfully!"
echo "Logs are available at:"
echo "  - api_server.log"
echo "  - kg_visualization.log"

tail -f api_server.log