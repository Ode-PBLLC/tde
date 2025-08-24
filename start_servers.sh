#!/bin/bash

# Start script for Climate Policy Radar API servers
# Kills any existing processes on ports 8098 and 8100, then starts both servers

echo "🔄 Starting Climate Policy Radar API servers..."

# Function to kill process on a specific port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "🔪 Killing process on port $port (PID: $pid)"
        kill -9 $pid
        sleep 1
    else
        echo "✅ Port $port is available"
    fi
}

# Kill existing processes
echo "🧹 Cleaning up existing processes..."
kill_port 8098
kill_port 8100

# Wait a moment for ports to be fully released
sleep 2

# Start KG visualization server in background
echo "🚀 Starting KG visualization server on port 8100..."
python kg_visualization_server.py &
KG_PID=$!
echo "📊 KG server started with PID: $KG_PID"

# Wait a moment for KG server to initialize
sleep 3

# Start main API server in background
echo "🚀 Starting main API server on port 8098..."
python api_server.py &
API_PID=$!
echo "🌐 API server started with PID: $API_PID"

# Wait a moment for servers to start
sleep 3

# Check if servers are running
echo "🔍 Checking server status..."
if lsof -ti:8100 >/dev/null 2>&1; then
    echo "✅ KG server running on port 8100"
else
    echo "❌ KG server failed to start on port 8100"
fi

if lsof -ti:8098 >/dev/null 2>&1; then
    echo "✅ API server running on port 8098"
else
    echo "❌ API server failed to start on port 8098"
fi

echo ""
echo "🎉 Server startup complete!"
echo "📍 API server: http://localhost:8098"
echo "📍 KG server: http://localhost:8100"
echo ""
echo "💡 To stop servers, run: ./stop_servers.sh"
echo "📝 To view logs, check nohup.out or run with foreground option"