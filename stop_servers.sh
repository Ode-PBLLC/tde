#!/bin/bash

# Stop script for Climate Policy Radar API servers
# Kills processes on ports 8098 and 8100

echo "🛑 Stopping Climate Policy Radar API servers..."

# Function to kill process on a specific port
kill_port() {
    local port=$1
    local service_name=$2
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "🔪 Stopping $service_name on port $port (PID: $pid)"
        kill -15 $pid  # Try graceful shutdown first
        sleep 2
        
        # Check if still running, force kill if needed
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "⚡ Force killing $service_name (PID: $pid)"
            kill -9 $pid
        fi
        echo "✅ $service_name stopped"
    else
        echo "ℹ️  No process found on port $port"
    fi
}

# Stop servers
kill_port 8098 "API server"
kill_port 8100 "KG server"

echo ""
echo "🎉 All servers stopped!"