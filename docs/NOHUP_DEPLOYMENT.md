# Simple Nohup Deployment Commands

## Quick Start with Nohup

### 1. Start Both Servers

```bash
# Navigate to your project directory
cd /home/ubuntu/tde

# Start KG Visualization Server (port 8100)
nohup python3 kg_visualization_server.py > kg_server.log 2>&1 &

# Start Main API Server (port 8098)
nohup python3 api_server.py > api_server.log 2>&1 &

# Check they're running
ps aux | grep python
```

### 2. View Logs in Real-Time

```bash
# Watch KG server logs
tail -f kg_server.log

# Watch API server logs
tail -f api_server.log

# Watch both logs at once
tail -f kg_server.log api_server.log
```

### 3. Stop the Servers

```bash
# Find the process IDs
ps aux | grep kg_visualization_server
ps aux | grep api_server

# Kill them
kill <PID_of_kg_server>
kill <PID_of_api_server>

# Or kill all Python processes (careful!)
pkill -f kg_visualization_server.py
pkill -f api_server.py
```

## One-Liner to Start Everything

```bash
cd /home/ubuntu/tde && nohup python3 kg_visualization_server.py > kg_server.log 2>&1 & nohup python3 api_server.py > api_server.log 2>&1 & echo "Both servers started! Check kg_server.log and api_server.log"
```

## Create a Simple Start Script

Create `start_servers.sh`:

```bash
#!/bin/bash
cd /home/ubuntu/tde

echo "Starting KG Visualization Server..."
nohup python3 kg_visualization_server.py > kg_server.log 2>&1 &
echo $! > kg_server.pid
echo "KG Server PID: $(cat kg_server.pid)"

echo "Starting API Server..."
nohup python3 api_server.py > api_server.log 2>&1 &
echo $! > api_server.pid
echo "API Server PID: $(cat api_server.pid)"

echo "Both servers started!"
echo "Logs: tail -f kg_server.log api_server.log"
```

Create `stop_servers.sh`:

```bash
#!/bin/bash
cd /home/ubuntu/tde

if [ -f kg_server.pid ]; then
    kill $(cat kg_server.pid)
    rm kg_server.pid
    echo "KG Server stopped"
fi

if [ -f api_server.pid ]; then
    kill $(cat api_server.pid)
    rm api_server.pid
    echo "API Server stopped"
fi
```

Make them executable:
```bash
chmod +x start_servers.sh stop_servers.sh

# Use them
./start_servers.sh
./stop_servers.sh
```

## Test Everything is Working

```bash
# Check processes are running
ps aux | grep python

# Test KG server
curl http://localhost:8100/api/kg/stats

# Test API server
curl http://localhost:8098/health

# Test integration (should include concepts)
curl -X POST "http://localhost:8098/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "solar energy"}' | jq '.concepts'
```

## Auto-Restart on Reboot (Simple Cron)

```bash
# Add to crontab
crontab -e

# Add these lines:
@reboot cd /home/ubuntu/tde && nohup python3 kg_visualization_server.py > kg_server.log 2>&1 &
@reboot cd /home/ubuntu/tde && nohup python3 api_server.py > api_server.log 2>&1 &
```

## Important Notes

1. **Update Public URLs**: Remember to update the KG URLs in your code:
   ```python
   # In api_server.py and mcp/mcp_chat.py
   "kg_visualization_url": "https://yourdomain.com:8100"  # or your actual domain
   ```

2. **Firewall**: Open the ports if needed:
   ```bash
   sudo ufw allow 8098
   sudo ufw allow 8100
   ```

3. **Monitor Logs**: The logs will grow over time. Consider rotating them:
   ```bash
   # Simple log rotation
   mv api_server.log api_server.log.old
   mv kg_server.log kg_server.log.old
   # Then restart servers
   ```

That's it! The nohup approach is simple and works well for testing or small deployments.