# Deploying KG Visualization Server with Main API Server

## Overview

You need to run two servers:
1. **Main API Server** (port 8098) - Your climate policy API
2. **KG Visualization Server** (port 8100) - Knowledge graph API and visualization

## Quick Start

### 1. Using systemd Services (Recommended for Production)

Create two service files:

#### `/etc/systemd/system/climate-api.service`
```ini
[Unit]
Description=Climate Policy API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tde
Environment="PATH=/home/ubuntu/.local/bin:/usr/bin"
ExecStart=/usr/bin/python3 api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### `/etc/systemd/system/kg-visualization.service`
```ini
[Unit]
Description=Knowledge Graph Visualization Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tde
Environment="PATH=/home/ubuntu/.local/bin:/usr/bin"
ExecStart=/usr/bin/python3 kg_visualization_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start both services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable climate-api kg-visualization
sudo systemctl start climate-api kg-visualization
sudo systemctl status climate-api kg-visualization
```

### 2. Using Screen/Tmux (Quick Testing)

```bash
# Start KG server in screen
screen -S kg-server
cd /home/ubuntu/tde
python3 kg_visualization_server.py
# Ctrl+A, D to detach

# Start API server in another screen
screen -S api-server
cd /home/ubuntu/tde
python3 api_server.py
# Ctrl+A, D to detach

# List screens
screen -ls

# Reattach to check logs
screen -r kg-server
screen -r api-server
```

### 3. Using PM2 (Node.js Process Manager)

```bash
# Install PM2
npm install -g pm2

# Start both servers
pm2 start kg_visualization_server.py --name kg-viz --interpreter python3
pm2 start api_server.py --name climate-api --interpreter python3

# Save configuration
pm2 save
pm2 startup
```

## Important Configuration Changes

### 1. Update API Server URLs

Since the servers communicate over localhost, you need to ensure the URLs are correct:

#### In `api_server.py`:
```python
# This should already be correct
kg_server_url = "http://localhost:8100/api/kg/query-subgraph"
```

#### In `mcp/mcp_chat.py`:
```python
# This should already be correct  
kg_server_url = "http://localhost:8100/api/kg/query-subgraph"
```

### 2. Update Public-Facing URLs

For the KG visualization URLs returned in API responses, update based on your domain:

#### In `api_server.py` (around line 167):
```python
"kg_visualization_url": "https://yourdomain.com:8100",
"kg_query_url": f"https://yourdomain.com:8100?query={query_text.replace(' ', '%20')}"
```

#### In `mcp/mcp_chat.py` (around line 2225):
```python
"kg_visualization_url": "https://yourdomain.com:8100",
"kg_query_url": f"https://yourdomain.com:8100?query={query.replace(' ', '%20')}"
```

## Nginx Configuration (Recommended)

To properly expose both services through a single domain:

```nginx
# /etc/nginx/sites-available/climate-api
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Main API
    location / {
        proxy_pass http://localhost:8098;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # KG Visualization (subdirectory)
    location /kg/ {
        proxy_pass http://localhost:8100/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Then update URLs to use subdirectory:
```python
"kg_visualization_url": "https://yourdomain.com/kg",
"kg_query_url": f"https://yourdomain.com/kg?query={query_text.replace(' ', '%20')}"
```

## Firewall Configuration

If accessing directly (not through nginx):

```bash
# Allow both ports
sudo ufw allow 8098/tcp
sudo ufw allow 8100/tcp
sudo ufw status
```

## Monitoring and Logs

### Check Service Status
```bash
# Systemd
sudo systemctl status climate-api
sudo systemctl status kg-visualization

# View logs
sudo journalctl -u climate-api -f
sudo journalctl -u kg-visualization -f

# PM2
pm2 status
pm2 logs climate-api
pm2 logs kg-viz
```

### Test Communication
```bash
# From the server itself
curl http://localhost:8098/health
curl http://localhost:8100/api/kg/stats

# Test KG integration
curl -X POST "http://localhost:8098/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "climate policy"}' | jq '.concepts'
```

## Troubleshooting

### Issue: KG data not appearing in API responses
```bash
# Check if KG server is running
curl http://localhost:8100/api/kg/stats

# Check logs for connection errors
sudo journalctl -u climate-api | grep "KG server"

# Test direct KG query
curl -X POST "http://localhost:8100/api/kg/query-subgraph" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### Issue: Port already in use
```bash
# Find what's using the port
sudo lsof -i :8098
sudo lsof -i :8100

# Kill if needed
sudo kill -9 <PID>
```

### Issue: Services won't start
```bash
# Check Python path
which python3

# Check dependencies
cd /home/ubuntu/tde
python3 -c "import fastapi; import aiohttp; print('Dependencies OK')"

# Check file permissions
ls -la api_server.py kg_visualization_server.py
```

## Production Checklist

- [ ] Both services configured with systemd
- [ ] Nginx proxy configured 
- [ ] SSL certificates installed
- [ ] Firewall rules updated
- [ ] Public URLs updated in code
- [ ] Services set to auto-restart
- [ ] Log rotation configured
- [ ] Monitoring alerts set up

## Environment Variables

Consider using environment variables for flexibility:

```bash
# /etc/environment or .env file
KG_SERVER_URL=http://localhost:8100
KG_PUBLIC_URL=https://yourdomain.com/kg
API_PUBLIC_URL=https://yourdomain.com
```

Then update code to use:
```python
import os
kg_public_url = os.getenv('KG_PUBLIC_URL', 'http://localhost:8100')
```

This setup ensures both servers can communicate locally while being accessible to your users through proper URLs.