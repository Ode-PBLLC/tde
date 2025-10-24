# Climate Policy API - Deployment Guide

Complete guide for local development and production deployment.

---

## Local Development Setup

### Prerequisites
- Python 3.11+
- Conda or venv
- Anthropic API key
- OpenAI API key (for embeddings)

### Installation

```bash
# Clone repository
git clone https://github.com/Ode-PBLLC/tde.git
cd tde

# Create environment
conda create -n tde-api python=3.11
conda activate tde-api

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "ANTHROPIC_API_KEY=your-anthropic-key" > .env
echo "OPENAI_API_KEY=your-openai-key" >> .env

# Start development server
python api_server.py
```

The API will be available at `http://localhost:8099`

### Development Commands

```bash
# Run with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8099

# Test health check
curl http://localhost:8099/health

# Test streaming endpoint
curl -X POST http://localhost:8099/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

---

## Production Deployment

### Current Production Environment
- **URL**: http://54.146.227.119:8099
- **Platform**: AWS EC2 Ubuntu
- **Environment**: Conda
- **Process**: Background service

### AWS EC2 Deployment

#### 1. Server Setup

```bash
# Connect to EC2 instance
ssh -i your-key.pem ubuntu@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3-pip python3-venv git curl

# Install Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### 2. Application Deployment

```bash
# Clone repository
git clone https://github.com/Ode-PBLLC/tde.git
cd tde

# Create environment
conda create -n tde-api python=3.11
conda activate tde-api

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
nano .env
# Add your API keys:
# ANTHROPIC_API_KEY=your-key
# OPENAI_API_KEY=your-key
```

#### 3. Service Configuration

Create systemd service for automatic startup:

```bash
sudo nano /etc/systemd/system/climate-api.service
```

Service file content:
```ini
[Unit]
Description=Climate Policy API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tde
Environment=PATH=/home/ubuntu/miniconda3/envs/tde-api/bin
ExecStart=/home/ubuntu/miniconda3/envs/tde-api/bin/python api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable climate-api
sudo systemctl start climate-api
sudo systemctl status climate-api
```

#### 4. Security & Firewall

```bash
# Configure firewall
sudo ufw allow 22    # SSH
sudo ufw allow 8099  # API port
sudo ufw enable

# Check service logs
sudo journalctl -u climate-api -f
```

---

## Deployment Updates

### Git-Based Updates

```bash
# On production server
cd /home/ubuntu/tde
git pull origin main

# Restart service
sudo systemctl restart climate-api
```

### Content Updates (Featured Queries)

```bash
# Update featured queries (no restart needed)
nano static/featured_queries.json

# Add new images
scp image.jpg ubuntu@server:/home/ubuntu/tde/static/images/
```

---

## Nginx Reverse Proxy (Optional)

For production with SSL and domain name:

### Install Nginx
```bash
sudo apt install nginx
sudo nano /etc/nginx/sites-available/climate-api
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8099;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

### Enable Site
```bash
sudo ln -s /etc/nginx/sites-available/climate-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Monitoring & Maintenance

### Service Status
```bash
# Check API service
sudo systemctl status climate-api

# View logs
sudo journalctl -u climate-api -f

# Restart if needed
sudo systemctl restart climate-api
```

### System Resources
```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check CPU usage
top
```

### API Health Monitoring
```bash
# Health check
curl http://localhost:8099/health

# Test streaming endpoint
curl -X POST http://localhost:8099/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' \
  --max-time 30
```

---

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
sudo journalctl -u climate-api -n 50

# Common causes:
# - Missing API keys in .env
# - Port already in use
# - Python environment issues
```

#### Dependencies Issues
```bash
# Recreate environment
conda deactivate
conda remove -n tde-api --all
conda create -n tde-api python=3.11
conda activate tde-api
pip install -r requirements.txt
```

#### Performance Issues
```bash
# Check if all MCP servers are responding
python -c "
import asyncio
from mcp_chat import run_query
result = asyncio.run(run_query('test'))
print('Success' if result else 'Failed')
"
```

### Log Analysis
```bash
# API server logs
sudo journalctl -u climate-api --since "1 hour ago"

# System logs  
sudo journalctl --since "1 hour ago" | grep climate-api

# Nginx logs (if using)
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## Backup & Recovery

### Configuration Backup
```bash
# Backup critical files
tar -czf climate-api-backup.tar.gz \
  tde/.env \
  tde/static/featured_queries.json \
  tde/static/images/ \
  /etc/systemd/system/climate-api.service
```

### Recovery
```bash
# Restore from backup
tar -xzf climate-api-backup.tar.gz
sudo systemctl daemon-reload
sudo systemctl restart climate-api
```

---

## Environment Variables

Required environment variables in `.env`:

```bash
# Required
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key

# Optional
PORT=8099
HOST=0.0.0.0
DEBUG=false
```

---

## Performance Tuning

### System Optimization
```bash
# Increase file limits for high concurrent connections
echo "ubuntu soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "ubuntu hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

### API Configuration
- Default timeout: 120 seconds
- CORS enabled for all origins
- Static file serving for images and maps
- Automatic error handling and recovery

---

For API integration and frontend development, see **[API_GUIDE.md](API_GUIDE.md)**.