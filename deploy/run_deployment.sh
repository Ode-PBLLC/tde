#!/bin/bash
# Simple deployment script for Climate API
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Climate API Deployment${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ Don't run this script as root${NC}"
   exit 1
fi

# Variables - EDIT THESE
DOMAIN="your-domain.com"  # Replace with your domain
EMAIL="your-email@domain.com"  # Replace with your email

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo "Current user: $(whoami)"
echo

# Update system
echo -e "${YELLOW}📦 Updating system...${NC}"
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
sudo apt install -y \
    python3 python3-pip python3-venv \
    nginx certbot python3-certbot-nginx \
    git htop curl unzip

# Create application user
echo -e "${YELLOW}👤 Creating application user...${NC}"
sudo useradd --system --shell /bin/bash --home /opt/climate-api climate-api || true
sudo mkdir -p /opt/climate-api
sudo chown climate-api:climate-api /opt/climate-api

# Move application code
echo -e "${YELLOW}📁 Setting up application...${NC}"
sudo mv /tmp/tde /opt/climate-api/app
sudo chown -R climate-api:climate-api /opt/climate-api/app

# Move data files
echo -e "${YELLOW}📊 Setting up data files...${NC}"
sudo mv /tmp/data /opt/climate-api/app/ || echo "No data directory found"
sudo mv /tmp/extras /opt/climate-api/app/ || echo "No extras directory found"
sudo chown -R climate-api:climate-api /opt/climate-api/app/data || true
sudo chown -R climate-api:climate-api /opt/climate-api/app/extras || true

# Create directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
sudo -u climate-api mkdir -p /opt/climate-api/app/static/maps
sudo -u climate-api mkdir -p /opt/climate-api/app/data/processed
sudo -u climate-api mkdir -p /opt/climate-api/logs

# Setup Python environment
echo -e "${YELLOW}🐍 Setting up Python environment...${NC}"
sudo -u climate-api python3 -m venv /opt/climate-api/venv
sudo -u climate-api /opt/climate-api/venv/bin/pip install --upgrade pip
sudo -u climate-api /opt/climate-api/venv/bin/pip install -r /opt/climate-api/app/requirements.txt
sudo -u climate-api /opt/climate-api/venv/bin/pip install polars pyarrow

# Setup environment file
echo -e "${YELLOW}⚙️ Creating environment file...${NC}"
if [ -f "/tmp/.env" ]; then
    sudo mv /tmp/.env /opt/climate-api/app/.env
    sudo chown climate-api:climate-api /opt/climate-api/app/.env
else
    echo "⚠️ No .env file found. Creating template..."
    sudo -u climate-api tee /opt/climate-api/app/.env << EOF
# API Keys - REPLACE WITH YOUR ACTUAL KEYS
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Application settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Large dataset settings
POLARS_MAX_THREADS=4
POLARS_STREAMING_CHUNK_SIZE=50000
CHUNK_SIZE=50000
MAX_CACHE_SIZE=5
EOF
    echo -e "${RED}⚠️ IMPORTANT: Edit /opt/climate-api/app/.env with your actual API keys!${NC}"
fi

# Create systemd service
echo -e "${YELLOW}⚙️ Creating systemd service...${NC}"
sudo tee /etc/systemd/system/climate-api.service << 'EOF'
[Unit]
Description=Climate Policy Radar API
After=network.target

[Service]
Type=exec
User=climate-api
Group=climate-api
WorkingDirectory=/opt/climate-api/app
Environment=PATH=/opt/climate-api/venv/bin
ExecStart=/opt/climate-api/venv/bin/uvicorn api_server:app --host 127.0.0.1 --port 8099 --workers 2
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=climate-api

# Security settings
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/climate-api/app/static /opt/climate-api/app/data
PrivateTmp=yes

# Memory settings for large datasets
MemoryHigh=6G
MemoryMax=7G
TimeoutStartSec=300
TimeoutStopSec=120

# Environment variables
EnvironmentFile=/opt/climate-api/app/.env

[Install]
WantedBy=multi-user.target
EOF

# Setup Nginx
echo -e "${YELLOW}🌐 Configuring Nginx...${NC}"
sudo tee /etc/nginx/sites-available/climate-api << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # CORS headers for API
    add_header Access-Control-Allow-Origin "*";
    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
    add_header Access-Control-Allow-Headers "Content-Type, Authorization";
    
    # Static files (GeoJSON)
    location /static/ {
        alias /opt/climate-api/app/static/;
        expires 1h;
        add_header Cache-Control "public, immutable";
        
        location ~* \.(geojson)$ {
            add_header Content-Type application/json;
            add_header Access-Control-Allow-Origin "*";
        }
    }
    
    # API endpoints
    location / {
        proxy_pass http://127.0.0.1:8099;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Extended timeouts for large dataset queries
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=5r/s;
    limit_req zone=api burst=10 nodelay;
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/climate-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx config
sudo nginx -t

# Setup firewall
echo -e "${YELLOW}🔥 Configuring firewall...${NC}"
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force reload

# Enable and start services
echo -e "${YELLOW}🚀 Starting services...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable climate-api
sudo systemctl start climate-api
sudo systemctl restart nginx

# Setup SSL if domain is provided and not default
if [ "$DOMAIN" != "your-domain.com" ]; then
    echo -e "${YELLOW}🔒 Setting up SSL...${NC}"
    sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --email $EMAIL --agree-tos --non-interactive || echo "SSL setup failed - you can run certbot manually later"
fi

# Setup cleanup cron
echo -e "${YELLOW}🧹 Setting up cleanup jobs...${NC}"
sudo -u climate-api crontab << 'EOF'
# Clean up old GeoJSON files daily at 2 AM
0 2 * * * find /opt/climate-api/app/static/maps -name "*.geojson" -mtime +7 -delete
EOF

# Final status check
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo
echo -e "${YELLOW}📊 Service Status:${NC}"
sudo systemctl status climate-api --no-pager || true
echo
echo -e "${YELLOW}🔍 Testing endpoints:${NC}"
echo "Health check:"
curl -s http://localhost:8099/health || echo "❌ Health check failed"
echo
echo
echo -e "${GREEN}🎉 Deployment Summary:${NC}"
echo "• Application: /opt/climate-api/app"
echo "• Logs: sudo journalctl -u climate-api -f"
echo "• Nginx config: /etc/nginx/sites-available/climate-api"
echo "• Environment: /opt/climate-api/app/.env"
echo
if [ "$DOMAIN" != "your-domain.com" ]; then
    echo "• Your API: https://$DOMAIN"
    echo "• Health check: https://$DOMAIN/health"
    echo "• Example: https://$DOMAIN/example-response"
else
    echo "• Update DOMAIN in this script and re-run for SSL setup"
    echo "• Your API: http://your-server-ip"
fi
echo
echo -e "${YELLOW}⚠️ Next steps:${NC}"
echo "1. Edit /opt/climate-api/app/.env with your actual API keys"
echo "2. sudo systemctl restart climate-api"
echo "3. Test your API endpoints"
echo
echo -e "${GREEN}🚀 Ready to serve climate policy data!${NC}"