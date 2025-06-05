# Complete Deployment Guide - Climate Policy Radar API

## Overview
Deploy a multi-dataset climate policy API with:
- **Sample datasets** (~25MB: solar, GIST, LSE)
- **Large knowledge base** (1.77GB HuggingFace dataset, 34M rows)
- **Static GeoJSON serving** for maps
- **15-20 second query response times** (acceptable for demo)

## Target Architecture
- **EC2 t3.large** (4 vCPU, 8GB RAM) - $60/month
- **50GB EBS storage** - $5/month
- **Nginx reverse proxy** + static file serving
- **6 MCP servers** for different datasets
- **Chunked processing** with Polars for large data

---

## Phase 1: EC2 Setup & Basic Infrastructure

### 1.1 Launch EC2 Instance
```bash
# EC2 Configuration:
# - Instance type: t3.large
# - AMI: Ubuntu 22.04 LTS
# - Storage: 50GB GP3 EBS
# - Security groups: SSH (22), HTTP (80), HTTPS (443)
# - Key pair: Your SSH key
```

### 1.2 Initial Server Setup
```bash
# Connect to your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    python3 python3-pip python3-venv \
    nginx certbot python3-certbot-nginx \
    git htop curl unzip

# Install Node.js (for any future frontend needs)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 1.3 Create Application User
```bash
# Create dedicated user for security
sudo useradd --system --shell /bin/bash --home /opt/climate-api climate-api
sudo mkdir -p /opt/climate-api
sudo chown climate-api:climate-api /opt/climate-api

# Create directory structure
sudo -u climate-api mkdir -p /opt/climate-api/{app,data,logs}
```

---

## Phase 2: Application Deployment

### 2.1 Clone and Setup Application
```bash
# Clone your repository (replace with your repo URL)
sudo -u climate-api git clone https://github.com/your-username/tde.git /opt/climate-api/app

# Navigate to app directory
cd /opt/climate-api/app

# Create Python virtual environment
sudo -u climate-api python3 -m venv /opt/climate-api/venv

# Activate virtual environment
sudo -u climate-api /opt/climate-api/venv/bin/pip install --upgrade pip

# Install requirements including Polars for large data processing
sudo -u climate-api /opt/climate-api/venv/bin/pip install -r requirements.txt
sudo -u climate-api /opt/climate-api/venv/bin/pip install polars pyarrow
```

### 2.2 Setup Data Directories
```bash
# Create data directories with proper permissions
sudo -u climate-api mkdir -p /opt/climate-api/app/data/{processed,cache}
sudo -u climate-api mkdir -p /opt/climate-api/app/static/maps

# Set permissions
sudo chmod 755 /opt/climate-api/app/static
sudo chmod 755 /opt/climate-api/app/static/maps
sudo chmod 755 /opt/climate-api/app/data
```

### 2.3 Environment Configuration
```bash
# Create production environment file
sudo -u climate-api tee /opt/climate-api/app/.env << EOF
# API Keys (replace with your actual keys)
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

# Static file settings
STATIC_FILE_MAX_AGE=3600
MAX_GEOJSON_FILES=1000
EOF
```

---

## Phase 3: Data Processing & Setup

### 3.1 Process HuggingFace Dataset
```bash
# Create data processing script
sudo -u climate-api tee /opt/climate-api/app/process_dataset.py << 'EOF'
#!/usr/bin/env python3
"""
Convert HuggingFace dataset to optimized Parquet format for chunked processing
"""
import os
import polars as pl
from datasets import load_dataset
import sys

def main():
    print("🔄 Loading HuggingFace dataset...")
    
    # Replace with your actual dataset name
    dataset_name = "your-huggingface-dataset-name"
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name)
        print(f"✅ Loaded dataset with {len(dataset['train'])} rows")
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(dataset["train"])
        
        # Optimize for chunked reading
        output_file = "/opt/climate-api/app/data/knowledge_base.parquet"
        
        print("🔄 Converting to optimized Parquet format...")
        df.write_parquet(
            output_file,
            compression="snappy",  # Good compression + speed balance
            row_group_size=50000   # Optimize for chunked reading
        )
        
        # Get file info
        file_size_gb = os.path.getsize(output_file) / (1024**3)
        print(f"✅ Saved {len(df)} rows to {output_file}")
        print(f"📊 File size: {file_size_gb:.2f} GB")
        
        # Create search index if needed
        print("🔄 Creating search optimizations...")
        df_optimized = df.with_columns([
            pl.col("text").str.to_lowercase().alias("text_lower") if "text" in df.columns else pl.lit("").alias("text_lower"),
            pl.col("text").str.len().alias("text_length") if "text" in df.columns else pl.lit(0).alias("text_length")
        ])
        
        optimized_file = "/opt/climate-api/app/data/knowledge_base_optimized.parquet"
        df_optimized.write_parquet(optimized_file, compression="snappy")
        
        print("✅ Dataset processing complete!")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Make executable
sudo chmod +x /opt/climate-api/app/process_dataset.py

# Run dataset processing (this may take 15-30 minutes)
echo "🚀 Processing large dataset - this may take 15-30 minutes..."
sudo -u climate-api /opt/climate-api/venv/bin/python /opt/climate-api/app/process_dataset.py
```

### 3.2 Copy Sample Datasets
```bash
# Copy your existing sample datasets
# (Adjust paths based on your local structure)

# Copy sample datasets from your local machine
# You'll need to scp these from your local machine:
# scp -i your-key.pem -r data/ ubuntu@your-ec2-ip:/tmp/

# Then move them to the proper location:
# sudo mv /tmp/data/* /opt/climate-api/app/data/
# sudo chown -R climate-api:climate-api /opt/climate-api/app/data/
```

---

## Phase 4: Service Configuration

### 4.1 Create Systemd Service
```bash
# Create systemd service file
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

# Memory and performance settings for large datasets
MemoryHigh=6G
MemoryMax=7G
TimeoutStartSec=300
TimeoutStopSec=120

# Environment variables
EnvironmentFile=/opt/climate-api/app/.env

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable climate-api
```

### 4.2 Configure Nginx
```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/climate-api << 'EOF'
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;  # Replace with your domain
    
    # Redirect HTTP to HTTPS (will be enabled after SSL setup)
    # return 301 https://$server_name$request_uri;
    
    # Temporary direct serving for initial setup
    include /etc/nginx/sites-available/climate-api-common;
}

# HTTPS configuration (uncomment after SSL setup)
# server {
#     listen 443 ssl http2;
#     server_name your-domain.com www.your-domain.com;
#     
#     # SSL Configuration (Let's Encrypt)
#     ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
#     ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
#     
#     include /etc/nginx/sites-available/climate-api-common;
# }
EOF

# Create common configuration
sudo tee /etc/nginx/sites-available/climate-api-common << 'EOF'
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # CORS headers for API
    add_header Access-Control-Allow-Origin "*";
    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
    add_header Access-Control-Allow-Headers "Content-Type, Authorization";
    
    # Static files (GeoJSON) - Nginx serves directly for performance
    location /static/ {
        alias /opt/climate-api/app/static/;
        expires 1h;  # Cache for 1 hour
        add_header Cache-Control "public, immutable";
        
        # Security for static files
        location ~* \.(geojson)$ {
            add_header Content-Type application/json;
            add_header Access-Control-Allow-Origin "*";
        }
        
        # Prevent access to sensitive files
        location ~ /\. {
            deny all;
        }
    }
    
    # API endpoints - Proxy to FastAPI
    location / {
        proxy_pass http://127.0.0.1:8099;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Extended timeouts for large dataset queries
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8099/health;
        access_log off;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=5r/s;
    limit_req zone=api burst=10 nodelay;
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/climate-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t
```

---

## Phase 5: SSL and Security Setup

### 5.1 Setup Let's Encrypt SSL
```bash
# Install SSL certificate (replace with your domain and email)
DOMAIN="your-domain.com"
EMAIL="your-email@domain.com"

# Get SSL certificate
sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --email $EMAIL --agree-tos --non-interactive

# Update Nginx configuration to enable HTTPS
sudo sed -i 's/# return 301/return 301/' /etc/nginx/sites-available/climate-api
sudo sed -i 's/# server {/server {/' /etc/nginx/sites-available/climate-api
sudo sed -i 's/#     /    /' /etc/nginx/sites-available/climate-api
sudo sed -i 's/# }/}/' /etc/nginx/sites-available/climate-api

# Test and reload Nginx
sudo nginx -t && sudo systemctl reload nginx
```

### 5.2 Setup Firewall
```bash
# Configure UFW firewall
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force reload

# Check firewall status
sudo ufw status
```

---

## Phase 6: Maintenance and Monitoring Setup

### 6.1 Log Rotation
```bash
# Setup log rotation
sudo tee /etc/logrotate.d/climate-api << 'EOF'
/var/log/climate-api/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 climate-api climate-api
}
EOF
```

### 6.2 Cleanup Jobs
```bash
# Setup automatic cleanup of old GeoJSON files
sudo -u climate-api crontab << 'EOF'
# Clean up old GeoJSON files daily at 2 AM
0 2 * * * find /opt/climate-api/app/static/maps -name "*.geojson" -mtime +7 -delete

# Clean up logs weekly
0 3 * * 0 find /opt/climate-api/logs -name "*.log" -mtime +30 -delete
EOF
```

### 6.3 Monitoring Script
```bash
# Create monitoring script
sudo tee /opt/climate-api/monitor.sh << 'EOF'
#!/bin/bash
echo "=== Climate API Status ==="
echo "Date: $(date)"
echo
echo "Service Status:"
sudo systemctl status climate-api --no-pager
echo
echo "Disk Usage:"
df -h /opt/climate-api
echo
echo "Memory Usage:"
free -h
echo
echo "Recent Logs:"
sudo journalctl -u climate-api -n 10 --no-pager
EOF

sudo chmod +x /opt/climate-api/monitor.sh
```

---

## Phase 7: Start Services and Test

### 7.1 Start All Services
```bash
# Start the application
sudo systemctl start climate-api
sudo systemctl start nginx

# Check service status
sudo systemctl status climate-api
sudo systemctl status nginx

# Check application logs
sudo journalctl -u climate-api -f
```

### 7.2 Test Deployment
```bash
# Test health endpoint
curl http://localhost:8099/health

# Test API endpoint
curl -X POST http://localhost:8099/query \
    -H "Content-Type: application/json" \
    -d '{"query": "climate change", "include_thinking": false}'

# Test static file serving
curl -I http://localhost/static/

# If using domain with SSL:
curl https://your-domain.com/health
curl https://your-domain.com/example-response
```

---

## Phase 8: Performance Verification

### 8.1 Test Large Dataset Queries
```bash
# Test query that uses large knowledge base
curl -X POST https://your-domain.com/query \
    -H "Content-Type: application/json" \
    -d '{"query": "renewable energy policy", "include_thinking": false}' \
    -w "Time: %{time_total}s\n"

# Should return in 15-20 seconds
```

### 8.2 Monitor Resources
```bash
# Check memory usage during heavy queries
htop

# Monitor disk space
df -h

# Check service logs
sudo journalctl -u climate-api -f
```

---

## Troubleshooting

### Common Issues:

**Service won't start:**
```bash
sudo journalctl -u climate-api -n 50
# Check for missing dependencies or permission issues
```

**Large dataset queries failing:**
```bash
# Check if Polars is installed
/opt/climate-api/venv/bin/python -c "import polars; print('Polars OK')"

# Check dataset file exists and is readable
ls -la /opt/climate-api/app/data/knowledge_base.parquet
```

**Static files not serving:**
```bash
# Check Nginx configuration
sudo nginx -t

# Check file permissions
ls -la /opt/climate-api/app/static/maps/
```

### Performance Tuning:

**If queries are too slow:**
- Reduce `CHUNK_SIZE` in environment
- Increase `POLARS_MAX_THREADS`
- Add SSD storage (GP3 with higher IOPS)

**If memory usage is too high:**
- Reduce `MAX_CACHE_SIZE`
- Decrease worker count in systemd service

---

## Deployment Summary

**Final Configuration:**
- ✅ **EC2 t3.large** with 50GB storage
- ✅ **Multi-dataset API** with 6 MCP servers
- ✅ **Large dataset support** (1.77GB, 34M rows) via chunked processing
- ✅ **Static GeoJSON serving** for maps
- ✅ **SSL/HTTPS** with Let's Encrypt
- ✅ **15-20 second** query response times
- ✅ **Automated maintenance** and monitoring

**Monthly Cost:** ~$65 (EC2 + storage)

**Your API endpoints:**
- `https://your-domain.com/health` - Health check
- `https://your-domain.com/query` - Main API
- `https://your-domain.com/example-response` - Example response
- `https://your-domain.com/static/maps/` - GeoJSON files

🚀 **Deployment Complete!**