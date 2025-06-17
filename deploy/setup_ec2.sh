#!/bin/bash
# EC2 Deployment Script for Climate Policy Radar API

set -e

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx git

# Create app user
sudo useradd --system --shell /bin/bash --home /opt/climate-api climate-api
sudo mkdir -p /opt/climate-api
sudo chown climate-api:climate-api /opt/climate-api

# Clone repository (replace with your repo)
sudo -u climate-api git clone https://github.com/your-username/tde.git /opt/climate-api/app
cd /opt/climate-api/app

# Create Python virtual environment
sudo -u climate-api python3 -m venv /opt/climate-api/venv
sudo -u climate-api /opt/climate-api/venv/bin/pip install --upgrade pip

# Install Python dependencies
sudo -u climate-api /opt/climate-api/venv/bin/pip install -r requirements.txt

# Create static directories with correct permissions
sudo -u climate-api mkdir -p /opt/climate-api/app/static/maps
sudo chmod 755 /opt/climate-api/app/static
sudo chmod 755 /opt/climate-api/app/static/maps

# Create logs directory
sudo mkdir -p /var/log/climate-api
sudo chown climate-api:climate-api /var/log/climate-api

echo "✅ Basic setup complete"