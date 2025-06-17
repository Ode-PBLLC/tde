#!/bin/bash
# Complete deployment script for Climate Policy Radar API

set -e

DOMAIN="your-domain.com"  # Replace with your domain
EMAIL="your-email@domain.com"  # Replace with your email

echo "🚀 Starting deployment..."

# 1. Install systemd service
echo "📦 Installing systemd service..."
sudo cp climate-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable climate-api

# 2. Setup environment variables
echo "🔐 Setting up environment..."
sudo -u climate-api cp .env.production /opt/climate-api/app/.env

# 3. Install nginx configuration
echo "🌐 Configuring Nginx..."
sudo cp nginx.conf /etc/nginx/sites-available/climate-api
sudo ln -sf /etc/nginx/sites-available/climate-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t

# 4. Setup SSL with Let's Encrypt
echo "🔒 Setting up SSL..."
sudo certbot --nginx -d $DOMAIN --email $EMAIL --agree-tos --non-interactive

# 5. Setup file cleanup cron job
echo "🧹 Setting up file cleanup..."
sudo -u climate-api crontab << EOF
# Clean up old GeoJSON files daily at 2 AM
0 2 * * * find /opt/climate-api/app/static/maps -name "*.geojson" -mtime +7 -delete
EOF

# 6. Setup log rotation
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/climate-api << EOF
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

# 7. Start services
echo "🚀 Starting services..."
sudo systemctl start climate-api
sudo systemctl reload nginx

# 8. Setup firewall
echo "🔥 Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force reload

echo "✅ Deployment complete!"
echo "🌐 Your API is available at: https://$DOMAIN"
echo "📊 Health check: https://$DOMAIN/health"
echo "📋 Example query: https://$DOMAIN/example-response"

# Display service status
echo "📊 Service Status:"
sudo systemctl status climate-api --no-pager