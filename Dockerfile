# Multi-stage build for TDE API v2.0.0
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (AWS CLI for S3 data sync)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create directories for data and generated assets
RUN mkdir -p data static/kg static/maps static/cache logs

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8098}/health || exit 1

# Expose API port
EXPOSE 8098

# Use custom entrypoint that handles S3 data download
ENTRYPOINT ["./docker-entrypoint.sh"]
