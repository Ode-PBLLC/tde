#!/bin/bash
set -e

echo "TDE API v2.0.0 Starting..."

# Check if data directory exists and is populated
if [ ! -f "/app/data/solar_facilities.db" ]; then
    echo "Data directory is empty. Checking for S3 data..."

    # If S3_DATA_BUCKET is set, download data from S3
    if [ -n "$S3_DATA_BUCKET" ]; then
        echo "Downloading datasets from S3 bucket: $S3_DATA_BUCKET"

        # Check if AWS CLI is available
        if ! command -v aws &> /dev/null; then
            echo "ERROR: aws CLI not found. Install it or mount data volume."
            exit 1
        fi

        # Download data from S3
        echo "Syncing data from s3://$S3_DATA_BUCKET/..."
        aws s3 sync "s3://$S3_DATA_BUCKET/" /app/data/ --no-progress

        if [ -f "/app/data/solar_facilities.db" ]; then
            echo "✓ Data downloaded successfully from S3"
        else
            echo "ERROR: Data download failed or incomplete"
            exit 1
        fi
    else
        echo "WARNING: No data found and S3_DATA_BUCKET not set"
        echo "Server will start but may have limited functionality"
        echo "Set S3_DATA_BUCKET env var or mount data volume to /app/data"
    fi
else
    echo "✓ Data directory found at /app/data"
fi

# Verify required environment variables
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY environment variable is required"
    exit 1
fi

echo "✓ Environment configured"
echo "Starting FastAPI server on port ${API_PORT:-8098}..."

# Start the application
exec python api_server.py
