#!/bin/bash
# Rollback Script for Quantum Hyper Search

set -e

echo "🔄 Starting rollback..."

# Configuration
CONTAINER_NAME="quantum-optimizer"
BACKUP_IMAGE="quantum-hyper-search:backup"

# Stop current container
echo "🛑 Stopping current container..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Start backup container
echo "🚀 Starting backup container..."
docker run -d \
  --name ${CONTAINER_NAME} \
  --restart unless-stopped \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e QUANTUM_BACKEND=simulated \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  ${BACKUP_IMAGE}

echo "✅ Rollback completed successfully!"
