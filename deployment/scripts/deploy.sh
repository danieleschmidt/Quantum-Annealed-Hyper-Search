#!/bin/bash
# Production Deployment Script for Quantum Hyper Search

set -e

echo "🚀 Starting production deployment..."

# Configuration
IMAGE_NAME="quantum-hyper-search"
IMAGE_TAG="production"
CONTAINER_NAME="quantum-optimizer"
HEALTH_CHECK_URL="http://localhost:8000/health"

# Build production image
echo "📦 Building production image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Stop existing container
echo "🛑 Stopping existing container..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Start new container
echo "🚀 Starting new container..."
docker run -d \
  --name ${CONTAINER_NAME} \
  --restart unless-stopped \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e QUANTUM_BACKEND=simulated \
  -e MONITORING_ENABLED=true \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  ${IMAGE_NAME}:${IMAGE_TAG}

# Wait for health check
echo "🏥 Waiting for health check..."
for i in {1..30}; do
  if curl -f ${HEALTH_CHECK_URL} > /dev/null 2>&1; then
    echo "✅ Health check passed"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "❌ Health check failed"
    exit 1
  fi
  echo "Attempt $i/30..."
  sleep 10
done

echo "🎉 Deployment completed successfully!"
echo "🌐 Application is running at: http://localhost:8000"
