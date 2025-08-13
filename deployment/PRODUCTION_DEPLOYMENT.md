# Production Deployment Guide

## Overview

This directory contains production-ready deployment configurations for the Quantum Hyperparameter Search system.

## Quick Start

### Docker
```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose logs -f quantum-optimizer
```

### Kubernetes
```bash
# Create namespace
kubectl create namespace quantum-optimization

# Apply manifests
kubectl apply -f kubernetes/

# Check status
kubectl get pods -n quantum-optimization
```

### Helm
```bash
# Install with Helm
helm install quantum-optimizer ./helm/quantum-hyper-search

# Upgrade
helm upgrade quantum-optimizer ./helm/quantum-hyper-search
```

## Configuration

### Environment Variables
- `ENVIRONMENT`: Deployment environment (production, staging, development)
- `QUANTUM_BACKEND`: Backend type (production, simple, simulator)
- `MONITORING_ENABLED`: Enable monitoring (true/false)
- `SECURITY_ENABLED`: Enable security features (true/false)

### Resource Requirements
- **CPU**: 500m - 2000m
- **Memory**: 1Gi - 4Gi
- **Replicas**: 1 - 10 (auto-scaling)

## Monitoring

The deployment includes:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Alerts**: Performance and health monitoring

Access Grafana at: http://localhost:3000 (admin/quantum123)

## Health Checks

- **Liveness**: `/health` endpoint
- **Readiness**: `/ready` endpoint
- **Metrics**: `/metrics` endpoint

## Scaling

Automatic scaling based on CPU utilization (70%).

Manual scaling:
```bash
kubectl scale deployment quantum-hyper-search --replicas=5
```

## Security

- Non-root container execution
- Resource limits enforced
- Network policies (when enabled)
- TLS encryption (when configured)

## Troubleshooting

### Common Issues
1. **Pod not starting**: Check resource limits and node capacity
2. **High memory usage**: Increase memory limits or reduce batch sizes
3. **Slow performance**: Check CPU allocation and scaling configuration

### Debug Commands
```bash
# Check pod status
kubectl describe pod <pod-name> -n quantum-optimization

# View logs
kubectl logs -f <pod-name> -n quantum-optimization

# Execute into container
kubectl exec -it <pod-name> -n quantum-optimization -- /bin/bash
```

## Performance Tuning

### Optimization Settings
- Batch size: Adjust based on available memory
- Worker count: Set to CPU cores * 2
- Cache size: Increase for better performance
- Quantum reads: Balance accuracy vs speed

### Monitoring Metrics
- `quantum_optimization_duration_seconds`: Optimization execution time
- `quantum_evaluations_total`: Total parameter evaluations
- `quantum_cache_hit_ratio`: Cache effectiveness
- `quantum_errors_total`: Error count

## Production Checklist

- [ ] Resource limits configured
- [ ] Health checks enabled
- [ ] Monitoring deployed
- [ ] Security policies applied
- [ ] Backup strategy implemented
- [ ] Disaster recovery tested
- [ ] Performance benchmarks validated
- [ ] Documentation updated

## Support

For production support:
- Email: support@terragonlabs.com
- Documentation: https://docs.terragonlabs.com
- Issues: https://github.com/terragon-labs/quantum-hyper-search/issues
