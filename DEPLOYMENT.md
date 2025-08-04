# Deployment Guide

## Quantum Annealed Hyperparameter Search - Production Deployment

This guide covers deploying the Quantum Annealed Hyperparameter Search system in production environments.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, or equivalent)
- **Python**: 3.8+
- **RAM**: 4 GB
- **CPU**: 4 cores
- **Storage**: 10 GB free space

#### Recommended Requirements
- **OS**: Linux (Ubuntu 22.04 LTS)
- **Python**: 3.9 or 3.10
- **RAM**: 16 GB
- **CPU**: 8+ cores
- **Storage**: 50 GB SSD
- **Network**: High-speed internet for D-Wave access

#### For Large-Scale Deployments
- **RAM**: 32+ GB
- **CPU**: 16+ cores
- **Storage**: 100+ GB NVMe SSD
- **Network**: Dedicated quantum cloud connectivity

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Git 2.30+

## 🚀 Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Quick Start
```bash
# Clone repository
git clone https://github.com/danieleschmidt/quantum-annealed-hyper-search.git
cd quantum-annealed-hyper-search

# Build and run with Docker Compose
docker-compose up -d

# Verify deployment
docker-compose logs quantum-hyper-search
```

#### Production Docker Deployment
```bash
# Build production image
docker build --target production -t quantum-hyper-search:latest .

# Run with production settings
docker run -d \
  --name quantum-hyper-search-prod \
  --restart unless-stopped \
  -v /opt/quantum-cache:/app/cache \
  -v /var/log/quantum:/app/logs \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  quantum-hyper-search:latest
```

#### D-Wave Enabled Deployment
```bash
# Build D-Wave enabled image
docker build --target dwave -t quantum-hyper-search:dwave .

# Run with D-Wave credentials
docker run -d \
  --name quantum-hyper-search-dwave \
  --restart unless-stopped \
  -v /opt/quantum-cache:/app/cache \
  -e DWAVE_API_TOKEN=your_dwave_token_here \
  -e DWAVE_SOLVER=Advantage_system4.1 \
  quantum-hyper-search:dwave
```

### Option 2: Native Installation

#### System Package Installation
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev \
  build-essential git curl

# Install from PyPI
pip install quantum-annealed-hyper-search[all]

# Or install from source
git clone https://github.com/danieleschmidt/quantum-annealed-hyper-search.git
cd quantum-annealed-hyper-search
pip install -e ".[all]"
```

#### Virtual Environment Setup
```bash
# Create virtual environment
python3.9 -m venv quantum-env
source quantum-env/bin/activate

# Install package
pip install --upgrade pip
pip install quantum-annealed-hyper-search[simulators,optimizers]

# For D-Wave support
pip install quantum-annealed-hyper-search[dwave]
```

### Option 3: Kubernetes Deployment

#### Basic Kubernetes Deployment
```yaml
# quantum-hyper-search-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-hyper-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-hyper-search
  template:
    metadata:
      labels:
        app: quantum-hyper-search
    spec:
      containers:
      - name: quantum-hyper-search
        image: danieleschmidt/quantum-hyper-search:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        volumeMounts:
        - name: cache-volume
          mountPath: /app/cache
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: quantum-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: quantum-hyper-search-service
spec:
  selector:
    app: quantum-hyper-search
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy with:
```bash
kubectl apply -f quantum-hyper-search-deployment.yaml
```

## ⚙️ Configuration

### Environment Variables

#### Core Configuration
```bash
# Environment
ENVIRONMENT=production              # production, development, testing
LOG_LEVEL=INFO                     # DEBUG, INFO, WARNING, ERROR
QUANTUM_BACKEND=simulator          # simulator, dwave
CACHE_ENABLED=true                 # Enable result caching
CACHE_SIZE=10000                   # Maximum cache entries
CACHE_TTL=3600                     # Cache TTL in seconds

# Performance
PARALLEL_ENABLED=true              # Enable parallel processing
MAX_WORKERS=8                      # Maximum parallel workers
AUTO_SCALING_ENABLED=true          # Enable auto-scaling
MONITORING_ENABLED=true            # Enable monitoring

# Security
SECURITY_ENABLED=true              # Enable security checks
INPUT_VALIDATION=strict            # strict, normal, permissive
```

#### D-Wave Configuration
```bash
# D-Wave Quantum Computer
DWAVE_API_TOKEN=your_token_here
DWAVE_SOLVER=Advantage_system4.1
DWAVE_ENDPOINT=https://cloud.dwavesys.com/sapi
DWAVE_REGION=na-west-1
```

#### Advanced Configuration
```bash
# Database (for result persistence)
DATABASE_URL=postgresql://user:pass@localhost:5432/quantum
REDIS_URL=redis://localhost:6379/0

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true

# API Configuration
API_ENABLED=true
API_PORT=8000
API_WORKERS=4
```

### Configuration Files

#### Production Configuration (`config/production.yaml`)
```yaml
quantum_hyper_search:
  backend: simulator
  logging:
    level: INFO
    structured: true
    file: /app/logs/quantum.log
  
  cache:
    enabled: true
    size: 50000
    ttl: 7200
    persistence: true
    directory: /app/cache
  
  performance:
    parallel: true
    max_workers: null  # Auto-detect
    auto_scaling: true
  
  security:
    enabled: true
    validation: strict
    
  monitoring:
    enabled: true
    metrics: true
    health_checks: true
```

#### D-Wave Configuration (`config/dwave.yaml`)
```yaml
dwave:
  api_token: ${DWAVE_API_TOKEN}
  solver: ${DWAVE_SOLVER:-Advantage_system4.1}
  endpoint: ${DWAVE_ENDPOINT:-https://cloud.dwavesys.com/sapi}
  
  optimization:
    num_reads: 1000
    annealing_time: 20
    chain_strength: null  # Auto-select
    
  advanced:
    embedding_method: minorminer
    topology: pegasus
    post_processing: optimization
```

## 🔧 Operational Procedures

### Health Monitoring

#### Health Check Endpoints
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status

# Metrics
curl http://localhost:8000/metrics
```

#### Monitoring Commands
```bash
# Check system status
docker exec quantum-hyper-search python -c "
from quantum_hyper_search.utils.monitoring import HealthChecker
hc = HealthChecker()
hc.setup_default_checks()
print(hc.run_checks())
"

# Monitor performance
docker stats quantum-hyper-search

# View logs
docker logs -f quantum-hyper-search
```

### Backup & Recovery

#### Data Backup
```bash
# Backup cache data
tar -czf quantum-cache-backup-$(date +%Y%m%d).tar.gz /opt/quantum-cache/

# Backup configuration
cp -r config/ config-backup-$(date +%Y%m%d)/

# Backup logs
tar -czf logs-backup-$(date +%Y%m%d).tar.gz /var/log/quantum/
```

#### Recovery Procedures
```bash
# Restore cache
tar -xzf quantum-cache-backup-YYYYMMDD.tar.gz -C /

# Restart services
docker-compose restart quantum-hyper-search

# Verify recovery
docker-compose logs quantum-hyper-search
```

### Scaling Operations

#### Horizontal Scaling
```bash
# Scale up replicas
docker-compose up -d --scale quantum-hyper-search=5

# Kubernetes scaling
kubectl scale deployment quantum-hyper-search --replicas=10
```

#### Vertical Scaling
```bash
# Update resource limits
docker update --memory=16g --cpus=8 quantum-hyper-search

# Restart with new limits
docker-compose restart quantum-hyper-search
```

## 🛡️ Security

### Security Checklist

#### Pre-deployment Security
- [ ] Enable security features in configuration
- [ ] Set strong authentication for D-Wave API
- [ ] Configure input validation and sanitization
- [ ] Set up secure networking (VPC, firewalls)
- [ ] Enable logging and monitoring
- [ ] Regular security updates scheduled

#### Runtime Security
- [ ] Monitor for unusual resource usage
- [ ] Check logs for security warnings
- [ ] Validate input parameters
- [ ] Monitor network connections
- [ ] Regular security scans

### Access Control
```bash
# Create service account (Kubernetes)
kubectl create serviceaccount quantum-service-account

# Apply RBAC
kubectl apply -f rbac.yaml

# Set resource quotas
kubectl apply -f resource-quota.yaml
```

## 📊 Performance Tuning

### Performance Optimization

#### Memory Optimization
```yaml
# Optimize cache settings
cache:
  size: 100000      # Increase for more memory
  ttl: 3600         # Reduce for less memory usage
  
# Optimize parallel processing
parallel:
  max_workers: 16   # Adjust based on CPU cores
  chunk_size: 4     # Optimize for workload
```

#### CPU Optimization
```yaml
# Configure threading
performance:
  parallel_enabled: true
  max_workers: null      # Auto-detect optimal
  thread_pool_size: 32   # For I/O bound tasks
```

#### Network Optimization
```yaml
# D-Wave connection optimization
dwave:
  connection_pool_size: 10
  timeout: 30
  retry_attempts: 3
```

### Performance Monitoring

#### Key Metrics to Monitor
- **Optimization Time**: Target < 60s for small problems
- **Memory Usage**: Keep < 80% of available
- **CPU Usage**: Target 70-80% utilization
- **Cache Hit Rate**: Target > 30%
- **Error Rate**: Target < 1%

#### Alerting Thresholds
```yaml
alerts:
  optimization_time: 300s    # Alert if > 5 minutes
  memory_usage: 85%          # Alert if > 85%
  cpu_usage: 90%             # Alert if > 90%
  error_rate: 5%             # Alert if > 5%
  cache_hit_rate: 10%        # Alert if < 10%
```

## 🚨 Troubleshooting

### Common Issues

#### Issue: Optimization Taking Too Long
```bash
# Check system resources
htop
df -h

# Check configuration
cat config/production.yaml | grep -A 10 performance

# Reduce problem size
# Increase cache size
# Enable parallel processing
```

#### Issue: Memory Usage Too High
```bash
# Check memory usage
docker stats

# Reduce cache size
# Optimize parallel workers
# Check for memory leaks in logs
```

#### Issue: D-Wave Connection Problems
```bash
# Test D-Wave connectivity
python -c "
from dwave.system import DWaveSampler
sampler = DWaveSampler()
print(sampler.properties)
"

# Check API token
echo $DWAVE_API_TOKEN

# Verify solver availability
dwave solvers --list
```

### Log Analysis

#### Important Log Patterns
```bash
# Error patterns
grep -i "error\|exception\|failed" /var/log/quantum/quantum.log

# Performance patterns  
grep -i "slow\|timeout\|memory" /var/log/quantum/quantum.log

# Security patterns
grep -i "security\|validation\|sanitiz" /var/log/quantum/quantum.log
```

## 📞 Support

### Getting Help
- **Documentation**: https://quantum-hyper-search.readthedocs.io
- **Issues**: https://github.com/danieleschmidt/quantum-annealed-hyper-search/issues
- **Discussions**: https://github.com/danieleschmidt/quantum-annealed-hyper-search/discussions

### Emergency Contacts
- **Critical Issues**: Create GitHub issue with "urgent" label
- **Security Issues**: Email security@quantum-hyper-search.io
- **D-Wave Support**: https://support.dwavesys.com

### Maintenance Windows
- **Planned Maintenance**: First Sunday of each month, 02:00-06:00 UTC
- **Emergency Maintenance**: As needed with 2-hour notice
- **Updates**: Released monthly with 1-week advance notice