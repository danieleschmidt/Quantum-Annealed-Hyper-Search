# Production Deployment Guide

This guide covers deploying Quantum-Annealed Hyperparameter Search in production environments.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- D-Wave API token (for quantum hardware)
- Minimum 4GB RAM, 2 CPU cores

### Installation

```bash
# Install from PyPI (when available)
pip install quantum-annealed-hyper-search[all]

# Or install from source
git clone https://github.com/danieleschmidt/quantum-annealed-hyper-search.git
cd quantum-annealed-hyper-search
pip install -e ".[all]"
```

### Configuration

```python
from quantum_hyper_search import QuantumHyperSearch
from quantum_hyper_search.optimization.caching import configure_global_cache

# Configure caching for production
configure_global_cache(
    cache_dir='/var/lib/quantum-cache',
    max_cache_size_mb=1000,
    ttl_hours=24
)

# Initialize with production settings
qhs = QuantumHyperSearch(
    backend='dwave',  # or 'simulator' for testing
    token=os.getenv('DWAVE_API_TOKEN'),
    verbose=True,
    log_file='/var/log/quantum-optimization.log'
)
```

## 🐳 Docker Deployment

### Basic Docker

```bash
# Build image
docker build -t quantum-hyper-search .

# Run container
docker run -d \
  --name quantum-search \
  -e DWAVE_API_TOKEN=your_token_here \
  -v /path/to/data:/app/data \
  -v /path/to/cache:/app/.quantum_cache \
  quantum-hyper-search
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f quantum-search

# Scale workers
docker-compose up -d --scale quantum-search=3

# Health check
docker-compose exec quantum-search python -m quantum_hyper_search.monitoring.health_check
```

## ☸️ Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# quantum-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quantum-search

---
# quantum-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-config
  namespace: quantum-search
data:
  cache_dir: "/var/lib/quantum-cache"
  max_cache_size_mb: "2000"
  log_level: "INFO"
```

### Secret for D-Wave Token

```yaml
# quantum-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: quantum-secrets
  namespace: quantum-search
type: Opaque
data:
  dwave-token: <base64-encoded-token>
```

### Deployment

```yaml
# quantum-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-search
  namespace: quantum-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-search
  template:
    metadata:
      labels:
        app: quantum-search
    spec:
      containers:
      - name: quantum-search
        image: quantum-hyper-search:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: DWAVE_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: quantum-secrets
              key: dwave-token
        - name: CACHE_DIR
          valueFrom:
            configMapKeyRef:
              name: quantum-config
              key: cache_dir
        volumeMounts:
        - name: cache-volume
          mountPath: /var/lib/quantum-cache
        - name: log-volume
          mountPath: /var/log
        livenessProbe:
          exec:
            command:
            - python
            - -m
            - quantum_hyper_search.monitoring.health_check
            - --quiet
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -m
            - quantum_hyper_search.monitoring.health_check
            - --quiet
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: quantum-cache-pvc
      - name: log-volume
        persistentVolumeClaim:
          claimName: quantum-logs-pvc
```

### Service

```yaml
# quantum-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: quantum-search-service
  namespace: quantum-search
spec:
  selector:
    app: quantum-search
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

### Deploy to Kubernetes

```bash
kubectl apply -f quantum-namespace.yaml
kubectl apply -f quantum-config.yaml
kubectl apply -f quantum-secret.yaml
kubectl apply -f quantum-deployment.yaml
kubectl apply -f quantum-service.yaml

# Check status
kubectl get pods -n quantum-search
kubectl logs -f deployment/quantum-search -n quantum-search
```

## 🏗️ Infrastructure as Code

### Terraform (AWS)

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# ECS Cluster
resource "aws_ecs_cluster" "quantum_cluster" {
  name = "quantum-search-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Task Definition
resource "aws_ecs_task_definition" "quantum_task" {
  family                   = "quantum-search"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048
  memory                   = 4096
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name  = "quantum-search"
      image = "quantum-hyper-search:latest"
      
      environment = [
        {
          name  = "CACHE_DIR"
          value = "/var/lib/quantum-cache"
        }
      ]
      
      secrets = [
        {
          name      = "DWAVE_API_TOKEN"
          valueFrom = aws_secretsmanager_secret.dwave_token.arn
        }
      ]
      
      mountPoints = [
        {
          sourceVolume  = "cache"
          containerPath = "/var/lib/quantum-cache"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.quantum_logs.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  volume {
    name = "cache"
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.quantum_cache.id
    }
  }
}

# ECS Service
resource "aws_ecs_service" "quantum_service" {
  name            = "quantum-search-service"
  cluster         = aws_ecs_cluster.quantum_cluster.id
  task_definition = aws_ecs_task_definition.quantum_task.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.quantum_sg.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.quantum_tg.arn
    container_name   = "quantum-search"
    container_port   = 8080
  }
}
```

## 📊 Monitoring and Observability

### Health Checks

```bash
# Basic health check
python -m quantum_hyper_search.monitoring.health_check

# With D-Wave connectivity check
python -m quantum_hyper_search.monitoring.health_check --token YOUR_DWAVE_TOKEN

# Quiet mode for automation
python -m quantum_hyper_search.monitoring.health_check --quiet
```

### Metrics Collection

```python
from quantum_hyper_search.monitoring import PerformanceMonitor
from quantum_hyper_search.utils.metrics import QuantumMetrics

# Initialize monitoring
monitor = PerformanceMonitor()
metrics = QuantumMetrics()

# Use in optimization
with monitor.time_block('optimization'):
    best_params, history = qhs.optimize(...)

# Get performance summary
summary = monitor.get_summary()
quantum_stats = metrics.get_summary_statistics()
```

### Prometheus Integration

```python
# Optional: Export metrics to Prometheus
from prometheus_client import start_http_server, Counter, Histogram

OPTIMIZATION_COUNTER = Counter('quantum_optimizations_total', 'Total optimizations')
OPTIMIZATION_DURATION = Histogram('quantum_optimization_duration_seconds', 'Optimization duration')

# Start metrics server
start_http_server(8000)

# In your optimization code
with OPTIMIZATION_DURATION.time():
    result = qhs.optimize(...)
    OPTIMIZATION_COUNTER.inc()
```

## 🔧 Production Configuration

### Environment Variables

```bash
# D-Wave Configuration
export DWAVE_API_TOKEN="your_token_here"
export DWAVE_ENDPOINT="https://cloud.dwavesys.com/sapi/"

# Cache Configuration
export QUANTUM_CACHE_DIR="/var/lib/quantum-cache"
export QUANTUM_CACHE_SIZE_MB="2000"
export QUANTUM_CACHE_TTL_HOURS="24"

# Logging Configuration
export QUANTUM_LOG_LEVEL="INFO"
export QUANTUM_LOG_FILE="/var/log/quantum-optimization.log"

# Performance Configuration
export QUANTUM_PARALLEL_JOBS="8"
export QUANTUM_DEFAULT_READS="1000"
```

### Configuration File

```yaml
# quantum-config.yaml
quantum_search:
  backend:
    default: "dwave"
    fallback: "simulator"
    timeout: 300
    
  cache:
    directory: "/var/lib/quantum-cache"
    max_size_mb: 2000
    ttl_hours: 24
    cleanup_interval: 3600
    
  optimization:
    default_iterations: 50
    default_quantum_reads: 1000
    parallel_jobs: 8
    early_stopping_patience: 10
    
  logging:
    level: "INFO"
    file: "/var/log/quantum-optimization.log"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
  monitoring:
    health_check_interval: 30
    metrics_port: 8000
    enable_prometheus: true
```

## 🔒 Security Considerations

### API Token Security

```bash
# Use secrets management
kubectl create secret generic dwave-secret --from-literal=token=YOUR_TOKEN

# Or use cloud provider secrets
aws secretsmanager create-secret --name quantum/dwave-token --secret-string YOUR_TOKEN
```

### Network Security

```yaml
# Kubernetes Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: quantum-search-netpol
spec:
  podSelector:
    matchLabels:
      app: quantum-search
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS for D-Wave API
```

## 🚨 Troubleshooting

### Common Issues

1. **D-Wave Connection Failures**
   ```bash
   # Check connectivity
   python -c "from dwave.cloud import Client; Client.from_config().ping()"
   
   # Verify token
   dwave ping --profile default
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   python -c "
   from quantum_hyper_search.monitoring.health_check import run_health_check
   print(run_health_check()['checks']['memory'])
   "
   ```

3. **Cache Problems**
   ```bash
   # Clear cache
   rm -rf /var/lib/quantum-cache/*
   
   # Check cache statistics
   python -c "
   from quantum_hyper_search.optimization.caching import get_global_cache
   print(get_global_cache().get_cache_statistics())
   "
   ```

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose mode
qhs = QuantumHyperSearch(verbose=True, log_file='debug.log')
```

## 📈 Performance Tuning

### Optimization Settings

```python
# For large problems
qhs = QuantumHyperSearch(
    backend='dwave',
    quantum_reads=5000,  # More reads for better solutions
    timeout=1800,        # 30 minute timeout
    early_stopping_patience=20
)

# For speed
qhs = QuantumHyperSearch(
    backend='simulator',
    quantum_reads=100,   # Fewer reads for speed
    timeout=300,         # 5 minute timeout
    early_stopping_patience=5
)
```

### Caching Strategy

```python
# Configure aggressive caching
configure_global_cache(
    max_cache_size_mb=5000,  # 5GB cache
    ttl_hours=72,            # 3 day TTL
    enable_disk_cache=True   # Persistent cache
)
```

## 🔄 Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      run: |
        docker build -t quantum-hyper-search:${{ github.ref_name }} .
        docker push quantum-hyper-search:${{ github.ref_name }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/quantum-search \
          quantum-search=quantum-hyper-search:${{ github.ref_name }}
```

## 📞 Support

For production support:
- Documentation: https://quantum-hyper-search.readthedocs.io
- Issues: https://github.com/danieleschmidt/quantum-annealed-hyper-search/issues
- Email: daniel@terragonlabs.com