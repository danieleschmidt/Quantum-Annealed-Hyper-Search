"""
Production-Ready Deployment Configuration

Enterprise-grade deployment orchestrator for quantum hyperparameter optimization
with Docker, Kubernetes, monitoring, and auto-scaling capabilities.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ProductionDeploymentOrchestrator:
    """Orchestrates production deployment for quantum optimization system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.deployment_dir = Path(__file__).parent
        self.repo_root = self.deployment_dir.parent
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default production configuration."""
        return {
            'app_name': 'quantum-hyper-search',
            'version': '1.0.0',
            'environment': 'production',
            'replicas': 3,
            'resources': {
                'cpu_request': '500m',
                'cpu_limit': '2000m',
                'memory_request': '1Gi',
                'memory_limit': '4Gi'
            },
            'scaling': {
                'min_replicas': 1,
                'max_replicas': 10,
                'target_cpu_utilization': 70
            },
            'monitoring': {
                'enabled': True,
                'prometheus': True,
                'grafana': True,
                'alerts': True
            },
            'security': {
                'enabled': True,
                'tls': True,
                'network_policies': True
            }
        }
    
    def generate_dockerfile(self) -> str:
        """Generate production Dockerfile."""
        dockerfile_content = f"""# Multi-stage production Dockerfile for Quantum Hyperparameter Search
# Built for: {self.config['app_name']} v{self.config['version']}

# Stage 1: Build Dependencies
FROM python:3.12-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE
ARG VCS_REF

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production Runtime
FROM python:3.12-slim as production

# Labels for metadata
LABEL maintainer="Terragon Labs <contact@terragonlabs.com>" \\
      version="{self.config['version']}" \\
      description="Enterprise Quantum Hyperparameter Optimization" \\
      build-date=$BUILD_DATE \\
      vcs-ref=$VCS_REF

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Create app user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=quantum:quantum . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \\
    chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Set environment variables
ENV PYTHONPATH=/app \\
    PYTHONUNBUFFERED=1 \\
    QUANTUM_LOG_LEVEL=INFO \\
    QUANTUM_CACHE_DIR=/app/cache \\
    QUANTUM_DATA_DIR=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD python -c "from quantum_hyper_search import QuantumHyperSearch; print('OK')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "quantum_hyper_search.main"]
"""
        return dockerfile_content
    
    def generate_kubernetes_deployment(self) -> str:
        """Generate Kubernetes deployment YAML."""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.config['app_name'],
                'namespace': 'quantum-optimization',
                'labels': {
                    'app': self.config['app_name'],
                    'version': self.config['version'],
                    'component': 'quantum-optimizer'
                }
            },
            'spec': {
                'replicas': self.config['replicas'],
                'strategy': {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxUnavailable': 1,
                        'maxSurge': 1
                    }
                },
                'selector': {
                    'matchLabels': {
                        'app': self.config['app_name']
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config['app_name'],
                            'version': self.config['version']
                        },
                        'annotations': {
                            'prometheus.io/scrape': 'true',
                            'prometheus.io/port': '8000',
                            'prometheus.io/path': '/metrics'
                        }
                    },
                    'spec': {
                        'serviceAccountName': f'{self.config["app_name"]}-service-account',
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        },
                        'containers': [{
                            'name': self.config['app_name'],
                            'image': f'{self.config["app_name"]}:{self.config["version"]}',
                            'imagePullPolicy': 'Always',
                            'ports': [{
                                'name': 'http',
                                'containerPort': 8000,
                                'protocol': 'TCP'
                            }],
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': self.config['environment']},
                                {'name': 'QUANTUM_BACKEND', 'value': 'production'},
                                {'name': 'MONITORING_ENABLED', 'value': 'true'},
                                {'name': 'SECURITY_ENABLED', 'value': 'true'}
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': self.config['resources']['cpu_request'],
                                    'memory': self.config['resources']['memory_request']
                                },
                                'limits': {
                                    'cpu': self.config['resources']['cpu_limit'],
                                    'memory': self.config['resources']['memory_limit']
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'successThreshold': 1,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5,
                                'timeoutSeconds': 3,
                                'successThreshold': 1,
                                'failureThreshold': 3
                            },
                            'volumeMounts': [
                                {
                                    'name': 'cache-volume',
                                    'mountPath': '/app/cache'
                                },
                                {
                                    'name': 'data-volume',
                                    'mountPath': '/app/data'
                                }
                            ]
                        }],
                        'volumes': [
                            {
                                'name': 'cache-volume',
                                'emptyDir': {'sizeLimit': '1Gi'}
                            },
                            {
                                'name': 'data-volume',
                                'persistentVolumeClaim': {
                                    'claimName': f'{self.config["app_name"]}-data-pvc'
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        return yaml.dump(deployment, default_flow_style=False)
    
    def generate_service(self) -> str:
        """Generate Kubernetes service YAML."""
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{self.config["app_name"]}-service',
                'namespace': 'quantum-optimization',
                'labels': {
                    'app': self.config['app_name'],
                    'component': 'quantum-optimizer'
                }
            },
            'spec': {
                'type': 'ClusterIP',
                'ports': [{
                    'name': 'http',
                    'port': 80,
                    'targetPort': 8000,
                    'protocol': 'TCP'
                }],
                'selector': {
                    'app': self.config['app_name']
                }
            }
        }
        
        return yaml.dump(service, default_flow_style=False)
    
    def generate_hpa(self) -> str:
        """Generate Horizontal Pod Autoscaler YAML."""
        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f'{self.config["app_name"]}-hpa',
                'namespace': 'quantum-optimization'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.config['app_name']
                },
                'minReplicas': self.config['scaling']['min_replicas'],
                'maxReplicas': self.config['scaling']['max_replicas'],
                'metrics': [{
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': self.config['scaling']['target_cpu_utilization']
                        }
                    }
                }],
                'behavior': {
                    'scaleUp': {
                        'stabilizationWindowSeconds': 60,
                        'policies': [{
                            'type': 'Percent',
                            'value': 100,
                            'periodSeconds': 15
                        }]
                    },
                    'scaleDown': {
                        'stabilizationWindowSeconds': 300,
                        'policies': [{
                            'type': 'Percent',
                            'value': 10,
                            'periodSeconds': 60
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(hpa, default_flow_style=False)
    
    def generate_monitoring_config(self) -> str:
        """Generate monitoring configuration."""
        monitoring_config = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f'{self.config["app_name"]}-monitoring',
                'namespace': 'quantum-optimization'
            },
            'data': {
                'prometheus.yml': yaml.dump({
                    'global': {
                        'scrape_interval': '15s',
                        'evaluation_interval': '15s'
                    },
                    'scrape_configs': [{
                        'job_name': 'quantum-optimizer',
                        'kubernetes_sd_configs': [{
                            'role': 'pod'
                        }],
                        'relabel_configs': [{
                            'source_labels': ['__meta_kubernetes_pod_annotation_prometheus_io_scrape'],
                            'action': 'keep',
                            'regex': 'true'
                        }]
                    }]
                }, default_flow_style=False),
                'alerts.yml': yaml.dump({
                    'groups': [{
                        'name': 'quantum-optimizer',
                        'rules': [
                            {
                                'alert': 'QuantumOptimizerDown',
                                'expr': 'up{job="quantum-optimizer"} == 0',
                                'for': '1m',
                                'labels': {'severity': 'critical'},
                                'annotations': {
                                    'summary': 'Quantum optimizer is down',
                                    'description': 'Quantum optimizer has been down for more than 1 minute'
                                }
                            },
                            {
                                'alert': 'HighErrorRate',
                                'expr': 'rate(quantum_errors_total[5m]) > 0.1',
                                'for': '2m',
                                'labels': {'severity': 'warning'},
                                'annotations': {
                                    'summary': 'High error rate detected',
                                    'description': 'Error rate is above 10% for 2 minutes'
                                }
                            }
                        ]
                    }]
                }, default_flow_style=False)
            }
        }
        
        return yaml.dump(monitoring_config, default_flow_style=False)
    
    def generate_docker_compose(self) -> str:
        """Generate Docker Compose for local development."""
        compose_config = {
            'version': '3.8',
            'services': {
                'quantum-optimizer': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile',
                        'target': 'production'
                    },
                    'image': f'{self.config["app_name"]}:{self.config["version"]}',
                    'container_name': 'quantum-optimizer',
                    'restart': 'unless-stopped',
                    'ports': ['8000:8000'],
                    'environment': {
                        'ENVIRONMENT': 'development',
                        'QUANTUM_LOG_LEVEL': 'DEBUG',
                        'MONITORING_ENABLED': 'true'
                    },
                    'volumes': [
                        './data:/app/data',
                        './logs:/app/logs',
                        './cache:/app/cache'
                    ],
                    'healthcheck': {
                        'test': ['CMD', 'python', '-c', 'from quantum_hyper_search import QuantumHyperSearch; print("OK")'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    },
                    'networks': ['quantum-network']
                },
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': 'quantum-prometheus',
                    'restart': 'unless-stopped',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml',
                        'prometheus-data:/prometheus'
                    ],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--web.enable-lifecycle'
                    ],
                    'networks': ['quantum-network']
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': 'quantum-grafana',
                    'restart': 'unless-stopped',
                    'ports': ['3000:3000'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'quantum123'
                    },
                    'volumes': [
                        'grafana-data:/var/lib/grafana',
                        './monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards',
                        './monitoring/grafana/datasources:/etc/grafana/provisioning/datasources'
                    ],
                    'networks': ['quantum-network']
                }
            },
            'networks': {
                'quantum-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'prometheus-data': {},
                'grafana-data': {}
            }
        }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def generate_helm_chart(self) -> Dict[str, str]:
        """Generate Helm chart files."""
        chart_yaml = f"""apiVersion: v2
name: {self.config['app_name']}
description: Enterprise Quantum Hyperparameter Optimization
type: application
version: {self.config['version']}
appVersion: "{self.config['version']}"
keywords:
  - quantum
  - optimization
  - machine-learning
  - hyperparameters
home: https://github.com/terragon-labs/quantum-hyper-search
sources:
  - https://github.com/terragon-labs/quantum-hyper-search
maintainers:
  - name: Terragon Labs
    email: contact@terragonlabs.com
"""
        
        values_yaml = f"""# Default values for {self.config['app_name']}
replicaCount: {self.config['replicas']}

image:
  repository: {self.config['app_name']}
  pullPolicy: Always
  tag: "{self.config['version']}"

serviceAccount:
  create: true
  annotations: {{}}
  name: ""

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: false
  className: ""
  annotations: {{}}
  hosts:
    - host: quantum-optimizer.local
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: {self.config['resources']['cpu_limit']}
    memory: {self.config['resources']['memory_limit']}
  requests:
    cpu: {self.config['resources']['cpu_request']}
    memory: {self.config['resources']['memory_request']}

autoscaling:
  enabled: true
  minReplicas: {self.config['scaling']['min_replicas']}
  maxReplicas: {self.config['scaling']['max_replicas']}
  targetCPUUtilizationPercentage: {self.config['scaling']['target_cpu_utilization']}

monitoring:
  enabled: {str(self.config['monitoring']['enabled']).lower()}
  prometheus: {str(self.config['monitoring']['prometheus']).lower()}
  grafana: {str(self.config['monitoring']['grafana']).lower()}

security:
  enabled: {str(self.config['security']['enabled']).lower()}
  networkPolicies: {str(self.config['security']['network_policies']).lower()}
"""
        
        return {
            'Chart.yaml': chart_yaml,
            'values.yaml': values_yaml
        }
    
    def create_deployment_files(self):
        """Create all deployment files."""
        deployment_dir = self.deployment_dir
        
        # Create directories
        (deployment_dir / 'kubernetes').mkdir(exist_ok=True)
        (deployment_dir / 'helm' / 'quantum-hyper-search').mkdir(parents=True, exist_ok=True)
        (deployment_dir / 'monitoring').mkdir(exist_ok=True)
        
        # Generate and write files
        files_to_create = {
            'Dockerfile': self.generate_dockerfile(),
            'docker-compose.production.yml': self.generate_docker_compose(),
            'kubernetes/deployment.yaml': self.generate_kubernetes_deployment(),
            'kubernetes/service.yaml': self.generate_service(),
            'kubernetes/hpa.yaml': self.generate_hpa(),
            'monitoring/monitoring-config.yaml': self.generate_monitoring_config(),
        }
        
        # Add Helm chart files
        helm_files = self.generate_helm_chart()
        for filename, content in helm_files.items():
            files_to_create[f'helm/quantum-hyper-search/{filename}'] = content
        
        # Write all files
        created_files = []
        for relative_path, content in files_to_create.items():
            file_path = self.deployment_dir.parent / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            created_files.append(str(file_path))
            logger.info(f"Created deployment file: {relative_path}")
        
        return created_files
    
    def generate_deployment_readme(self) -> str:
        """Generate deployment README."""
        readme_content = f"""# Production Deployment Guide

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
- **CPU**: {self.config['resources']['cpu_request']} - {self.config['resources']['cpu_limit']}
- **Memory**: {self.config['resources']['memory_request']} - {self.config['resources']['memory_limit']}
- **Replicas**: {self.config['scaling']['min_replicas']} - {self.config['scaling']['max_replicas']} (auto-scaling)

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

Automatic scaling based on CPU utilization ({self.config['scaling']['target_cpu_utilization']}%).

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
"""
        return readme_content


def main():
    """Main deployment preparation function."""
    print("🚀 PRODUCTION DEPLOYMENT PREPARATION")
    print("=" * 60)
    
    orchestrator = ProductionDeploymentOrchestrator()
    
    try:
        # Create all deployment files
        created_files = orchestrator.create_deployment_files()
        
        # Create deployment README
        readme_content = orchestrator.generate_deployment_readme()
        readme_path = orchestrator.deployment_dir / 'PRODUCTION_DEPLOYMENT.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        created_files.append(str(readme_path))
        
        print(f"✅ Successfully created {len(created_files)} deployment files:")
        for file_path in created_files:
            print(f"   📄 {file_path}")
        
        print(f"\n🎯 DEPLOYMENT READY!")
        print(f"   🐳 Docker: {orchestrator.config['app_name']}:{orchestrator.config['version']}")
        print(f"   ☸️  Kubernetes: quantum-optimization namespace")
        print(f"   📊 Monitoring: Prometheus + Grafana included")
        print(f"   🔒 Security: Enterprise features enabled")
        print(f"   📈 Auto-scaling: {orchestrator.config['scaling']['min_replicas']}-{orchestrator.config['scaling']['max_replicas']} replicas")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)