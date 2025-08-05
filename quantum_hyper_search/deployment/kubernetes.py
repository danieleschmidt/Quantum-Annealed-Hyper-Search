"""
Kubernetes deployment utilities for quantum hyperparameter search.
"""

import json
import yaml
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KubernetesConfig:
    """Kubernetes deployment configuration."""
    namespace: str = "quantum-hyper-search"
    image: str = "quantum-hyper-search:latest"
    replicas: int = 3
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    enable_autoscaling: bool = True
    enable_monitoring: bool = True
    storage_class: str = "standard"
    storage_size: str = "10Gi"


class KubernetesDeployment:
    """Kubernetes deployment manager for quantum hyperparameter search."""
    
    def __init__(self, config: KubernetesConfig = None, kubeconfig: str = None):
        """
        Initialize Kubernetes deployment manager.
        
        Args:
            config: Kubernetes deployment configuration
            kubeconfig: Path to kubeconfig file
        """
        self.config = config or KubernetesConfig()
        self.kubeconfig = kubeconfig
        
    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace manifest."""
        return {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': self.config.namespace,
                'labels': {
                    'app': 'quantum-hyper-search',
                    'tier': 'compute'
                }
            }
        }
    
    def generate_configmap(self, config_data: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Generate ConfigMap for application configuration.
        
        Args:
            config_data: Configuration key-value pairs
            
        Returns:
            ConfigMap manifest
        """
        config_data = config_data or {
            'QUANTUM_LOG_LEVEL': 'INFO',
            'QUANTUM_CACHE_SIZE': '10000',
            'QUANTUM_PARALLEL_WORKERS': '4',
            'QUANTUM_OPTIMIZATION_LEVEL': 'production',
            'QUANTUM_ENABLE_MONITORING': 'true'
        }
        
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'quantum-config',
                'namespace': self.config.namespace
            },
            'data': config_data
        }
    
    def generate_secret(self, secret_data: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Generate Secret for sensitive configuration.
        
        Args:
            secret_data: Secret key-value pairs (base64 encoded)
            
        Returns:
            Secret manifest
        """
        import base64
        
        # Default secrets (these should be replaced with real values)
        raw_secrets = secret_data or {
            'quantum-api-key': 'your-quantum-api-key-here',
            'dwave-token': 'your-dwave-token-here'
        }
        
        # Base64 encode secrets
        encoded_secrets = {
            key: base64.b64encode(value.encode()).decode()
            for key, value in raw_secrets.items()
        }
        
        return {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': 'quantum-secrets',
                'namespace': self.config.namespace
            },
            'type': 'Opaque',
            'data': encoded_secrets
        }
    
    def generate_persistent_volume_claim(self) -> Dict[str, Any]:
        """Generate PersistentVolumeClaim for data storage."""
        return {
            'apiVersion': 'v1',
            'kind': 'PersistentVolumeClaim',
            'metadata': {
                'name': 'quantum-data-pvc',
                'namespace': self.config.namespace
            },
            'spec': {
                'accessModes': ['ReadWriteOnce'],
                'storageClassName': self.config.storage_class,
                'resources': {
                    'requests': {
                        'storage': self.config.storage_size
                    }
                }
            }
        }
    
    def generate_deployment(self) -> Dict[str, Any]:
        """Generate Deployment manifest."""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'quantum-hyper-search',
                'namespace': self.config.namespace,
                'labels': {
                    'app': 'quantum-hyper-search',
                    'version': 'v1'
                }
            },
            'spec': {
                'replicas': self.config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': 'quantum-hyper-search'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'quantum-hyper-search',
                            'version': 'v1'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'quantum-api',
                            'image': self.config.image,
                            'ports': [{
                                'containerPort': 8080,
                                'name': 'http'
                            }],
                            'resources': {
                                'requests': {
                                    'cpu': self.config.cpu_request,
                                    'memory': self.config.memory_request
                                },
                                'limits': {
                                    'cpu': self.config.cpu_limit,
                                    'memory': self.config.memory_limit
                                }
                            },
                            'envFrom': [{
                                'configMapRef': {
                                    'name': 'quantum-config'
                                }
                            }],
                            'env': [{
                                'name': 'QUANTUM_API_KEY',
                                'valueFrom': {
                                    'secretKeyRef': {
                                        'name': 'quantum-secrets',
                                        'key': 'quantum-api-key'
                                    }
                                }
                            }, {
                                'name': 'DWAVE_TOKEN',
                                'valueFrom': {
                                    'secretKeyRef': {
                                        'name': 'quantum-secrets',
                                        'key': 'dwave-token'
                                    }
                                }
                            }],
                            'volumeMounts': [{
                                'name': 'data-storage',
                                'mountPath': '/app/data'
                            }, {
                                'name': 'cache-storage',
                                'mountPath': '/app/cache'
                            }],
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8080
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30,
                                'timeoutSeconds': 10,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8080
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            }
                        }],
                        'volumes': [{
                            'name': 'data-storage',
                            'persistentVolumeClaim': {
                                'claimName': 'quantum-data-pvc'
                            }
                        }, {
                            'name': 'cache-storage',
                            'emptyDir': {
                                'sizeLimit': '5Gi'
                            }
                        }]
                    }
                }
            }
        }
    
    def generate_service(self) -> Dict[str, Any]:
        """Generate Service manifest."""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'quantum-hyper-search-service',
                'namespace': self.config.namespace,
                'labels': {
                    'app': 'quantum-hyper-search'
                }
            },
            'spec': {
                'selector': {
                    'app': 'quantum-hyper-search'
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8080,
                    'name': 'http'
                }],
                'type': 'ClusterIP'
            }
        }
    
    def generate_horizontal_pod_autoscaler(self) -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler manifest."""
        if not self.config.enable_autoscaling:
            return {}
            
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'quantum-hpa',
                'namespace': self.config.namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'quantum-hyper-search'
                },
                'minReplicas': self.config.replicas,
                'maxReplicas': self.config.max_replicas,
                'metrics': [{
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': self.config.target_cpu_utilization
                        }
                    }
                }, {
                    'type': 'Resource',
                    'resource': {
                        'name': 'memory',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': 80
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
    
    def generate_ingress(self, 
                        hostname: str = "quantum-api.example.com",
                        enable_tls: bool = True,
                        ingress_class: str = "nginx") -> Dict[str, Any]:
        """
        Generate Ingress manifest.
        
        Args:
            hostname: Hostname for the ingress
            enable_tls: Enable TLS/SSL
            ingress_class: Ingress controller class
            
        Returns:
            Ingress manifest
        """
        ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': 'quantum-ingress',
                'namespace': self.config.namespace,
                'annotations': {
                    'kubernetes.io/ingress.class': ingress_class,
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true' if enable_tls else 'false'
                }
            },
            'spec': {
                'rules': [{
                    'host': hostname,
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': 'quantum-hyper-search-service',
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        if enable_tls:
            ingress['spec']['tls'] = [{
                'hosts': [hostname],
                'secretName': 'quantum-tls-secret'
            }]
        
        return ingress
    
    def generate_network_policy(self) -> Dict[str, Any]:
        """Generate NetworkPolicy for enhanced security."""
        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'quantum-network-policy',
                'namespace': self.config.namespace
            },
            'spec': {
                'podSelector': {
                    'matchLabels': {
                        'app': 'quantum-hyper-search'
                    }
                },
                'policyTypes': ['Ingress', 'Egress'],
                'ingress': [{
                    'from': [{
                        'namespaceSelector': {
                            'matchLabels': {
                                'name': 'ingress-nginx'
                            }
                        }
                    }],
                    'ports': [{
                        'protocol': 'TCP',
                        'port': 8080
                    }]
                }],
                'egress': [{
                    'to': [],  # Allow all egress for quantum service calls
                    'ports': [{
                        'protocol': 'TCP',
                        'port': 443
                    }, {
                        'protocol': 'TCP',
                        'port': 80
                    }]
                }]
            }
        }
    
    def generate_monitoring_manifests(self) -> List[Dict[str, Any]]:
        """Generate monitoring manifests (ServiceMonitor, etc.)."""
        if not self.config.enable_monitoring:
            return []
        
        service_monitor = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': 'quantum-metrics',
                'namespace': self.config.namespace,
                'labels': {
                    'app': 'quantum-hyper-search'
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': 'quantum-hyper-search'
                    }
                },
                'endpoints': [{
                    'port': 'http',
                    'path': '/metrics',
                    'interval': '30s'
                }]
            }
        }
        
        return [service_monitor]
    
    def generate_all_manifests(self, 
                              hostname: str = "quantum-api.example.com",
                              config_data: Dict[str, str] = None,
                              secret_data: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Generate all Kubernetes manifests.
        
        Args:
            hostname: Hostname for ingress
            config_data: ConfigMap data
            secret_data: Secret data
            
        Returns:
            List of all manifests
        """
        manifests = [
            self.generate_namespace(),
            self.generate_configmap(config_data),
            self.generate_secret(secret_data),
            self.generate_persistent_volume_claim(),
            self.generate_deployment(),
            self.generate_service(),
            self.generate_ingress(hostname),
            self.generate_network_policy()
        ]
        
        # Add HPA if enabled
        hpa = self.generate_horizontal_pod_autoscaler()
        if hpa:
            manifests.append(hpa)
        
        # Add monitoring manifests
        manifests.extend(self.generate_monitoring_manifests())
        
        return manifests
    
    def save_manifests(self, 
                      manifests: List[Dict[str, Any]], 
                      output_dir: str = "./k8s") -> None:
        """
        Save manifests to YAML files.
        
        Args:
            manifests: List of Kubernetes manifests
            output_dir: Output directory for YAML files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, manifest in enumerate(manifests):
            if not manifest:  # Skip empty manifests
                continue
                
            kind = manifest.get('kind', 'unknown').lower()
            name = manifest.get('metadata', {}).get('name', f'resource-{i}')
            
            filename = f"{kind}-{name}.yaml"
            filepath = output_path / filename
            
            with open(filepath, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved manifest: {filepath}")
    
    def apply_manifests(self, manifest_dir: str = "./k8s") -> None:
        """
        Apply Kubernetes manifests using kubectl.
        
        Args:
            manifest_dir: Directory containing YAML manifests
        """
        cmd = ["kubectl", "apply", "-f", manifest_dir]
        
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        
        logger.info(f"Applying Kubernetes manifests from: {manifest_dir}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Successfully applied manifests: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply manifests: {e.stderr}")
            raise RuntimeError(f"Failed to apply Kubernetes manifests: {e.stderr}")
    
    def delete_deployment(self) -> None:
        """Delete the entire deployment."""
        cmd = ["kubectl", "delete", "namespace", self.config.namespace]
        
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        
        logger.info(f"Deleting namespace: {self.config.namespace}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully deleted namespace: {self.config.namespace}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to delete namespace: {e.stderr}")
            raise RuntimeError(f"Failed to delete deployment: {e.stderr}")
    
    def scale_deployment(self, replicas: int) -> None:
        """
        Scale the deployment.
        
        Args:
            replicas: Number of replicas
        """
        cmd = [
            "kubectl", "scale", "deployment", "quantum-hyper-search",
            "--replicas", str(replicas),
            "-n", self.config.namespace
        ]
        
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        
        logger.info(f"Scaling deployment to {replicas} replicas")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully scaled to {replicas} replicas")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to scale deployment: {e.stderr}")
            raise RuntimeError(f"Failed to scale deployment: {e.stderr}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """
        Get deployment status.
        
        Returns:
            Deployment status information
        """
        cmd = [
            "kubectl", "get", "deployment", "quantum-hyper-search",
            "-n", self.config.namespace,
            "-o", "json"
        ]
        
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return json.loads(result.stdout)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get deployment status: {e.stderr}")
            return {}
    
    def get_pod_logs(self, pod_name: str = None, lines: int = 100) -> str:
        """
        Get pod logs.
        
        Args:
            pod_name: Specific pod name (if None, gets logs from first pod)
            lines: Number of lines to retrieve
            
        Returns:
            Pod logs
        """
        if not pod_name:
            # Get first pod
            cmd = [
                "kubectl", "get", "pods",
                "-n", self.config.namespace,
                "-l", "app=quantum-hyper-search",
                "-o", "jsonpath={.items[0].metadata.name}"
            ]
            
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                pod_name = result.stdout.strip()
            except subprocess.CalledProcessError:
                return "No pods found"
        
        # Get logs
        cmd = [
            "kubectl", "logs", pod_name,
            "-n", self.config.namespace,
            "--tail", str(lines)
        ]
        
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get pod logs: {e.stderr}")
            return f"Error getting logs: {e.stderr}"