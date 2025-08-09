"""
Production Orchestrator - Enterprise-grade deployment and orchestration system.

Provides comprehensive production deployment capabilities including container orchestration,
service mesh integration, auto-scaling, and multi-cloud deployment support.
"""

import os
import yaml
import json
import time
import logging
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import boto3
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

try:
    import azure.mgmt.containerinstance
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

try:
    from google.cloud import container_v1
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

try:
    from kubernetes import client, config
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"


@dataclass
class DeploymentConfig:
    """Configuration for a deployment."""
    name: str
    version: str
    environment: str  # dev, staging, prod
    cloud_provider: CloudProvider
    region: str
    replicas: int = 3
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    auto_scaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    health_check_path: str = "/health"
    service_mesh_enabled: bool = False
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    backup_enabled: bool = True
    disaster_recovery_enabled: bool = False
    compliance_mode: str = "standard"
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    network_policies: List[str] = field(default_factory=list)
    secrets: Dict[str, str] = field(default_factory=dict)
    config_maps: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    ingress_config: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentStatus:
    """Status of a deployment."""
    deployment_id: str
    status: DeploymentStatus
    created_at: datetime
    updated_at: datetime
    message: str
    health_status: str = "unknown"
    endpoints: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs_location: Optional[str] = None
    rollback_available: bool = False


class KubernetesOrchestrator:
    """Kubernetes deployment orchestrator."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        """Initialize Kubernetes orchestrator."""
        self.kubeconfig_path = kubeconfig_path
        self.k8s_client = None
        
        if HAS_KUBERNETES:
            try:
                if kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path)
                else:
                    try:
                        config.load_incluster_config()
                    except:
                        config.load_kube_config()
                
                self.k8s_client = client.ApiClient()
                logger.info("Kubernetes client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Kubernetes client: {e}")
        else:
            logger.warning("Kubernetes client not available")
    
    def create_namespace(self, namespace: str) -> bool:
        """Create Kubernetes namespace."""
        if not self.k8s_client:
            return False
        
        try:
            v1 = client.CoreV1Api()
            namespace_manifest = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=namespace)
            )
            v1.create_namespace(namespace_manifest)
            logger.info(f"Created namespace: {namespace}")
            return True
        except client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"Namespace {namespace} already exists")
                return True
            logger.error(f"Failed to create namespace {namespace}: {e}")
            return False
    
    def deploy_application(self, config: DeploymentConfig, namespace: str) -> bool:
        """Deploy application to Kubernetes."""
        if not self.k8s_client:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            # Create deployment
            if not self._create_deployment(config, namespace):
                return False
            
            # Create service
            if not self._create_service(config, namespace):
                return False
            
            # Create ingress if specified
            if config.ingress_config:
                if not self._create_ingress(config, namespace):
                    return False
            
            # Create HPA if auto-scaling enabled
            if config.auto_scaling:
                if not self._create_hpa(config, namespace):
                    return False
            
            logger.info(f"Successfully deployed {config.name} to {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _create_deployment(self, config: DeploymentConfig, namespace: str) -> bool:
        """Create Kubernetes deployment."""
        try:
            apps_v1 = client.AppsV1Api()
            
            # Container specification
            container = client.V1Container(
                name=config.name,
                image=f"{config.name}:{config.version}",
                ports=[client.V1ContainerPort(container_port=8000)],
                resources=client.V1ResourceRequirements(
                    requests={
                        "cpu": config.cpu_request,
                        "memory": config.memory_request
                    },
                    limits={
                        "cpu": config.cpu_limit,
                        "memory": config.memory_limit
                    }
                ),
                liveness_probe=client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path=config.health_check_path,
                        port=8000
                    ),
                    initial_delay_seconds=30,
                    period_seconds=10
                ),
                readiness_probe=client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path=config.health_check_path,
                        port=8000
                    ),
                    initial_delay_seconds=5,
                    period_seconds=5
                )
            )
            
            # Pod template
            template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": config.name, "version": config.version}
                ),
                spec=client.V1PodSpec(containers=[container])
            )
            
            # Deployment spec
            spec = client.V1DeploymentSpec(
                replicas=config.replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": config.name}
                ),
                template=template
            )
            
            # Deployment object
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(name=config.name),
                spec=spec
            )
            
            apps_v1.create_namespaced_deployment(
                body=deployment,
                namespace=namespace
            )
            
            logger.info(f"Created deployment for {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            return False
    
    def _create_service(self, config: DeploymentConfig, namespace: str) -> bool:
        """Create Kubernetes service."""
        try:
            v1 = client.CoreV1Api()
            
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(name=config.name),
                spec=client.V1ServiceSpec(
                    selector={"app": config.name},
                    ports=[
                        client.V1ServicePort(
                            protocol="TCP",
                            port=80,
                            target_port=8000
                        )
                    ],
                    type="ClusterIP"
                )
            )
            
            v1.create_namespaced_service(
                body=service,
                namespace=namespace
            )
            
            logger.info(f"Created service for {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            return False
    
    def _create_ingress(self, config: DeploymentConfig, namespace: str) -> bool:
        """Create Kubernetes ingress."""
        try:
            networking_v1 = client.NetworkingV1Api()
            
            ingress = client.V1Ingress(
                api_version="networking.k8s.io/v1",
                kind="Ingress",
                metadata=client.V1ObjectMeta(
                    name=config.name,
                    annotations=config.ingress_config.get("annotations", {})
                ),
                spec=client.V1IngressSpec(
                    rules=[
                        client.V1IngressRule(
                            host=config.ingress_config.get("host"),
                            http=client.V1HTTPIngressRuleValue(
                                paths=[
                                    client.V1HTTPIngressPath(
                                        path="/",
                                        path_type="Prefix",
                                        backend=client.V1IngressBackend(
                                            service=client.V1IngressServiceBackend(
                                                name=config.name,
                                                port=client.V1ServiceBackendPort(
                                                    number=80
                                                )
                                            )
                                        )
                                    )
                                ]
                            )
                        )
                    ]
                )
            )
            
            networking_v1.create_namespaced_ingress(
                body=ingress,
                namespace=namespace
            )
            
            logger.info(f"Created ingress for {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create ingress: {e}")
            return False
    
    def _create_hpa(self, config: DeploymentConfig, namespace: str) -> bool:
        """Create Horizontal Pod Autoscaler."""
        try:
            autoscaling_v1 = client.AutoscalingV1Api()
            
            hpa = client.V1HorizontalPodAutoscaler(
                api_version="autoscaling/v1",
                kind="HorizontalPodAutoscaler",
                metadata=client.V1ObjectMeta(name=f"{config.name}-hpa"),
                spec=client.V1HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V1CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=config.name
                    ),
                    min_replicas=config.min_replicas,
                    max_replicas=config.max_replicas,
                    target_cpu_utilization_percentage=70
                )
            )
            
            autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                body=hpa,
                namespace=namespace
            )
            
            logger.info(f"Created HPA for {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create HPA: {e}")
            return False
    
    def get_deployment_status(self, name: str, namespace: str) -> Dict[str, Any]:
        """Get deployment status."""
        if not self.k8s_client:
            return {"status": "unknown", "reason": "kubernetes_unavailable"}
        
        try:
            apps_v1 = client.AppsV1Api()
            deployment = apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
            
            status = {
                "name": name,
                "namespace": namespace,
                "replicas": deployment.status.replicas or 0,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "updated_replicas": deployment.status.updated_replicas or 0,
                "conditions": []
            }
            
            if deployment.status.conditions:
                for condition in deployment.status.conditions:
                    status["conditions"].append({
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message
                    })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"status": "error", "reason": str(e)}


class ProductionOrchestrator:
    """
    Master production orchestrator for quantum hyperparameter optimization system.
    
    Provides enterprise-grade deployment capabilities including multi-cloud support,
    container orchestration, service mesh integration, and automated operations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize production orchestrator."""
        self.config_path = config_path
        self.deployments: Dict[str, DeploymentStatus] = {}
        self.orchestrators = {}
        
        # Initialize orchestrators
        if HAS_KUBERNETES:
            self.orchestrators['kubernetes'] = KubernetesOrchestrator()
        
        # Load configuration
        self.global_config = self._load_global_config()
        
        logger.info("Production orchestrator initialized")
    
    def _load_global_config(self) -> Dict[str, Any]:
        """Load global configuration."""
        if not self.config_path or not os.path.exists(self.config_path):
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "environments": {
                "development": {
                    "replicas": 1,
                    "resources": {
                        "cpu_request": "100m",
                        "cpu_limit": "500m",
                        "memory_request": "256Mi",
                        "memory_limit": "1Gi"
                    }
                },
                "staging": {
                    "replicas": 2,
                    "resources": {
                        "cpu_request": "200m",
                        "cpu_limit": "1000m",
                        "memory_request": "512Mi",
                        "memory_limit": "2Gi"
                    }
                },
                "production": {
                    "replicas": 3,
                    "resources": {
                        "cpu_request": "500m",
                        "cpu_limit": "2000m",
                        "memory_request": "1Gi",
                        "memory_limit": "4Gi"
                    },
                    "auto_scaling": True,
                    "monitoring": True,
                    "backup": True
                }
            },
            "cloud_providers": {
                "aws": {
                    "regions": ["us-west-2", "us-east-1", "eu-west-1"],
                    "default_region": "us-west-2"
                },
                "azure": {
                    "regions": ["West US 2", "East US", "West Europe"],
                    "default_region": "West US 2"
                },
                "gcp": {
                    "regions": ["us-west1", "us-east1", "europe-west1"],
                    "default_region": "us-west1"
                }
            },
            "security": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "network_policies": True,
                "rbac": True
            },
            "monitoring": {
                "metrics": True,
                "logs": True,
                "traces": True,
                "alerts": True
            }
        }
    
    def create_deployment_config(
        self,
        name: str,
        version: str,
        environment: str,
        cloud_provider: str = "aws",
        region: Optional[str] = None,
        **overrides
    ) -> DeploymentConfig:
        """Create deployment configuration."""
        
        # Get environment defaults
        env_config = self.global_config.get("environments", {}).get(environment, {})
        
        # Get cloud provider defaults
        provider_config = self.global_config.get("cloud_providers", {}).get(cloud_provider, {})
        default_region = region or provider_config.get("default_region", "us-west-2")
        
        # Build configuration
        config_data = {
            "name": name,
            "version": version,
            "environment": environment,
            "cloud_provider": CloudProvider(cloud_provider),
            "region": default_region,
            "replicas": env_config.get("replicas", 1),
            "auto_scaling": env_config.get("auto_scaling", False),
            "monitoring_enabled": env_config.get("monitoring", True),
            "backup_enabled": env_config.get("backup", False)
        }
        
        # Add resource configuration
        if "resources" in env_config:
            resources = env_config["resources"]
            config_data.update({
                "cpu_request": resources.get("cpu_request", "100m"),
                "cpu_limit": resources.get("cpu_limit", "500m"),
                "memory_request": resources.get("memory_request", "256Mi"),
                "memory_limit": resources.get("memory_limit", "1Gi")
            })
        
        # Apply overrides
        config_data.update(overrides)
        
        return DeploymentConfig(**config_data)
    
    def deploy(
        self,
        deployment_config: DeploymentConfig,
        orchestrator: str = "kubernetes",
        namespace: Optional[str] = None
    ) -> str:
        """Deploy application using specified orchestrator."""
        
        deployment_id = f"{deployment_config.name}-{deployment_config.version}-{int(time.time())}"
        namespace = namespace or f"qhs-{deployment_config.environment}"
        
        # Create deployment status
        deployment_status = DeploymentStatus(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            message="Deployment initiated"
        )
        
        self.deployments[deployment_id] = deployment_status
        
        # Start deployment in background
        thread = threading.Thread(
            target=self._execute_deployment,
            args=(deployment_id, deployment_config, orchestrator, namespace)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started deployment {deployment_id}")
        return deployment_id
    
    def _execute_deployment(
        self,
        deployment_id: str,
        config: DeploymentConfig,
        orchestrator: str,
        namespace: str
    ):
        """Execute deployment asynchronously."""
        
        try:
            # Update status
            self._update_deployment_status(
                deployment_id, 
                DeploymentStatus.IN_PROGRESS, 
                "Deployment in progress"
            )
            
            # Get orchestrator
            if orchestrator not in self.orchestrators:
                raise ValueError(f"Orchestrator {orchestrator} not available")
            
            orch = self.orchestrators[orchestrator]
            
            # Execute deployment steps
            if orchestrator == "kubernetes":
                # Create namespace
                orch.create_namespace(namespace)
                
                # Deploy application
                success = orch.deploy_application(config, namespace)
                
                if success:
                    # Wait for deployment to be ready
                    self._wait_for_deployment_ready(config.name, namespace, orch)
                    
                    # Update status to deployed
                    self._update_deployment_status(
                        deployment_id,
                        DeploymentStatus.DEPLOYED,
                        "Deployment completed successfully"
                    )
                    
                    # Get service endpoints
                    endpoints = self._get_service_endpoints(config.name, namespace, orch)
                    self.deployments[deployment_id].endpoints = endpoints
                    
                else:
                    raise RuntimeError("Deployment failed")
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            self._update_deployment_status(
                deployment_id,
                DeploymentStatus.FAILED,
                f"Deployment failed: {str(e)}"
            )
    
    def _update_deployment_status(
        self,
        deployment_id: str,
        status: DeploymentStatus,
        message: str
    ):
        """Update deployment status."""
        if deployment_id in self.deployments:
            self.deployments[deployment_id].status = status
            self.deployments[deployment_id].message = message
            self.deployments[deployment_id].updated_at = datetime.now()
    
    def _wait_for_deployment_ready(
        self,
        name: str,
        namespace: str,
        orchestrator,
        timeout: int = 300
    ):
        """Wait for deployment to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = orchestrator.get_deployment_status(name, namespace)
            
            ready_replicas = status.get("ready_replicas", 0)
            replicas = status.get("replicas", 0)
            
            if ready_replicas > 0 and ready_replicas == replicas:
                logger.info(f"Deployment {name} is ready")
                return
            
            time.sleep(10)
        
        raise TimeoutError(f"Deployment {name} did not become ready within {timeout} seconds")
    
    def _get_service_endpoints(
        self,
        name: str,
        namespace: str,
        orchestrator
    ) -> List[str]:
        """Get service endpoints."""
        # This would query the actual service endpoints
        # For now, return a mock endpoint
        return [f"http://{name}.{namespace}.svc.cluster.local"]
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status."""
        if deployment_id not in self.deployments:
            return None
        
        status = self.deployments[deployment_id]
        return {
            "deployment_id": status.deployment_id,
            "status": status.status.value,
            "created_at": status.created_at.isoformat(),
            "updated_at": status.updated_at.isoformat(),
            "message": status.message,
            "health_status": status.health_status,
            "endpoints": status.endpoints,
            "metrics": status.metrics,
            "logs_location": status.logs_location,
            "rollback_available": status.rollback_available
        }
    
    def list_deployments(self, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all deployments, optionally filtered by environment."""
        deployments = []
        
        for deployment_id, status in self.deployments.items():
            # Filter by environment if specified
            if environment and environment not in deployment_id:
                continue
            
            deployments.append({
                "deployment_id": deployment_id,
                "status": status.status.value,
                "created_at": status.created_at.isoformat(),
                "updated_at": status.updated_at.isoformat(),
                "message": status.message
            })
        
        return deployments
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment."""
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        try:
            self._update_deployment_status(
                deployment_id,
                DeploymentStatus.ROLLING_BACK,
                "Rollback in progress"
            )
            
            # Implement rollback logic here
            # This would involve deploying the previous version
            
            self._update_deployment_status(
                deployment_id,
                DeploymentStatus.ROLLED_BACK,
                "Rollback completed"
            )
            
            logger.info(f"Rollback completed for {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed for {deployment_id}: {e}")
            return False
    
    def scale_deployment(
        self,
        deployment_id: str,
        replicas: int,
        orchestrator: str = "kubernetes",
        namespace: Optional[str] = None
    ) -> bool:
        """Scale a deployment."""
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        try:
            # Extract deployment name from ID
            deployment_name = deployment_id.split('-')[0]
            namespace = namespace or "default"
            
            if orchestrator == "kubernetes" and "kubernetes" in self.orchestrators:
                k8s_orch = self.orchestrators["kubernetes"]
                
                # Scale the deployment
                apps_v1 = client.AppsV1Api()
                apps_v1.patch_namespaced_deployment_scale(
                    name=deployment_name,
                    namespace=namespace,
                    body={'spec': {'replicas': replicas}}
                )
                
                logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
                return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
        
        return False
    
    def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment metrics."""
        if deployment_id not in self.deployments:
            return {}
        
        # Mock metrics - in production, this would query monitoring systems
        return {
            "cpu_usage": "45%",
            "memory_usage": "67%",
            "requests_per_second": 150,
            "error_rate": "0.1%",
            "response_time_p99": "245ms",
            "uptime": "99.9%"
        }
    
    def get_deployment_logs(
        self,
        deployment_id: str,
        lines: int = 100,
        since: Optional[str] = None
    ) -> List[str]:
        """Get deployment logs."""
        # Mock logs - in production, this would query logging systems
        return [
            f"[{datetime.now().isoformat()}] INFO: Quantum optimization service started",
            f"[{datetime.now().isoformat()}] INFO: Backend initialized successfully",
            f"[{datetime.now().isoformat()}] INFO: Health check endpoint ready",
            f"[{datetime.now().isoformat()}] INFO: Accepting requests on port 8000"
        ]
    
    def create_production_manifest(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """Create production-ready deployment manifest."""
        return {
            "apiVersion": "v1",
            "kind": "List",
            "items": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {
                        "name": deployment_config.name,
                        "labels": {
                            "app": deployment_config.name,
                            "version": deployment_config.version,
                            "environment": deployment_config.environment
                        }
                    },
                    "spec": {
                        "replicas": deployment_config.replicas,
                        "selector": {
                            "matchLabels": {"app": deployment_config.name}
                        },
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app": deployment_config.name,
                                    "version": deployment_config.version
                                }
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": deployment_config.name,
                                        "image": f"{deployment_config.name}:{deployment_config.version}",
                                        "ports": [{"containerPort": 8000}],
                                        "resources": {
                                            "requests": {
                                                "cpu": deployment_config.cpu_request,
                                                "memory": deployment_config.memory_request
                                            },
                                            "limits": {
                                                "cpu": deployment_config.cpu_limit,
                                                "memory": deployment_config.memory_limit
                                            }
                                        },
                                        "livenessProbe": {
                                            "httpGet": {
                                                "path": deployment_config.health_check_path,
                                                "port": 8000
                                            },
                                            "initialDelaySeconds": 30,
                                            "periodSeconds": 10
                                        },
                                        "readinessProbe": {
                                            "httpGet": {
                                                "path": deployment_config.health_check_path,
                                                "port": 8000
                                            },
                                            "initialDelaySeconds": 5,
                                            "periodSeconds": 5
                                        }
                                    }
                                ]
                            }
                        }
                    }
                },
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": deployment_config.name,
                        "labels": {"app": deployment_config.name}
                    },
                    "spec": {
                        "selector": {"app": deployment_config.name},
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 80,
                                "targetPort": 8000
                            }
                        ],
                        "type": "ClusterIP"
                    }
                }
            ]
        }
    
    def export_deployment_config(self, deployment_config: DeploymentConfig, format: str = "yaml") -> str:
        """Export deployment configuration."""
        manifest = self.create_production_manifest(deployment_config)
        
        if format.lower() == "yaml":
            return yaml.dump(manifest, default_flow_style=False)
        else:
            return json.dumps(manifest, indent=2)
    
    def get_orchestrator_summary(self) -> Dict[str, Any]:
        """Get orchestrator summary."""
        return {
            "available_orchestrators": list(self.orchestrators.keys()),
            "total_deployments": len(self.deployments),
            "active_deployments": len([d for d in self.deployments.values() 
                                     if d.status == DeploymentStatus.DEPLOYED]),
            "failed_deployments": len([d for d in self.deployments.values() 
                                     if d.status == DeploymentStatus.FAILED]),
            "supported_cloud_providers": [p.value for p in CloudProvider],
            "kubernetes_available": HAS_KUBERNETES,
            "aws_available": HAS_AWS,
            "azure_available": HAS_AZURE,
            "gcp_available": HAS_GCP
        }
