"""
Docker deployment utilities for quantum hyperparameter search.
"""

import os
import json
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DockerDeployment:
    """Docker deployment manager for quantum hyperparameter search."""
    
    def __init__(self, 
                 project_root: str = ".",
                 registry: str = "localhost:5000",
                 image_name: str = "quantum-hyper-search"):
        """
        Initialize Docker deployment manager.
        
        Args:
            project_root: Root directory of the project
            registry: Docker registry URL
            image_name: Base name for Docker images
        """
        self.project_root = Path(project_root)
        self.registry = registry
        self.image_name = image_name
        self.version = "latest"
        
    def generate_dockerfile(self, 
                          optimization_type: str = "standard",
                          quantum_backends: List[str] = None) -> str:
        """
        Generate optimized Dockerfile for different deployment scenarios.
        
        Args:
            optimization_type: Type of optimization ('standard', 'gpu', 'distributed')
            quantum_backends: List of quantum backends to support
            
        Returns:
            Dockerfile content as string
        """
        quantum_backends = quantum_backends or ['simulator']
        
        base_dockerfile = self._get_base_dockerfile()
        
        # Add quantum backend specific installations
        quantum_deps = self._get_quantum_dependencies(quantum_backends)
        
        # Add optimization specific configurations
        optimization_config = self._get_optimization_config(optimization_type)
        
        dockerfile = f"""
{base_dockerfile}

# Quantum backend dependencies
{quantum_deps}

# Optimization specific configuration
{optimization_config}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "from quantum_hyper_search.utils.monitoring import HealthChecker; hc = HealthChecker(); hc.setup_default_checks(); result = hc.run_checks(); exit(0 if result['overall_health'] else 1)"

# Entry point
ENTRYPOINT ["python", "-m", "quantum_hyper_search.server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
"""
        
        return dockerfile.strip()
    
    def _get_base_dockerfile(self) -> str:
        """Get base Dockerfile configuration."""
        return """
# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    gfortran \\
    libblas-dev \\
    liblapack-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libblas3 \\
    liblapack3 \\
    libgfortran5 \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Set working directory
WORKDIR /app

# Copy application code
COPY quantum_hyper_search/ ./quantum_hyper_search/
COPY setup.py README.md ./

# Install the package
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/cache && \\
    chown -R quantum:quantum /app

# Switch to non-root user
USER quantum
"""
    
    def _get_quantum_dependencies(self, backends: List[str]) -> str:
        """Get quantum backend specific dependencies."""
        deps = []
        
        if 'dwave' in backends:
            deps.append("RUN pip install --no-cache-dir dwave-ocean-sdk")
        
        if 'qiskit' in backends:
            deps.append("RUN pip install --no-cache-dir qiskit qiskit-aer")
        
        if 'cirq' in backends:
            deps.append("RUN pip install --no-cache-dir cirq")
            
        return "\n".join(deps)
    
    def _get_optimization_config(self, optimization_type: str) -> str:
        """Get optimization specific configuration."""
        if optimization_type == "gpu":
            return """
# GPU optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV QUANTUM_GPU_ENABLED=true
RUN pip install --no-cache-dir cupy-cuda11x
"""
        elif optimization_type == "distributed":
            return """
# Distributed computing configuration
ENV QUANTUM_DISTRIBUTED=true
ENV QUANTUM_WORKER_NODES=auto
RUN pip install --no-cache-dir dask[distributed] ray[default]
"""
        else:
            return """
# Standard configuration
ENV QUANTUM_OPTIMIZATION_LEVEL=standard
ENV QUANTUM_PARALLEL_WORKERS=auto
"""
    
    def build_image(self, 
                   version: str = None,
                   optimization_type: str = "standard",
                   quantum_backends: List[str] = None,
                   build_args: Dict[str, str] = None) -> str:
        """
        Build Docker image for quantum hyperparameter search.
        
        Args:
            version: Image version tag
            optimization_type: Type of optimization to build for
            quantum_backends: Quantum backends to include
            build_args: Additional build arguments
            
        Returns:
            Full image name with tag
        """
        version = version or self.version
        build_args = build_args or {}
        
        # Generate Dockerfile
        dockerfile_content = self.generate_dockerfile(optimization_type, quantum_backends)
        
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write Dockerfile
            dockerfile_path = temp_path / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Copy necessary files
            self._copy_build_context(temp_path)
            
            # Build image
            full_image_name = f"{self.registry}/{self.image_name}:{version}"
            
            build_cmd = [
                "docker", "build",
                "-t", full_image_name,
                "-f", str(dockerfile_path)
            ]
            
            # Add build arguments
            for key, value in build_args.items():
                build_cmd.extend(["--build-arg", f"{key}={value}"])
            
            build_cmd.append(str(temp_path))
            
            logger.info(f"Building Docker image: {full_image_name}")
            
            try:
                result = subprocess.run(
                    build_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                logger.info(f"Successfully built image: {full_image_name}")
                return full_image_name
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Docker build failed: {e.stderr}")
                raise RuntimeError(f"Failed to build Docker image: {e.stderr}")
    
    def _copy_build_context(self, temp_path: Path) -> None:
        """Copy necessary files to build context."""
        import shutil
        
        # Copy source code
        if (self.project_root / "quantum_hyper_search").exists():
            shutil.copytree(
                self.project_root / "quantum_hyper_search",
                temp_path / "quantum_hyper_search"
            )
        
        # Copy setup files
        for file_name in ["setup.py", "requirements.txt", "README.md", "pyproject.toml"]:
            src_file = self.project_root / file_name
            if src_file.exists():
                shutil.copy2(src_file, temp_path / file_name)
    
    def push_image(self, image_name: str) -> None:
        """
        Push Docker image to registry.
        
        Args:
            image_name: Full image name with tag
        """
        logger.info(f"Pushing image to registry: {image_name}")
        
        try:
            subprocess.run(
                ["docker", "push", image_name],
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info(f"Successfully pushed image: {image_name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker push failed: {e.stderr}")
            raise RuntimeError(f"Failed to push Docker image: {e.stderr}")
    
    def run_container(self,
                     image_name: str,
                     container_name: str = None,
                     ports: Dict[int, int] = None,
                     environment: Dict[str, str] = None,
                     volumes: Dict[str, str] = None,
                     detached: bool = True) -> str:
        """
        Run Docker container.
        
        Args:
            image_name: Docker image to run
            container_name: Name for the container
            ports: Port mappings (container_port: host_port)
            environment: Environment variables
            volumes: Volume mounts (host_path: container_path)
            detached: Run in detached mode
            
        Returns:
            Container ID
        """
        container_name = container_name or f"quantum-hyper-search-{os.urandom(4).hex()}"
        ports = ports or {8080: 8080}
        environment = environment or {}
        volumes = volumes or {}
        
        run_cmd = ["docker", "run"]
        
        if detached:
            run_cmd.append("-d")
        
        run_cmd.extend(["--name", container_name])
        
        # Add port mappings
        for container_port, host_port in ports.items():
            run_cmd.extend(["-p", f"{host_port}:{container_port}"])
        
        # Add environment variables
        for key, value in environment.items():
            run_cmd.extend(["-e", f"{key}={value}"])
        
        # Add volume mounts
        for host_path, container_path in volumes.items():
            run_cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        run_cmd.append(image_name)
        
        logger.info(f"Starting container: {container_name}")
        
        try:
            result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            container_id = result.stdout.strip()
            logger.info(f"Container started: {container_id}")
            
            return container_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start container: {e.stderr}")
            raise RuntimeError(f"Failed to start Docker container: {e.stderr}")
    
    def generate_docker_compose(self,
                               services: Dict[str, Any] = None,
                               include_monitoring: bool = True,
                               include_load_balancer: bool = True) -> str:
        """
        Generate Docker Compose configuration.
        
        Args:
            services: Custom service configurations
            include_monitoring: Include monitoring services
            include_load_balancer: Include load balancer
            
        Returns:
            Docker Compose YAML content
        """
        services = services or {}
        
        base_config = {
            'version': '3.8',
            'services': {
                'quantum-api': {
                    'image': f"{self.registry}/{self.image_name}:latest",
                    'ports': ['8080:8080'],
                    'environment': {
                        'QUANTUM_LOG_LEVEL': 'INFO',
                        'QUANTUM_CACHE_SIZE': '10000',
                        'QUANTUM_PARALLEL_WORKERS': '4'
                    },
                    'volumes': [
                        './data:/app/data',
                        './logs:/app/logs',
                        './cache:/app/cache'
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    },
                    'deploy': {
                        'resources': {
                            'limits': {
                                'cpus': '2.0',
                                'memory': '4G'
                            },
                            'reservations': {
                                'cpus': '1.0',
                                'memory': '2G'
                            }
                        }
                    }
                }
            }
        }
        
        # Add monitoring services
        if include_monitoring:
            base_config['services'].update({
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'ports': ['9090:9090'],
                    'volumes': ['./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml'],
                    'restart': 'unless-stopped'
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'ports': ['3000:3000'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'quantum123'
                    },
                    'volumes': ['grafana-storage:/var/lib/grafana'],
                    'restart': 'unless-stopped'
                }
            })
        
        # Add load balancer
        if include_load_balancer:
            base_config['services']['nginx'] = {
                'image': 'nginx:alpine',
                'ports': ['80:80', '443:443'],
                'volumes': [
                    './nginx/nginx.conf:/etc/nginx/nginx.conf',
                    './nginx/ssl:/etc/nginx/ssl'
                ],
                'depends_on': ['quantum-api'],
                'restart': 'unless-stopped'
            }
        
        # Add volumes section
        if include_monitoring:
            base_config['volumes'] = {
                'grafana-storage': {}
            }
        
        # Merge custom services
        base_config['services'].update(services)
        
        # Convert to YAML
        import yaml
        return yaml.dump(base_config, default_flow_style=False, sort_keys=False)
    
    def deploy_stack(self,
                    compose_file: str = "docker-compose.yml",
                    env_file: str = None) -> None:
        """
        Deploy Docker Compose stack.
        
        Args:
            compose_file: Path to docker-compose.yml file
            env_file: Path to environment file
        """
        cmd = ["docker-compose", "-f", compose_file]
        
        if env_file:
            cmd.extend(["--env-file", env_file])
        
        cmd.extend(["up", "-d"])
        
        logger.info(f"Deploying Docker Compose stack: {compose_file}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Successfully deployed Docker Compose stack")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker Compose deployment failed: {e.stderr}")
            raise RuntimeError(f"Failed to deploy stack: {e.stderr}")
    
    def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """
        Get container resource usage statistics.
        
        Args:
            container_name: Name of the container
            
        Returns:
            Container statistics
        """
        try:
            result = subprocess.run(
                ["docker", "stats", container_name, "--no-stream", "--format", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            return json.loads(result.stdout)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get container stats: {e.stderr}")
            return {}
    
    def scale_service(self, service_name: str, replicas: int, compose_file: str = "docker-compose.yml") -> None:
        """
        Scale Docker Compose service.
        
        Args:
            service_name: Name of the service to scale
            replicas: Number of replicas
            compose_file: Path to docker-compose.yml file
        """
        cmd = ["docker-compose", "-f", compose_file, "up", "-d", "--scale", f"{service_name}={replicas}"]
        
        logger.info(f"Scaling service {service_name} to {replicas} replicas")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully scaled {service_name} to {replicas} replicas")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Service scaling failed: {e.stderr}")
            raise RuntimeError(f"Failed to scale service: {e.stderr}")