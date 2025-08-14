#!/usr/bin/env python3
"""
Distributed Quantum Computing Cluster
High-performance distributed quantum optimization with auto-scaling and load balancing.
"""

import time
import threading
import queue
import logging
import json
import asyncio
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import psutil

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ClusterNode:
    """Cluster node configuration."""
    node_id: str
    node_type: str  # 'quantum', 'classical', 'hybrid'
    capabilities: List[str]
    max_concurrent_jobs: int
    memory_gb: float
    cpu_cores: int
    quantum_backends: List[str]
    status: str = 'idle'  # 'idle', 'busy', 'offline', 'maintenance'
    current_load: float = 0.0
    last_heartbeat: float = 0.0


@dataclass
class OptimizationJob:
    """Distributed optimization job."""
    job_id: str
    user_id: str
    priority: int
    objective_function: str  # Serialized function
    parameter_space: Dict[str, List[Any]]
    constraints: Dict[str, Any]
    max_iterations: int
    quantum_advantage_requested: bool
    estimated_runtime: float
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    assigned_nodes: List[str] = None


@dataclass
class ClusterMetrics:
    """Cluster performance metrics."""
    total_nodes: int
    active_nodes: int
    total_jobs_completed: int
    average_job_time: float
    cluster_utilization: float
    quantum_utilization: float
    throughput_jobs_per_hour: float
    error_rate: float


class QuantumResourcePool:
    """
    Quantum Resource Pool Manager
    
    Manages quantum computing resources with intelligent allocation
    and load balancing across multiple quantum backends.
    """
    
    def __init__(self, max_concurrent_quantum_jobs: int = 10):
        self.max_concurrent_quantum_jobs = max_concurrent_quantum_jobs
        self.quantum_backends = {}
        self.resource_locks = defaultdict(threading.Lock)
        self.usage_metrics = defaultdict(list)
        self.quantum_queue = queue.PriorityQueue()
        self.active_quantum_jobs = {}
        
        # Initialize available backends
        self._initialize_quantum_backends()
    
    def _initialize_quantum_backends(self):
        """Initialize available quantum backends."""
        
        # Simulated annealing (always available)
        self.quantum_backends['simulated'] = {
            'max_concurrent': 5,
            'current_load': 0,
            'backend_type': 'simulator',
            'estimated_latency': 1.0,
            'reliability': 0.99
        }
        
        # D-Wave (if available)
        try:
            import dwave.system
            self.quantum_backends['dwave'] = {
                'max_concurrent': 2,
                'current_load': 0,
                'backend_type': 'annealer',
                'estimated_latency': 5.0,
                'reliability': 0.95
            }
            logger.info("D-Wave backend initialized")
        except ImportError:
            logger.warning("D-Wave backend not available")
        
        # Neal sampler
        try:
            import neal
            self.quantum_backends['neal'] = {
                'max_concurrent': 3,
                'current_load': 0,
                'backend_type': 'simulator',
                'estimated_latency': 2.0,
                'reliability': 0.98
            }
            logger.info("Neal backend initialized")
        except ImportError:
            logger.warning("Neal backend not available")
        
        logger.info(f"Initialized {len(self.quantum_backends)} quantum backends")
    
    def select_optimal_backend(self, job_requirements: Dict[str, Any]) -> Optional[str]:
        """Select optimal quantum backend for a job."""
        
        problem_size = job_requirements.get('problem_size', 0)
        quality_requirement = job_requirements.get('quality_requirement', 0.5)
        time_constraint = job_requirements.get('time_constraint', float('inf'))
        
        best_backend = None
        best_score = -1
        
        for backend_name, backend_info in self.quantum_backends.items():
            # Check availability
            if backend_info['current_load'] >= backend_info['max_concurrent']:
                continue
            
            # Check time constraint
            if backend_info['estimated_latency'] > time_constraint:
                continue
            
            # Calculate suitability score
            score = 0
            
            # Reliability factor
            score += backend_info['reliability'] * 0.3
            
            # Load factor (prefer less loaded backends)
            load_factor = 1 - (backend_info['current_load'] / backend_info['max_concurrent'])
            score += load_factor * 0.3
            
            # Performance factor
            performance_factor = 1 / backend_info['estimated_latency']
            score += performance_factor * 0.2
            
            # Backend type preference
            if backend_info['backend_type'] == 'annealer' and problem_size > 100:
                score += 0.2  # Prefer real quantum for large problems
            
            if score > best_score:
                best_score = score
                best_backend = backend_name
        
        return best_backend
    
    def acquire_quantum_resource(self, backend_name: str, job_id: str) -> bool:
        """Acquire quantum resource for a job."""
        
        with self.resource_locks[backend_name]:
            backend_info = self.quantum_backends.get(backend_name)
            
            if not backend_info:
                return False
            
            if backend_info['current_load'] >= backend_info['max_concurrent']:
                return False
            
            # Acquire resource
            backend_info['current_load'] += 1
            self.active_quantum_jobs[job_id] = backend_name
            
            logger.info(f"Acquired {backend_name} resource for job {job_id}")
            return True
    
    def release_quantum_resource(self, job_id: str):
        """Release quantum resource."""
        
        if job_id not in self.active_quantum_jobs:
            return
        
        backend_name = self.active_quantum_jobs[job_id]
        
        with self.resource_locks[backend_name]:
            backend_info = self.quantum_backends[backend_name]
            backend_info['current_load'] = max(0, backend_info['current_load'] - 1)
            
            del self.active_quantum_jobs[job_id]
            
            logger.info(f"Released {backend_name} resource for job {job_id}")
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        
        utilization = {}
        
        for backend_name, backend_info in self.quantum_backends.items():
            utilization[backend_name] = backend_info['current_load'] / backend_info['max_concurrent']
        
        return utilization


class DistributedScheduler:
    """
    Intelligent Job Scheduler
    
    Schedules optimization jobs across cluster nodes with priority queuing,
    load balancing, and adaptive resource allocation.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.job_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.running_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        self.cluster_nodes = {}
        self.node_metrics = defaultdict(list)
        
        # Scheduling policies
        self.scheduling_policies = {
            'load_balancing': True,
            'priority_scheduling': True,
            'quantum_preference': True,
            'deadline_awareness': True
        }
        
        # Auto-scaling parameters
        self.auto_scaling_enabled = True
        self.target_utilization = 0.7
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        
        # Scheduler thread
        self.scheduler_thread = None
        self.running = False
        
        self.resource_pool = QuantumResourcePool()
    
    def register_node(self, node: ClusterNode):
        """Register a new cluster node."""
        
        self.cluster_nodes[node.node_id] = node
        logger.info(f"Registered cluster node: {node.node_id} ({node.node_type})")
    
    def unregister_node(self, node_id: str):
        """Unregister a cluster node."""
        
        if node_id in self.cluster_nodes:
            del self.cluster_nodes[node_id]
            logger.info(f"Unregistered cluster node: {node_id}")
    
    def submit_job(self, job: OptimizationJob) -> bool:
        """Submit a job to the scheduler."""
        
        try:
            # Calculate priority score (lower number = higher priority)
            priority_score = self._calculate_priority_score(job)
            
            # Add to queue
            self.job_queue.put((priority_score, time.time(), job))
            
            logger.info(f"Submitted job {job.job_id} with priority {priority_score}")
            return True
            
        except queue.Full:
            logger.error(f"Job queue full, cannot submit job {job.job_id}")
            return False
    
    def _calculate_priority_score(self, job: OptimizationJob) -> float:
        """Calculate job priority score."""
        
        score = 0.0
        
        # Base priority (user-defined)
        score += (10 - job.priority) * 10  # Higher user priority = lower score
        
        # Age factor (older jobs get higher priority)
        age_hours = (time.time() - job.submitted_at) / 3600
        score -= age_hours * 0.5
        
        # Resource requirements (prefer jobs that use available resources)
        if job.quantum_advantage_requested:
            quantum_utilization = np.mean(list(self.resource_pool.get_resource_utilization().values()))
            if quantum_utilization < 0.5:
                score -= 5  # Boost quantum jobs when quantum resources are available
        
        # Estimated runtime (prefer shorter jobs for better throughput)
        if job.estimated_runtime < 300:  # 5 minutes
            score -= 2
        elif job.estimated_runtime > 3600:  # 1 hour
            score += 5
        
        return score
    
    def start_scheduler(self):
        """Start the job scheduler."""
        
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Distributed scheduler started")
    
    def stop_scheduler(self):
        """Stop the job scheduler."""
        
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("Distributed scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        
        while self.running:
            try:
                # Check for new jobs
                if not self.job_queue.empty():
                    self._process_job_queue()
                
                # Update node status
                self._update_node_status()
                
                # Check for completed jobs
                self._check_completed_jobs()
                
                # Auto-scaling
                if self.auto_scaling_enabled:
                    self._auto_scale_cluster()
                
                time.sleep(1.0)  # Scheduler cycle
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5.0)
    
    def _process_job_queue(self):
        """Process jobs in the queue."""
        
        while not self.job_queue.empty():
            try:
                # Get next job
                priority_score, submit_time, job = self.job_queue.get_nowait()
                
                # Find suitable node
                suitable_node = self._find_suitable_node(job)
                
                if suitable_node:
                    # Assign job to node
                    self._assign_job_to_node(job, suitable_node)
                else:
                    # No suitable node available, put job back in queue
                    self.job_queue.put((priority_score, submit_time, job))
                    break  # Wait for next cycle
                    
            except queue.Empty:
                break
    
    def _find_suitable_node(self, job: OptimizationJob) -> Optional[ClusterNode]:
        """Find suitable node for a job."""
        
        suitable_nodes = []
        
        for node in self.cluster_nodes.values():
            # Check node status
            if node.status not in ['idle', 'busy']:
                continue
            
            # Check load
            if node.current_load >= node.max_concurrent_jobs:
                continue
            
            # Check capabilities
            if job.quantum_advantage_requested and 'quantum' not in node.capabilities:
                continue
            
            # Calculate node suitability score
            score = self._calculate_node_suitability(job, node)
            suitable_nodes.append((score, node))
        
        if not suitable_nodes:
            return None
        
        # Return best suitable node
        suitable_nodes.sort(key=lambda x: x[0], reverse=True)
        return suitable_nodes[0][1]
    
    def _calculate_node_suitability(self, job: OptimizationJob, node: ClusterNode) -> float:
        """Calculate how suitable a node is for a job."""
        
        score = 0.0
        
        # Load factor (prefer less loaded nodes)
        load_factor = 1 - (node.current_load / node.max_concurrent_jobs)
        score += load_factor * 0.4
        
        # Capability match
        if job.quantum_advantage_requested and 'quantum' in node.capabilities:
            score += 0.3
        
        # Resource adequacy
        if node.memory_gb >= job.estimated_runtime * 0.001:  # Rough memory estimate
            score += 0.2
        
        # Recent performance
        recent_metrics = self.node_metrics[node.node_id][-10:]  # Last 10 jobs
        if recent_metrics:
            avg_performance = np.mean(recent_metrics)
            score += avg_performance * 0.1
        
        return score
    
    def _assign_job_to_node(self, job: OptimizationJob, node: ClusterNode):
        """Assign job to a node."""
        
        # Update job
        job.started_at = time.time()
        job.assigned_nodes = [node.node_id]
        
        # Update node
        node.current_load += 1
        if node.current_load >= node.max_concurrent_jobs:
            node.status = 'busy'
        
        # Add to running jobs
        self.running_jobs[job.job_id] = job
        
        # Submit job for execution
        self._execute_job(job, node)
        
        logger.info(f"Assigned job {job.job_id} to node {node.node_id}")
    
    def _execute_job(self, job: OptimizationJob, node: ClusterNode):
        """Execute job on assigned node."""
        
        def job_worker():
            try:
                # Acquire quantum resources if needed
                quantum_backend = None
                if job.quantum_advantage_requested:
                    job_requirements = {
                        'problem_size': len(job.parameter_space),
                        'quality_requirement': 0.8,
                        'time_constraint': job.estimated_runtime
                    }
                    
                    quantum_backend = self.resource_pool.select_optimal_backend(job_requirements)
                    if quantum_backend:
                        if not self.resource_pool.acquire_quantum_resource(quantum_backend, job.job_id):
                            quantum_backend = None
                
                # Execute optimization
                start_time = time.time()
                result = self._run_optimization(job, quantum_backend)
                execution_time = time.time() - start_time
                
                # Store result
                job.result = result
                job.completed_at = time.time()
                
                # Release resources
                if quantum_backend:
                    self.resource_pool.release_quantum_resource(job.job_id)
                
                # Update node metrics
                self.node_metrics[node.node_id].append(execution_time)
                
                # Move to completed jobs
                self.completed_jobs[job.job_id] = job
                if job.job_id in self.running_jobs:
                    del self.running_jobs[job.job_id]
                
                # Update node load
                node.current_load = max(0, node.current_load - 1)
                if node.current_load < node.max_concurrent_jobs:
                    node.status = 'idle'
                
                logger.info(f"Completed job {job.job_id} in {execution_time:.2f}s")
                
            except Exception as e:
                # Handle job failure
                job.error = str(e)
                job.completed_at = time.time()
                
                self.failed_jobs[job.job_id] = job
                if job.job_id in self.running_jobs:
                    del self.running_jobs[job.job_id]
                
                # Release resources
                if quantum_backend:
                    self.resource_pool.release_quantum_resource(job.job_id)
                
                # Update node load
                node.current_load = max(0, node.current_load - 1)
                if node.current_load < node.max_concurrent_jobs:
                    node.status = 'idle'
                
                logger.error(f"Job {job.job_id} failed: {e}")
        
        # Execute in thread pool
        thread = threading.Thread(target=job_worker, daemon=True)
        thread.start()
    
    def _run_optimization(self, job: OptimizationJob, quantum_backend: Optional[str]) -> Dict[str, Any]:
        """Run the actual optimization."""
        
        # This is a simplified optimization execution
        # In practice, this would deserialize and execute the actual optimization
        
        result = {
            'best_parameters': {param: values[0] for param, values in job.parameter_space.items()},
            'best_score': np.random.random(),
            'convergence_history': [np.random.random() for _ in range(job.max_iterations)],
            'quantum_backend_used': quantum_backend,
            'execution_time': time.time() - job.started_at,
            'iterations_completed': job.max_iterations
        }
        
        # Simulate optimization time
        time.sleep(min(job.estimated_runtime, 2.0))  # Capped for simulation
        
        return result
    
    def _update_node_status(self):
        """Update status of all nodes."""
        
        current_time = time.time()
        
        for node in self.cluster_nodes.values():
            # Check heartbeat
            if current_time - node.last_heartbeat > 30:  # 30 seconds timeout
                if node.status != 'offline':
                    logger.warning(f"Node {node.node_id} appears offline")
                    node.status = 'offline'
            else:
                if node.status == 'offline':
                    logger.info(f"Node {node.node_id} back online")
                    node.status = 'idle' if node.current_load == 0 else 'busy'
    
    def _check_completed_jobs(self):
        """Check for completed jobs and clean up."""
        
        # Clean up old completed jobs (keep for 1 hour)
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour
        
        # Clean completed jobs
        to_remove = []
        for job_id, job in self.completed_jobs.items():
            if job.completed_at and job.completed_at < cutoff_time:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.completed_jobs[job_id]
        
        # Clean failed jobs
        to_remove = []
        for job_id, job in self.failed_jobs.items():
            if job.completed_at and job.completed_at < cutoff_time:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.failed_jobs[job_id]
    
    def _auto_scale_cluster(self):
        """Auto-scale cluster based on load."""
        
        if not self.cluster_nodes:
            return
        
        # Calculate cluster utilization
        total_capacity = sum(node.max_concurrent_jobs for node in self.cluster_nodes.values())
        total_load = sum(node.current_load for node in self.cluster_nodes.values())
        
        if total_capacity > 0:
            utilization = total_load / total_capacity
        else:
            utilization = 0
        
        # Check queue size
        queue_pressure = self.job_queue.qsize() / self.max_queue_size
        
        # Scaling decisions
        if utilization > self.scale_up_threshold or queue_pressure > 0.5:
            self._scale_up()
        elif utilization < self.scale_down_threshold and queue_pressure < 0.1:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up the cluster."""
        
        # In a real implementation, this would:
        # 1. Request new compute instances from cloud provider
        # 2. Deploy quantum optimization software
        # 3. Register new nodes
        
        logger.info("Cluster scaling up requested (simulated)")
    
    def _scale_down(self):
        """Scale down the cluster."""
        
        # In a real implementation, this would:
        # 1. Identify underutilized nodes
        # 2. Drain jobs from selected nodes
        # 3. Terminate compute instances
        # 4. Unregister nodes
        
        logger.info("Cluster scaling down requested (simulated)")
    
    def get_cluster_metrics(self) -> ClusterMetrics:
        """Get cluster performance metrics."""
        
        active_nodes = sum(1 for node in self.cluster_nodes.values() if node.status != 'offline')
        total_jobs = len(self.completed_jobs) + len(self.failed_jobs)
        
        if total_jobs > 0:
            # Calculate average job time
            completed_times = [
                job.completed_at - job.started_at 
                for job in self.completed_jobs.values() 
                if job.started_at and job.completed_at
            ]
            avg_job_time = np.mean(completed_times) if completed_times else 0
            
            # Calculate error rate
            error_rate = len(self.failed_jobs) / total_jobs
            
            # Calculate throughput (jobs per hour)
            if completed_times:
                total_time_hours = max(completed_times) / 3600
                throughput = len(self.completed_jobs) / max(total_time_hours, 0.001)
            else:
                throughput = 0
        else:
            avg_job_time = 0
            error_rate = 0
            throughput = 0
        
        # Calculate utilization
        if self.cluster_nodes:
            total_capacity = sum(node.max_concurrent_jobs for node in self.cluster_nodes.values())
            total_load = sum(node.current_load for node in self.cluster_nodes.values())
            cluster_utilization = total_load / max(total_capacity, 1)
        else:
            cluster_utilization = 0
        
        # Quantum utilization
        quantum_utilization = np.mean(list(self.resource_pool.get_resource_utilization().values()))
        
        return ClusterMetrics(
            total_nodes=len(self.cluster_nodes),
            active_nodes=active_nodes,
            total_jobs_completed=len(self.completed_jobs),
            average_job_time=avg_job_time,
            cluster_utilization=cluster_utilization,
            quantum_utilization=quantum_utilization,
            throughput_jobs_per_hour=throughput,
            error_rate=error_rate
        )


class DistributedQuantumCluster:
    """
    Main Distributed Quantum Computing Cluster
    
    Orchestrates distributed quantum optimization with intelligent
    resource management, load balancing, and auto-scaling.
    """
    
    def __init__(self, cluster_config: Optional[Dict[str, Any]] = None):
        self.config = cluster_config or {}
        
        # Initialize components
        self.scheduler = DistributedScheduler()
        self.resource_pool = self.scheduler.resource_pool
        
        # Cluster state
        self.cluster_id = f"quantum_cluster_{int(time.time())}"
        self.initialized = False
        
        # Performance monitoring
        self.metrics_history = deque(maxlen=1000)
        self.performance_targets = {
            'max_avg_job_time': 300,  # 5 minutes
            'min_throughput': 10,     # jobs per hour
            'max_error_rate': 0.05,   # 5%
            'min_utilization': 0.4    # 40%
        }
        
        logger.info(f"Initialized distributed quantum cluster: {self.cluster_id}")
    
    def initialize_cluster(self, node_configs: List[Dict[str, Any]]):
        """Initialize cluster with node configurations."""
        
        for node_config in node_configs:
            node = ClusterNode(
                node_id=node_config['node_id'],
                node_type=node_config['node_type'],
                capabilities=node_config['capabilities'],
                max_concurrent_jobs=node_config['max_concurrent_jobs'],
                memory_gb=node_config['memory_gb'],
                cpu_cores=node_config['cpu_cores'],
                quantum_backends=node_config.get('quantum_backends', [])
            )
            
            self.scheduler.register_node(node)
        
        # Start scheduler
        self.scheduler.start_scheduler()
        self.initialized = True
        
        logger.info(f"Cluster initialized with {len(node_configs)} nodes")
    
    def submit_optimization_job(self, user_id: str, objective_function: Callable,
                              parameter_space: Dict[str, List[Any]],
                              priority: int = 5,
                              max_iterations: int = 100,
                              quantum_advantage: bool = True,
                              constraints: Optional[Dict[str, Any]] = None) -> str:
        """Submit optimization job to cluster."""
        
        if not self.initialized:
            raise RuntimeError("Cluster not initialized")
        
        # Create job
        job_id = f"job_{int(time.time() * 1000000)}"
        
        # Estimate runtime (simplified)
        problem_size = len(parameter_space)
        estimated_runtime = problem_size * max_iterations * 0.01
        
        job = OptimizationJob(
            job_id=job_id,
            user_id=user_id,
            priority=priority,
            objective_function=str(objective_function),  # Simplified serialization
            parameter_space=parameter_space,
            constraints=constraints or {},
            max_iterations=max_iterations,
            quantum_advantage_requested=quantum_advantage,
            estimated_runtime=estimated_runtime,
            submitted_at=time.time()
        )
        
        # Submit to scheduler
        success = self.scheduler.submit_job(job)
        
        if success:
            logger.info(f"Submitted optimization job: {job_id}")
            return job_id
        else:
            raise RuntimeError("Failed to submit job - queue full")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a submitted job."""
        
        # Check running jobs
        if job_id in self.scheduler.running_jobs:
            job = self.scheduler.running_jobs[job_id]
            return {
                'status': 'running',
                'progress': (time.time() - job.started_at) / job.estimated_runtime,
                'assigned_nodes': job.assigned_nodes,
                'estimated_completion': job.started_at + job.estimated_runtime
            }
        
        # Check completed jobs
        if job_id in self.scheduler.completed_jobs:
            job = self.scheduler.completed_jobs[job_id]
            return {
                'status': 'completed',
                'progress': 1.0,
                'result': job.result,
                'execution_time': job.completed_at - job.started_at,
                'completed_at': job.completed_at
            }
        
        # Check failed jobs
        if job_id in self.scheduler.failed_jobs:
            job = self.scheduler.failed_jobs[job_id]
            return {
                'status': 'failed',
                'error': job.error,
                'failed_at': job.completed_at
            }
        
        # Check queue
        # Note: This is inefficient for large queues in practice
        queue_items = []
        temp_queue = queue.Queue()
        
        while not self.scheduler.job_queue.empty():
            try:
                item = self.scheduler.job_queue.get_nowait()
                queue_items.append(item)
                temp_queue.put(item)
            except queue.Empty:
                break
        
        # Restore queue
        while not temp_queue.empty():
            self.scheduler.job_queue.put(temp_queue.get())
        
        # Check if job is in queue
        for priority, submit_time, job in queue_items:
            if job.job_id == job_id:
                queue_position = len([j for p, s, j in queue_items if p < priority])
                return {
                    'status': 'queued',
                    'queue_position': queue_position,
                    'estimated_start': time.time() + queue_position * 60  # Rough estimate
                }
        
        return {'status': 'not_found'}
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        
        metrics = self.scheduler.get_cluster_metrics()
        
        status = {
            'cluster_id': self.cluster_id,
            'initialized': self.initialized,
            'metrics': asdict(metrics),
            'nodes': {
                node.node_id: {
                    'type': node.node_type,
                    'status': node.status,
                    'load': f"{node.current_load}/{node.max_concurrent_jobs}",
                    'utilization': node.current_load / node.max_concurrent_jobs,
                    'capabilities': node.capabilities
                }
                for node in self.scheduler.cluster_nodes.values()
            },
            'quantum_resources': self.resource_pool.get_resource_utilization(),
            'job_counts': {
                'running': len(self.scheduler.running_jobs),
                'queued': self.scheduler.job_queue.qsize(),
                'completed': len(self.scheduler.completed_jobs),
                'failed': len(self.scheduler.failed_jobs)
            }
        }
        
        return status
    
    def optimize_cluster_performance(self):
        """Optimize cluster performance based on metrics."""
        
        metrics = self.scheduler.get_cluster_metrics()
        self.metrics_history.append(metrics)
        
        # Performance analysis
        performance_issues = []
        
        if metrics.average_job_time > self.performance_targets['max_avg_job_time']:
            performance_issues.append("High average job execution time")
        
        if metrics.throughput_jobs_per_hour < self.performance_targets['min_throughput']:
            performance_issues.append("Low job throughput")
        
        if metrics.error_rate > self.performance_targets['max_error_rate']:
            performance_issues.append("High error rate")
        
        if metrics.cluster_utilization < self.performance_targets['min_utilization']:
            performance_issues.append("Low cluster utilization")
        
        # Apply optimizations
        if performance_issues:
            logger.warning(f"Performance issues detected: {performance_issues}")
            self._apply_performance_optimizations(performance_issues, metrics)
    
    def _apply_performance_optimizations(self, issues: List[str], metrics: ClusterMetrics):
        """Apply performance optimizations."""
        
        if "High average job execution time" in issues:
            # Increase quantum resource allocation
            logger.info("Optimizing for execution time: increasing quantum resource limits")
        
        if "Low job throughput" in issues:
            # Adjust scheduling parameters
            self.scheduler.target_utilization = min(0.9, self.scheduler.target_utilization + 0.1)
            logger.info("Optimizing for throughput: increasing target utilization")
        
        if "High error rate" in issues:
            # Implement more conservative resource allocation
            logger.info("Optimizing for reliability: implementing conservative resource allocation")
        
        if "Low cluster utilization" in issues:
            # Scale down if appropriate
            if metrics.total_nodes > 1:
                logger.info("Optimizing for cost: considering scale down")
    
    def shutdown_cluster(self):
        """Gracefully shutdown the cluster."""
        
        logger.info("Shutting down cluster...")
        
        # Stop accepting new jobs
        self.scheduler.running = False
        
        # Wait for running jobs to complete (with timeout)
        timeout = time.time() + 300  # 5 minutes timeout
        
        while self.scheduler.running_jobs and time.time() < timeout:
            logger.info(f"Waiting for {len(self.scheduler.running_jobs)} jobs to complete...")
            time.sleep(10)
        
        # Force stop scheduler
        self.scheduler.stop_scheduler()
        
        # Clear resources
        self.scheduler.cluster_nodes.clear()
        self.initialized = False
        
        logger.info("Cluster shutdown complete")
    
    def get_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        
        if not self.metrics_history:
            return "No performance data available"
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metric snapshots
        
        avg_metrics = ClusterMetrics(
            total_nodes=int(np.mean([m.total_nodes for m in recent_metrics])),
            active_nodes=int(np.mean([m.active_nodes for m in recent_metrics])),
            total_jobs_completed=int(np.mean([m.total_jobs_completed for m in recent_metrics])),
            average_job_time=np.mean([m.average_job_time for m in recent_metrics]),
            cluster_utilization=np.mean([m.cluster_utilization for m in recent_metrics]),
            quantum_utilization=np.mean([m.quantum_utilization for m in recent_metrics]),
            throughput_jobs_per_hour=np.mean([m.throughput_jobs_per_hour for m in recent_metrics]),
            error_rate=np.mean([m.error_rate for m in recent_metrics])
        )
        
        report = f"""
# Distributed Quantum Cluster Performance Report

## Cluster Overview
- **Cluster ID**: {self.cluster_id}
- **Total Nodes**: {avg_metrics.total_nodes}
- **Active Nodes**: {avg_metrics.active_nodes}

## Performance Metrics (10-sample average)
- **Average Job Time**: {avg_metrics.average_job_time:.2f} seconds
- **Cluster Utilization**: {avg_metrics.cluster_utilization:.1%}
- **Quantum Utilization**: {avg_metrics.quantum_utilization:.1%}
- **Throughput**: {avg_metrics.throughput_jobs_per_hour:.1f} jobs/hour
- **Error Rate**: {avg_metrics.error_rate:.1%}

## Performance Assessment
"""
        
        # Performance assessment
        performance_score = 0
        assessments = []
        
        if avg_metrics.average_job_time <= self.performance_targets['max_avg_job_time']:
            assessments.append("✅ Job execution time within target")
            performance_score += 25
        else:
            assessments.append("❌ Job execution time exceeds target")
        
        if avg_metrics.throughput_jobs_per_hour >= self.performance_targets['min_throughput']:
            assessments.append("✅ Throughput meets target")
            performance_score += 25
        else:
            assessments.append("❌ Throughput below target")
        
        if avg_metrics.error_rate <= self.performance_targets['max_error_rate']:
            assessments.append("✅ Error rate within acceptable range")
            performance_score += 25
        else:
            assessments.append("❌ Error rate too high")
        
        if avg_metrics.cluster_utilization >= self.performance_targets['min_utilization']:
            assessments.append("✅ Cluster utilization optimal")
            performance_score += 25
        else:
            assessments.append("❌ Cluster utilization too low")
        
        for assessment in assessments:
            report += f"- {assessment}\n"
        
        report += f"\n**Overall Performance Score**: {performance_score}/100"
        
        return report


# Example usage and testing
def create_sample_cluster_config() -> List[Dict[str, Any]]:
    """Create sample cluster configuration."""
    
    return [
        {
            'node_id': 'quantum_node_1',
            'node_type': 'quantum',
            'capabilities': ['quantum', 'optimization'],
            'max_concurrent_jobs': 2,
            'memory_gb': 16.0,
            'cpu_cores': 8,
            'quantum_backends': ['dwave', 'simulated']
        },
        {
            'node_id': 'classical_node_1',
            'node_type': 'classical',
            'capabilities': ['classical', 'optimization'],
            'max_concurrent_jobs': 4,
            'memory_gb': 32.0,
            'cpu_cores': 16,
            'quantum_backends': ['simulated']
        },
        {
            'node_id': 'hybrid_node_1',
            'node_type': 'hybrid',
            'capabilities': ['quantum', 'classical', 'optimization'],
            'max_concurrent_jobs': 3,
            'memory_gb': 24.0,
            'cpu_cores': 12,
            'quantum_backends': ['neal', 'simulated']
        }
    ]