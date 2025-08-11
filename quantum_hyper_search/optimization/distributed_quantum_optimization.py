"""
Distributed Quantum Optimization Framework

Advanced distributed computing framework for quantum optimization
with auto-scaling, load balancing, and fault tolerance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
import asyncio
import concurrent.futures
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle
from queue import Queue, Empty
import threading
from ..core.base import QuantumBackend
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class WorkerStatus(Enum):
    """Worker node status"""
    IDLE = "idle"
    BUSY = "busy" 
    FAILED = "failed"
    OFFLINE = "offline"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class OptimizationTask:
    """Distributed optimization task"""
    task_id: str
    problem_data: Dict[str, Any]
    parameters: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = 0.0
    max_retries: int = 3
    timeout_seconds: int = 300
    quantum_required: bool = False
    estimated_runtime: float = 60.0

@dataclass
class WorkerNode:
    """Distributed worker node"""
    worker_id: str
    hostname: str
    port: int
    status: WorkerStatus = WorkerStatus.OFFLINE
    capabilities: List[str] = None
    current_task_id: Optional[str] = None
    last_heartbeat: float = 0.0
    total_tasks_completed: int = 0
    average_task_time: float = 0.0
    quantum_backend_available: bool = False
    cpu_count: int = 1
    memory_gb: float = 4.0
    load_factor: float = 0.0

@dataclass
class OptimizationResult:
    """Result from distributed optimization"""
    task_id: str
    worker_id: str
    best_solution: Dict[str, Any]
    best_value: float
    execution_time: float
    iterations_completed: int
    convergence_achieved: bool
    quantum_advantage_used: bool
    error_message: Optional[str] = None

class DistributedQuantumOptimizer:
    """
    Advanced distributed quantum optimization system with automatic
    scaling, intelligent load balancing, and fault tolerance.
    """
    
    def __init__(
        self,
        cluster_config: Optional[Dict[str, Any]] = None,
        enable_auto_scaling: bool = True,
        max_workers: int = 50,
        enable_fault_tolerance: bool = True
    ):
        self.cluster_config = cluster_config or {}
        self.enable_auto_scaling = enable_auto_scaling
        self.max_workers = max_workers
        self.enable_fault_tolerance = enable_fault_tolerance
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.active_tasks: Dict[str, OptimizationTask] = {}
        
        # Performance tracking
        self.cluster_stats = {
            'total_tasks_processed': 0,
            'total_execution_time': 0.0,
            'average_task_time': 0.0,
            'quantum_advantage_ratio': 0.0,
            'worker_utilization': 0.0,
            'fault_recovery_count': 0
        }
        
        # Task scheduling
        self.scheduler_running = False
        self.scheduler_thread = None
        self.heartbeat_thread = None
        
    async def optimize_distributed(
        self,
        optimization_tasks: List[OptimizationTask],
        collect_timeout: float = 3600.0
    ) -> List[OptimizationResult]:
        """
        Execute distributed quantum optimization across cluster
        
        Args:
            optimization_tasks: List of tasks to distribute
            collect_timeout: Maximum time to wait for all results
            
        Returns:
            List of optimization results from all workers
        """
        
        logger.info(f"Starting distributed optimization with {len(optimization_tasks)} tasks")
        
        # Initialize cluster if needed
        if not self.workers:
            await self._discover_workers()
        
        # Start scheduler and monitoring
        await self._start_cluster_services()
        
        try:
            # Submit all tasks
            for task in optimization_tasks:
                self._submit_task(task)
            
            # Auto-scaling based on workload
            if self.enable_auto_scaling:
                await self._auto_scale_cluster(len(optimization_tasks))
            
            # Collect results
            results = await self._collect_results(
                expected_count=len(optimization_tasks),
                timeout=collect_timeout
            )
            
            # Update cluster statistics
            self._update_cluster_statistics(results)
            
            return results
            
        finally:
            await self._stop_cluster_services()
    
    async def _discover_workers(self):
        """Discover available worker nodes in the cluster"""
        
        logger.info("Discovering worker nodes...")
        
        # Local workers (simulate multiple local workers)
        local_worker_count = min(8, max(2, self.cluster_config.get('local_workers', 4)))
        
        for i in range(local_worker_count):
            worker = WorkerNode(
                worker_id=f"local_worker_{i}",
                hostname="localhost",
                port=8000 + i,
                capabilities=["quantum_simulation", "classical_optimization"],
                quantum_backend_available=(i < 2),  # First 2 workers have quantum
                cpu_count=2,
                memory_gb=8.0
            )
            
            # Simulate worker registration
            await self._register_worker(worker)
        
        # Distributed workers from config
        if 'remote_workers' in self.cluster_config:
            for worker_config in self.cluster_config['remote_workers']:
                worker = WorkerNode(**worker_config)
                await self._register_worker(worker)
        
        logger.info(f"Discovered {len(self.workers)} worker nodes")
    
    async def _register_worker(self, worker: WorkerNode):
        """Register a worker node with the cluster"""
        
        try:
            # Simulate worker health check
            await self._health_check_worker(worker)
            
            worker.status = WorkerStatus.IDLE
            worker.last_heartbeat = time.time()
            
            self.workers[worker.worker_id] = worker
            
            logger.info(f"Registered worker {worker.worker_id} on {worker.hostname}:{worker.port}")
            
        except Exception as e:
            logger.warning(f"Failed to register worker {worker.worker_id}: {e}")
            worker.status = WorkerStatus.FAILED
    
    async def _health_check_worker(self, worker: WorkerNode):
        """Perform health check on worker node"""
        
        # Simulate network check and capability verification
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # In real implementation, this would check:
        # - Network connectivity
        # - Available memory and CPU
        # - Quantum backend availability
        # - Software dependencies
        
        return True
    
    async def _start_cluster_services(self):
        """Start cluster management services"""
        
        if not self.scheduler_running:
            self.scheduler_running = True
            
            # Start task scheduler
            self.scheduler_thread = threading.Thread(
                target=self._task_scheduler_loop,
                daemon=True
            )
            self.scheduler_thread.start()
            
            # Start heartbeat monitor
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_monitor_loop,
                daemon=True
            )
            self.heartbeat_thread.start()
            
            logger.info("Started cluster services")
    
    async def _stop_cluster_services(self):
        """Stop cluster management services"""
        
        self.scheduler_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5.0)
        
        logger.info("Stopped cluster services")
    
    def _submit_task(self, task: OptimizationTask):
        """Submit optimization task to the queue"""
        
        task.created_at = time.time()
        task.task_id = self._generate_task_id(task)
        
        self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        
        logger.debug(f"Submitted task {task.task_id} with priority {task.priority.name}")
    
    def _generate_task_id(self, task: OptimizationTask) -> str:
        """Generate unique task ID"""
        
        task_data = json.dumps(asdict(task), default=str, sort_keys=True)
        task_hash = hashlib.md5(task_data.encode()).hexdigest()[:12]
        timestamp = int(time.time() * 1000)
        
        return f"task_{timestamp}_{task_hash}"
    
    def _task_scheduler_loop(self):
        """Main task scheduling loop"""
        
        logger.info("Started task scheduler loop")
        
        while self.scheduler_running:
            try:
                # Get next task from queue
                try:
                    task = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Find best worker for task
                worker_id = self._select_best_worker(task)
                
                if worker_id:
                    # Assign task to worker
                    self._assign_task_to_worker(task, worker_id)
                else:
                    # No available workers, requeue task
                    self.task_queue.put(task)
                    time.sleep(0.5)  # Wait before retrying
                
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                time.sleep(1.0)
    
    def _select_best_worker(self, task: OptimizationTask) -> Optional[str]:
        """Select best available worker for the task"""
        
        available_workers = [
            w for w in self.workers.values() 
            if w.status == WorkerStatus.IDLE
        ]
        
        if not available_workers:
            return None
        
        # Score workers based on task requirements
        worker_scores = []
        
        for worker in available_workers:
            score = self._calculate_worker_score(worker, task)
            worker_scores.append((worker.worker_id, score))
        
        # Sort by score (higher is better)
        worker_scores.sort(key=lambda x: x[1], reverse=True)
        
        return worker_scores[0][0] if worker_scores else None
    
    def _calculate_worker_score(self, worker: WorkerNode, task: OptimizationTask) -> float:
        """Calculate worker suitability score for task"""
        
        score = 0.0
        
        # Quantum capability bonus
        if task.quantum_required and worker.quantum_backend_available:
            score += 50.0
        elif task.quantum_required and not worker.quantum_backend_available:
            return -100.0  # Cannot handle quantum tasks
        
        # Load factor penalty
        score -= worker.load_factor * 20.0
        
        # Performance history bonus
        if worker.average_task_time > 0:
            efficiency = max(0.1, 60.0 / worker.average_task_time)  # Faster = better
            score += efficiency * 10.0
        
        # Resource capacity
        estimated_memory_usage = task.estimated_runtime * 0.1  # Rough estimate
        if worker.memory_gb > estimated_memory_usage * 2:
            score += 10.0
        
        # CPU capacity
        score += worker.cpu_count * 5.0
        
        return score
    
    def _assign_task_to_worker(self, task: OptimizationTask, worker_id: str):
        """Assign task to specific worker"""
        
        worker = self.workers[worker_id]
        worker.status = WorkerStatus.BUSY
        worker.current_task_id = task.task_id
        worker.load_factor = min(1.0, worker.load_factor + 0.3)
        
        # Execute task asynchronously
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._execute_task_on_worker, task, worker)
        
        # Handle completion
        def task_completed(fut):
            try:
                result = fut.result()
                self.result_queue.put(result)
                self._release_worker(worker_id, task.task_id)
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed on worker {worker_id}: {e}")
                self._handle_task_failure(task, worker_id, str(e))
        
        future.add_done_callback(task_completed)
        
        logger.debug(f"Assigned task {task.task_id} to worker {worker_id}")
    
    def _execute_task_on_worker(
        self, 
        task: OptimizationTask, 
        worker: WorkerNode
    ) -> OptimizationResult:
        """Execute optimization task on worker node"""
        
        start_time = time.time()
        
        try:
            # Simulate quantum optimization execution
            best_solution, best_value = self._simulate_optimization(task, worker)
            
            execution_time = time.time() - start_time
            
            # Update worker performance statistics
            worker.total_tasks_completed += 1
            if worker.average_task_time == 0:
                worker.average_task_time = execution_time
            else:
                worker.average_task_time = (
                    worker.average_task_time * 0.9 + execution_time * 0.1
                )
            
            result = OptimizationResult(
                task_id=task.task_id,
                worker_id=worker.worker_id,
                best_solution=best_solution,
                best_value=best_value,
                execution_time=execution_time,
                iterations_completed=task.parameters.get('max_iterations', 100),
                convergence_achieved=True,
                quantum_advantage_used=worker.quantum_backend_available and task.quantum_required
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return OptimizationResult(
                task_id=task.task_id,
                worker_id=worker.worker_id,
                best_solution={},
                best_value=float('inf'),
                execution_time=execution_time,
                iterations_completed=0,
                convergence_achieved=False,
                quantum_advantage_used=False,
                error_message=str(e)
            )
    
    def _simulate_optimization(
        self, 
        task: OptimizationTask, 
        worker: WorkerNode
    ) -> Tuple[Dict[str, Any], float]:
        """Simulate optimization execution (replace with actual implementation)"""
        
        # Simulate computational work
        computation_time = min(task.estimated_runtime, 10.0)  # Cap for simulation
        time.sleep(computation_time * 0.01)  # Scaled down for testing
        
        # Generate mock solution
        n_vars = task.problem_data.get('dimension', 10)
        solution = {
            'variables': {
                str(i): np.random.choice([0, 1]) 
                for i in range(n_vars)
            }
        }
        
        # Mock objective value with quantum advantage
        base_value = np.random.uniform(0.1, 1.0)
        if worker.quantum_backend_available and task.quantum_required:
            base_value *= 0.85  # 15% improvement with quantum
        
        return solution, base_value
    
    def _release_worker(self, worker_id: str, task_id: str):
        """Release worker after task completion"""
        
        worker = self.workers[worker_id]
        worker.status = WorkerStatus.IDLE
        worker.current_task_id = None
        worker.load_factor = max(0.0, worker.load_factor - 0.3)
        worker.last_heartbeat = time.time()
        
        # Remove from active tasks
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
    
    def _handle_task_failure(self, task: OptimizationTask, worker_id: str, error: str):
        """Handle task failure with retry logic"""
        
        logger.warning(f"Task {task.task_id} failed on worker {worker_id}: {error}")
        
        # Release worker
        self._release_worker(worker_id, task.task_id)
        
        # Mark worker as failed if multiple failures
        worker = self.workers[worker_id]
        if "network" in error.lower() or "timeout" in error.lower():
            worker.status = WorkerStatus.FAILED
        
        # Retry task if retries remaining
        task.max_retries -= 1
        if task.max_retries > 0:
            logger.info(f"Retrying task {task.task_id}, {task.max_retries} retries remaining")
            self.task_queue.put(task)
        else:
            # Task failed permanently
            failed_result = OptimizationResult(
                task_id=task.task_id,
                worker_id=worker_id,
                best_solution={},
                best_value=float('inf'),
                execution_time=0.0,
                iterations_completed=0,
                convergence_achieved=False,
                quantum_advantage_used=False,
                error_message=f"Task failed after max retries: {error}"
            )
            self.result_queue.put(failed_result)
    
    def _heartbeat_monitor_loop(self):
        """Monitor worker heartbeats and health"""
        
        logger.info("Started heartbeat monitor")
        
        while self.scheduler_running:
            try:
                current_time = time.time()
                
                for worker in self.workers.values():
                    # Check for stale workers
                    if current_time - worker.last_heartbeat > 60.0:  # 1 minute timeout
                        if worker.status != WorkerStatus.OFFLINE:
                            logger.warning(f"Worker {worker.worker_id} appears offline")
                            worker.status = WorkerStatus.OFFLINE
                            
                            # Handle any active task on this worker
                            if worker.current_task_id and worker.current_task_id in self.active_tasks:
                                task = self.active_tasks[worker.current_task_id]
                                self._handle_task_failure(task, worker.worker_id, "Worker timeout")
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                time.sleep(10.0)
    
    async def _auto_scale_cluster(self, task_count: int):
        """Automatically scale cluster based on workload"""
        
        if not self.enable_auto_scaling:
            return
        
        current_workers = len([w for w in self.workers.values() 
                             if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]])
        
        # Calculate desired worker count
        desired_workers = min(self.max_workers, max(2, task_count // 5))
        
        if desired_workers > current_workers:
            # Scale up
            new_workers_needed = desired_workers - current_workers
            logger.info(f"Scaling up cluster: adding {new_workers_needed} workers")
            
            for i in range(new_workers_needed):
                worker_id = f"autoscale_worker_{int(time.time())}_{i}"
                worker = WorkerNode(
                    worker_id=worker_id,
                    hostname="localhost",
                    port=9000 + i,
                    capabilities=["classical_optimization"],
                    quantum_backend_available=False,  # Basic workers
                    cpu_count=1,
                    memory_gb=4.0
                )
                await self._register_worker(worker)
        
        elif desired_workers < current_workers * 0.5:
            # Scale down (only if significantly over-provisioned)
            workers_to_remove = current_workers - desired_workers
            logger.info(f"Scaling down cluster: removing {workers_to_remove} workers")
            
            # Remove idle autoscale workers first
            idle_autoscale_workers = [
                w for w in self.workers.values()
                if w.status == WorkerStatus.IDLE and "autoscale" in w.worker_id
            ][:workers_to_remove]
            
            for worker in idle_autoscale_workers:
                worker.status = WorkerStatus.OFFLINE
                logger.info(f"Removed autoscale worker {worker.worker_id}")
    
    async def _collect_results(
        self, 
        expected_count: int, 
        timeout: float
    ) -> List[OptimizationResult]:
        """Collect optimization results from workers"""
        
        results = []
        start_time = time.time()
        
        logger.info(f"Collecting {expected_count} results with {timeout}s timeout")
        
        while len(results) < expected_count and (time.time() - start_time) < timeout:
            try:
                result = self.result_queue.get(timeout=1.0)
                results.append(result)
                
                logger.debug(f"Collected result {len(results)}/{expected_count} "
                           f"from worker {result.worker_id}")
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
        
        if len(results) < expected_count:
            logger.warning(f"Collected only {len(results)}/{expected_count} results "
                          f"within {timeout}s timeout")
        
        return results
    
    def _update_cluster_statistics(self, results: List[OptimizationResult]):
        """Update cluster performance statistics"""
        
        if not results:
            return
        
        self.cluster_stats['total_tasks_processed'] += len(results)
        
        total_time = sum(r.execution_time for r in results)
        self.cluster_stats['total_execution_time'] += total_time
        
        if len(results) > 0:
            avg_time = total_time / len(results)
            if self.cluster_stats['average_task_time'] == 0:
                self.cluster_stats['average_task_time'] = avg_time
            else:
                self.cluster_stats['average_task_time'] = (
                    self.cluster_stats['average_task_time'] * 0.9 + avg_time * 0.1
                )
        
        # Quantum advantage ratio
        quantum_results = sum(1 for r in results if r.quantum_advantage_used)
        if len(results) > 0:
            quantum_ratio = quantum_results / len(results)
            self.cluster_stats['quantum_advantage_ratio'] = quantum_ratio
        
        # Worker utilization
        active_workers = len([w for w in self.workers.values() 
                            if w.status == WorkerStatus.BUSY])
        total_workers = len([w for w in self.workers.values() 
                           if w.status != WorkerStatus.OFFLINE])
        
        if total_workers > 0:
            self.cluster_stats['worker_utilization'] = active_workers / total_workers
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        
        worker_summary = {}
        for status in WorkerStatus:
            worker_summary[status.value] = len([
                w for w in self.workers.values() if w.status == status
            ])
        
        return {
            'cluster_stats': self.cluster_stats.copy(),
            'worker_summary': worker_summary,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'total_workers': len(self.workers),
            'quantum_capable_workers': len([
                w for w in self.workers.values() if w.quantum_backend_available
            ])
        }