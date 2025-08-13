"""
Enterprise Scaling - Advanced scaling and performance optimization for quantum systems.

Provides distributed computing, auto-scaling, load balancing, and resource management
for enterprise-scale quantum hyperparameter optimization.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import json
import logging
import psutil
import queue

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    disk_io: Dict[str, int]
    network_io: Dict[str, int]
    timestamp: float
    
    @classmethod
    def current(cls) -> 'ResourceUsage':
        """Get current resource usage."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent
            disk_io = psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
            net_io = psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            
            return cls(
                cpu_percent=cpu,
                memory_percent=memory,
                disk_io=disk_io,
                network_io=net_io,
                timestamp=time.time()
            )
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {e}")
            return cls(0, 0, {}, {}, time.time())


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_workers: int = 1
    max_workers: int = mp.cpu_count()
    target_cpu_utilization: float = 0.75
    target_memory_utilization: float = 0.80
    scale_up_threshold: float = 0.85
    scale_down_threshold: float = 0.50
    scale_cooldown: float = 60.0  # seconds
    evaluation_window: int = 5  # number of metrics to evaluate


class AdaptiveResourceManager:
    """Manages computational resources adaptively based on load and performance."""
    
    def __init__(self, scaling_policy: Optional[ScalingPolicy] = None):
        self.policy = scaling_policy or ScalingPolicy()
        self.current_workers = self.policy.min_workers
        self.resource_history = deque(maxlen=self.policy.evaluation_window * 2)
        self.last_scale_time = 0
        self.performance_metrics = deque(maxlen=100)
        self._lock = threading.Lock()
        
        # Resource monitoring thread
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
    
    def start_monitoring(self):
        """Start resource monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        def monitor_resources():
            while not self._stop_monitoring.wait(1.0):
                try:
                    usage = ResourceUsage.current()
                    with self._lock:
                        self.resource_history.append(usage)
                        self._evaluate_scaling()
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self._monitor_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring thread."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("Stopped resource monitoring")
    
    def _evaluate_scaling(self):
        """Evaluate whether to scale up or down."""
        if len(self.resource_history) < self.policy.evaluation_window:
            return
        
        # Check cooldown
        if time.time() - self.last_scale_time < self.policy.scale_cooldown:
            return
        
        # Calculate average resource usage
        recent_usage = list(self.resource_history)[-self.policy.evaluation_window:]
        avg_cpu = np.mean([r.cpu_percent for r in recent_usage]) / 100.0
        avg_memory = np.mean([r.memory_percent for r in recent_usage]) / 100.0
        
        # Determine scaling action
        should_scale_up = (
            (avg_cpu > self.policy.scale_up_threshold or 
             avg_memory > self.policy.scale_up_threshold) and
            self.current_workers < self.policy.max_workers
        )
        
        should_scale_down = (
            avg_cpu < self.policy.scale_down_threshold and
            avg_memory < self.policy.scale_down_threshold and
            self.current_workers > self.policy.min_workers
        )
        
        if should_scale_up:
            self._scale_up()
        elif should_scale_down:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up worker count."""
        new_workers = min(self.current_workers + 1, self.policy.max_workers)
        if new_workers > self.current_workers:
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            logger.info(f"Scaled up to {self.current_workers} workers")
    
    def _scale_down(self):
        """Scale down worker count."""
        new_workers = max(self.current_workers - 1, self.policy.min_workers)
        if new_workers < self.current_workers:
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            logger.info(f"Scaled down to {self.current_workers} workers")
    
    def get_optimal_worker_count(self) -> int:
        """Get optimal worker count based on current conditions."""
        return self.current_workers
    
    def record_performance(self, metric_name: str, value: float, worker_count: int):
        """Record performance metric for scaling decisions."""
        metric = {
            'name': metric_name,
            'value': value,
            'worker_count': worker_count,
            'timestamp': time.time()
        }
        
        with self._lock:
            self.performance_metrics.append(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for analysis."""
        if not self.performance_metrics:
            return {}
        
        with self._lock:
            metrics = list(self.performance_metrics)
        
        throughput_metrics = [m for m in metrics if m['name'] == 'throughput']
        if throughput_metrics:
            throughputs = [m['value'] for m in throughput_metrics]
            return {
                'avg_throughput': np.mean(throughputs),
                'max_throughput': np.max(throughputs),
                'current_workers': self.current_workers,
                'total_metrics': len(metrics)
            }
        
        return {'current_workers': self.current_workers, 'total_metrics': len(metrics)}


class DistributedOptimizer:
    """Distributed optimization coordinator for large-scale problems."""
    
    def __init__(self, max_parallel_jobs: int = None):
        self.max_parallel_jobs = max_parallel_jobs or mp.cpu_count()
        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.is_running = False
        
    def start_workers(self, worker_count: int):
        """Start distributed worker processes."""
        if self.is_running:
            return
        
        self.workers = []
        self.is_running = True
        
        for i in range(worker_count):
            worker = mp.Process(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {worker_count} distributed workers")
    
    def stop_workers(self):
        """Stop all worker processes."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Signal workers to stop
        for _ in self.workers:
            self.job_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        
        self.workers = []
        logger.info("Stopped all distributed workers")
    
    def _worker_loop(self, worker_id: int):
        """Worker process main loop."""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                job = self.job_queue.get(timeout=1)
                if job is None:  # Stop signal
                    break
                
                func, args, kwargs = job
                result = func(*args, **kwargs)
                self.result_queue.put(('success', result))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put(('error', str(e)))
        
        logger.info(f"Worker {worker_id} stopped")
    
    def submit_job(self, func: Callable, *args, **kwargs):
        """Submit a job for distributed execution."""
        if not self.is_running:
            raise RuntimeError("Workers not started")
        
        self.job_queue.put((func, args, kwargs))
    
    def get_results(self, count: int, timeout: float = None) -> List[Any]:
        """Get results from completed jobs."""
        results = []
        deadline = time.time() + (timeout or float('inf'))
        
        while len(results) < count and time.time() < deadline:
            try:
                remaining_timeout = max(0, deadline - time.time())
                status, result = self.result_queue.get(timeout=remaining_timeout)
                
                if status == 'success':
                    results.append(result)
                else:
                    logger.error(f"Job failed: {result}")
                    
            except queue.Empty:
                break
        
        return results


class LoadBalancer:
    """Load balancer for distributing optimization tasks."""
    
    def __init__(self, backends: List[str]):
        self.backends = backends
        self.backend_loads = defaultdict(int)
        self.backend_performance = defaultdict(list)
        self._lock = threading.Lock()
    
    def select_backend(self) -> str:
        """Select optimal backend based on load and performance."""
        with self._lock:
            if not self.backends:
                raise ValueError("No backends available")
            
            # For now, use round-robin with load awareness
            min_load = min(self.backend_loads[b] for b in self.backends)
            candidates = [b for b in self.backends if self.backend_loads[b] == min_load]
            
            # Among equal-load backends, prefer better performing ones
            if len(candidates) > 1 and any(self.backend_performance[b] for b in candidates):
                best_backend = max(
                    candidates,
                    key=lambda b: np.mean(self.backend_performance[b]) if self.backend_performance[b] else 0
                )
                return best_backend
            
            return candidates[0]
    
    def record_usage(self, backend: str, start_load: bool = True):
        """Record backend usage."""
        with self._lock:
            if start_load:
                self.backend_loads[backend] += 1
            else:
                self.backend_loads[backend] = max(0, self.backend_loads[backend] - 1)
    
    def record_performance(self, backend: str, performance_score: float):
        """Record backend performance."""
        with self._lock:
            self.backend_performance[backend].append(performance_score)
            # Keep only recent performance data
            if len(self.backend_performance[backend]) > 100:
                self.backend_performance[backend] = self.backend_performance[backend][-50:]
    
    def get_load_summary(self) -> Dict[str, Any]:
        """Get load balancing summary."""
        with self._lock:
            return {
                'backend_loads': dict(self.backend_loads),
                'backend_performance': {
                    k: {
                        'count': len(v),
                        'avg': np.mean(v) if v else 0,
                        'recent': v[-10:] if v else []
                    }
                    for k, v in self.backend_performance.items()
                }
            }


class PerformanceOptimizer:
    """Advanced performance optimization and tuning."""
    
    def __init__(self):
        self.optimization_history = defaultdict(list)
        self.performance_patterns = defaultdict(dict)
        self._lock = threading.Lock()
    
    def optimize_batch_size(self, current_size: int, performance_history: List[float]) -> int:
        """Optimize batch size based on performance history."""
        if len(performance_history) < 3:
            return current_size
        
        recent_performance = np.mean(performance_history[-3:])
        older_performance = np.mean(performance_history[-6:-3]) if len(performance_history) >= 6 else recent_performance
        
        if recent_performance > older_performance * 1.1:
            # Performance improving, try larger batch
            return min(current_size * 2, 1000)
        elif recent_performance < older_performance * 0.9:
            # Performance degrading, try smaller batch
            return max(current_size // 2, 1)
        else:
            # Performance stable, keep current size
            return current_size
    
    def optimize_quantum_reads(self, current_reads: int, accuracy_trend: List[float]) -> int:
        """Optimize quantum reads based on accuracy trends."""
        if len(accuracy_trend) < 2:
            return current_reads
        
        accuracy_improvement = accuracy_trend[-1] - accuracy_trend[-2]
        
        if accuracy_improvement > 0.01:  # Good improvement
            return current_reads  # Keep current reads
        elif accuracy_improvement < -0.01:  # Degrading
            return min(current_reads * 2, 10000)  # Increase reads for better sampling
        else:  # Plateau
            return max(current_reads // 2, 10)  # Reduce reads for efficiency
    
    def suggest_parallel_strategy(self, problem_size: int, resource_usage: ResourceUsage) -> Dict[str, Any]:
        """Suggest optimal parallelization strategy."""
        cpu_cores = mp.cpu_count()
        
        # Base strategy on problem size and resource availability
        if problem_size < 50:
            # Small problems: minimal parallelization
            return {
                'strategy': 'sequential',
                'workers': 1,
                'batch_size': problem_size
            }
        elif problem_size < 500:
            # Medium problems: thread-based parallelization
            optimal_workers = min(4, cpu_cores // 2)
            return {
                'strategy': 'threaded',
                'workers': optimal_workers,
                'batch_size': max(1, problem_size // optimal_workers)
            }
        else:
            # Large problems: process-based parallelization
            if resource_usage.memory_percent > 80:
                # Memory constrained
                optimal_workers = max(1, cpu_cores // 4)
            else:
                optimal_workers = min(cpu_cores, problem_size // 10)
            
            return {
                'strategy': 'process_based',
                'workers': optimal_workers,
                'batch_size': max(1, problem_size // optimal_workers)
            }
    
    def record_optimization_result(self, config: Dict[str, Any], performance: float):
        """Record optimization result for learning."""
        with self._lock:
            config_key = str(sorted(config.items()))
            self.optimization_history[config_key].append({
                'performance': performance,
                'timestamp': time.time()
            })
    
    def get_best_configuration(self, similar_configs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get best configuration based on historical performance."""
        with self._lock:
            best_config = None
            best_performance = 0
            
            for config in similar_configs:
                config_key = str(sorted(config.items()))
                if config_key in self.optimization_history:
                    avg_performance = np.mean([
                        r['performance'] for r in self.optimization_history[config_key]
                    ])
                    if avg_performance > best_performance:
                        best_performance = avg_performance
                        best_config = config
            
            return best_config


class EnterpriseScalingManager:
    """Main enterprise scaling coordinator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.resource_manager = AdaptiveResourceManager(
            ScalingPolicy(**config.get('scaling_policy', {}))
        )
        
        self.performance_optimizer = PerformanceOptimizer()
        
        # Available backends for load balancing
        backends = config.get('backends', ['simple', 'simulator'])
        self.load_balancer = LoadBalancer(backends)
        
        # Distributed optimizer
        self.distributed_optimizer = DistributedOptimizer(
            max_parallel_jobs=config.get('max_parallel_jobs')
        )
        
        self.is_started = False
    
    def start(self):
        """Start enterprise scaling services."""
        if self.is_started:
            return
        
        self.resource_manager.start_monitoring()
        
        # Start distributed workers
        optimal_workers = self.resource_manager.get_optimal_worker_count()
        self.distributed_optimizer.start_workers(optimal_workers)
        
        self.is_started = True
        logger.info("Enterprise scaling manager started")
    
    def stop(self):
        """Stop enterprise scaling services."""
        if not self.is_started:
            return
        
        self.resource_manager.stop_monitoring()
        self.distributed_optimizer.stop_workers()
        
        self.is_started = False
        logger.info("Enterprise scaling manager stopped")
    
    def optimize_configuration(self, problem_size: int) -> Dict[str, Any]:
        """Get optimized configuration for a given problem size."""
        resource_usage = ResourceUsage.current()
        
        strategy = self.performance_optimizer.suggest_parallel_strategy(
            problem_size, resource_usage
        )
        
        backend = self.load_balancer.select_backend()
        
        return {
            'backend': backend,
            'parallel_strategy': strategy,
            'optimal_workers': self.resource_manager.get_optimal_worker_count(),
            'resource_usage': resource_usage
        }
    
    def execute_distributed_optimization(self, tasks: List[Callable], timeout: float = None) -> List[Any]:
        """Execute optimization tasks in distributed fashion."""
        if not self.is_started:
            self.start()
        
        # Submit all tasks
        for task in tasks:
            self.distributed_optimizer.submit_job(task)
        
        # Collect results
        results = self.distributed_optimizer.get_results(len(tasks), timeout)
        
        return results
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get comprehensive scaling summary."""
        return {
            'resource_manager': {
                'current_workers': self.resource_manager.current_workers,
                'performance_summary': self.resource_manager.get_performance_summary()
            },
            'load_balancer': self.load_balancer.get_load_summary(),
            'distributed_optimizer': {
                'is_running': self.distributed_optimizer.is_running,
                'worker_count': len(self.distributed_optimizer.workers)
            },
            'current_resource_usage': ResourceUsage.current().__dict__
        }


# Global enterprise scaling manager
enterprise_scaling_manager = EnterpriseScalingManager()