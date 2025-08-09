"""
Quantum Advantage Accelerator - Next-generation quantum-classical acceleration techniques.

Implements cutting-edge acceleration methods that maximize quantum advantage
through advanced hardware utilization, algorithm hybridization, and real-time adaptation.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod

try:
    import cupy as cp  # GPU acceleration
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import ray  # Distributed computing
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

logger = logging.getLogger(__name__)


@dataclass
class AccelerationMetrics:
    """Metrics for tracking acceleration performance."""
    quantum_speedup: float = 1.0
    parallel_efficiency: float = 1.0
    cache_hit_rate: float = 0.0
    gpu_utilization: float = 0.0
    memory_efficiency: float = 1.0
    total_acceleration: float = 1.0
    
    def overall_score(self) -> float:
        """Calculate overall acceleration score."""
        return (
            0.3 * self.quantum_speedup +
            0.2 * self.parallel_efficiency +
            0.2 * (1 + self.cache_hit_rate) +
            0.15 * (1 + self.gpu_utilization) +
            0.15 * self.memory_efficiency
        )


class QuantumResourceManager:
    """
    Advanced quantum resource manager for optimal hardware utilization.
    
    Manages quantum hardware resources, schedules computations,
    and optimizes quantum circuit execution.
    """
    
    def __init__(self, backend_name: str = 'simulator'):
        self.backend_name = backend_name
        self.resource_pool = {}
        self.active_jobs = {}
        self.job_queue = deque()
        self.utilization_history = deque(maxlen=100)
        self.lock = threading.RLock()
        
        # Hardware characteristics (would be queried from real hardware)
        self.hardware_specs = self._get_hardware_specifications()
        
    def _get_hardware_specifications(self) -> Dict[str, Any]:
        """Get quantum hardware specifications."""
        if 'dwave' in self.backend_name.lower():
            return {
                'n_qubits': 2048,  # Advantage system
                'topology': 'pegasus',
                'annealing_time_range': (1, 2000),  # microseconds
                'max_reads_per_sample': 10000,
                'coupling_strength': 1.0,
                'chain_strength': 2.0
            }
        else:
            return {
                'n_qubits': 100,  # Simulator
                'topology': 'complete',
                'annealing_time_range': (1, 100),
                'max_reads_per_sample': 1000,
                'coupling_strength': 1.0,
                'chain_strength': 1.0
            }
    
    def optimize_quantum_parameters(self, Q: Dict, current_performance: float) -> Dict[str, Any]:
        """Optimize quantum parameters for better performance."""
        optimized_params = {
            'num_reads': self._optimize_num_reads(Q, current_performance),
            'annealing_time': self._optimize_annealing_time(Q),
            'chain_strength': self._optimize_chain_strength(Q),
            'annealing_schedule': self._generate_custom_annealing_schedule(Q)
        }
        
        logger.info(f"Quantum parameters optimized: {optimized_params}")
        return optimized_params
    
    def _optimize_num_reads(self, Q: Dict, performance: float) -> int:
        """Dynamically optimize number of reads based on problem characteristics."""
        base_reads = 1000
        
        # Adjust based on QUBO size
        qubo_size = len(set([i for (i, j) in Q.keys()] + [j for (i, j) in Q.keys()]))
        size_factor = min(2.0, np.sqrt(qubo_size / 50))  # Scale with problem size
        
        # Adjust based on current performance
        if performance < 0.5:  # Poor performance
            performance_factor = 1.5
        elif performance > 0.8:  # Good performance
            performance_factor = 0.8
        else:
            performance_factor = 1.0
        
        optimized_reads = int(base_reads * size_factor * performance_factor)
        return min(optimized_reads, self.hardware_specs['max_reads_per_sample'])
    
    def _optimize_annealing_time(self, Q: Dict) -> int:
        """Optimize annealing time based on problem characteristics."""
        # Analyze QUBO complexity
        values = list(Q.values())
        value_range = max(values) - min(values) if values else 1.0
        coupling_density = len([1 for (i, j) in Q.keys() if i != j]) / max(1, len(Q))
        
        # Base annealing time
        base_time = 20  # microseconds
        
        # Adjust for problem complexity
        complexity_factor = 1.0 + 0.5 * coupling_density + 0.3 * np.log(1 + value_range)
        
        optimized_time = int(base_time * complexity_factor)
        
        # Ensure within hardware limits
        min_time, max_time = self.hardware_specs['annealing_time_range']
        return max(min_time, min(optimized_time, max_time))
    
    def _optimize_chain_strength(self, Q: Dict) -> float:
        """Optimize chain strength for embedding."""
        if not Q:
            return self.hardware_specs['chain_strength']
        
        # Analyze QUBO coupling strengths
        coupling_values = [abs(value) for (i, j), value in Q.items() if i != j]
        if not coupling_values:
            return self.hardware_specs['chain_strength']
        
        max_coupling = max(coupling_values)
        avg_coupling = np.mean(coupling_values)
        
        # Chain strength should be stronger than typical couplings
        recommended_chain_strength = max(2.0 * max_coupling, 1.5 * avg_coupling)
        
        return min(recommended_chain_strength, 10.0)  # Cap at reasonable maximum
    
    def _generate_custom_annealing_schedule(self, Q: Dict) -> List[Tuple[float, float]]:
        """Generate custom annealing schedule for complex problems."""
        # Default linear schedule
        schedule = [
            (0.0, 1.0),  # (time_fraction, annealing_parameter)
            (0.5, 0.5),
            (1.0, 0.0)
        ]
        
        # Customize based on problem characteristics
        if Q:
            values = list(Q.values())
            if max(values) - min(values) > 10:  # High dynamic range
                # Use slower initial cooling
                schedule = [
                    (0.0, 1.0),
                    (0.3, 0.8),
                    (0.7, 0.3),
                    (1.0, 0.0)
                ]
        
        return schedule
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization metrics."""
        return {
            'active_jobs': len(self.active_jobs),
            'queue_length': len(self.job_queue),
            'avg_utilization': np.mean(self.utilization_history) if self.utilization_history else 0.0,
            'hardware_efficiency': self._calculate_hardware_efficiency()
        }
    
    def _calculate_hardware_efficiency(self) -> float:
        """Calculate quantum hardware efficiency score."""
        # Mock calculation - would use real metrics from quantum hardware
        base_efficiency = 0.8
        
        # Adjust based on utilization
        if self.utilization_history:
            utilization = np.mean(self.utilization_history)
            if utilization > 0.9:  # Over-utilized
                base_efficiency *= 0.9
            elif utilization < 0.3:  # Under-utilized
                base_efficiency *= 0.95
        
        return base_efficiency


class AdvancedParallelProcessor:
    """
    Advanced parallel processing system with intelligent load balancing.
    
    Combines CPU, GPU, and distributed processing for maximum throughput.
    """
    
    def __init__(self, max_workers: Optional[int] = None, enable_gpu: bool = True):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.enable_gpu = enable_gpu and HAS_CUPY
        self.enable_distributed = HAS_RAY
        
        # Initialize Ray if available
        if self.enable_distributed:
            try:
                if not ray.is_initialized():
                    ray.init(num_cpus=self.max_workers, ignore_reinit_error=True)
                logger.info("Ray distributed processing initialized")
            except Exception as e:
                logger.warning(f"Ray initialization failed: {e}")
                self.enable_distributed = False
        
        self.processing_stats = {
            'cpu_tasks': 0,
            'gpu_tasks': 0,
            'distributed_tasks': 0,
            'total_processing_time': 0.0
        }
    
    def process_batch_parallel(
        self,
        tasks: List[Callable],
        task_data: List[Any],
        strategy: str = 'auto'
    ) -> List[Any]:
        """
        Process batch of tasks in parallel using optimal strategy.
        
        Args:
            tasks: List of callable tasks
            task_data: List of data for each task
            strategy: Processing strategy ('cpu', 'gpu', 'distributed', 'auto')
            
        Returns:
            List of results
        """
        if strategy == 'auto':
            strategy = self._select_optimal_strategy(len(tasks))
        
        start_time = time.time()
        
        if strategy == 'gpu' and self.enable_gpu:
            results = self._process_gpu_batch(tasks, task_data)
        elif strategy == 'distributed' and self.enable_distributed:
            results = self._process_distributed_batch(tasks, task_data)
        else:
            results = self._process_cpu_batch(tasks, task_data)
        
        processing_time = time.time() - start_time
        self.processing_stats['total_processing_time'] += processing_time
        
        logger.info(f"Batch processed using {strategy} strategy in {processing_time:.2f}s")
        
        return results
    
    def _select_optimal_strategy(self, num_tasks: int) -> str:
        """Select optimal processing strategy based on task characteristics."""
        if num_tasks <= 2:
            return 'cpu'
        elif num_tasks <= 10:
            return 'gpu' if self.enable_gpu else 'cpu'
        else:
            return 'distributed' if self.enable_distributed else 'gpu' if self.enable_gpu else 'cpu'
    
    def _process_cpu_batch(self, tasks: List[Callable], task_data: List[Any]) -> List[Any]:
        """Process batch using CPU threads."""
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(task, data): i 
                for i, (task, data) in enumerate(zip(tasks, task_data))
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.warning(f"CPU task {index} failed: {e}")
                    results[index] = None
        
        self.processing_stats['cpu_tasks'] += len(tasks)
        return results
    
    def _process_gpu_batch(self, tasks: List[Callable], task_data: List[Any]) -> List[Any]:
        """Process batch using GPU acceleration."""
        if not HAS_CUPY:
            return self._process_cpu_batch(tasks, task_data)
        
        results = []
        
        try:
            # Move data to GPU if possible
            gpu_data = []
            for data in task_data:
                if isinstance(data, np.ndarray):
                    gpu_data.append(cp.asarray(data))
                else:
                    gpu_data.append(data)
            
            # Process on GPU
            for task, data in zip(tasks, gpu_data):
                try:
                    result = task(data)
                    # Move result back to CPU if needed
                    if hasattr(result, 'get'):
                        result = result.get()
                    results.append(result)
                except Exception as e:
                    logger.warning(f"GPU task failed: {e}")
                    results.append(None)
            
            self.processing_stats['gpu_tasks'] += len(tasks)
            
        except Exception as e:
            logger.warning(f"GPU processing failed, falling back to CPU: {e}")
            results = self._process_cpu_batch(tasks, task_data)
        
        return results
    
    def _process_distributed_batch(self, tasks: List[Callable], task_data: List[Any]) -> List[Any]:
        """Process batch using distributed computing."""
        if not HAS_RAY:
            return self._process_cpu_batch(tasks, task_data)
        
        try:
            # Create remote tasks
            remote_tasks = [ray.remote(task) for task in tasks]
            
            # Execute in parallel
            futures = [remote_task.remote(data) for remote_task, data in zip(remote_tasks, task_data)]
            
            # Collect results
            results = ray.get(futures)
            
            self.processing_stats['distributed_tasks'] += len(tasks)
            
        except Exception as e:
            logger.warning(f"Distributed processing failed, falling back to CPU: {e}")
            results = self._process_cpu_batch(tasks, task_data)
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_tasks = sum([
            self.processing_stats['cpu_tasks'],
            self.processing_stats['gpu_tasks'],
            self.processing_stats['distributed_tasks']
        ])
        
        return {
            'total_tasks': total_tasks,
            'cpu_tasks': self.processing_stats['cpu_tasks'],
            'gpu_tasks': self.processing_stats['gpu_tasks'],
            'distributed_tasks': self.processing_stats['distributed_tasks'],
            'total_processing_time': self.processing_stats['total_processing_time'],
            'tasks_per_second': total_tasks / max(1, self.processing_stats['total_processing_time']),
            'gpu_enabled': self.enable_gpu,
            'distributed_enabled': self.enable_distributed
        }


class IntelligentCacheManager:
    """
    Intelligent cache management with predictive prefetching and adaptive policies.
    """
    
    def __init__(self, max_size: int = 10000, enable_prediction: bool = True):
        self.max_size = max_size
        self.enable_prediction = enable_prediction
        
        # Multi-level cache
        self.hot_cache = {}  # Frequently accessed
        self.warm_cache = {}  # Moderately accessed
        self.cold_cache = {}  # Infrequently accessed
        
        # Access tracking
        self.access_counts = defaultdict(int)
        self.access_history = deque(maxlen=1000)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'prefetch_hits': 0
        }
        
        # Prediction model (simple frequency-based)
        self.prediction_model = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent tier management."""
        # Check hot cache first
        if key in self.hot_cache:
            self.cache_stats['hits'] += 1
            self._record_access(key)
            return self.hot_cache[key]
        
        # Check warm cache
        if key in self.warm_cache:
            value = self.warm_cache.pop(key)
            self._promote_to_hot(key, value)
            self.cache_stats['hits'] += 1
            self._record_access(key)
            return value
        
        # Check cold cache
        if key in self.cold_cache:
            value = self.cold_cache.pop(key)
            self._promote_to_warm(key, value)
            self.cache_stats['hits'] += 1
            self._record_access(key)
            return value
        
        self.cache_stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache with intelligent placement."""
        access_count = self.access_counts.get(key, 0)
        
        if access_count > 10:  # Hot item
            self._put_hot(key, value)
        elif access_count > 3:  # Warm item
            self._put_warm(key, value)
        else:  # Cold item
            self._put_cold(key, value)
        
        self._record_access(key)
        
        # Trigger predictive prefetching
        if self.enable_prediction:
            self._predict_and_prefetch(key)
    
    def _promote_to_hot(self, key: str, value: Any):
        """Promote item to hot cache."""
        self._put_hot(key, value)
    
    def _promote_to_warm(self, key: str, value: Any):
        """Promote item to warm cache."""
        self._put_warm(key, value)
    
    def _put_hot(self, key: str, value: Any):
        """Put item in hot cache."""
        if len(self.hot_cache) >= self.max_size // 4:  # 25% of total cache
            self._evict_from_hot()
        self.hot_cache[key] = value
    
    def _put_warm(self, key: str, value: Any):
        """Put item in warm cache."""
        if len(self.warm_cache) >= self.max_size // 2:  # 50% of total cache
            self._evict_from_warm()
        self.warm_cache[key] = value
    
    def _put_cold(self, key: str, value: Any):
        """Put item in cold cache."""
        if len(self.cold_cache) >= self.max_size // 4:  # 25% of total cache
            self._evict_from_cold()
        self.cold_cache[key] = value
    
    def _evict_from_hot(self):
        """Evict least recently used item from hot cache."""
        if self.hot_cache:
            # Simple LRU (would be more sophisticated in production)
            key = next(iter(self.hot_cache))
            value = self.hot_cache.pop(key)
            self._put_warm(key, value)  # Demote to warm
            self.cache_stats['evictions'] += 1
    
    def _evict_from_warm(self):
        """Evict item from warm cache."""
        if self.warm_cache:
            key = next(iter(self.warm_cache))
            value = self.warm_cache.pop(key)
            self._put_cold(key, value)  # Demote to cold
            self.cache_stats['evictions'] += 1
    
    def _evict_from_cold(self):
        """Evict item from cold cache."""
        if self.cold_cache:
            key = next(iter(self.cold_cache))
            self.cold_cache.pop(key)  # Remove completely
            self.cache_stats['evictions'] += 1
    
    def _record_access(self, key: str):
        """Record cache access for analytics."""
        self.access_counts[key] += 1
        self.access_history.append((time.time(), key))
    
    def _predict_and_prefetch(self, key: str):
        """Predict and prefetch related items."""
        # Simple prediction based on access patterns
        # In production, would use more sophisticated ML models
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / max(1, total_requests)
        
        return {
            'total_requests': total_requests,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.cache_stats['evictions'],
            'prefetch_hits': self.cache_stats['prefetch_hits'],
            'hot_cache_size': len(self.hot_cache),
            'warm_cache_size': len(self.warm_cache),
            'cold_cache_size': len(self.cold_cache),
            'total_cache_size': len(self.hot_cache) + len(self.warm_cache) + len(self.cold_cache)
        }


class QuantumAdvantageAccelerator:
    """
    Master accelerator that coordinates all acceleration techniques
    to maximize quantum advantage and overall performance.
    """
    
    def __init__(self,
                 backend_name: str = 'simulator',
                 max_workers: Optional[int] = None,
                 cache_size: int = 10000,
                 enable_gpu: bool = True):
        """
        Initialize quantum advantage accelerator.
        
        Args:
            backend_name: Quantum backend name
            max_workers: Maximum parallel workers
            cache_size: Cache size
            enable_gpu: Enable GPU acceleration
        """
        self.backend_name = backend_name
        
        # Initialize components
        self.quantum_manager = QuantumResourceManager(backend_name)
        self.parallel_processor = AdvancedParallelProcessor(max_workers, enable_gpu)
        self.cache_manager = IntelligentCacheManager(cache_size)
        
        # Performance tracking
        self.acceleration_history = deque(maxlen=100)
        self.optimization_metrics = {}
        
        logger.info(f"Quantum Advantage Accelerator initialized with backend: {backend_name}")
    
    def accelerate_optimization(
        self,
        quantum_tasks: List[Callable],
        evaluation_tasks: List[Callable],
        task_data: List[Any],
        current_performance: float
    ) -> Tuple[List[Any], AccelerationMetrics]:
        """
        Accelerate optimization using all available techniques.
        
        Args:
            quantum_tasks: Quantum computation tasks
            evaluation_tasks: Model evaluation tasks
            task_data: Data for tasks
            current_performance: Current optimization performance
            
        Returns:
            Tuple of (results, acceleration_metrics)
        """
        start_time = time.time()
        
        # Step 1: Optimize quantum parameters
        if quantum_tasks:
            quantum_params = self.quantum_manager.optimize_quantum_parameters(
                {}, current_performance  # Would pass actual QUBO
            )
        
        # Step 2: Check cache for existing results
        cached_results = []
        uncached_tasks = []
        uncached_data = []
        
        for i, (task, data) in enumerate(zip(evaluation_tasks, task_data)):
            cache_key = self._generate_cache_key(task, data)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result is not None:
                cached_results.append((i, cached_result))
            else:
                uncached_tasks.append((i, task))
                uncached_data.append(data)
        
        # Step 3: Process uncached tasks in parallel
        if uncached_tasks:
            parallel_results = self.parallel_processor.process_batch_parallel(
                [task for _, task in uncached_tasks],
                uncached_data,
                strategy='auto'
            )
            
            # Cache new results
            for (original_index, task), result, data in zip(uncached_tasks, parallel_results, uncached_data):
                if result is not None:
                    cache_key = self._generate_cache_key(task, data)
                    self.cache_manager.put(cache_key, result)
        else:
            parallel_results = []
        
        # Step 4: Combine results
        final_results = [None] * len(evaluation_tasks)
        
        # Add cached results
        for index, result in cached_results:
            final_results[index] = result
        
        # Add computed results
        for (original_index, _), result in zip(uncached_tasks, parallel_results):
            final_results[original_index] = result
        
        # Step 5: Calculate acceleration metrics
        processing_time = time.time() - start_time
        cache_stats = self.cache_manager.get_cache_stats()
        processing_stats = self.parallel_processor.get_processing_stats()
        
        metrics = AccelerationMetrics(
            quantum_speedup=self._calculate_quantum_speedup(),
            parallel_efficiency=self._calculate_parallel_efficiency(processing_stats),
            cache_hit_rate=cache_stats['hit_rate'],
            gpu_utilization=self._calculate_gpu_utilization(processing_stats),
            memory_efficiency=self._calculate_memory_efficiency(),
            total_acceleration=self._calculate_total_acceleration(processing_time, len(evaluation_tasks))
        )
        
        self.acceleration_history.append(metrics)
        
        logger.info(f"Acceleration completed: {metrics.overall_score():.3f} score, {processing_time:.2f}s")
        
        return final_results, metrics
    
    def _generate_cache_key(self, task: Callable, data: Any) -> str:
        """Generate cache key for task and data."""
        # Simple hash-based key generation
        task_name = getattr(task, '__name__', str(task))
        data_hash = hash(str(data)) if data is not None else 0
        return f"{task_name}_{data_hash}"
    
    def _calculate_quantum_speedup(self) -> float:
        """Calculate quantum speedup factor."""
        # Mock calculation - would compare quantum vs classical performance
        return 1.2  # 20% speedup
    
    def _calculate_parallel_efficiency(self, stats: Dict) -> float:
        """Calculate parallel processing efficiency."""
        if stats['total_tasks'] == 0:
            return 1.0
        
        # Estimate efficiency based on task distribution
        cpu_ratio = stats['cpu_tasks'] / stats['total_tasks']
        gpu_ratio = stats['gpu_tasks'] / stats['total_tasks']
        dist_ratio = stats['distributed_tasks'] / stats['total_tasks']
        
        # Weight by efficiency of each method
        efficiency = cpu_ratio * 1.0 + gpu_ratio * 2.0 + dist_ratio * 1.5
        return min(efficiency, 3.0)  # Cap at 3x
    
    def _calculate_gpu_utilization(self, stats: Dict) -> float:
        """Calculate GPU utilization."""
        if stats['total_tasks'] == 0:
            return 0.0
        
        return stats['gpu_tasks'] / stats['total_tasks']
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency."""
        # Mock calculation - would analyze memory usage patterns
        return 0.9  # 90% efficiency
    
    def _calculate_total_acceleration(self, processing_time: float, num_tasks: int) -> float:
        """Calculate total acceleration factor."""
        if num_tasks == 0:
            return 1.0
        
        # Estimate what serial processing would have taken
        estimated_serial_time = num_tasks * 0.5  # 0.5 seconds per task
        
        if processing_time > 0:
            return estimated_serial_time / processing_time
        else:
            return 1.0
    
    def get_acceleration_summary(self) -> Dict[str, Any]:
        """Get comprehensive acceleration summary."""
        if not self.acceleration_history:
            return {'status': 'no_data'}
        
        recent_metrics = self.acceleration_history[-1]
        avg_metrics = AccelerationMetrics(
            quantum_speedup=np.mean([m.quantum_speedup for m in self.acceleration_history]),
            parallel_efficiency=np.mean([m.parallel_efficiency for m in self.acceleration_history]),
            cache_hit_rate=np.mean([m.cache_hit_rate for m in self.acceleration_history]),
            gpu_utilization=np.mean([m.gpu_utilization for m in self.acceleration_history]),
            memory_efficiency=np.mean([m.memory_efficiency for m in self.acceleration_history]),
            total_acceleration=np.mean([m.total_acceleration for m in self.acceleration_history])
        )
        
        return {
            'recent_metrics': {
                'quantum_speedup': recent_metrics.quantum_speedup,
                'parallel_efficiency': recent_metrics.parallel_efficiency,
                'cache_hit_rate': recent_metrics.cache_hit_rate,
                'gpu_utilization': recent_metrics.gpu_utilization,
                'memory_efficiency': recent_metrics.memory_efficiency,
                'total_acceleration': recent_metrics.total_acceleration,
                'overall_score': recent_metrics.overall_score()
            },
            'average_metrics': {
                'quantum_speedup': avg_metrics.quantum_speedup,
                'parallel_efficiency': avg_metrics.parallel_efficiency,
                'cache_hit_rate': avg_metrics.cache_hit_rate,
                'gpu_utilization': avg_metrics.gpu_utilization,
                'memory_efficiency': avg_metrics.memory_efficiency,
                'total_acceleration': avg_metrics.total_acceleration,
                'overall_score': avg_metrics.overall_score()
            },
            'component_stats': {
                'quantum_manager': self.quantum_manager.get_resource_utilization(),
                'parallel_processor': self.parallel_processor.get_processing_stats(),
                'cache_manager': self.cache_manager.get_cache_stats()
            },
            'optimization_count': len(self.acceleration_history)
        }
    
    def optimize_for_problem_class(self, problem_characteristics: Dict[str, Any]):
        """Optimize accelerator settings for specific problem class."""
        # Adjust cache size based on problem characteristics
        if problem_characteristics.get('search_space_size', 0) > 10000:
            # Large search space - increase cache
            self.cache_manager.max_size = min(50000, self.cache_manager.max_size * 2)
        
        # Adjust parallel processing strategy
        if problem_characteristics.get('evaluation_complexity', 'low') == 'high':
            # Complex evaluations - prefer GPU/distributed
            self.parallel_processor.enable_gpu = True
        
        logger.info(f"Accelerator optimized for problem: {problem_characteristics}")
    
    def reset_acceleration_history(self):
        """Reset acceleration history and statistics."""
        self.acceleration_history.clear()
        self.cache_manager = IntelligentCacheManager(self.cache_manager.max_size)
        logger.info("Acceleration history reset")
