"""
Parallel and concurrent processing for quantum hyperparameter search.
"""

import asyncio
import concurrent.futures
import threading
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator
import time
import numpy as np
from dataclasses import dataclass
from queue import Queue, Empty
import psutil


@dataclass
class EvaluationTask:
    """Single parameter evaluation task."""
    task_id: str
    parameters: Dict[str, Any]
    model_class: type
    X: np.ndarray
    y: np.ndarray
    cv_folds: int
    scoring: str
    model_kwargs: Dict[str, Any]
    
    def __hash__(self):
        """Make task hashable for deduplication."""
        return hash(self.task_id)


@dataclass 
class EvaluationResult:
    """Result of parameter evaluation."""
    task_id: str
    parameters: Dict[str, Any]
    score: float
    success: bool
    error: Optional[str] = None
    computation_time: float = 0.0


class ParallelEvaluator:
    """
    Parallel evaluator for parameter configurations using multiple processes.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        chunk_size: int = 1,
        enable_deduplication: bool = True
    ):
        """
        Initialize parallel evaluator.
        
        Args:
            max_workers: Maximum number of worker processes (None for auto)
            chunk_size: Number of tasks per chunk for batch processing
            enable_deduplication: Enable deduplication of identical tasks
        """
        # Auto-detect optimal worker count
        if max_workers is None:
            cpu_count = psutil.cpu_count(logical=False) or mp.cpu_count()
            # Use 75% of available cores, but at least 1
            max_workers = max(1, int(cpu_count * 0.75))
        
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.enable_deduplication = enable_deduplication
        
        # Task deduplication
        self._seen_tasks = set() if enable_deduplication else None
        
        # Performance tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_time = 0.0
    
    def evaluate_batch(
        self,
        tasks: List[EvaluationTask],
        timeout: Optional[float] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate batch of parameter configurations in parallel.
        
        Args:
            tasks: List of evaluation tasks
            timeout: Timeout in seconds for each task
            
        Returns:
            List of evaluation results
        """
        if not tasks:
            return []
        
        start_time = time.time()
        
        # Deduplicate tasks if enabled
        if self.enable_deduplication:
            unique_tasks = []
            for task in tasks:
                if task.task_id not in self._seen_tasks:
                    unique_tasks.append(task)
                    self._seen_tasks.add(task.task_id)
            tasks = unique_tasks
        
        if not tasks:
            return []
        
        self.total_tasks += len(tasks)
        results = []
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context('spawn')  # More reliable across platforms
        ) as executor:
            
            # Submit tasks in chunks for better load balancing
            task_chunks = [
                tasks[i:i + self.chunk_size] 
                for i in range(0, len(tasks), self.chunk_size)
            ]
            
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._evaluate_chunk, chunk): chunk
                for chunk in task_chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(
                future_to_chunk, timeout=timeout
            ):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    self.completed_tasks += len(chunk_results)
                    
                except Exception as e:
                    # Handle chunk failure
                    failed_chunk = future_to_chunk[future]
                    chunk_results = [
                        EvaluationResult(
                            task_id=task.task_id,
                            parameters=task.parameters,
                            score=0.0,
                            success=False,
                            error=str(e)
                        )
                        for task in failed_chunk
                    ]
                    results.extend(chunk_results)
                    self.failed_tasks += len(failed_chunk)
        
        self.total_time += time.time() - start_time
        return results
    
    @staticmethod
    def _evaluate_chunk(tasks: List[EvaluationTask]) -> List[EvaluationResult]:
        """Evaluate a chunk of tasks in a single process."""
        results = []
        
        for task in tasks:
            start_time = time.time()
            
            try:
                # Import here to avoid serialization issues
                from sklearn.model_selection import cross_val_score
                
                # Create model instance
                model = task.model_class(**task.model_kwargs)
                
                # Perform cross-validation
                scores = cross_val_score(
                    model, task.X, task.y,
                    cv=task.cv_folds,
                    scoring=task.scoring,
                    n_jobs=1  # Single job per process to avoid nested parallelism
                )
                
                score = float(scores.mean())
                computation_time = time.time() - start_time
                
                results.append(EvaluationResult(
                    task_id=task.task_id,
                    parameters=task.parameters,
                    score=score,
                    success=True,
                    computation_time=computation_time
                ))
                
            except Exception as e:
                computation_time = time.time() - start_time
                
                results.append(EvaluationResult(
                    task_id=task.task_id,
                    parameters=task.parameters,
                    score=0.0,
                    success=False,
                    error=str(e),
                    computation_time=computation_time
                ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        success_rate = self.completed_tasks / max(self.total_tasks, 1)
        avg_time_per_task = self.total_time / max(self.total_tasks, 1)
        
        return {
            'max_workers': self.max_workers,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': success_rate,
            'total_time': self.total_time,
            'avg_time_per_task': avg_time_per_task,
            'throughput_tasks_per_second': self.total_tasks / max(self.total_time, 1e-6)
        }


class ConcurrentSampler:
    """
    Concurrent quantum sampler for handling multiple QUBO problems simultaneously.
    """
    
    def __init__(
        self,
        max_concurrent_samples: int = 4,
        queue_size: int = 100
    ):
        """
        Initialize concurrent sampler.
        
        Args:
            max_concurrent_samples: Maximum concurrent sampling operations
            queue_size: Maximum size of sampling queue
        """
        self.max_concurrent_samples = max_concurrent_samples
        self.sampling_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue()
        
        # Worker threads for sampling
        self.workers = []
        self.shutdown_event = threading.Event()
        
        # Start worker threads
        for i in range(max_concurrent_samples):
            worker = threading.Thread(
                target=self._sampling_worker,
                name=f"SamplingWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def submit_sampling_task(
        self,
        backend,
        Q: np.ndarray,
        num_reads: int,
        task_id: str,
        **kwargs
    ) -> None:
        """
        Submit quantum sampling task.
        
        Args:
            backend: Quantum backend instance
            Q: QUBO matrix
            num_reads: Number of reads
            task_id: Unique task identifier
            **kwargs: Additional backend parameters
        """
        task = {
            'backend': backend,
            'Q': Q,
            'num_reads': num_reads,
            'task_id': task_id,
            'kwargs': kwargs,
            'timestamp': time.time()
        }
        
        try:
            self.sampling_queue.put(task, timeout=1.0)
        except:
            # Queue full - could implement backpressure here
            pass
    
    def get_completed_samples(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get all completed sampling results.
        
        Args:
            timeout: Timeout for getting results
            
        Returns:
            List of completed sampling results
        """
        results = []
        
        try:
            while True:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
                self.result_queue.task_done()
        except Empty:
            pass
        
        return results
    
    def _sampling_worker(self) -> None:
        """Worker thread for quantum sampling."""
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue
                task = self.sampling_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                try:
                    # Perform quantum sampling
                    samples = task['backend'].sample_qubo(
                        task['Q'],
                        num_reads=task['num_reads'],
                        **task['kwargs']
                    )
                    
                    sampling_time = time.time() - start_time
                    
                    # Submit result
                    result = {
                        'task_id': task['task_id'],
                        'samples': samples,
                        'success': True,
                        'sampling_time': sampling_time,
                        'timestamp': task['timestamp']
                    }
                    
                except Exception as e:
                    sampling_time = time.time() - start_time
                    
                    result = {
                        'task_id': task['task_id'],
                        'samples': [],
                        'success': False,
                        'error': str(e),
                        'sampling_time': sampling_time,
                        'timestamp': task['timestamp']
                    }
                
                self.result_queue.put(result)
                self.sampling_queue.task_done()
                
            except Empty:
                continue
            except Exception:
                # Log error but continue working
                continue
    
    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Shutdown concurrent sampler.
        
        Args:
            timeout: Timeout for graceful shutdown
        """
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)


class AdaptiveScheduler:
    """
    Adaptive scheduler that dynamically adjusts parallelization based on system load.
    """
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: Optional[int] = None,
        target_cpu_usage: float = 0.8,
        adaptation_interval: float = 10.0
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (None for auto)
            target_cpu_usage: Target CPU utilization (0.0 to 1.0)
            adaptation_interval: How often to adapt in seconds
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or psutil.cpu_count()
        self.target_cpu_usage = target_cpu_usage
        self.adaptation_interval = adaptation_interval
        
        self.current_workers = min_workers
        self.last_adaptation = time.time()
        
        # Performance history
        self.performance_history = []
        self.max_history = 100
    
    def get_optimal_workers(self) -> int:
        """
        Get optimal number of workers based on current system state.
        
        Returns:
            Optimal number of workers
        """
        current_time = time.time()
        
        # Only adapt if enough time has passed
        if current_time - self.last_adaptation < self.adaptation_interval:
            return self.current_workers
        
        # Get current system metrics
        cpu_usage = psutil.cpu_percent(interval=1.0)
        memory_usage = psutil.virtual_memory().percent
        
        # Adaptive logic
        if cpu_usage < self.target_cpu_usage * 100 * 0.7:
            # System underutilized - increase workers
            new_workers = min(self.current_workers + 1, self.max_workers)
        elif cpu_usage > self.target_cpu_usage * 100 * 1.3:
            # System overutilized - decrease workers
            new_workers = max(self.current_workers - 1, self.min_workers)
        else:
            # System well-utilized - maintain current level
            new_workers = self.current_workers
        
        # Also consider memory pressure
        if memory_usage > 90:
            new_workers = max(new_workers - 1, self.min_workers)
        
        self.current_workers = new_workers
        self.last_adaptation = current_time
        
        # Record performance metrics
        self.performance_history.append({
            'timestamp': current_time,
            'workers': new_workers,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage
        })
        
        # Trim history
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
        
        return new_workers
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and history."""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-10:]  # Last 10 measurements
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'target_cpu_usage': self.target_cpu_usage,
            'avg_cpu_usage': np.mean([m['cpu_usage'] for m in recent_metrics]),
            'avg_memory_usage': np.mean([m['memory_usage'] for m in recent_metrics]),
            'adaptation_count': len(self.performance_history)
        }