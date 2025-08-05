"""
Parallel and distributed quantum optimization capabilities.
"""

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging_config import get_logger

logger = get_logger('parallel_optimization')


class ParallelQuantumOptimizer:
    """
    Parallel quantum optimization using multiple processes and quantum backends.
    
    Enables distributed hyperparameter search across multiple quantum annealers
    or classical simulators simultaneously.
    """
    
    def __init__(
        self,
        n_parallel_jobs: int = None,
        use_processes: bool = True,
        backend_configs: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize parallel quantum optimizer.
        
        Args:
            n_parallel_jobs: Number of parallel jobs (default: CPU count)
            use_processes: Use processes instead of threads
            backend_configs: List of backend configurations for parallel execution
        """
        self.n_parallel_jobs = n_parallel_jobs or multiprocessing.cpu_count()
        self.use_processes = use_processes
        self.backend_configs = backend_configs or [{'backend': 'simulator'}]
        
        logger.info(f"Initialized parallel optimizer with {self.n_parallel_jobs} jobs")
    
    def parallel_parameter_evaluation(
        self,
        parameter_sets: List[Dict[str, Any]],
        evaluation_function: Callable[[Dict[str, Any]], float],
        max_workers: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Evaluate multiple parameter sets in parallel.
        
        Args:
            parameter_sets: List of parameter dictionaries to evaluate
            evaluation_function: Function to evaluate each parameter set
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of (parameters, score) tuples
        """
        max_workers = max_workers or self.n_parallel_jobs
        results = []
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with executor_class(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_params = {
                    executor.submit(evaluation_function, params): params
                    for params in parameter_sets
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    
                    try:
                        score = future.result()
                        results.append((params, score))
                        logger.debug(f"Completed evaluation: {score:.4f}")
                        
                    except Exception as e:
                        logger.warning(f"Parameter evaluation failed: {e}")
                        results.append((params, float('-inf')))
        
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}")
            # Fallback to sequential evaluation
            for params in parameter_sets:
                try:
                    score = evaluation_function(params)
                    results.append((params, score))
                except Exception as eval_e:
                    logger.warning(f"Sequential evaluation failed: {eval_e}")
                    results.append((params, float('-inf')))
        
        logger.info(f"Completed {len(results)} parallel evaluations")
        return results
    
    def parallel_quantum_sampling(
        self,
        qubo_matrices: List[Dict[Tuple[int, int], float]],
        quantum_reads_per_matrix: int = 1000,
        backend_rotation: bool = True
    ) -> List[Any]:
        """
        Sample multiple QUBO matrices in parallel using different backends.
        
        Args:
            qubo_matrices: List of QUBO matrices to sample
            quantum_reads_per_matrix: Number of reads per matrix
            backend_rotation: Rotate between different backends
            
        Returns:
            List of SampleSet results
        """
        results = []
        
        def sample_qubo(args):
            qubo, backend_config, reads = args
            try:
                from ..backends.backend_factory import get_backend
                backend = get_backend(**backend_config)
                return backend.sample_qubo(qubo, num_reads=reads)
            except Exception as e:
                logger.warning(f"Quantum sampling failed: {e}")
                return None
        
        # Prepare sampling arguments
        sampling_args = []
        for i, qubo in enumerate(qubo_matrices):
            # Select backend configuration
            if backend_rotation:
                backend_config = self.backend_configs[i % len(self.backend_configs)]
            else:
                backend_config = self.backend_configs[0]
            
            sampling_args.append((qubo, backend_config, quantum_reads_per_matrix))
        
        # Execute parallel sampling
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with executor_class(max_workers=self.n_parallel_jobs) as executor:
                results = list(executor.map(sample_qubo, sampling_args))
        
        except Exception as e:
            logger.error(f"Parallel quantum sampling failed: {e}")
            # Fallback to sequential sampling
            for args in sampling_args:
                results.append(sample_qubo(args))
        
        # Filter out failed samples
        valid_results = [r for r in results if r is not None]
        logger.info(f"Completed {len(valid_results)}/{len(qubo_matrices)} quantum samplings")
        
        return valid_results
    
    def distributed_optimization(
        self,
        search_spaces: List[Dict[str, List[Any]]],
        evaluation_function: Callable,
        n_iterations_per_space: int = 20,
        merge_strategy: str = 'best_global'
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run distributed optimization across multiple search spaces.
        
        Args:
            search_spaces: List of parameter search spaces
            evaluation_function: Function to evaluate parameters
            n_iterations_per_space: Iterations per search space
            merge_strategy: Strategy for merging results ('best_global', 'pareto_front')
            
        Returns:
            Tuple of (best_parameters, all_results)
        """
        def optimize_subspace(args):
            """Optimize a single search space."""
            subspace, iterations, space_id = args
            
            try:
                from .. import QuantumHyperSearch
                
                # Create optimizer for this subspace
                qhs = QuantumHyperSearch(
                    backend='simulator',  # Use simulator for parallel jobs
                    verbose=False
                )
                
                # Generate random parameter combinations for this subspace
                results = []
                for _ in range(iterations):
                    params = {
                        param: np.random.choice(values)
                        for param, values in subspace.items()
                    }
                    
                    score = evaluation_function(params)
                    results.append({
                        'params': params,
                        'score': score,
                        'space_id': space_id
                    })
                
                return results
                
            except Exception as e:
                logger.warning(f"Subspace optimization failed: {e}")
                return []
        
        # Prepare optimization arguments
        optimization_args = [
            (space, n_iterations_per_space, i)
            for i, space in enumerate(search_spaces)
        ]
        
        # Execute distributed optimization
        all_results = []
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with executor_class(max_workers=min(len(search_spaces), self.n_parallel_jobs)) as executor:
                subspace_results = list(executor.map(optimize_subspace, optimization_args))
                
                # Flatten results
                for results in subspace_results:
                    all_results.extend(results)
        
        except Exception as e:
            logger.error(f"Distributed optimization failed: {e}")
            # Fallback to sequential optimization
            for args in optimization_args:
                results = optimize_subspace(args)
                all_results.extend(results)
        
        # Merge results according to strategy
        if merge_strategy == 'best_global':
            best_result = max(all_results, key=lambda x: x['score'])
            best_params = best_result['params']
        else:
            # For now, just return best global (can implement Pareto front later)
            best_result = max(all_results, key=lambda x: x['score'])
            best_params = best_result['params']
        
        logger.info(f"Distributed optimization completed: {len(all_results)} total evaluations")
        logger.info(f"Best score: {best_result['score']:.4f} from space {best_result['space_id']}")
        
        return best_params, all_results
    
    def adaptive_load_balancing(
        self,
        task_queue: List[Any],
        worker_function: Callable,
        performance_monitor: Optional[Callable] = None
    ) -> List[Any]:
        """
        Execute tasks with adaptive load balancing based on worker performance.
        
        Args:
            task_queue: List of tasks to execute
            worker_function: Function to execute each task
            performance_monitor: Optional function to monitor worker performance
            
        Returns:
            List of task results
        """
        results = []
        worker_performance = {}  # Track performance per worker
        
        def monitored_worker(task_with_worker_id):
            task, worker_id = task_with_worker_id
            start_time = time.time()
            
            try:
                result = worker_function(task)
                execution_time = time.time() - start_time
                
                # Update performance tracking
                if worker_id not in worker_performance:
                    worker_performance[worker_id] = []
                worker_performance[worker_id].append(execution_time)
                
                return result, worker_id, execution_time
                
            except Exception as e:
                logger.warning(f"Worker {worker_id} failed: {e}")
                return None, worker_id, time.time() - start_time
        
        # Assign tasks to workers with load balancing
        tasks_with_workers = []
        for i, task in enumerate(task_queue):
            worker_id = i % self.n_parallel_jobs
            tasks_with_workers.append((task, worker_id))
        
        # Execute with monitoring
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with executor_class(max_workers=self.n_parallel_jobs) as executor:
                future_results = list(executor.map(monitored_worker, tasks_with_workers))
                
                for result, worker_id, exec_time in future_results:
                    if result is not None:
                        results.append(result)
                    
                    if performance_monitor:
                        performance_monitor(worker_id, exec_time, result is not None)
        
        except Exception as e:
            logger.error(f"Adaptive load balancing failed: {e}")
            # Fallback to sequential execution
            for task, worker_id in tasks_with_workers:
                try:
                    result = worker_function(task)
                    results.append(result)
                except Exception as task_e:
                    logger.warning(f"Sequential task execution failed: {task_e}")
        
        # Log performance statistics
        if worker_performance:
            for worker_id, times in worker_performance.items():
                avg_time = np.mean(times)
                logger.debug(f"Worker {worker_id}: {len(times)} tasks, avg {avg_time:.2f}s")
        
        return results
    
    def get_optimal_parallelization(
        self,
        problem_size: int,
        available_resources: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Determine optimal parallelization strategy for given problem size.
        
        Args:
            problem_size: Size of the optimization problem
            available_resources: Available computational resources
            
        Returns:
            Dictionary with optimal parallelization parameters
        """
        # Simple heuristics for parallelization
        cpu_cores = available_resources.get('cpu_cores', multiprocessing.cpu_count())
        memory_gb = available_resources.get('memory_gb', 8)
        
        # Estimate memory per worker
        memory_per_worker = max(1, memory_gb // cpu_cores)
        
        # Adjust for problem size
        if problem_size < 100:
            # Small problems: use fewer workers to avoid overhead
            optimal_workers = min(4, cpu_cores)
        elif problem_size < 1000:
            # Medium problems: use most cores
            optimal_workers = cpu_cores
        else:
            # Large problems: may need to limit based on memory
            optimal_workers = min(cpu_cores, memory_gb // 2)
        
        return {
            'n_workers': optimal_workers,
            'use_processes': problem_size > 100,  # Processes for larger problems
            'batch_size': max(1, problem_size // optimal_workers),
            'memory_per_worker': memory_per_worker
        }


def parallel_hyperparameter_search(
    search_space: Dict[str, List[Any]],
    evaluation_function: Callable[[Dict[str, Any]], float],
    n_parallel_evaluations: int = 50,
    n_workers: Optional[int] = None
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Convenience function for parallel hyperparameter search.
    
    Args:
        search_space: Parameter search space
        evaluation_function: Function to evaluate parameters
        n_parallel_evaluations: Number of parallel evaluations
        n_workers: Number of parallel workers
        
    Returns:
        List of (parameters, score) tuples sorted by score
    """
    optimizer = ParallelQuantumOptimizer(n_parallel_jobs=n_workers)
    
    # Generate random parameter combinations
    parameter_sets = []
    for _ in range(n_parallel_evaluations):
        params = {
            param: np.random.choice(values)
            for param, values in search_space.items()
        }
        parameter_sets.append(params)
    
    # Evaluate in parallel
    results = optimizer.parallel_parameter_evaluation(parameter_sets, evaluation_function)
    
    # Sort by score (best first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results