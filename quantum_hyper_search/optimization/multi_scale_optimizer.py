#!/usr/bin/env python3
"""
Multi-Scale Quantum Optimization System

This module implements a comprehensive multi-scale optimization framework
that combines quantum annealing, classical optimization, and machine learning
to achieve unprecedented hyperparameter optimization performance.

Key Features:
1. Multi-scale problem decomposition
2. Adaptive algorithm selection
3. Real-time performance optimization
4. Quantum-classical hybrid optimization
5. Predictive resource allocation
"""

import time
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Import our quantum components
try:
    from ..research.quantum_advantage_accelerator import (
        QuantumAdvantageAccelerator, 
        QuantumParallelTempering,
        AdaptiveQuantumWalk,
        QuantumEnhancedBayesianOptimization
    )
    from ..research.novel_encodings import (
        HierarchicalEncoder,
        ConstraintAwareEncoder, 
        MultiObjectiveEncoder
    )
    from .quantum_advantage_accelerator import QuantumAdvantageAccelerator as ProductionAccelerator
    HAS_QUANTUM_RESEARCH = True
except ImportError:
    HAS_QUANTUM_RESEARCH = False

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class OptimizationScale:
    """Defines an optimization scale with specific characteristics."""
    name: str
    parameter_count_range: Tuple[int, int]
    time_budget_seconds: float
    preferred_algorithms: List[str]
    quantum_advantage_threshold: float = 1.2
    parallelization_factor: int = 1
    
    def is_suitable_for_problem(self, param_count: int, time_budget: float) -> bool:
        """Check if this scale is suitable for the given problem."""
        min_params, max_params = self.parameter_count_range
        return (min_params <= param_count <= max_params and 
                time_budget >= self.time_budget_seconds * 0.5)


@dataclass
class OptimizationTask:
    """Represents a single optimization task."""
    task_id: str
    algorithm: str
    parameters: Dict[str, Any]
    search_space: Dict[str, List[Any]]
    objective_function: Callable
    scale: OptimizationScale
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiScaleMetrics:
    """Comprehensive metrics for multi-scale optimization."""
    total_optimization_time: float
    algorithm_performance: Dict[str, float]
    scale_utilization: Dict[str, int]
    quantum_advantage_achieved: float
    resource_efficiency: float
    convergence_rate: float
    solution_quality: float
    adaptive_decisions: int
    
    def overall_score(self) -> float:
        """Calculate overall optimization score."""
        return (
            (1.0 / max(0.1, self.total_optimization_time)) * 0.2 +
            np.mean(list(self.algorithm_performance.values())) * 0.3 +
            self.quantum_advantage_achieved * 0.2 +
            self.resource_efficiency * 0.15 +
            self.solution_quality * 0.15
        )


class AlgorithmSelector:
    """Intelligent algorithm selection based on problem characteristics."""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.algorithm_registry = self._initialize_algorithm_registry()
        self.problem_classifier = None
        self._train_problem_classifier()
        
    def _initialize_algorithm_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize registry of available algorithms."""
        return {
            'quantum_annealing': {
                'suitable_scales': ['fine', 'medium'],
                'min_parameters': 2,
                'max_parameters': 100,
                'quantum_advantage': True,
                'parallelizable': True,
                'complexity': 'medium'
            },
            'quantum_walk': {
                'suitable_scales': ['fine', 'medium'],
                'min_parameters': 5,
                'max_parameters': 50,
                'quantum_advantage': True,
                'parallelizable': True,
                'complexity': 'high'
            },
            'bayesian_optimization': {
                'suitable_scales': ['fine', 'medium', 'coarse'],
                'min_parameters': 1,
                'max_parameters': 20,
                'quantum_advantage': False,
                'parallelizable': False,
                'complexity': 'high'
            },
            'quantum_bayesian': {
                'suitable_scales': ['fine', 'medium'],
                'min_parameters': 2,
                'max_parameters': 30,
                'quantum_advantage': True,
                'parallelizable': True,
                'complexity': 'high'
            },
            'parallel_tempering': {
                'suitable_scales': ['medium', 'coarse'],
                'min_parameters': 5,
                'max_parameters': 200,
                'quantum_advantage': True,
                'parallelizable': True,
                'complexity': 'medium'
            },
            'genetic_algorithm': {
                'suitable_scales': ['medium', 'coarse'],
                'min_parameters': 3,
                'max_parameters': 1000,
                'quantum_advantage': False,
                'parallelizable': True,
                'complexity': 'low'
            },
            'random_search': {
                'suitable_scales': ['coarse'],
                'min_parameters': 1,
                'max_parameters': 10000,
                'quantum_advantage': False,
                'parallelizable': True,
                'complexity': 'low'
            }
        }
    
    def _train_problem_classifier(self):
        """Train a classifier to categorize problems."""
        # In a real implementation, this would use historical data
        # For now, we'll use a simple rule-based approach
        pass
    
    def select_algorithm(self, problem_characteristics: Dict[str, Any], 
                        scale: OptimizationScale,
                        performance_history: Optional[Dict[str, float]] = None) -> str:
        """Select the best algorithm for the given problem and scale."""
        
        param_count = problem_characteristics.get('parameter_count', 10)
        time_budget = problem_characteristics.get('time_budget', 300)
        complexity = problem_characteristics.get('complexity', 'medium')
        
        # Filter algorithms suitable for the scale and problem size
        suitable_algorithms = []
        
        for algo_name, algo_info in self.algorithm_registry.items():
            if (scale.name in algo_info['suitable_scales'] and
                algo_info['min_parameters'] <= param_count <= algo_info['max_parameters']):
                suitable_algorithms.append(algo_name)
        
        if not suitable_algorithms:
            return 'random_search'  # Fallback
        
        # Score algorithms based on expected performance
        algorithm_scores = {}
        for algo_name in suitable_algorithms:
            score = self._calculate_algorithm_score(
                algo_name, problem_characteristics, scale, performance_history
            )
            algorithm_scores[algo_name] = score
        
        # Select algorithm with highest score
        best_algorithm = max(algorithm_scores, key=algorithm_scores.get)
        
        logger.info(f"Selected algorithm {best_algorithm} with score {algorithm_scores[best_algorithm]:.3f}")
        
        return best_algorithm
    
    def _calculate_algorithm_score(self, algorithm_name: str, 
                                 problem_characteristics: Dict[str, Any],
                                 scale: OptimizationScale,
                                 performance_history: Optional[Dict[str, float]]) -> float:
        """Calculate score for an algorithm on a specific problem."""
        
        algo_info = self.algorithm_registry[algorithm_name]
        score = 0.0
        
        # Base score from algorithm characteristics
        if algo_info['quantum_advantage'] and problem_characteristics.get('enable_quantum', True):
            score += 2.0
        
        if algo_info['parallelizable'] and scale.parallelization_factor > 1:
            score += 1.5
        
        # Complexity matching
        problem_complexity = problem_characteristics.get('complexity', 'medium')
        if algo_info['complexity'] == problem_complexity:
            score += 1.0
        
        # Historical performance
        if performance_history and algorithm_name in performance_history:
            historical_performance = performance_history[algorithm_name]
            score += historical_performance * 2.0
        
        # Scale preference
        if scale.name in algo_info['suitable_scales']:
            scale_index = algo_info['suitable_scales'].index(scale.name)
            score += (len(algo_info['suitable_scales']) - scale_index) * 0.5
        
        return score
    
    def update_performance(self, algorithm_name: str, performance_score: float):
        """Update performance history for an algorithm."""
        self.performance_history[algorithm_name].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[algorithm_name]) > 100:
            self.performance_history[algorithm_name] = self.performance_history[algorithm_name][-50:]


class ResourceManager:
    """Manages computational resources across multiple scales."""
    
    def __init__(self, max_concurrent_tasks: int = 8):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks = {}
        self.resource_usage = {
            'cpu_cores': 0,
            'memory_gb': 0,
            'gpu_memory_gb': 0,
            'quantum_queue_position': 0
        }
        self.resource_limits = {
            'cpu_cores': min(16, max(4, mp.cpu_count())),
            'memory_gb': 32,
            'gpu_memory_gb': 8,
            'quantum_queue_position': 10
        }
        self.lock = threading.RLock()
    
    def allocate_resources(self, task: OptimizationTask) -> bool:
        """Allocate resources for a task."""
        with self.lock:
            required_resources = self._estimate_resource_requirements(task)
            
            if self._can_allocate(required_resources):
                task_id = task.task_id
                self.active_tasks[task_id] = {
                    'task': task,
                    'allocated_resources': required_resources,
                    'start_time': time.time()
                }
                
                # Update resource usage
                for resource, amount in required_resources.items():
                    self.resource_usage[resource] += amount
                
                logger.debug(f"Resources allocated for task {task_id}: {required_resources}")
                return True
            
            return False
    
    def deallocate_resources(self, task_id: str):
        """Deallocate resources for a completed task."""
        with self.lock:
            if task_id in self.active_tasks:
                allocated_resources = self.active_tasks[task_id]['allocated_resources']
                
                # Update resource usage
                for resource, amount in allocated_resources.items():
                    self.resource_usage[resource] -= amount
                
                del self.active_tasks[task_id]
                logger.debug(f"Resources deallocated for task {task_id}")
    
    def _estimate_resource_requirements(self, task: OptimizationTask) -> Dict[str, float]:
        """Estimate resource requirements for a task."""
        algorithm = task.algorithm
        param_count = len(task.search_space)
        
        requirements = {
            'cpu_cores': 1.0,
            'memory_gb': 1.0,
            'gpu_memory_gb': 0.0,
            'quantum_queue_position': 0
        }
        
        # Algorithm-specific requirements
        if 'quantum' in algorithm:
            requirements['quantum_queue_position'] = 1
            requirements['cpu_cores'] = 2.0
            requirements['memory_gb'] = 2.0
        
        if 'parallel' in algorithm or 'distributed' in algorithm:
            requirements['cpu_cores'] = min(4.0, param_count * 0.5)
            requirements['memory_gb'] = min(8.0, param_count * 0.2)
        
        if 'gpu' in algorithm.lower():
            requirements['gpu_memory_gb'] = 2.0
        
        # Scale-based adjustments
        if task.scale.name == 'coarse':
            requirements['cpu_cores'] *= 1.5
            requirements['memory_gb'] *= 1.3
        elif task.scale.name == 'fine':
            requirements['cpu_cores'] *= 0.7
            requirements['memory_gb'] *= 0.8
        
        return requirements
    
    def _can_allocate(self, required_resources: Dict[str, float]) -> bool:
        """Check if required resources can be allocated."""
        for resource, amount in required_resources.items():
            if self.resource_usage[resource] + amount > self.resource_limits[resource]:
                return False
        
        return len(self.active_tasks) < self.max_concurrent_tasks
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        utilization = {}
        for resource in self.resource_usage:
            usage = self.resource_usage[resource]
            limit = self.resource_limits[resource]
            utilization[resource] = usage / max(limit, 1.0)
        
        utilization['task_slots'] = len(self.active_tasks) / self.max_concurrent_tasks
        
        return utilization


class MultiScaleOptimizer:
    """
    Main multi-scale quantum optimization system.
    
    Coordinates multiple optimization scales, algorithms, and resources
    to achieve optimal hyperparameter optimization performance.
    """
    
    def __init__(self, 
                 quantum_backend: str = 'simulator',
                 max_concurrent_tasks: int = 8,
                 enable_quantum_acceleration: bool = True):
        """
        Initialize multi-scale optimizer.
        
        Args:
            quantum_backend: Quantum backend to use
            max_concurrent_tasks: Maximum concurrent optimization tasks
            enable_quantum_acceleration: Enable quantum acceleration techniques
        """
        self.quantum_backend = quantum_backend
        self.enable_quantum_acceleration = enable_quantum_acceleration
        
        # Initialize scales
        self.scales = self._initialize_scales()
        
        # Initialize components
        self.algorithm_selector = AlgorithmSelector()
        self.resource_manager = ResourceManager(max_concurrent_tasks)
        
        # Initialize quantum accelerator if available
        if HAS_QUANTUM_RESEARCH and enable_quantum_acceleration:
            self.quantum_accelerator = QuantumAdvantageAccelerator(
                techniques=['parallel_tempering', 'quantum_walk', 'bayesian_opt'],
                backend=quantum_backend
            )
        else:
            self.quantum_accelerator = ProductionAccelerator(quantum_backend)
        
        # Task management
        self.task_queue = deque()
        self.completed_tasks = []
        self.active_optimizations = {}
        
        # Performance tracking
        self.optimization_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.is_running = False
        self.optimization_thread = None
        
        logger.info(f"MultiScaleOptimizer initialized with {len(self.scales)} scales")
    
    def _initialize_scales(self) -> List[OptimizationScale]:
        """Initialize optimization scales."""
        return [
            OptimizationScale(
                name='fine',
                parameter_count_range=(1, 20),
                time_budget_seconds=60,
                preferred_algorithms=['quantum_annealing', 'quantum_walk', 'bayesian_optimization'],
                quantum_advantage_threshold=1.1,
                parallelization_factor=2
            ),
            OptimizationScale(
                name='medium',
                parameter_count_range=(10, 100),
                time_budget_seconds=300,
                preferred_algorithms=['quantum_bayesian', 'parallel_tempering', 'genetic_algorithm'],
                quantum_advantage_threshold=1.3,
                parallelization_factor=4
            ),
            OptimizationScale(
                name='coarse',
                parameter_count_range=(50, 1000),
                time_budget_seconds=1800,
                preferred_algorithms=['parallel_tempering', 'genetic_algorithm', 'random_search'],
                quantum_advantage_threshold=1.5,
                parallelization_factor=8
            )
        ]
    
    def optimize(self, 
                objective_function: Callable,
                search_space: Dict[str, List[Any]],
                optimization_budget: Dict[str, Any] = None,
                problem_characteristics: Dict[str, Any] = None) -> Tuple[Dict[str, Any], MultiScaleMetrics]:
        """
        Multi-scale optimization of hyperparameters.
        
        Args:
            objective_function: Function to optimize
            search_space: Parameter search space
            optimization_budget: Budget constraints (time, evaluations, etc.)
            problem_characteristics: Characteristics of the optimization problem
            
        Returns:
            Tuple of (best_parameters, optimization_metrics)
        """
        start_time = time.time()
        
        # Set defaults
        optimization_budget = optimization_budget or {'time_seconds': 300, 'max_evaluations': 100}
        problem_characteristics = problem_characteristics or {}
        problem_characteristics.update({
            'parameter_count': len(search_space),
            'time_budget': optimization_budget.get('time_seconds', 300),
            'max_evaluations': optimization_budget.get('max_evaluations', 100)
        })
        
        # Select appropriate scale
        scale = self._select_optimization_scale(problem_characteristics)
        logger.info(f"Selected optimization scale: {scale.name}")
        
        # Decompose problem if necessary
        subproblems = self._decompose_problem(search_space, scale, problem_characteristics)
        
        # Create optimization tasks
        tasks = []
        for i, subproblem in enumerate(subproblems):
            algorithm = self.algorithm_selector.select_algorithm(
                problem_characteristics, scale, 
                self._get_algorithm_performance_history()
            )
            
            task = OptimizationTask(
                task_id=f"opt_task_{int(time.time())}_{i}",
                algorithm=algorithm,
                parameters={},
                search_space=subproblem,
                objective_function=objective_function,
                scale=scale,
                priority=1
            )
            tasks.append(task)
        
        # Execute optimization tasks
        results = self._execute_optimization_tasks(tasks, optimization_budget)
        
        # Combine results from multiple scales/subproblems
        best_result = self._combine_optimization_results(results)
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        metrics = self._calculate_optimization_metrics(results, total_time, scale)
        
        # Update performance history
        self.optimization_history.append({
            'timestamp': time.time(),
            'scale': scale.name,
            'algorithm_performance': {task.algorithm: task.performance_metrics.get('score', 0.0) for task in results},
            'total_time': total_time,
            'best_score': best_result[1] if len(best_result) > 1 else 0.0,
            'metrics': metrics
        })
        
        logger.info(f"Multi-scale optimization completed in {total_time:.2f}s with score {metrics.overall_score():.3f}")
        
        return best_result[0] if best_result else {}, metrics
    
    def _select_optimization_scale(self, problem_characteristics: Dict[str, Any]) -> OptimizationScale:
        """Select appropriate optimization scale for the problem."""
        param_count = problem_characteristics['parameter_count']
        time_budget = problem_characteristics['time_budget']
        
        # Find suitable scales
        suitable_scales = [
            scale for scale in self.scales 
            if scale.is_suitable_for_problem(param_count, time_budget)
        ]
        
        if not suitable_scales:
            return self.scales[-1]  # Use coarse scale as fallback
        
        # Prefer scale that best matches the problem size
        best_scale = suitable_scales[0]
        best_match_score = float('inf')
        
        for scale in suitable_scales:
            min_params, max_params = scale.parameter_count_range
            mid_params = (min_params + max_params) / 2
            match_score = abs(param_count - mid_params)
            
            if match_score < best_match_score:
                best_match_score = match_score
                best_scale = scale
        
        return best_scale
    
    def _decompose_problem(self, 
                          search_space: Dict[str, List[Any]], 
                          scale: OptimizationScale,
                          problem_characteristics: Dict[str, Any]) -> List[Dict[str, List[Any]]]:
        """Decompose large problems into smaller subproblems."""
        
        param_count = len(search_space)
        max_params_per_subproblem = scale.parameter_count_range[1]
        
        if param_count <= max_params_per_subproblem:
            return [search_space]  # No decomposition needed
        
        # Intelligent parameter grouping
        subproblems = self._group_parameters_intelligently(search_space, max_params_per_subproblem)
        
        logger.info(f"Problem decomposed into {len(subproblems)} subproblems")
        
        return subproblems
    
    def _group_parameters_intelligently(self, 
                                      search_space: Dict[str, List[Any]], 
                                      max_params_per_group: int) -> List[Dict[str, List[Any]]]:
        """Group parameters intelligently for decomposition."""
        
        param_names = list(search_space.keys())
        
        # Simple grouping by parameter name similarity and type
        groups = []
        current_group = {}
        
        for param_name in param_names:
            if len(current_group) >= max_params_per_group:
                groups.append(current_group)
                current_group = {}
            
            current_group[param_name] = search_space[param_name]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _execute_optimization_tasks(self, 
                                   tasks: List[OptimizationTask], 
                                   budget: Dict[str, Any]) -> List[OptimizationTask]:
        """Execute optimization tasks with resource management."""
        
        completed_tasks = []
        futures = {}
        
        # Submit tasks to executor
        for task in tasks:
            if self.resource_manager.allocate_resources(task):
                future = self.executor.submit(self._execute_single_task, task, budget)
                futures[future] = task
                task.started_at = time.time()
            else:
                logger.warning(f"Could not allocate resources for task {task.task_id}")
        
        # Collect results
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                task.result = result
                task.completed_at = time.time()
                completed_tasks.append(task)
                
                # Update algorithm performance
                if 'score' in task.performance_metrics:
                    self.algorithm_selector.update_performance(
                        task.algorithm, 
                        task.performance_metrics['score']
                    )
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                task.completed_at = time.time()
                task.performance_metrics['error'] = str(e)
                completed_tasks.append(task)
            
            finally:
                self.resource_manager.deallocate_resources(task.task_id)
        
        return completed_tasks
    
    def _execute_single_task(self, 
                           task: OptimizationTask, 
                           budget: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Execute a single optimization task."""
        
        try:
            if 'quantum' in task.algorithm and self.enable_quantum_acceleration:
                result = self._execute_quantum_optimization(task, budget)
            else:
                result = self._execute_classical_optimization(task, budget)
            
            # Record performance metrics
            if result and len(result) > 1:
                task.performance_metrics['score'] = result[1]
                task.performance_metrics['algorithm'] = task.algorithm
                task.performance_metrics['execution_time'] = time.time() - task.started_at
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {}, 0.0
    
    def _execute_quantum_optimization(self, 
                                    task: OptimizationTask, 
                                    budget: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Execute quantum-enhanced optimization."""
        
        if not HAS_QUANTUM_RESEARCH:
            logger.warning("Quantum research modules not available, falling back to classical")
            return self._execute_classical_optimization(task, budget)
        
        try:
            # Use quantum advantage accelerator
            if hasattr(self.quantum_accelerator, 'optimize_with_quantum_advantage'):
                best_params, metrics = self.quantum_accelerator.optimize_with_quantum_advantage(
                    task.objective_function,
                    task.search_space,
                    n_iterations=budget.get('max_evaluations', 50)
                )
                
                # Evaluate best parameters
                best_score = task.objective_function(best_params)
                
                # Store quantum metrics
                task.performance_metrics.update({
                    'quantum_advantage_ratio': metrics.quantum_advantage_score() if hasattr(metrics, 'quantum_advantage_score') else 1.0,
                    'quantum_coherence_utilization': getattr(metrics, 'quantum_coherence_utilization', 0.5)
                })
                
                return best_params, best_score
                
            else:
                # Fallback to basic quantum optimization
                return self._basic_quantum_optimization(task, budget)
                
        except Exception as e:
            logger.warning(f"Quantum optimization failed: {e}, falling back to classical")
            return self._execute_classical_optimization(task, budget)
    
    def _basic_quantum_optimization(self, 
                                  task: OptimizationTask, 
                                  budget: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Basic quantum optimization implementation."""
        
        # Simple quantum-inspired random search with annealing
        best_params = None
        best_score = float('-inf')
        
        n_iterations = budget.get('max_evaluations', 50)
        temperature_schedule = np.linspace(1.0, 0.1, n_iterations)
        
        for i in range(n_iterations):
            # Quantum-inspired sampling with temperature
            params = self._quantum_inspired_sample(task.search_space, temperature_schedule[i])
            score = task.objective_function(params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params or self._random_sample(task.search_space), best_score
    
    def _quantum_inspired_sample(self, 
                               search_space: Dict[str, List[Any]], 
                               temperature: float) -> Dict[str, Any]:
        """Sample parameters using quantum-inspired probabilistic method."""
        
        params = {}
        
        for param_name, param_values in search_space.items():
            if isinstance(param_values[0], (int, float)):
                # Continuous parameter - use quantum-inspired distribution
                min_val, max_val = min(param_values), max(param_values)
                
                # Create superposition-like distribution with temperature
                n_peaks = max(2, int(len(param_values) * temperature))
                peak_positions = np.linspace(min_val, max_val, n_peaks)
                
                # Select peak with quantum-like probability
                peak_weights = np.exp(-np.arange(n_peaks) / (temperature + 0.1))
                peak_weights /= np.sum(peak_weights)
                
                selected_peak = np.random.choice(n_peaks, p=peak_weights)
                peak_center = peak_positions[selected_peak]
                
                # Add quantum noise
                noise_scale = (max_val - min_val) * temperature * 0.1
                value = np.random.normal(peak_center, noise_scale)
                
                # Ensure within bounds
                value = max(min_val, min(max_val, value))
                
                # Find closest valid value
                closest_idx = np.argmin([abs(value - v) for v in param_values])
                params[param_name] = param_values[closest_idx]
            else:
                # Categorical parameter - use temperature-weighted selection
                if temperature > 0.5:
                    # High temperature - more exploration
                    weights = np.ones(len(param_values))
                else:
                    # Low temperature - more exploitation (prefer certain values)
                    weights = np.exp(-np.arange(len(param_values)) * (1 - temperature))
                
                weights /= np.sum(weights)
                selected_idx = np.random.choice(len(param_values), p=weights)
                params[param_name] = param_values[selected_idx]
        
        return params
    
    def _execute_classical_optimization(self, 
                                      task: OptimizationTask, 
                                      budget: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Execute classical optimization algorithms."""
        
        algorithm = task.algorithm
        
        if algorithm == 'bayesian_optimization':
            return self._bayesian_optimization(task, budget)
        elif algorithm == 'genetic_algorithm':
            return self._genetic_algorithm(task, budget)
        elif algorithm == 'random_search':
            return self._random_search(task, budget)
        else:
            # Default to random search
            return self._random_search(task, budget)
    
    def _bayesian_optimization(self, 
                             task: OptimizationTask, 
                             budget: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Bayesian optimization implementation."""
        
        # Simple Bayesian optimization using Gaussian Process
        best_params = None
        best_score = float('-inf')
        
        # Collect initial samples
        X_samples = []
        y_samples = []
        
        n_initial = min(5, budget.get('max_evaluations', 20) // 4)
        
        for _ in range(n_initial):
            params = self._random_sample(task.search_space)
            score = task.objective_function(params)
            
            X_samples.append(self._params_to_vector(params, task.search_space))
            y_samples.append(score)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        # Bayesian optimization loop
        if len(X_samples) > 1:
            gp = GaussianProcessRegressor()
            
            for _ in range(budget.get('max_evaluations', 20) - n_initial):
                # Fit GP
                gp.fit(np.array(X_samples), np.array(y_samples))
                
                # Acquisition function optimization
                next_params = self._optimize_acquisition(gp, task.search_space, X_samples)
                score = task.objective_function(next_params)
                
                # Update samples
                X_samples.append(self._params_to_vector(next_params, task.search_space))
                y_samples.append(score)
                
                if score > best_score:
                    best_score = score
                    best_params = next_params
        
        return best_params or self._random_sample(task.search_space), best_score
    
    def _genetic_algorithm(self, 
                         task: OptimizationTask, 
                         budget: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Genetic algorithm implementation."""
        
        population_size = min(20, budget.get('max_evaluations', 50) // 5)
        n_generations = budget.get('max_evaluations', 50) // population_size
        
        # Initialize population
        population = [self._random_sample(task.search_space) for _ in range(population_size)]
        fitness_scores = [task.objective_function(individual) for individual in population]
        
        best_idx = np.argmax(fitness_scores)
        best_params = population[best_idx]
        best_score = fitness_scores[best_idx]
        
        # Evolution
        for generation in range(n_generations):
            # Selection
            selected_indices = self._tournament_selection(fitness_scores, population_size // 2)
            selected_population = [population[i] for i in selected_indices]
            
            # Crossover and mutation
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(len(selected_population), 2, replace=False)
                child = self._crossover(
                    selected_population[parent1], 
                    selected_population[parent2], 
                    task.search_space
                )
                child = self._mutate(child, task.search_space)
                new_population.append(child)
            
            # Evaluate new population
            population = new_population
            fitness_scores = [task.objective_function(individual) for individual in population]
            
            # Update best
            generation_best_idx = np.argmax(fitness_scores)
            if fitness_scores[generation_best_idx] > best_score:
                best_score = fitness_scores[generation_best_idx]
                best_params = population[generation_best_idx]
        
        return best_params, best_score
    
    def _random_search(self, 
                      task: OptimizationTask, 
                      budget: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Random search optimization."""
        
        best_params = None
        best_score = float('-inf')
        
        for _ in range(budget.get('max_evaluations', 50)):
            params = self._random_sample(task.search_space)
            score = task.objective_function(params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params or self._random_sample(task.search_space), best_score
    
    def _random_sample(self, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        return {param: np.random.choice(values) for param, values in search_space.items()}
    
    def _params_to_vector(self, params: Dict[str, Any], search_space: Dict[str, List[Any]]) -> np.ndarray:
        """Convert parameter dictionary to vector."""
        vector = []
        for param_name, param_values in search_space.items():
            if param_name in params:
                try:
                    idx = param_values.index(params[param_name])
                    vector.append(idx / len(param_values))  # Normalize
                except ValueError:
                    vector.append(0.5)  # Default to middle if not found
            else:
                vector.append(0.5)
        return np.array(vector)
    
    def _optimize_acquisition(self, gp, search_space, X_samples) -> Dict[str, Any]:
        """Optimize acquisition function (Expected Improvement)."""
        
        best_ei = -np.inf
        best_params = None
        
        # Sample candidates and evaluate acquisition
        for _ in range(100):
            candidate_params = self._random_sample(search_space)
            candidate_vector = self._params_to_vector(candidate_params, search_space)
            
            # Calculate Expected Improvement
            if len(X_samples) > 0:
                mu, sigma = gp.predict([candidate_vector], return_std=True)
                current_best = max([gp.predict([x])[0] for x in X_samples])
                
                if sigma[0] > 0:
                    z = (mu[0] - current_best) / sigma[0]
                    ei = sigma[0] * (z * norm.cdf(z) + norm.pdf(z))
                else:
                    ei = 0
                
                if ei > best_ei:
                    best_ei = ei
                    best_params = candidate_params
        
        return best_params or self._random_sample(search_space)
    
    def _tournament_selection(self, fitness_scores: List[float], n_selected: int) -> List[int]:
        """Tournament selection for genetic algorithm."""
        selected = []
        for _ in range(n_selected):
            tournament_indices = np.random.choice(len(fitness_scores), 3, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(winner_idx)
        return selected
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                  search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Crossover operation for genetic algorithm."""
        child = {}
        for param_name in search_space:
            if np.random.random() < 0.5:
                child[param_name] = parent1.get(param_name, np.random.choice(search_space[param_name]))
            else:
                child[param_name] = parent2.get(param_name, np.random.choice(search_space[param_name]))
        return child
    
    def _mutate(self, individual: Dict[str, Any], search_space: Dict[str, List[Any]], 
               mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        for param_name, param_values in search_space.items():
            if np.random.random() < mutation_rate:
                mutated[param_name] = np.random.choice(param_values)
        return mutated
    
    def _combine_optimization_results(self, results: List[OptimizationTask]) -> Tuple[Dict[str, Any], float]:
        """Combine results from multiple optimization tasks."""
        
        if not results:
            return {}, 0.0
        
        # Find best result
        best_task = None
        best_score = float('-inf')
        
        for task in results:
            if task.result and len(task.result) > 1:
                if task.result[1] > best_score:
                    best_score = task.result[1]
                    best_task = task
        
        if best_task and best_task.result:
            return best_task.result
        else:
            return {}, 0.0
    
    def _calculate_optimization_metrics(self, 
                                      results: List[OptimizationTask], 
                                      total_time: float,
                                      scale: OptimizationScale) -> MultiScaleMetrics:
        """Calculate comprehensive optimization metrics."""
        
        # Algorithm performance
        algorithm_performance = {}
        for task in results:
            if task.algorithm not in algorithm_performance:
                algorithm_performance[task.algorithm] = []
            if 'score' in task.performance_metrics:
                algorithm_performance[task.algorithm].append(task.performance_metrics['score'])
        
        # Average performance per algorithm
        avg_algorithm_performance = {}
        for algo, scores in algorithm_performance.items():
            avg_algorithm_performance[algo] = np.mean(scores) if scores else 0.0
        
        # Scale utilization
        scale_utilization = {scale.name: len(results)}
        
        # Quantum advantage
        quantum_advantage = 1.0
        quantum_tasks = [t for t in results if 'quantum' in t.algorithm]
        if quantum_tasks:
            quantum_ratios = [
                t.performance_metrics.get('quantum_advantage_ratio', 1.0) 
                for t in quantum_tasks
            ]
            quantum_advantage = np.mean(quantum_ratios) if quantum_ratios else 1.0
        
        # Resource efficiency
        resource_utilization = self.resource_manager.get_resource_utilization()
        resource_efficiency = 1.0 - np.mean([
            util for util in resource_utilization.values() 
            if isinstance(util, (int, float))
        ])
        
        # Convergence rate (simplified)
        convergence_rate = 1.0 / max(total_time, 1.0)
        
        # Solution quality (best score normalized)
        solution_quality = 0.0
        if results:
            best_scores = [
                task.performance_metrics.get('score', 0.0) 
                for task in results if 'score' in task.performance_metrics
            ]
            if best_scores:
                solution_quality = max(best_scores)
        
        return MultiScaleMetrics(
            total_optimization_time=total_time,
            algorithm_performance=avg_algorithm_performance,
            scale_utilization=scale_utilization,
            quantum_advantage_achieved=quantum_advantage,
            resource_efficiency=max(0.0, resource_efficiency),
            convergence_rate=convergence_rate,
            solution_quality=solution_quality,
            adaptive_decisions=len(results)  # Number of algorithm selections made
        )
    
    def _get_algorithm_performance_history(self) -> Dict[str, float]:
        """Get recent algorithm performance history."""
        performance_history = {}
        
        for algo_name, scores in self.algorithm_selector.performance_history.items():
            if scores:
                performance_history[algo_name] = np.mean(scores[-10:])  # Last 10 scores
        
        return performance_history
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        
        if not self.optimization_history:
            return {'status': 'no_optimizations'}
        
        recent_optimization = self.optimization_history[-1]
        
        # Calculate averages across all optimizations
        all_scores = [opt['best_score'] for opt in self.optimization_history]
        all_times = [opt['total_time'] for opt in self.optimization_history]
        
        # Scale usage statistics
        scale_usage = defaultdict(int)
        for opt in self.optimization_history:
            scale_usage[opt['scale']] += 1
        
        # Algorithm success rates
        algorithm_stats = defaultdict(lambda: {'uses': 0, 'scores': []})
        for opt in self.optimization_history:
            for algo, score in opt['algorithm_performance'].items():
                algorithm_stats[algo]['uses'] += 1
                algorithm_stats[algo]['scores'].append(score)
        
        algorithm_summary = {}
        for algo, stats in algorithm_stats.items():
            algorithm_summary[algo] = {
                'usage_count': stats['uses'],
                'average_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                'success_rate': len([s for s in stats['scores'] if s > 0]) / max(1, len(stats['scores']))
            }
        
        return {
            'total_optimizations': len(self.optimization_history),
            'recent_performance': {
                'best_score': recent_optimization['best_score'],
                'total_time': recent_optimization['total_time'],
                'scale_used': recent_optimization['scale'],
                'algorithms_used': list(recent_optimization['algorithm_performance'].keys())
            },
            'overall_statistics': {
                'average_score': np.mean(all_scores),
                'average_time': np.mean(all_times),
                'best_score_ever': max(all_scores) if all_scores else 0.0,
                'fastest_optimization': min(all_times) if all_times else 0.0
            },
            'scale_usage': dict(scale_usage),
            'algorithm_performance': algorithm_summary,
            'resource_utilization': self.resource_manager.get_resource_utilization(),
            'quantum_accelerator_stats': (
                self.quantum_accelerator.get_acceleration_summary() 
                if hasattr(self.quantum_accelerator, 'get_acceleration_summary') 
                else {}
            )
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.executor.shutdown(wait=True)


# Helper functions for easy setup
def create_multi_scale_optimizer(
    quantum_backend: str = 'simulator',
    enable_quantum_acceleration: bool = True,
    max_concurrent_tasks: int = 4
) -> MultiScaleOptimizer:
    """Create and configure a multi-scale optimizer."""
    
    return MultiScaleOptimizer(
        quantum_backend=quantum_backend,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_quantum_acceleration=enable_quantum_acceleration
    )


def optimize_hyperparameters(
    objective_function: Callable,
    search_space: Dict[str, List[Any]],
    time_budget: int = 300,
    quantum_backend: str = 'simulator'
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    High-level function for hyperparameter optimization.
    
    Args:
        objective_function: Function to optimize (higher is better)
        search_space: Dictionary defining parameter search space
        time_budget: Time budget in seconds
        quantum_backend: Quantum backend to use
        
    Returns:
        Tuple of (best_parameters, optimization_summary)
    """
    
    optimizer = create_multi_scale_optimizer(
        quantum_backend=quantum_backend,
        enable_quantum_acceleration=True
    )
    
    with optimizer:
        best_params, metrics = optimizer.optimize(
            objective_function=objective_function,
            search_space=search_space,
            optimization_budget={'time_seconds': time_budget, 'max_evaluations': time_budget // 3}
        )
        
        summary = optimizer.get_optimization_summary()
        summary['final_metrics'] = {
            'total_time': metrics.total_optimization_time,
            'quantum_advantage': metrics.quantum_advantage_achieved,
            'overall_score': metrics.overall_score()
        }
    
    return best_params, summary