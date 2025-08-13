"""
Highly optimized QuantumHyperSearch implementation with performance enhancements,
caching, parallel processing, and advanced optimization strategies.

Generation 3: MAKE IT SCALE - Enhanced with enterprise-grade scaling,
distributed computing, auto-scaling, and performance optimization.
"""

import logging
import time
import warnings
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import numpy as np
from sklearn.model_selection import cross_val_score

from .backends.backend_factory import get_backend
from .core.qubo_encoder import QUBOEncoder
from .utils.validation import (
    validate_search_space, validate_model_class, validate_data,
    validate_optimization_params, ValidationError
)
from .utils.security import (
    sanitize_parameters, check_safety, generate_session_id, SecurityError
)
from .utils.enterprise_scaling import (
    EnterpriseScalingManager, AdaptiveResourceManager, PerformanceOptimizer,
    LoadBalancer, DistributedOptimizer, enterprise_scaling_manager
)
from .utils.robust_error_handling import (
    handle_optimization_errors, handle_quantum_errors
)
from .utils.comprehensive_monitoring import (
    monitor_performance, global_monitor
)

logger = logging.getLogger(__name__)


class CachedEvaluator:
    """High-performance cached evaluator with smart caching strategies."""
    
    def __init__(self, max_cache_size: int = 10000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
    def _generate_cache_key(self, params: Dict, model_class: type, X_shape: tuple, 
                          y_shape: tuple, cv_folds: int, scoring: str) -> str:
        """Generate deterministic cache key."""
        key_parts = [
            str(sorted(params.items())),
            model_class.__name__,
            str(X_shape),
            str(y_shape),
            str(cv_folds),
            str(scoring)
        ]
        return hash(tuple(key_parts))
    
    def get(self, params: Dict, model_class: type, X_shape: tuple, 
           y_shape: tuple, cv_folds: int, scoring: str) -> Optional[float]:
        """Get cached evaluation result."""
        key = self._generate_cache_key(params, model_class, X_shape, y_shape, cv_folds, scoring)
        
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, params: Dict, model_class: type, X_shape: tuple, y_shape: tuple, 
           cv_folds: int, scoring: str, score: float):
        """Cache evaluation result."""
        key = self._generate_cache_key(params, model_class, X_shape, y_shape, cv_folds, scoring)
        
        with self.lock:
            # Implement LRU-like behavior
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entries (simplified)
                keys_to_remove = list(self.cache.keys())[:self.max_cache_size // 4]
                for k in keys_to_remove:
                    del self.cache[k]
            
            self.cache[key] = score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        return {
            'cache_size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.hit_count / max(1, total)
        }


class AdaptiveQuantumStrategy:
    """Adaptive quantum sampling strategy that learns and optimizes."""
    
    def __init__(self):
        self.success_history = []
        self.energy_history = []
        self.param_success_rate = {}
        self.iteration_count = 0
        
    def update_results(self, params: Dict, score: float, quantum_energy: float, 
                      was_successful: bool):
        """Update strategy based on results."""
        self.iteration_count += 1
        self.success_history.append(was_successful)
        self.energy_history.append(quantum_energy)
        
        # Track parameter success rates
        param_key = str(sorted(params.items()))
        if param_key not in self.param_success_rate:
            self.param_success_rate[param_key] = {'success': 0, 'total': 0}
        
        self.param_success_rate[param_key]['total'] += 1
        if was_successful:
            self.param_success_rate[param_key]['success'] += 1
    
    def get_adaptive_reads(self, base_reads: int) -> int:
        """Get adaptive number of quantum reads based on history."""
        if len(self.success_history) < 3:
            return base_reads
        
        # Increase reads if success rate is low
        recent_success_rate = np.mean(self.success_history[-5:])
        
        if recent_success_rate < 0.3:
            return min(base_reads * 2, 1000)
        elif recent_success_rate > 0.8:
            return max(base_reads // 2, 10)
        else:
            return base_reads
    
    def should_focus_exploration(self) -> bool:
        """Determine if we should focus on exploration vs exploitation."""
        if len(self.success_history) < 10:
            return True
        
        # If we haven't had improvements recently, explore more
        recent_improvements = np.sum(np.diff(self.energy_history[-10:]) < 0)
        return recent_improvements < 2


class OptimizedOptimizationHistory:
    """Optimized history tracking with efficient data structures."""
    
    def __init__(self):
        # Use efficient data structures
        self.trials = []
        self.scores = np.array([])
        self.timestamps = np.array([])
        self.quantum_energies = np.array([])
        self.best_score = float('-inf')
        self.best_params = None
        self.best_iteration = -1
        self.n_evaluations = 0
        self.start_time = None
        self.end_time = None
        
        # Performance tracking
        self.evaluation_times = []
        self.quantum_sample_times = []
        
        # Use sets for fast lookups
        self._evaluated_configs = set()
    
    def start_optimization(self):
        """Mark optimization start."""
        self.start_time = time.time()
    
    def end_optimization(self):
        """Mark optimization end."""
        self.end_time = time.time()
    
    def add_evaluation(self, params: Dict[str, Any], score: float, iteration: int, 
                      quantum_energy: float = None, evaluation_time: float = None):
        """Add evaluation with optimized storage."""
        # Check for duplicates efficiently
        param_key = tuple(sorted(params.items()))
        if param_key in self._evaluated_configs:
            return False  # Skip duplicate
        
        self._evaluated_configs.add(param_key)
        
        # Append to efficient arrays
        self.trials.append(params)
        self.scores = np.append(self.scores, score)
        self.timestamps = np.append(self.timestamps, time.time())
        
        if quantum_energy is not None:
            self.quantum_energies = np.append(self.quantum_energies, quantum_energy)
        
        if evaluation_time is not None:
            self.evaluation_times.append(evaluation_time)
        
        self.n_evaluations += 1
        
        # Update best efficiently
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            self.best_iteration = iteration
        
        return True  # Successfully added
    
    def get_top_configurations(self, n: int = 5) -> List[Tuple[Dict, float]]:
        """Get top N configurations efficiently."""
        if len(self.scores) == 0:
            return []
        
        top_indices = np.argsort(self.scores)[-n:][::-1]
        return [(self.trials[i], self.scores[i]) for i in top_indices]
    
    def get_recent_trend(self, window: int = 10) -> str:
        """Get recent optimization trend."""
        if len(self.scores) < window:
            return "insufficient_data"
        
        recent_scores = self.scores[-window:]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.001:
            return "improving"
        elif trend < -0.001:
            return "declining"
        else:
            return "stable"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        stats = {
            'n_evaluations': self.n_evaluations,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'best_iteration': self.best_iteration,
            'duration_seconds': duration,
            'evaluations_per_second': self.n_evaluations / max(1, duration),
            'unique_configurations': len(self._evaluated_configs),
            'duplicate_rate': 1 - len(self._evaluated_configs) / max(1, self.n_evaluations),
        }
        
        if len(self.scores) > 1:
            stats.update({
                'score_mean': float(np.mean(self.scores)),
                'score_std': float(np.std(self.scores)),
                'score_range': float(np.ptp(self.scores)),
                'improvement_rate': np.sum(np.diff(self.scores) > 0) / max(1, len(self.scores) - 1),
                'recent_trend': self.get_recent_trend()
            })
        
        if self.evaluation_times:
            stats['avg_evaluation_time'] = np.mean(self.evaluation_times)
            stats['total_evaluation_time'] = np.sum(self.evaluation_times)
        
        return stats


class QuantumHyperSearchOptimized:
    """
    Highly optimized quantum hyperparameter search with performance enhancements,
    caching, parallel processing, and advanced optimization strategies.
    """
    
    def __init__(
        self,
        backend: str = "simple",
        encoding: str = "one_hot",
        penalty_strength: float = 2.0,
        enable_security: bool = True,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        enable_enterprise_scaling: bool = True,
        enable_monitoring: bool = True,
        max_workers: int = None,
        cache_size: int = 10000,
        adaptive_strategy: bool = True,
        **kwargs
    ):
        """
        Initialize optimized quantum hyperparameter search.
        
        Args:
            backend: Backend name
            encoding: QUBO encoding method
            penalty_strength: Penalty strength for constraints
            enable_security: Enable security validation
            enable_caching: Enable result caching
            enable_parallel: Enable parallel evaluation
            enable_enterprise_scaling: Enable enterprise scaling features
            enable_monitoring: Enable comprehensive monitoring
            max_workers: Maximum parallel workers
            cache_size: Maximum cache size
            adaptive_strategy: Use adaptive quantum strategy
            **kwargs: Additional backend parameters
        """
        self.backend_name = backend
        self.encoding = encoding
        self.penalty_strength = penalty_strength
        self.enable_security = enable_security
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.enable_enterprise_scaling = enable_enterprise_scaling
        self.enable_monitoring = enable_monitoring
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.adaptive_strategy = adaptive_strategy
        self.session_id = generate_session_id()
        
        # Initialize enterprise scaling
        if enable_enterprise_scaling:
            self.scaling_manager = EnterpriseScalingManager()
            self.scaling_manager.start()
            self.performance_optimizer = PerformanceOptimizer()
        else:
            self.scaling_manager = None
            self.performance_optimizer = None
        
        # Initialize monitoring
        if enable_monitoring:
            self.monitor = global_monitor
        else:
            self.monitor = None
        
        # Initialize performance components
        if self.enable_caching:
            self.cache = CachedEvaluator(max_cache_size=cache_size)
        else:
            self.cache = None
            
        if self.adaptive_strategy:
            self.strategy = AdaptiveQuantumStrategy()
        else:
            self.strategy = None
        
        # Initialize core components
        self._initialize_components(**kwargs)
        
        logger.info(f"Initialized optimized QuantumHyperSearch with {self.max_workers} workers")
        
    def _initialize_components(self, **kwargs):
        """Initialize components with optimized settings."""
        # Initialize backend
        self.backend = get_backend(self.backend_name, **kwargs)
        
        # Initialize QUBO encoder
        self.encoder = QUBOEncoder(
            encoding=self.encoding, 
            penalty_strength=self.penalty_strength
        )
        
        # Initialize optimized history
        self.history = OptimizedOptimizationHistory()
    
    @handle_optimization_errors
    @monitor_performance("optimized_quantum_optimization", "seconds", {"component": "optimized_main"})
    def optimize(
        self,
        model_class: type,
        param_space: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 10,
        quantum_reads: int = 100,
        cv_folds: int = 3,
        scoring: str = "accuracy",
        batch_size: int = 10,
        **model_kwargs
    ) -> Tuple[Dict[str, Any], OptimizedOptimizationHistory]:
        """
        Run optimized quantum hyperparameter search.
        
        Args:
            model_class: Model class to optimize
            param_space: Parameter search space
            X: Training features
            y: Training targets
            n_iterations: Number of optimization iterations
            quantum_reads: Number of quantum reads per iteration
            cv_folds: Cross-validation folds
            scoring: Scoring metric
            batch_size: Batch size for parallel evaluation
            **model_kwargs: Additional model parameters
            
        Returns:
            Tuple of (best_parameters, optimization_history)
        """
        self.history.start_optimization()
        
        try:
            # Optimized input validation
            self._validate_inputs_optimized(model_class, param_space, X, y, n_iterations, 
                                          quantum_reads, cv_folds, model_kwargs)
            
            print(f"🚀 Starting optimized quantum search (session: {self.session_id})")
            print(f"⚡ Performance: Caching={self.enable_caching}, Parallel={self.enable_parallel}")
            print(f"👥 Workers: {self.max_workers}, Batch size: {batch_size}")
            
            # Fast preliminary evaluation
            preliminary_scores = self._get_preliminary_scores_optimized(
                model_class, param_space, X, y, cv_folds, scoring, **model_kwargs
            )
            
            # Optimized QUBO encoding
            Q, offset, variable_map = self.encoder.encode_search_space(param_space, self.history)
            print(f"🔢 QUBO optimization: {len(variable_map)} variables, {len(Q)} terms")
            
            # Main optimization loop with performance enhancements
            return self._run_optimized_loop(
                model_class, param_space, X, y, n_iterations, quantum_reads,
                cv_folds, scoring, Q, variable_map, batch_size, **model_kwargs
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._get_best_result()
        finally:
            self.history.end_optimization()
            self._print_performance_summary()
    
    def _validate_inputs_optimized(self, model_class, param_space, X, y, n_iterations, 
                                 quantum_reads, cv_folds, model_kwargs):
        """Fast optimized input validation."""
        # Use cached validation where possible
        param_space = validate_search_space(param_space)
        X, y = validate_data(X, y)
        validate_model_class(model_class)
        validate_optimization_params(n_iterations, quantum_reads, cv_folds)
        
        if self.enable_security:
            check_safety(param_space, model_class)
    
    def _get_preliminary_scores_optimized(self, model_class, param_space, X, y, 
                                        cv_folds, scoring, **model_kwargs):
        """Fast preliminary evaluation with smart sampling."""
        print("⚡ Fast preliminary evaluation...")
        
        # Use smart sampling for preliminary evaluation
        n_samples = min(3, len(list(self._generate_smart_samples(param_space, 5))))
        samples = list(self._generate_smart_samples(param_space, n_samples))
        
        if self.enable_parallel and len(samples) > 1:
            # Parallel preliminary evaluation
            results = self._evaluate_batch_parallel(
                samples, model_class, X, y, cv_folds, scoring, **model_kwargs
            )
        else:
            # Sequential evaluation
            results = []
            for params in samples:
                score = self._evaluate_single_cached(
                    params, model_class, X, y, cv_folds, scoring, **model_kwargs
                )
                if score is not None:
                    results.append((params, score))
        
        preliminary_scores = {str(sorted(params.items())): score 
                            for params, score in results if score is not None}
        
        print(f"📊 Preliminary: {len(results)}/{n_samples} successful")
        return preliminary_scores
    
    def _generate_smart_samples(self, param_space: Dict, n_samples: int):
        """Generate smart parameter samples using various strategies."""
        strategies = [
            self._sample_corners,      # Corner cases
            self._sample_random,       # Random sampling
            self._sample_balanced,     # Balanced across parameters
        ]
        
        samples_per_strategy = max(1, n_samples // len(strategies))
        
        for strategy in strategies:
            for _ in range(samples_per_strategy):
                yield strategy(param_space)
    
    def _sample_corners(self, param_space: Dict) -> Dict:
        """Sample corner/extreme values."""
        return {param: np.random.choice([values[0], values[-1]]) 
               for param, values in param_space.items()}
    
    def _sample_random(self, param_space: Dict) -> Dict:
        """Random sampling."""
        return {param: np.random.choice(values) 
               for param, values in param_space.items()}
    
    def _sample_balanced(self, param_space: Dict) -> Dict:
        """Sample balanced across parameter ranges."""
        return {param: values[len(values) // 2] if values else values[0]
               for param, values in param_space.items()}
    
    def _run_optimized_loop(self, model_class, param_space, X, y, n_iterations,
                          quantum_reads, cv_folds, scoring, Q, variable_map,
                          batch_size, **model_kwargs):
        """Run optimization loop with performance optimizations."""
        
        for iteration in range(n_iterations):
            print(f"\n⚡ Iteration {iteration + 1}/{n_iterations}")
            
            # Adaptive quantum reads
            if self.adaptive_strategy:
                current_reads = self.strategy.get_adaptive_reads(quantum_reads)
                print(f"🎯 Adaptive reads: {current_reads}")
            else:
                current_reads = quantum_reads
            
            # Quantum sampling
            samples = self.backend.sample_qubo(Q, num_reads=current_reads)
            
            if not samples or not hasattr(samples, 'record'):
                continue
            
            # Batch decode samples
            param_configs = self._decode_samples_batch(
                samples.record[:batch_size * 2], variable_map, param_space
            )
            
            # Remove duplicates efficiently
            unique_configs = []
            seen_configs = set()
            for params in param_configs:
                param_key = tuple(sorted(params.items()))
                if param_key not in seen_configs:
                    seen_configs.add(param_key)
                    unique_configs.append(params)
                if len(unique_configs) >= batch_size:
                    break
            
            # Parallel batch evaluation
            if self.enable_parallel and len(unique_configs) > 1:
                results = self._evaluate_batch_parallel(
                    unique_configs, model_class, X, y, cv_folds, scoring, **model_kwargs
                )
            else:
                results = self._evaluate_batch_sequential(
                    unique_configs, model_class, X, y, cv_folds, scoring, **model_kwargs
                )
            
            # Update history and strategy
            improvements = 0
            for params, score in results:
                if score is not None:
                    was_added = self.history.add_evaluation(params, score, iteration)
                    if was_added and score > self.history.best_score - score:
                        improvements += 1
                        print(f"🎉 New best: {score:.4f} - {params}")
                    
                    # Update adaptive strategy
                    if self.adaptive_strategy:
                        quantum_energy = getattr(samples.record[0], 'energy', 0.0)
                        self.strategy.update_results(params, score, quantum_energy, score > 0.5)
            
            print(f"📊 Batch: {len(results)} evaluated, {improvements} improvements")
            
            # Early termination based on trend
            if iteration > 5:
                trend = self.history.get_recent_trend()
                if trend == "declining" and self.history.n_evaluations > 20:
                    print("📈 Performance trend suggests early termination")
                    break
        
        return self._get_best_result()
    
    def _decode_samples_batch(self, samples, variable_map, param_space):
        """Decode multiple samples efficiently."""
        param_configs = []
        for sample_result in samples:
            try:
                params = self.encoder.decode_sample(
                    sample_result.sample, variable_map, param_space
                )
                if params:
                    param_configs.append(params)
            except Exception as e:
                logger.debug(f"Sample decode failed: {e}")
                continue
        return param_configs
    
    def _evaluate_batch_parallel(self, param_configs, model_class, X, y, 
                                cv_folds, scoring, **model_kwargs):
        """Evaluate parameter configurations in parallel."""
        results = []
        
        # Use ThreadPoolExecutor for I/O bound sklearn operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(
                    self._evaluate_single_cached, params, model_class, 
                    X, y, cv_folds, scoring, **model_kwargs
                ): params 
                for params in param_configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params, timeout=60):
                params = future_to_params[future]
                try:
                    score = future.result(timeout=30)
                    if score is not None:
                        results.append((params, score))
                except Exception as e:
                    logger.debug(f"Parallel evaluation failed for {params}: {e}")
        
        return results
    
    def _evaluate_batch_sequential(self, param_configs, model_class, X, y,
                                 cv_folds, scoring, **model_kwargs):
        """Evaluate parameter configurations sequentially."""
        results = []
        
        for params in param_configs:
            score = self._evaluate_single_cached(
                params, model_class, X, y, cv_folds, scoring, **model_kwargs
            )
            if score is not None:
                results.append((params, score))
        
        return results
    
    def _evaluate_single_cached(self, params, model_class, X, y, cv_folds, scoring, **model_kwargs):
        """Evaluate single configuration with caching."""
        # Check cache first
        if self.cache:
            cached_score = self.cache.get(params, model_class, X.shape, y.shape, cv_folds, scoring)
            if cached_score is not None:
                return cached_score
        
        # Evaluate
        start_time = time.time()
        try:
            if self.enable_security:
                params = sanitize_parameters(params)
            
            all_params = {**params, **model_kwargs}
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model_class(**all_params)
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                score = float(np.mean(scores))
            
            # Cache result
            if self.cache:
                self.cache.put(params, model_class, X.shape, y.shape, cv_folds, scoring, score)
            
            # Record timing
            evaluation_time = time.time() - start_time
            self.history.evaluation_times.append(evaluation_time)
            
            return score
            
        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")
            return None
    
    def _get_best_result(self):
        """Get best optimization result."""
        if self.history.best_params is not None:
            return self.history.best_params, self.history
        else:
            return {}, self.history
    
    def _print_performance_summary(self):
        """Print comprehensive performance summary."""
        stats = self.history.get_statistics()
        
        print(f"\n⚡ Performance Summary:")
        print(f"   Duration: {stats['duration_seconds']:.1f}s")
        print(f"   Evaluations: {stats['n_evaluations']} ({stats['evaluations_per_second']:.2f}/sec)")
        print(f"   Unique configs: {stats['unique_configurations']} (duplicate rate: {stats.get('duplicate_rate', 0):.1%})")
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            print(f"   Cache: {cache_stats['cache_size']} entries, {cache_stats['hit_rate']:.1%} hit rate")
        
        if 'avg_evaluation_time' in stats:
            print(f"   Avg evaluation time: {stats['avg_evaluation_time']:.3f}s")
        
        if self.history.best_params:
            print(f"\n🏆 Optimization Results:")
            print(f"   Best score: {stats['best_score']:.4f} (iteration {stats['best_iteration']})")
            if 'recent_trend' in stats:
                print(f"   Recent trend: {stats['recent_trend']}")


# Main alias
QuantumHyperSearch = QuantumHyperSearchOptimized