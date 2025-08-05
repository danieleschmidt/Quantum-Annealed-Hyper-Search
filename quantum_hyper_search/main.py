"""
Main QuantumHyperSearch class for quantum-enhanced hyperparameter optimization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

from .backends import get_backend
from .core.qubo_formulation import QUBOEncoder
from .core.optimization_history import OptimizationHistory
from .utils.logging import OptimizationLogger, get_logger
from .utils.validation import (
    validate_search_space, validate_model_class, validate_data,
    validate_optimization_params, validate_constraints, ValidationError
)
from .utils.security import sanitize_parameters, check_safety, SecurityError, generate_session_id
from .utils.monitoring import OptimizationMonitor
from .optimization.caching import ResultCache, generate_cache_key, adaptive_cache
from .optimization.parallel import ParallelEvaluator, EvaluationTask, AdaptiveScheduler
from .optimization.scaling import AutoScaler, ResourceManager, ScalingPolicy
from .optimization.strategies import AdaptiveStrategy, HybridQuantumClassical, StrategyConfig


class QuantumHyperSearch:
    """
    Quantum-enhanced hyperparameter optimization using D-Wave quantum annealers.
    
    Combines quantum annealing with classical optimization for efficient
    exploration of hyperparameter spaces.
    """
    
    def __init__(
        self,
        backend: str = "simulator",
        token: Optional[str] = None,
        encoding: str = "one_hot",
        penalty_strength: float = 2.0,
        enable_logging: bool = True,
        enable_monitoring: bool = True,
        enable_security: bool = True,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        enable_auto_scaling: bool = True,
        max_parallel_workers: Optional[int] = None,
        cache_size: int = 10000,
        optimization_strategy: str = "adaptive",
        log_level: str = "INFO",
        **backend_kwargs
    ):
        """
        Initialize QuantumHyperSearch.
        
        Args:
            backend: Quantum backend to use ('dwave', 'simulator', etc.)
            token: API token for quantum hardware access
            encoding: QUBO encoding method ('one_hot', 'binary', 'domain_wall')
            penalty_strength: Strength of constraint penalties in QUBO
            enable_logging: Enable comprehensive logging
            enable_monitoring: Enable performance monitoring
            enable_security: Enable security checks
            enable_caching: Enable result caching for performance
            enable_parallel: Enable parallel evaluation of configurations
            enable_auto_scaling: Enable automatic resource scaling
            max_parallel_workers: Maximum parallel workers (None for auto)
            cache_size: Maximum number of cached results
            optimization_strategy: Strategy ('adaptive', 'hybrid', 'bayesian')
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            **backend_kwargs: Additional backend-specific parameters
        """
        # Generate session ID for this optimization run
        self.session_id = generate_session_id()
        
        # Initialize logging
        self.enable_logging = enable_logging
        if enable_logging:
            import logging
            log_level_int = getattr(logging, log_level.upper(), logging.INFO)
            self.logger = get_logger("quantum_hyper_search")
            self.logger.setLevel(log_level_int)
            self.opt_logger = OptimizationLogger(self.logger)
        else:
            self.logger = None
            self.opt_logger = None
        
        # Initialize security
        self.enable_security = enable_security
        
        # Initialize monitoring
        self.monitor = OptimizationMonitor(
            enable_performance=enable_monitoring,
            enable_health=enable_monitoring
        ) if enable_monitoring else None
        
        # Initialize caching
        self.enable_caching = enable_caching
        self.cache = ResultCache(max_size=cache_size) if enable_caching else None
        
        # Initialize parallel processing
        self.enable_parallel = enable_parallel
        self.parallel_evaluator = ParallelEvaluator(
            max_workers=max_parallel_workers
        ) if enable_parallel else None
        
        # Initialize auto-scaling
        self.enable_auto_scaling = enable_auto_scaling
        self.auto_scaler = None
        self.resource_manager = None
        
        if enable_auto_scaling:
            self.resource_manager = ResourceManager()
            self.auto_scaler = AutoScaler()
            
            # Connect auto-scaler to parallel evaluator
            if self.parallel_evaluator:
                def scale_up_callback(new_workers):
                    self.parallel_evaluator.max_workers = new_workers
                def scale_down_callback(new_workers):
                    self.parallel_evaluator.max_workers = new_workers
                    
                self.auto_scaler.set_scaling_callbacks(scale_up_callback, scale_down_callback)
        
        # Initialize optimization strategy
        self.optimization_strategy = self._create_optimization_strategy(optimization_strategy)
        
        # Initialize core components
        try:
            self.backend_name = backend
            self.backend = get_backend(backend)(token=token, **backend_kwargs)
            self.encoder = QUBOEncoder(encoding=encoding, penalty_strength=penalty_strength)
            self.history = OptimizationHistory()
            
            if self.logger:
                self.logger.info(
                    f"QuantumHyperSearch initialized (session: {self.session_id})",
                    extra={
                        'session_id': self.session_id,
                        'backend': backend,
                        'encoding': encoding,
                        'penalty_strength': penalty_strength
                    }
                )
                
        except Exception as e:
            if self.opt_logger:
                self.opt_logger.log_error(e, {'component': 'initialization'})
            raise
    
    def _create_optimization_strategy(self, strategy_name: str):
        """Create optimization strategy based on name."""
        if strategy_name == "adaptive":
            return AdaptiveStrategy()
        elif strategy_name == "hybrid":
            return HybridQuantumClassical()
        else:
            # Default to adaptive
            return AdaptiveStrategy()
        
    def optimize(
        self,
        model_class: type,
        param_space: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 20,
        quantum_reads: int = 1000,
        cv_folds: int = 5,
        scoring: str = "accuracy",
        constraints: Optional[Dict] = None,
        objective_function: Optional[Callable] = None,
        **model_kwargs
    ) -> Tuple[Dict[str, Any], OptimizationHistory]:
        """
        Run quantum-enhanced hyperparameter optimization.
        
        Args:
            model_class: Scikit-learn compatible model class
            param_space: Dictionary of parameter names and possible values
            X: Training features
            y: Training targets
            n_iterations: Number of optimization iterations
            quantum_reads: Number of quantum annealer reads per iteration
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for model evaluation
            constraints: Optional constraints for parameter combinations
            objective_function: Custom objective function (overrides default CV)
            **model_kwargs: Additional model constructor arguments
            
        Returns:
            Tuple of (best_parameters, optimization_history)
            
        Raises:
            ValidationError: If input parameters are invalid
            SecurityError: If security issues are detected
        """
        start_time = time.time()
        
        try:
            # Input validation and security checks
            self._validate_and_secure_inputs(
                model_class, param_space, X, y, n_iterations,
                quantum_reads, cv_folds, constraints, model_kwargs
            )
            
            # Start monitoring and scaling
            if self.monitor:
                if not self.monitor.check_health():
                    raise RuntimeError("System health check failed - cannot start optimization")
                self.monitor.start_monitoring()
                
            if self.auto_scaler:
                self.auto_scaler.start_monitoring()
            
            if self.resource_manager:
                resource_status = self.resource_manager.check_resource_constraints()
                if resource_status['status'] == 'constrained':
                    if self.opt_logger:
                        self.opt_logger.log_warning(
                            "Resource constraints detected",
                            {'resource_warnings': resource_status['warnings']}
                        )
            
            # Log optimization start
            if self.opt_logger:
                self.opt_logger.log_optimization_start(
                    self.backend_name, param_space, n_iterations, quantum_reads
                )
            
            print(f"🌌 Starting quantum hyperparameter optimization with {self.backend_name}")
            print(f"📊 Parameter space: {len(param_space)} parameters")
            print(f"🔢 Total combinations: {np.prod([len(v) for v in param_space.values()])}")
            print(f"🆔 Session ID: {self.session_id}")
            
            return self._run_optimization_loop(
                model_class, param_space, X, y, n_iterations,
                quantum_reads, cv_folds, scoring, constraints,
                objective_function, **model_kwargs
            )
            
        except Exception as e:
            if self.opt_logger:
                self.opt_logger.log_error(e, {
                    'session_id': self.session_id,
                    'backend': self.backend_name,
                    'elapsed_time': time.time() - start_time
                })
            raise
        finally:
            # Stop monitoring and scaling
            if self.monitor:
                self.monitor.stop_monitoring()
                
                # Log final performance report
                if self.logger:
                    report = self.monitor.get_report()
                    self.logger.info(
                        "Optimization monitoring report",
                        extra={
                            'session_id': self.session_id,
                            'performance_report': report
                        }
                    )
            
            if self.auto_scaler:
                self.auto_scaler.stop_monitoring_system()
                
                if self.logger and self.resource_manager:
                    resource_summary = self.resource_manager.get_resource_summary()
                    self.logger.info(
                        "Resource management summary",
                        extra={
                            'session_id': self.session_id,
                            'resource_summary': resource_summary
                        }
                    )
    
    def _validate_and_secure_inputs(
        self,
        model_class: type,
        param_space: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int,
        quantum_reads: int,
        cv_folds: int,
        constraints: Optional[Dict],
        model_kwargs: Dict[str, Any]
    ) -> None:
        """Validate and secure all inputs."""
        if self.logger:
            self.logger.debug("Starting input validation and security checks")
        
        # Validate model class
        validate_model_class(model_class)
        
        # Validate and sanitize search space
        validated_space = validate_search_space(param_space)
        
        # Validate data
        validate_data(X, y)
        
        # Validate optimization parameters
        validate_optimization_params(n_iterations, quantum_reads, cv_folds)
        
        # Validate constraints
        validated_constraints = validate_constraints(constraints, validated_space)
        
        # Security checks
        if self.enable_security:
            check_safety(validated_space, model_class, validated_constraints)
            
            # Sanitize model kwargs
            if model_kwargs:
                sanitized_kwargs = sanitize_parameters(model_kwargs)
                model_kwargs.clear()
                model_kwargs.update(sanitized_kwargs)
        
        if self.logger:
            self.logger.debug("Input validation and security checks passed")
    
    def _run_optimization_loop(
        self,
        model_class: type,
        param_space: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int,
        quantum_reads: int,
        cv_folds: int,
        scoring: str,
        constraints: Optional[Dict],
        objective_function: Optional[Callable],
        **model_kwargs
    ) -> Tuple[Dict[str, Any], OptimizationHistory]:
        """Run the main optimization loop with monitoring."""
        
        # Initialize with random samples for preliminary scoring
        if self.monitor:
            with self.monitor.time_operation("preliminary_scoring"):
                preliminary_scores = self._get_preliminary_scores(
                    model_class, param_space, X, y, cv_folds, scoring, 
                    objective_function, **model_kwargs
                )
        else:
            preliminary_scores = self._get_preliminary_scores(
                model_class, param_space, X, y, cv_folds, scoring, 
                objective_function, **model_kwargs
            )
        
        # Encode to QUBO
        if self.monitor:
            with self.monitor.time_operation("qubo_encoding"):
                Q, offset, variable_map = self.encoder.encode(
                    search_space=param_space,
                    objective_estimates=preliminary_scores,
                    constraints=constraints or {}
                )
        else:
            Q, offset, variable_map = self.encoder.encode(
                search_space=param_space,
                objective_estimates=preliminary_scores,
                constraints=constraints or {}
            )
        
        print(f"🔢 QUBO matrix size: {Q.shape}")
        
        best_score = float('-inf')
        best_params = None
        
        for iteration in range(n_iterations):
            if self.opt_logger:
                self.opt_logger.log_iteration_start(iteration + 1, n_iterations)
            
            print(f"\n🔄 Iteration {iteration + 1}/{n_iterations}")
            
            # Health check
            if self.monitor and not self.monitor.check_health():
                if self.opt_logger:
                    self.opt_logger.log_warning("Health check failed during optimization")
                print("⚠️  System health check failed - continuing with caution")
            
            # Sample from quantum annealer
            if self.monitor:
                with self.monitor.time_operation("quantum_sampling"):
                    samples = self.backend.sample_qubo(Q, num_reads=quantum_reads)
            else:
                samples = self.backend.sample_qubo(Q, num_reads=quantum_reads)
            
            if self.opt_logger and samples:
                # Estimate energy of best sample
                best_energy = self._estimate_sample_energy(samples[0], Q) if samples else float('inf')
                self.opt_logger.log_quantum_sampling(self.backend_name, quantum_reads, best_energy)
            
            # Decode and evaluate samples
            best_score, best_params = self._evaluate_samples(
                samples, variable_map, param_space, model_class, X, y,
                cv_folds, scoring, objective_function, iteration,
                best_score, best_params, **model_kwargs
            )
            
            # Adaptive QUBO update
            if iteration < n_iterations - 1 and best_params:
                Q = self._update_qubo_bias(Q, variable_map, best_params, param_space)
        
        # Log optimization completion
        elapsed_time = time.time() - (self.monitor.performance_monitor.metrics.start_time.timestamp() 
                                     if self.monitor and self.monitor.performance_monitor 
                                     else time.time())
        
        if self.opt_logger:
            self.opt_logger.log_optimization_complete(
                best_score, best_params, self.history.n_evaluations, elapsed_time
            )
        
        print(f"\n✅ Optimization complete!")
        print(f"🏆 Best score: {best_score:.4f}")
        print(f"🎯 Best parameters: {best_params}")
        
        return best_params, self.history
    
    def _evaluate_samples(
        self,
        samples: List[Dict[int, int]],
        variable_map: Dict[str, int],
        param_space: Dict[str, List[Any]],
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        scoring: str,
        objective_function: Optional[Callable],
        iteration: int,
        current_best_score: float,
        current_best_params: Optional[Dict[str, Any]],
        **model_kwargs
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Evaluate samples using advanced optimization strategies."""
        best_score = current_best_score
        best_params = current_best_params
        
        # Use optimization strategy to select configurations
        if self.optimization_strategy:
            param_configs = self.optimization_strategy.select_configurations(
                samples, variable_map, param_space, self.history, n_select=10
            )
        else:
            # Fallback to simple decoding
            param_configs = []
            for sample in samples[:min(10, len(samples))]:
                try:
                    params = self.encoder.decode_sample(sample, variable_map, param_space)
                    if params not in param_configs:
                        param_configs.append(params)
                except Exception as e:
                    if self.opt_logger:
                        self.opt_logger.log_warning(f"Could not decode sample: {e}")
                    continue
        
        # Use parallel evaluation if enabled
        if self.enable_parallel and self.parallel_evaluator and len(param_configs) > 1:
            results = self._evaluate_parallel(
                param_configs, model_class, X, y, cv_folds, scoring,
                objective_function, iteration, **model_kwargs
            )
        else:
            # Sequential evaluation
            results = self._evaluate_sequential(
                param_configs, model_class, X, y, cv_folds, scoring,
                objective_function, iteration, **model_kwargs
            )
        
        # Process results and update best
        current_results = []
        for result in results:
            if result.success:
                # Track in history and monitoring
                self.history.add_evaluation(result.parameters, result.score, iteration)
                if self.monitor:
                    self.monitor.record_evaluation(result.score, success=True)
                
                # Check if this is a new best
                is_best = result.score > best_score
                if is_best:
                    best_score = result.score
                    best_params = result.parameters.copy()
                    print(f"🎉 New best score: {best_score:.4f}")
                    print(f"🎯 Best params: {best_params}")
                
                # Log evaluation
                if self.opt_logger:
                    self.opt_logger.log_evaluation(result.parameters, result.score, iteration, is_best)
                
                current_results.append((result.parameters, result.score))
            else:
                if self.opt_logger:
                    self.opt_logger.log_warning(f"Evaluation failed: {result.error}")
                if self.monitor:
                    self.monitor.record_evaluation(0.0, success=False)
        
        # Update optimization strategy
        if self.optimization_strategy and current_results:
            self.optimization_strategy.update_strategy(current_results, self.history)
        
        return best_score, best_params
    
    def _evaluate_parallel(
        self,
        param_configs: List[Dict[str, Any]],
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        scoring: str,
        objective_function: Optional[Callable],
        iteration: int,
        **model_kwargs
    ) -> List:
        """Evaluate configurations in parallel."""
        if objective_function:
            # Custom objective function - evaluate sequentially for safety
            return self._evaluate_sequential(
                param_configs, model_class, X, y, cv_folds, scoring,
                objective_function, iteration, **model_kwargs
            )
        
        # Create evaluation tasks and collect cached results
        tasks = []
        cached_results = []
        
        for i, params in enumerate(param_configs):
            # Check cache first
            if self.cache:
                cache_key = generate_cache_key(
                    params, model_class, X.shape, y.shape, cv_folds, scoring
                )
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    # Create mock result for cached value
                    from .optimization.parallel import EvaluationResult
                    result = EvaluationResult(
                        task_id=f"cached_{iteration}_{i}",
                        parameters=params,
                        score=cached_result,
                        success=True
                    )
                    cached_results.append(result)
                    continue
            
            # Create evaluation task
            task = EvaluationTask(
                task_id=f"eval_{iteration}_{i}",
                parameters=params,
                model_class=model_class,
                X=X,
                y=y,
                cv_folds=cv_folds,
                scoring=scoring,
                model_kwargs=model_kwargs
            )
            tasks.append(task)
        
        # Evaluate in parallel
        results = cached_results.copy()  # Start with cached results
        
        if tasks:
            new_results = self.parallel_evaluator.evaluate_batch(tasks)
            results.extend(new_results)
            
            # Cache successful results
            if self.cache:
                for result in new_results:
                    if result.success:
                        cache_key = generate_cache_key(
                            result.parameters, model_class, X.shape, y.shape, cv_folds, scoring
                        )
                        self.cache.put(cache_key, result.score, result.computation_time)
        
        return results
    
    def _evaluate_sequential(
        self,
        param_configs: List[Dict[str, Any]],
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        scoring: str,
        objective_function: Optional[Callable],
        iteration: int,
        **model_kwargs
    ) -> List:
        """Evaluate configurations sequentially."""
        from .optimization.parallel import EvaluationResult
        results = []
        
        for i, params in enumerate(param_configs):
            try:
                # Check cache first
                score = None
                if self.cache and not objective_function:
                    cache_key = generate_cache_key(
                        params, model_class, X.shape, y.shape, cv_folds, scoring
                    )
                    score = self.cache.get(cache_key)
                
                if score is None:
                    # Evaluate configuration
                    start_time = time.time()
                    if self.monitor:
                        with self.monitor.time_operation("evaluation"):
                            score = self._evaluate_configuration(
                                params, model_class, X, y, cv_folds, scoring,
                                objective_function, **model_kwargs
                            )
                    else:
                        score = self._evaluate_configuration(
                            params, model_class, X, y, cv_folds, scoring,
                            objective_function, **model_kwargs
                        )
                    computation_time = time.time() - start_time
                    
                    # Cache result
                    if self.cache and not objective_function:
                        self.cache.put(cache_key, score, computation_time)
                
                result = EvaluationResult(
                    task_id=f"seq_{iteration}_{i}",
                    parameters=params,
                    score=score,
                    success=True
                )
                results.append(result)
                
            except Exception as e:
                result = EvaluationResult(
                    task_id=f"seq_{iteration}_{i}",
                    parameters=params,
                    score=0.0,
                    success=False,
                    error=str(e)
                )
                results.append(result)
        
        return results
    
    def _evaluate_configuration(
        self,
        params: Dict[str, Any],
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        scoring: str,
        objective_function: Optional[Callable],
        **model_kwargs
    ) -> float:
        """Evaluate a single parameter configuration."""
        if objective_function:
            return objective_function(params)
        else:
            # Safely create model with security checks
            if self.enable_security:
                safe_params = sanitize_parameters(params)
                safe_model_kwargs = sanitize_parameters(model_kwargs)
            else:
                safe_params = params
                safe_model_kwargs = model_kwargs
            
            model = model_class(**safe_params, **safe_model_kwargs)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            return scores.mean()
    
    def _estimate_sample_energy(self, sample: Dict[int, int], Q: np.ndarray) -> float:
        """Estimate energy of a sample for logging purposes."""
        try:
            state = np.array([sample.get(i, 0) for i in range(Q.shape[0])])
            return float(state.T @ Q @ state)
        except Exception:
            return float('inf')
    
    def _get_preliminary_scores(
        self,
        model_class: type,
        param_space: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        scoring: str,
        objective_function: Optional[Callable],
        **model_kwargs
    ) -> Dict[str, float]:
        """Get preliminary scores for QUBO initialization."""
        print("Getting preliminary scores for QUBO initialization...")
        
        preliminary_scores = {}
        
        # Sample a few random configurations
        for _ in range(min(5, np.prod([len(v) for v in param_space.values()]))):
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = np.random.choice(param_values)
            
            param_key = str(sorted(params.items()))
            if param_key in preliminary_scores:
                continue
                
            try:
                if objective_function:
                    score = objective_function(params)
                else:
                    model = model_class(**params, **model_kwargs)
                    scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                    score = scores.mean()
                
                preliminary_scores[param_key] = score
                
            except Exception as e:
                print(f"Warning: Preliminary evaluation failed - {e}")
                preliminary_scores[param_key] = 0.0
        
        return preliminary_scores
    
    def _update_qubo_bias(
        self,
        Q: np.ndarray,
        variable_map: Dict,
        best_params: Dict[str, Any],
        param_space: Dict[str, List[Any]]
    ) -> np.ndarray:
        """Update QUBO bias towards current best solution."""
        if not best_params:
            return Q
            
        # Small bias towards current best
        bias_strength = 0.1
        
        for param_name, param_value in best_params.items():
            if param_name in param_space:
                try:
                    param_idx = param_space[param_name].index(param_value)
                    var_name = f"{param_name}_{param_idx}"
                    if var_name in variable_map:
                        var_idx = variable_map[var_name]
                        Q[var_idx, var_idx] -= bias_strength
                except (ValueError, KeyError):
                    continue
        
        return Q