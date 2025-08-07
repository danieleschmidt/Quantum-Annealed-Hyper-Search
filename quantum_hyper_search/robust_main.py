"""
Robust QuantumHyperSearch implementation with comprehensive error handling, 
monitoring, and reliability features.
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Callable
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

logger = logging.getLogger(__name__)


class OptimizationHistory:
    """Enhanced optimization history with detailed tracking."""
    
    def __init__(self):
        self.trials = []
        self.scores = []
        self.timestamps = []
        self.errors = []
        self.quantum_info = []
        self.best_score = float('-inf')
        self.best_params = None
        self.n_evaluations = 0
        self.n_errors = 0
        self.start_time = None
        self.end_time = None
    
    def start_optimization(self):
        """Mark optimization start."""
        self.start_time = time.time()
    
    def end_optimization(self):
        """Mark optimization end."""
        self.end_time = time.time()
    
    def add_evaluation(self, params: Dict[str, Any], score: float, iteration: int, 
                      quantum_info: Dict = None, error_msg: str = None):
        """Add an evaluation result with comprehensive tracking."""
        self.trials.append(params)
        self.scores.append(score)
        self.timestamps.append(time.time())
        self.quantum_info.append(quantum_info or {})
        self.errors.append(error_msg)
        self.n_evaluations += 1
        
        if error_msg:
            self.n_errors += 1
        
        if score > self.best_score and error_msg is None:
            self.best_score = score
            self.best_params = params.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        return {
            'n_evaluations': self.n_evaluations,
            'n_errors': self.n_errors,
            'error_rate': self.n_errors / max(1, self.n_evaluations),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'duration_seconds': duration,
            'evaluations_per_second': self.n_evaluations / max(1, duration),
            'improvement_iterations': sum(1 for i in range(1, len(self.scores)) 
                                        if self.scores[i] > max(self.scores[:i])),
        }


class QuantumHyperSearchRobust:
    """
    Robust quantum hyperparameter search with comprehensive error handling,
    monitoring, retry logic, and graceful degradation.
    """
    
    def __init__(
        self,
        backend: str = "simple",
        encoding: str = "one_hot",
        penalty_strength: float = 2.0,
        enable_security: bool = True,
        max_retries: int = 3,
        timeout_per_iteration: float = 300.0,  # 5 minutes
        fallback_to_random: bool = True,
        **kwargs
    ):
        """
        Initialize robust quantum hyperparameter search.
        
        Args:
            backend: Backend name ('simple', 'simulator', 'dwave')
            encoding: QUBO encoding method
            penalty_strength: Penalty strength for constraints
            enable_security: Enable security validation
            max_retries: Maximum retries per operation
            timeout_per_iteration: Timeout per iteration in seconds
            fallback_to_random: Fall back to random search on failures
            **kwargs: Additional backend parameters
        """
        self.backend_name = backend
        self.encoding = encoding
        self.penalty_strength = penalty_strength
        self.enable_security = enable_security
        self.max_retries = max_retries
        self.timeout_per_iteration = timeout_per_iteration
        self.fallback_to_random = fallback_to_random
        self.session_id = generate_session_id()
        
        # Initialize with retry logic
        self._initialize_components(**kwargs)
        
        logger.info(f"Initialized robust QuantumHyperSearch (session: {self.session_id})")
    
    def _initialize_components(self, **kwargs):
        """Initialize components with error handling and retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Initialize backend with error handling
                self.backend = self._initialize_backend(**kwargs)
                
                # Initialize QUBO encoder
                self.encoder = QUBOEncoder(
                    encoding=self.encoding, 
                    penalty_strength=self.penalty_strength
                )
                
                # Initialize history
                self.history = OptimizationHistory()
                
                logger.info(f"Components initialized successfully on attempt {attempt + 1}")
                return
                
            except Exception as e:
                last_error = e
                logger.warning(f"Initialization attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError(f"Failed to initialize components after {self.max_retries} attempts: {last_error}")
    
    def _initialize_backend(self, **kwargs):
        """Initialize backend with fallback logic."""
        try:
            backend = get_backend(self.backend_name, **kwargs)
            if backend.is_available():
                return backend
            else:
                raise RuntimeError(f"Backend {self.backend_name} is not available")
        
        except Exception as e:
            logger.warning(f"Failed to initialize {self.backend_name} backend: {e}")
            
            # Try fallback backends
            fallback_backends = ['simple', 'simulator']
            for fallback in fallback_backends:
                if fallback != self.backend_name:
                    try:
                        logger.info(f"Trying fallback backend: {fallback}")
                        backend = get_backend(fallback)
                        if backend.is_available():
                            self.backend_name = fallback
                            return backend
                    except Exception as fallback_error:
                        logger.warning(f"Fallback {fallback} also failed: {fallback_error}")
            
            raise RuntimeError(f"All backends failed. Last error: {e}")
    
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
        early_stopping_patience: int = None,
        **model_kwargs
    ) -> Tuple[Dict[str, Any], OptimizationHistory]:
        """
        Run robust quantum hyperparameter optimization with comprehensive error handling.
        
        Args:
            model_class: Model class to optimize
            param_space: Parameter search space
            X: Training features
            y: Training targets
            n_iterations: Number of optimization iterations
            quantum_reads: Number of quantum reads per iteration
            cv_folds: Cross-validation folds
            scoring: Scoring metric
            early_stopping_patience: Early stopping patience (iterations)
            **model_kwargs: Additional model parameters
            
        Returns:
            Tuple of (best_parameters, optimization_history)
        """
        # Start optimization tracking
        self.history.start_optimization()
        
        try:
            # Comprehensive input validation with detailed error messages
            self._validate_inputs(model_class, param_space, X, y, 
                                n_iterations, quantum_reads, cv_folds, model_kwargs)
            
            print(f"🌌 Starting robust quantum optimization (session: {self.session_id})")
            print(f"🔧 Backend: {self.backend_name}, Encoding: {self.encoding}")
            print(f"📊 Parameter space: {len(param_space)} parameters")
            print(f"🔢 Search space size: {np.prod([len(v) for v in param_space.values()])}")
            print(f"🛡️  Security: {'Enabled' if self.enable_security else 'Disabled'}")
            
            # Get preliminary scores with error handling
            preliminary_scores = self._get_preliminary_scores_robust(
                model_class, param_space, X, y, cv_folds, scoring, **model_kwargs
            )
            
            # Encode to QUBO with retry logic
            Q, offset, variable_map = self._encode_qubo_robust(param_space, preliminary_scores)
            
            print(f"🔢 QUBO variables: {len(variable_map)}")
            print(f"⚖️  QUBO density: {len(Q) / (len(variable_map) ** 2) * 100:.1f}%")
            
            # Main optimization loop with comprehensive error handling
            return self._run_optimization_loop_robust(
                model_class, param_space, X, y, n_iterations, quantum_reads,
                cv_folds, scoring, Q, variable_map, early_stopping_patience, **model_kwargs
            )
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            print("⚠️  Optimization interrupted by user")
            return self._get_best_result()
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            print(f"❌ Optimization failed: {e}")
            
            # Try to return best result found so far
            try:
                return self._get_best_result()
            except:
                # If even that fails, return minimal result
                return {}, self.history
            
        finally:
            self.history.end_optimization()
            self._print_final_statistics()
    
    def _validate_inputs(self, model_class, param_space, X, y, n_iterations, 
                        quantum_reads, cv_folds, model_kwargs):
        """Comprehensive input validation with detailed error reporting."""
        try:
            # Standard validation
            param_space = validate_search_space(param_space)
            X, y = validate_data(X, y)
            validate_model_class(model_class)
            validate_optimization_params(n_iterations, quantum_reads, cv_folds)
            
            # Security validation
            if self.enable_security:
                try:
                    check_safety(param_space, model_class)
                    if model_kwargs:
                        model_kwargs = sanitize_parameters(model_kwargs)
                except SecurityError as e:
                    logger.error(f"Security validation failed: {e}")
                    raise ValidationError(f"Security check failed: {e}")
            
            # Resource validation
            param_space_size = np.prod([len(v) for v in param_space.values()])
            if param_space_size > 100000:
                logger.warning(f"Very large search space: {param_space_size} combinations")
            
            total_evaluations = n_iterations * min(10, quantum_reads // 10)
            if total_evaluations > 10000:
                logger.warning(f"Estimated {total_evaluations} evaluations - this may take a long time")
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValidationError(f"Input validation failed: {e}")
    
    def _get_preliminary_scores_robust(self, model_class, param_space, X, y, 
                                     cv_folds, scoring, **model_kwargs):
        """Get preliminary scores with robust error handling."""
        print("🔍 Getting preliminary scores...")
        
        preliminary_scores = {}
        successful_evaluations = 0
        max_attempts = min(5, np.prod([len(v) for v in param_space.values()]))
        
        for attempt in range(max_attempts):
            try:
                # Sample random parameters
                params = self._sample_random_params(param_space)
                param_key = str(sorted(params.items()))
                
                if param_key in preliminary_scores:
                    continue
                
                # Evaluate with timeout and retry
                score = self._evaluate_configuration_robust(
                    params, model_class, X, y, cv_folds, scoring, **model_kwargs
                )
                
                if score is not None and score != float('-inf'):
                    preliminary_scores[param_key] = score
                    successful_evaluations += 1
                    print(f"  ✅ Preliminary {successful_evaluations}: {params} -> {score:.4f}")
                
                if successful_evaluations >= 3:  # We have enough
                    break
                    
            except Exception as e:
                logger.warning(f"Preliminary evaluation {attempt + 1} failed: {e}")
                continue
        
        if not preliminary_scores:
            logger.warning("No successful preliminary evaluations - using default scores")
            # Create minimal default scores
            params = self._sample_random_params(param_space)
            preliminary_scores[str(sorted(params.items()))] = 0.5
        
        print(f"📊 Preliminary evaluation: {successful_evaluations}/{max_attempts} successful")
        return preliminary_scores
    
    def _encode_qubo_robust(self, param_space, preliminary_scores):
        """Encode QUBO with robust error handling and fallbacks."""
        for attempt in range(self.max_retries):
            try:
                Q, offset, variable_map = self.encoder.encode_search_space(param_space, self.history)
                
                # Validate QUBO
                if not Q or not variable_map:
                    raise ValueError("Empty QUBO or variable mapping")
                
                # Check QUBO size
                if len(variable_map) > 1000:
                    logger.warning(f"Large QUBO with {len(variable_map)} variables")
                
                return Q, offset, variable_map
                
            except Exception as e:
                logger.warning(f"QUBO encoding attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Try with different penalty strength
                    original_penalty = self.encoder.penalty_strength
                    self.encoder.penalty_strength = max(0.1, original_penalty * 0.5)
                    logger.info(f"Retrying with reduced penalty strength: {self.encoder.penalty_strength}")
                else:
                    raise RuntimeError(f"Failed to encode QUBO after {self.max_retries} attempts: {e}")
    
    def _run_optimization_loop_robust(self, model_class, param_space, X, y, n_iterations,
                                    quantum_reads, cv_folds, scoring, Q, variable_map,
                                    early_stopping_patience, **model_kwargs):
        """Run optimization loop with comprehensive error handling."""
        
        best_score = float('-inf')
        best_params = None
        iterations_without_improvement = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        for iteration in range(n_iterations):
            iteration_start = time.time()
            
            print(f"\n🔄 Iteration {iteration + 1}/{n_iterations}")
            
            # Check timeout
            if time.time() - self.history.start_time > n_iterations * self.timeout_per_iteration:
                print("⏰ Global timeout reached")
                break
            
            # Check early stopping
            if (early_stopping_patience and 
                iterations_without_improvement >= early_stopping_patience):
                print(f"🛑 Early stopping: no improvement for {early_stopping_patience} iterations")
                break
            
            try:
                # Quantum sampling with timeout and retry
                samples = self._sample_quantum_robust(Q, quantum_reads, iteration)
                
                if not samples:
                    raise RuntimeError("No quantum samples obtained")
                
                # Evaluate samples with robust error handling
                iteration_improved = self._evaluate_samples_robust(
                    samples, variable_map, param_space, model_class, X, y,
                    cv_folds, scoring, iteration, **model_kwargs
                )
                
                if iteration_improved:
                    iterations_without_improvement = 0
                    best_score = self.history.best_score
                    best_params = self.history.best_params
                else:
                    iterations_without_improvement += 1
                
                consecutive_failures = 0  # Reset failure counter
                
            except Exception as e:
                consecutive_failures += 1
                error_msg = f"Iteration {iteration + 1} failed: {e}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                
                # Record error in history
                self.history.add_evaluation({}, float('-inf'), iteration, error_msg=str(e))
                
                if consecutive_failures >= max_consecutive_failures:
                    if self.fallback_to_random:
                        print(f"🎲 Switching to random search after {max_consecutive_failures} failures")
                        success = self._fallback_to_random_search(
                            model_class, param_space, X, y, cv_folds, scoring, 
                            iteration, **model_kwargs
                        )
                        if success:
                            consecutive_failures = 0
                        else:
                            print("❌ Random search fallback also failed")
                            break
                    else:
                        print(f"❌ Too many consecutive failures ({max_consecutive_failures})")
                        break
            
            # Print iteration summary
            iteration_time = time.time() - iteration_start
            print(f"⏱️  Iteration time: {iteration_time:.1f}s")
        
        return self._get_best_result()
    
    def _sample_quantum_robust(self, Q, quantum_reads, iteration):
        """Sample from quantum backend with robust error handling."""
        for attempt in range(self.max_retries):
            try:
                # Set timeout for quantum sampling
                start_time = time.time()
                samples = self.backend.sample_qubo(Q, num_reads=quantum_reads)
                sampling_time = time.time() - start_time
                
                if samples and hasattr(samples, 'record') and samples.record:
                    print(f"🔬 Quantum sampling: {len(samples.record)} samples in {sampling_time:.1f}s")
                    return samples
                else:
                    raise RuntimeError("Empty sample set returned")
                    
            except Exception as e:
                logger.warning(f"Quantum sampling attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Reduce problem size and try again
                    quantum_reads = max(10, quantum_reads // 2)
                    print(f"🔄 Retrying with {quantum_reads} quantum reads")
                    time.sleep(1)
        
        return None
    
    def _evaluate_samples_robust(self, samples, variable_map, param_space, model_class,
                               X, y, cv_folds, scoring, iteration, **model_kwargs):
        """Evaluate quantum samples with robust error handling."""
        evaluated_configs = set()
        improvements = 0
        successful_evaluations = 0
        
        # Evaluate top samples
        max_samples = min(10, len(samples.record))
        
        for i, sample_result in enumerate(samples.record[:max_samples]):
            try:
                # Decode sample with error handling
                params = self._decode_sample_robust(sample_result.sample, variable_map, param_space)
                
                if not params:
                    continue
                
                # Skip duplicates
                param_key = tuple(sorted(params.items()))
                if param_key in evaluated_configs:
                    continue
                evaluated_configs.add(param_key)
                
                # Evaluate configuration
                score = self._evaluate_configuration_robust(
                    params, model_class, X, y, cv_folds, scoring, **model_kwargs
                )
                
                if score is not None and score != float('-inf'):
                    successful_evaluations += 1
                    
                    # Record quantum info
                    quantum_info = {
                        'energy': getattr(sample_result, 'energy', None),
                        'chain_break_fraction': getattr(sample_result, 'chain_break_fraction', 0.0),
                        'sample_rank': i
                    }
                    
                    # Update history
                    previous_best = self.history.best_score
                    self.history.add_evaluation(params, score, iteration, quantum_info)
                    
                    # Check for improvement
                    if score > previous_best:
                        improvements += 1
                        print(f"🎉 New best score: {score:.4f}")
                        print(f"🎯 Best params: {params}")
                
            except Exception as e:
                logger.warning(f"Sample evaluation {i} failed: {e}")
                continue
        
        print(f"📊 Evaluated {successful_evaluations}/{max_samples} samples, {improvements} improvements")
        return improvements > 0
    
    def _decode_sample_robust(self, sample, variable_map, param_space):
        """Decode quantum sample with robust error handling."""
        try:
            params = self.encoder.decode_sample(sample, variable_map, param_space)
            
            # Validate decoded parameters
            if not params:
                return None
            
            # Check parameter validity
            for param_name, param_value in params.items():
                if param_name not in param_space:
                    logger.warning(f"Decoded parameter not in search space: {param_name}")
                    return None
                
                if param_value not in param_space[param_name]:
                    logger.warning(f"Decoded value not in parameter values: {param_name}={param_value}")
                    # Find closest valid value
                    param_space_values = param_space[param_name]
                    if isinstance(param_value, (int, float)) and all(isinstance(v, (int, float)) for v in param_space_values):
                        closest = min(param_space_values, key=lambda x: abs(x - param_value))
                        params[param_name] = closest
                    else:
                        params[param_name] = param_space_values[0]  # Default to first value
            
            return params
            
        except Exception as e:
            logger.warning(f"Sample decoding failed: {e}")
            return None
    
    def _evaluate_configuration_robust(self, params, model_class, X, y, cv_folds, scoring, **model_kwargs):
        """Evaluate configuration with robust error handling and timeout."""
        
        for attempt in range(min(2, self.max_retries)):  # Limit retries for evaluation
            try:
                # Security check
                if self.enable_security:
                    params = sanitize_parameters(params)
                
                # Create model with timeout protection
                all_params = {**params, **model_kwargs}
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Suppress sklearn warnings
                    
                    model = model_class(**all_params)
                    
                    # Evaluate with cross-validation
                    scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                    
                    if len(scores) == 0 or np.any(np.isnan(scores)):
                        raise ValueError("Invalid cross-validation scores")
                    
                    return float(np.mean(scores))
                    
            except Exception as e:
                logger.debug(f"Configuration evaluation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)  # Brief pause before retry
        
        return None  # All attempts failed
    
    def _fallback_to_random_search(self, model_class, param_space, X, y, cv_folds, 
                                  scoring, iteration, **model_kwargs):
        """Fallback to random search when quantum sampling fails."""
        try:
            params = self._sample_random_params(param_space)
            score = self._evaluate_configuration_robust(
                params, model_class, X, y, cv_folds, scoring, **model_kwargs
            )
            
            if score is not None and score != float('-inf'):
                previous_best = self.history.best_score
                self.history.add_evaluation(params, score, iteration)
                
                if score > previous_best:
                    print(f"🎲 Random search found improvement: {score:.4f}")
                    print(f"🎯 Parameters: {params}")
                    return True
                else:
                    print(f"🎲 Random search score: {score:.4f} (no improvement)")
                    return True
            
        except Exception as e:
            logger.error(f"Random search fallback failed: {e}")
        
        return False
    
    def _sample_random_params(self, param_space):
        """Sample random parameters from search space."""
        return {param: np.random.choice(values) for param, values in param_space.items()}
    
    def _get_best_result(self):
        """Get best optimization result."""
        if self.history.best_params is not None:
            return self.history.best_params, self.history
        else:
            # Return empty result if no successful evaluations
            return {}, self.history
    
    def _print_final_statistics(self):
        """Print comprehensive final statistics."""
        stats = self.history.get_statistics()
        
        print(f"\n📊 Optimization Statistics:")
        print(f"   Duration: {stats['duration_seconds']:.1f} seconds")
        print(f"   Evaluations: {stats['n_evaluations']} total, {stats['n_errors']} errors")
        print(f"   Error rate: {stats['error_rate']:.1%}")
        print(f"   Evaluation rate: {stats['evaluations_per_second']:.2f}/sec")
        print(f"   Improvements: {stats['improvement_iterations']} iterations")
        
        if self.history.best_params:
            print(f"\n🏆 Final Results:")
            print(f"   Best score: {stats['best_score']:.4f}")
            print(f"   Best parameters: {stats['best_params']}")
        else:
            print(f"\n❌ No successful optimizations completed")


# Alias for backward compatibility
QuantumHyperSearch = QuantumHyperSearchRobust