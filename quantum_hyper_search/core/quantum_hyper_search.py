"""
Main QuantumHyperSearch class - the primary interface for quantum hyperparameter optimization.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import cross_val_score

from ..backends.backend_factory import get_backend
from .qubo_encoder import QUBOEncoder
from ..utils.validation import (
    validate_search_space, validate_data, validate_model_class,
    validate_optimization_params, validate_backend_config, ValidationError
)
from ..utils.logging_config import get_logger, log_optimization_start, log_optimization_result
from ..utils.metrics import QuantumMetrics

logger = get_logger('quantum_hyper_search')


class OptimizationHistory:
    """Track optimization history and results."""
    
    def __init__(self):
        self.trials = []
        self.scores = []
        self.quantum_samples = []
        self.timestamps = []
        self.best_score = float('-inf')
        self.best_params = None
    
    def add_trial(self, params: Dict[str, Any], score: float, quantum_sample: Optional[Dict] = None):
        """Add a trial result to the history."""
        self.trials.append(params)
        self.scores.append(score)
        self.quantum_samples.append(quantum_sample)
        self.timestamps.append(time.time())
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
    
    def get_convergence_data(self) -> Tuple[List[float], List[float]]:
        """Get data for plotting convergence."""
        best_so_far = []
        current_best = float('-inf')
        
        for score in self.scores:
            if score > current_best:
                current_best = score
            best_so_far.append(current_best)
        
        return list(range(len(self.scores))), best_so_far


class QuantumHyperSearch:
    """
    Main quantum hyperparameter search interface.
    
    Combines quantum annealing with classical machine learning to optimize
    hyperparameters using D-Wave quantum computers or simulators.
    """
    
    def __init__(
        self,
        backend: str = 'simulator',
        token: Optional[str] = None,
        embedding_method: str = 'minorminer',
        penalty_strength: float = 2.0,
        verbose: bool = True,
        log_file: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize quantum hyperparameter search.
        
        Args:
            backend: Quantum backend to use ('dwave', 'simulator', 'neal')
            token: D-Wave API token (required for 'dwave' backend)
            embedding_method: Method for minor embedding ('minorminer', 'clique')
            penalty_strength: Strength of constraint penalties in QUBO
            verbose: Enable verbose logging
            log_file: Optional file path for logging output
            random_seed: Random seed for reproducibility
            
        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate configuration
        validate_backend_config(backend, token=token, penalty_strength=penalty_strength)
        
        # Setup logging
        if verbose:
            from ..utils.logging_config import setup_logging, configure_third_party_loggers
            setup_logging(level='INFO', log_file=log_file, console=True)
            configure_third_party_loggers()
        
        # Initialize components with error handling
        try:
            self.backend = get_backend(backend, token=token)
            self.encoder = QUBOEncoder(penalty_strength=penalty_strength)
            self.metrics = QuantumMetrics()
        except Exception as e:
            logger.error(f"Failed to initialize backend or encoder: {e}")
            raise ValidationError(f"Initialization failed: {e}")
        
        self.embedding_method = embedding_method
        self.verbose = verbose
        self.random_seed = random_seed
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(f"Initialized QuantumHyperSearch with {backend} backend")
        logger.info(f"Backend properties: {self.backend.get_properties()}")
        
        # Validate backend availability
        if not self.backend.is_available():
            logger.warning("Backend may not be fully available. Some features may be limited.")
    
    def optimize(
        self,
        model_class: type,
        param_space: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 20,
        quantum_reads: int = 1000,
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        random_seed: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Tuple[Dict[str, Any], OptimizationHistory]:
        """
        Run quantum-enhanced hyperparameter optimization.
        
        Args:
            model_class: Scikit-learn model class to optimize
            param_space: Dictionary defining hyperparameter search space
            X: Training features
            y: Training labels
            n_iterations: Number of optimization iterations
            quantum_reads: Number of quantum annealer reads per iteration
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for model evaluation
            random_seed: Random seed for reproducibility
            early_stopping_patience: Stop if no improvement for N iterations
            timeout: Maximum optimization time in seconds
            
        Returns:
            Tuple of (best_parameters, optimization_history)
            
        Raises:
            ValidationError: If inputs are invalid
        """
        start_time = time.time()
        
        # Comprehensive input validation
        try:
            param_space = validate_search_space(param_space)
            X, y = validate_data(X, y)
            validate_model_class(model_class)
            validate_optimization_params(n_iterations, quantum_reads, cv_folds, scoring)
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            raise
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        elif self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        history = OptimizationHistory()
        self.metrics.reset()
        self.metrics.start_optimization()
        
        # Log optimization start
        dataset_info = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'cv_folds': cv_folds,
            'scoring': scoring
        }
        
        if self.verbose:
            log_optimization_start(param_space, n_iterations, 
                                 self.backend.get_properties()['name'], dataset_info)
        
        # Initial random sampling for warm-start
        warmup_iterations = min(5, n_iterations // 4)
        successful_trials = 0
        
        for i in range(warmup_iterations):
            try:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning("Timeout reached during warmup phase")
                    break
                
                params = self._sample_random_params(param_space)
                score = self._evaluate_params(model_class, params, X, y, cv_folds, scoring)
                history.add_trial(params, score)
                successful_trials += 1
                
                if self.verbose:
                    logger.info(f"Warmup {i+1}: Score = {score:.4f}, Params = {params}")
                    
            except Exception as e:
                logger.error(f"Warmup iteration {i+1} failed: {e}")
                # Continue with remaining warmup iterations
        
        if successful_trials == 0:
            raise ValidationError("All warmup iterations failed. Check your data and model configuration.")
        
        # Main quantum optimization loop
        consecutive_failures = 0
        max_consecutive_failures = 3
        iterations_without_improvement = 0
        
        for iteration in range(n_iterations - len(history.trials)):
            iteration_start_time = time.time()
            
            try:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.info(f"Timeout reached after {len(history.trials)} iterations")
                    break
                
                # Check early stopping
                if early_stopping_patience and iterations_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping: no improvement for {early_stopping_patience} iterations")
                    break
                
                # Encode current knowledge into QUBO
                Q, offset, param_mapping = self._create_qubo(param_space, history)
                
                # Sample from quantum annealer with timeout protection
                quantum_samples = self.backend.sample_qubo(Q, num_reads=quantum_reads)
                
                # Decode best quantum sample to parameters
                try:
                    best_sample = min(quantum_samples.record, key=lambda x: x.energy)
                    
                    # Handle different sample formats
                    if hasattr(best_sample, 'sample'):
                        sample_dict = best_sample.sample
                    else:
                        # Fallback: use the sample directly if it's a dict
                        sample_dict = best_sample if isinstance(best_sample, dict) else {}
                    
                    params = self._decode_sample(sample_dict, param_mapping)
                    sample_energy = best_sample.energy
                    chain_break_fraction = getattr(best_sample, 'chain_break_fraction', 0.0)
                    
                except (AttributeError, TypeError):
                    # Fallback: extract from sampleset directly
                    best_energy_idx = np.argmin(quantum_samples.data_vectors['energy'])
                    sample_dict = dict(quantum_samples.samples()[best_energy_idx])
                    params = self._decode_sample(sample_dict, param_mapping)
                    sample_energy = quantum_samples.data_vectors['energy'][best_energy_idx]
                    chain_break_fraction = 0.0
                
                # Record quantum metrics
                self.metrics.add_quantum_sample(
                    sample_dict, sample_energy, chain_break_fraction
                )
                
                # Evaluate parameters
                score = self._evaluate_params(model_class, params, X, y, cv_folds, scoring)
                previous_best = history.best_score
                history.add_trial(params, score, {'energy': sample_energy})
                
                # Track improvement
                if score > previous_best:
                    iterations_without_improvement = 0
                    logger.info(f"New best score: {score:.4f} (improvement: {score - previous_best:.4f})")
                else:
                    iterations_without_improvement += 1
                
                consecutive_failures = 0  # Reset failure counter on success
                
                if self.verbose:
                    current_iteration = len(history.trials)
                    logger.info(f"Iteration {current_iteration}: Score = {score:.4f}, "
                              f"Energy = {sample_energy:.2f}, "
                              f"Chain breaks = {chain_break_fraction:.3f}")
                    logger.info(f"Current best: {history.best_score:.4f}")
                
            except KeyboardInterrupt:
                logger.info("Optimization interrupted by user")
                break
                
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Quantum sampling failed (attempt {consecutive_failures}): {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Maximum consecutive failures ({max_consecutive_failures}) reached. "
                               "Switching to random sampling mode.")
                
                # Fallback to random sampling
                try:
                    params = self._sample_random_params(param_space)
                    score = self._evaluate_params(model_class, params, X, y, cv_folds, scoring)
                    previous_best = history.best_score
                    history.add_trial(params, score)
                    
                    # Track improvement
                    if score > previous_best:
                        iterations_without_improvement = 0
                    else:
                        iterations_without_improvement += 1
                        
                    if self.verbose:
                        logger.info(f"Fallback iteration {len(history.trials)}: Score = {score:.4f}")
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback sampling also failed: {fallback_error}")
                    # Skip this iteration entirely
                    continue
        
        # Finalize metrics
        self.metrics.end_optimization()
        execution_time = time.time() - start_time
        
        # Log comprehensive results
        if self.verbose:
            log_optimization_result(history.best_params, history.best_score, 
                                  history, execution_time)
            self.metrics.log_performance_summary()
        
        # Final validation
        if history.best_params is None:
            raise ValidationError("Optimization failed to find any valid parameters")
        
        return history.best_params, history
    
    def _sample_random_params(self, param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Sample random parameters from the search space."""
        return {
            param: np.random.choice(values)
            for param, values in param_space.items()
        }
    
    def _evaluate_params(
        self,
        model_class: type,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        scoring: str
    ) -> float:
        """Evaluate model with given parameters using cross-validation."""
        try:
            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            return float(np.mean(scores))
        except Exception as e:
            logger.warning(f"Model evaluation failed with params {params}: {e}")
            return float('-inf')
    
    def _create_qubo(
        self,
        param_space: Dict[str, List[Any]],
        history: OptimizationHistory
    ) -> Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]]:
        """Create QUBO matrix from current optimization state."""
        return self.encoder.encode_search_space(param_space, history)
    
    def _decode_sample(
        self,
        sample: Dict[int, int],
        param_mapping: Dict[int, Tuple[str, Any]]
    ) -> Dict[str, Any]:
        """Decode quantum sample back to parameter values."""
        params = {}
        
        # Group by parameter name
        param_groups = {}
        for var_idx, (param_name, param_value) in param_mapping.items():
            if param_name not in param_groups:
                param_groups[param_name] = []
            param_groups[param_name].append((var_idx, param_value))
        
        # For each parameter, select the value with highest activation
        for param_name, var_list in param_groups.items():
            best_activation = -1
            best_value = None
            
            for var_idx, param_value in var_list:
                activation = sample.get(var_idx, 0)
                if activation > best_activation:
                    best_activation = activation
                    best_value = param_value
            
            if best_value is not None:
                params[param_name] = best_value
        
        return params