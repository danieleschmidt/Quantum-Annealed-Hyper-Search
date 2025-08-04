"""
Input validation and data sanitization utilities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_search_space(param_space: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Validate and normalize hyperparameter search space.
    
    Args:
        param_space: Dictionary mapping parameter names to possible values
        
    Returns:
        Normalized search space with consistent formatting
        
    Raises:
        ValidationError: If search space is invalid
    """
    if not isinstance(param_space, dict):
        raise ValidationError("Search space must be a dictionary")
    
    if not param_space:
        raise ValidationError("Search space cannot be empty")
    
    normalized_space = {}
    
    for param_name, param_values in param_space.items():
        # Validate parameter name
        if not isinstance(param_name, str):
            raise ValidationError(f"Parameter name must be string, got {type(param_name)}")
        
        if not param_name.strip():
            raise ValidationError("Parameter name cannot be empty")
        
        if not param_name.isidentifier():
            logger.warning(f"Parameter name '{param_name}' is not a valid Python identifier")
        
        # Validate parameter values
        if not isinstance(param_values, (list, tuple, np.ndarray)):
            raise ValidationError(f"Parameter values for '{param_name}' must be a list, tuple, or array")
        
        param_values = list(param_values)  # Convert to list
        
        if len(param_values) == 0:
            raise ValidationError(f"Parameter '{param_name}' must have at least one possible value")
        
        if len(param_values) == 1:
            logger.warning(f"Parameter '{param_name}' has only one value, no optimization possible")
        
        # Check for duplicates
        unique_values = []
        seen = set()
        for value in param_values:
            # Handle unhashable types like numpy arrays
            try:
                if value not in seen:
                    unique_values.append(value)
                    seen.add(value)
                else:
                    logger.warning(f"Duplicate value {value} in parameter '{param_name}'")
            except TypeError:
                # For unhashable types, just add to list
                unique_values.append(value)
        
        normalized_space[param_name] = unique_values
    
    # Check total search space size
    total_combinations = 1
    for param_values in normalized_space.values():
        total_combinations *= len(param_values)
    
    if total_combinations > 1e6:
        logger.warning(f"Large search space: {total_combinations:,} combinations. "
                      "Consider reducing parameter ranges.")
    
    return normalized_space


def validate_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate training data.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        Validated and potentially cleaned data
        
    Raises:
        ValidationError: If data is invalid
    """
    # Convert to numpy arrays if needed
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X)
        except Exception as e:
            raise ValidationError(f"Cannot convert X to numpy array: {e}")
    
    if not isinstance(y, np.ndarray):
        try:
            y = np.array(y)
        except Exception as e:
            raise ValidationError(f"Cannot convert y to numpy array: {e}")
    
    # Basic shape validation
    if X.ndim != 2:
        raise ValidationError(f"X must be 2-dimensional, got shape {X.shape}")
    
    if y.ndim != 1:
        raise ValidationError(f"y must be 1-dimensional, got shape {y.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise ValidationError(f"X and y must have same number of samples: "
                            f"X has {X.shape[0]}, y has {y.shape[0]}")
    
    # Check for minimum samples
    n_samples = X.shape[0]
    if n_samples < 10:
        logger.warning(f"Very few samples ({n_samples}). Results may be unreliable.")
    
    # Check for missing values
    if np.any(np.isnan(X)):
        logger.warning("X contains NaN values. Consider preprocessing your data.")
    
    if np.any(np.isnan(y)):
        raise ValidationError("y contains NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(X)):
        logger.warning("X contains infinite values. Consider preprocessing your data.")
    
    if np.any(np.isinf(y)):
        raise ValidationError("y contains infinite values")
    
    # Check feature variance
    feature_vars = np.var(X, axis=0)
    zero_var_features = np.sum(feature_vars == 0)
    if zero_var_features > 0:
        logger.warning(f"{zero_var_features} features have zero variance")
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = min(class_counts)
    
    if min_class_count < 2:
        logger.warning(f"Some classes have very few samples (min: {min_class_count})")
    
    logger.info(f"Data validation passed: {X.shape[0]} samples, {X.shape[1]} features, "
               f"{len(unique_classes)} classes")
    
    return X, y


def validate_model_class(model_class: type) -> None:
    """
    Validate that model class is compatible with scikit-learn.
    
    Args:
        model_class: Class to validate
        
    Raises:
        ValidationError: If model class is invalid
    """
    if not isinstance(model_class, type):
        raise ValidationError("model_class must be a class, not an instance")
    
    # Check if it's a scikit-learn estimator
    try:
        # Try to instantiate with default parameters
        model_instance = model_class()
        
        # Check if it has required methods
        required_methods = ['fit', 'predict']
        for method in required_methods:
            if not hasattr(model_instance, method):
                raise ValidationError(f"Model class must have '{method}' method")
        
        # Check if it's a proper sklearn estimator
        if not isinstance(model_instance, BaseEstimator):
            logger.warning("Model class is not a scikit-learn BaseEstimator. "
                         "Compatibility may be limited.")
        
    except Exception as e:
        raise ValidationError(f"Cannot instantiate model class: {e}")


def validate_optimization_params(
    n_iterations: int,
    quantum_reads: int,
    cv_folds: int,
    scoring: str
) -> None:
    """
    Validate optimization parameters.
    
    Args:
        n_iterations: Number of optimization iterations
        quantum_reads: Number of quantum annealer reads
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric name
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(n_iterations, int) or n_iterations < 1:
        raise ValidationError("n_iterations must be a positive integer")
    
    if n_iterations > 1000:
        logger.warning(f"Large number of iterations ({n_iterations}). "
                      "Consider smaller values for faster execution.")
    
    if not isinstance(quantum_reads, int) or quantum_reads < 1:
        raise ValidationError("quantum_reads must be a positive integer")
    
    if quantum_reads > 10000:
        logger.warning(f"Large number of quantum reads ({quantum_reads}). "
                      "This may be expensive on real quantum hardware.")
    
    if not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValidationError("cv_folds must be an integer >= 2")
    
    if cv_folds > 20:
        logger.warning(f"Large number of CV folds ({cv_folds}). "
                      "This will increase computation time.")
    
    if not isinstance(scoring, str):
        raise ValidationError("scoring must be a string")
    
    # Common sklearn scoring metrics
    valid_scores = {
        'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
        'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
        'recall', 'recall_macro', 'recall_micro', 'recall_weighted',
        'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'average_precision',
        'neg_log_loss', 'neg_mean_squared_error', 'neg_mean_absolute_error',
        'r2'
    }
    
    if scoring not in valid_scores:
        logger.warning(f"Scoring metric '{scoring}' not in common sklearn metrics. "
                      f"Valid options include: {sorted(valid_scores)}")


def validate_backend_config(backend: str, **kwargs) -> None:
    """
    Validate backend configuration.
    
    Args:
        backend: Backend name
        **kwargs: Backend-specific configuration
        
    Raises:
        ValidationError: If configuration is invalid
    """
    valid_backends = ['dwave', 'simulator', 'neal']
    
    if backend not in valid_backends:
        raise ValidationError(f"Backend '{backend}' not supported. "
                            f"Valid options: {valid_backends}")
    
    if backend == 'dwave':
        if 'token' not in kwargs or not kwargs['token']:
            logger.warning("D-Wave backend requires API token. "
                         "Set token parameter or DWAVE_API_TOKEN environment variable.")
    
    # Validate numeric parameters
    for param, value in kwargs.items():
        if param in ['penalty_strength', 'chain_strength']:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValidationError(f"{param} must be a positive number")
        
        if param in ['quantum_reads', 'num_reads']:
            if not isinstance(value, int) or value < 1:
                raise ValidationError(f"{param} must be a positive integer")