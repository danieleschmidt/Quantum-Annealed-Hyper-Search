"""
Validation utilities for quantum hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Type, Callable
from sklearn.base import BaseEstimator
import inspect
import re
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_search_space(param_space: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Validate and normalize hyperparameter search space.
    
    Args:
        param_space: Dictionary of parameter names and possible values
        
    Returns:
        Validated and normalized search space
        
    Raises:
        ValidationError: If search space is invalid
    """
    if not isinstance(param_space, dict):
        raise ValidationError("Parameter space must be a dictionary")
    
    if not param_space:
        raise ValidationError("Parameter space cannot be empty")
    
    validated_space = {}
    
    for param_name, param_values in param_space.items():
        # Validate parameter name
        if not isinstance(param_name, str):
            raise ValidationError(f"Parameter name must be string, got {type(param_name)}")
        
        if not param_name or not param_name.replace('_', '').isalnum():
            raise ValidationError(f"Invalid parameter name: {param_name}")
        
        # Validate parameter values
        if not isinstance(param_values, (list, tuple)):
            param_values = [param_values]
        
        if not param_values:
            raise ValidationError(f"Parameter {param_name} cannot have empty values")
        
        # Convert to list and validate values
        clean_values = []
        for value in param_values:
            if value is None:
                clean_values.append(None)
            elif isinstance(value, (int, float, str, bool)):
                clean_values.append(value)
            elif isinstance(value, np.number):
                clean_values.append(value.item())
            else:
                try:
                    # Try to convert to basic type
                    if hasattr(value, '__float__'):
                        clean_values.append(float(value))
                    elif hasattr(value, '__int__'):
                        clean_values.append(int(value))
                    else:
                        clean_values.append(str(value))
                except Exception:
                    raise ValidationError(f"Invalid value type for {param_name}: {type(value)}")
        
        # Remove duplicates while preserving order
        unique_values = []
        seen = set()
        for value in clean_values:
            if value not in seen:
                unique_values.append(value)
                seen.add(value)
        
        if len(unique_values) < 2:
            logger.warning(f"Parameter {param_name} has only one unique value")
        
        validated_space[param_name] = unique_values
    
    return validated_space


def validate_model_class(model_class: Type) -> None:
    """
    Validate scikit-learn compatible model class.
    
    Args:
        model_class: Model class to validate
        
    Raises:
        ValidationError: If model class is invalid
    """
    if not inspect.isclass(model_class):
        raise ValidationError("model_class must be a class")
    
    # Check if it's a sklearn estimator or has required methods
    required_methods = ['fit', 'predict']
    if hasattr(model_class, '__bases__'):
        # Check if it inherits from BaseEstimator
        if not any(issubclass(base, BaseEstimator) for base in model_class.__mro__):
            # If not sklearn estimator, check for required methods
            for method in required_methods:
                if not hasattr(model_class, method):
                    raise ValidationError(f"Model class must have {method} method")
    
    # Try to create instance to check constructor
    try:
        # Get constructor signature
        sig = inspect.signature(model_class.__init__)
        required_params = [
            p for p in sig.parameters.values() 
            if p.default == inspect.Parameter.empty and p.name != 'self'
        ]
        
        if required_params:
            logger.warning(f"Model class has required parameters: {[p.name for p in required_params]}")
    except Exception as e:
        raise ValidationError(f"Cannot analyze model class constructor: {e}")


def validate_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Validate training data.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Validated (X, y) tuple
        
    Raises:
        ValidationError: If data is invalid
    """
    # Convert to numpy arrays
    if not isinstance(X, np.ndarray):
        try:
            if hasattr(X, 'values'):  # pandas DataFrame
                X = X.values
            else:
                X = np.array(X)
        except Exception as e:
            raise ValidationError(f"Cannot convert X to numpy array: {e}")
    
    if not isinstance(y, np.ndarray):
        try:
            if hasattr(y, 'values'):  # pandas Series
                y = y.values
            else:
                y = np.array(y)
        except Exception as e:
            raise ValidationError(f"Cannot convert y to numpy array: {e}")
    
    # Basic shape validation
    if X.ndim != 2:
        raise ValidationError(f"X must be 2D array, got shape {X.shape}")
    
    if y.ndim != 1:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        else:
            raise ValidationError(f"y must be 1D array, got shape {y.shape}")
    
    if len(X) != len(y):
        raise ValidationError(f"X and y have different lengths: {len(X)} vs {len(y)}")
    
    if len(X) == 0:
        raise ValidationError("Empty dataset provided")
    
    # Check for NaN/inf values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValidationError("X contains NaN or infinite values")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValidationError("y contains NaN or infinite values")
    
    # Minimum samples check
    if len(X) < 3:
        raise ValidationError("At least 3 samples required for cross-validation")
    
    return X, y


def validate_optimization_params(
    n_iterations: int,
    quantum_reads: int,
    cv_folds: int,
    scoring: str = 'accuracy'
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
    if not isinstance(n_iterations, int) or n_iterations <= 0:
        raise ValidationError(f"n_iterations must be positive integer, got {n_iterations}")
    
    if n_iterations > 1000:
        logger.warning(f"Large number of iterations ({n_iterations}) may take very long")
    
    if not isinstance(quantum_reads, int) or quantum_reads <= 0:
        raise ValidationError(f"quantum_reads must be positive integer, got {quantum_reads}")
    
    if quantum_reads > 100000:
        logger.warning(f"Large number of quantum reads ({quantum_reads}) may be expensive")
    
    if not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValidationError(f"cv_folds must be integer >= 2, got {cv_folds}")
    
    if cv_folds > 20:
        logger.warning(f"Large number of CV folds ({cv_folds}) may slow down optimization")
    
    # Validate scoring metric
    valid_metrics = {
        'accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
        'roc_auc', 'average_precision', 'neg_mean_squared_error', 'neg_mean_absolute_error',
        'neg_log_loss', 'r2'
    }
    
    if scoring not in valid_metrics:
        logger.warning(f"Scoring metric '{scoring}' not in common metrics: {valid_metrics}")


def validate_constraints(
    constraints: Optional[Dict],
    param_space: Dict[str, List[Any]]
) -> Optional[Dict]:
    """
    Validate parameter constraints.
    
    Args:
        constraints: Optional constraint dictionary
        param_space: Parameter search space
        
    Returns:
        Validated constraints or None
        
    Raises:
        ValidationError: If constraints are invalid
    """
    if constraints is None:
        return None
    
    if not isinstance(constraints, dict):
        raise ValidationError("Constraints must be dictionary")
    
    validated_constraints = {}
    
    for constraint_name, constraint_def in constraints.items():
        if constraint_name == 'mutual_exclusion':
            if not isinstance(constraint_def, list):
                raise ValidationError("mutual_exclusion must be list of parameter groups")
            
            validated_groups = []
            for group in constraint_def:
                if not isinstance(group, (list, tuple)):
                    raise ValidationError("Each mutual exclusion group must be list")
                
                # Validate parameter names in group
                valid_group = []
                for param_ref in group:
                    if isinstance(param_ref, str):
                        # Format: "param_name_value_idx" or "param_name"
                        if '_' in param_ref:
                            param_name = '_'.join(param_ref.split('_')[:-1])
                        else:
                            param_name = param_ref
                        
                        if param_name not in param_space:
                            raise ValidationError(f"Unknown parameter in constraint: {param_name}")
                        
                        valid_group.append(param_ref)
                
                if valid_group:
                    validated_groups.append(valid_group)
            
            validated_constraints['mutual_exclusion'] = validated_groups
        
        elif constraint_name == 'conditional':
            if not isinstance(constraint_def, list):
                raise ValidationError("conditional must be list of (if, then) pairs")
            
            validated_conditionals = []
            for condition in constraint_def:
                if not isinstance(condition, (tuple, list)) or len(condition) != 2:
                    raise ValidationError("Each conditional must be (if, then) pair")
                
                if_param, then_param = condition
                
                # Basic validation of parameter references
                if isinstance(if_param, str) and isinstance(then_param, str):
                    validated_conditionals.append((if_param, then_param))
                else:
                    raise ValidationError("Conditional parameters must be strings")
            
            validated_constraints['conditional'] = validated_conditionals
        
        elif constraint_name == 'budget':
            if not isinstance(constraint_def, dict):
                raise ValidationError("budget constraint must be dictionary")
            
            validated_budget = {}
            for resource, limit in constraint_def.items():
                if not isinstance(limit, (int, float)):
                    raise ValidationError(f"Budget limit for {resource} must be numeric")
                validated_budget[resource] = float(limit)
            
            validated_constraints['budget'] = validated_budget
        
        else:
            logger.warning(f"Unknown constraint type: {constraint_name}")
            validated_constraints[constraint_name] = constraint_def
    
    return validated_constraints


def validate_backend_config(
    backend: str,
    token: Optional[str] = None,
    penalty_strength: float = 2.0,
    **kwargs
) -> None:
    """
    Validate backend configuration.
    
    Args:
        backend: Backend name
        token: API token (required for some backends)
        penalty_strength: QUBO penalty strength
        **kwargs: Additional backend parameters
        
    Raises:
        ValidationError: If configuration is invalid
    """
    valid_backends = ['dwave', 'simulator', 'neal', 'tabu', 'qbsolv']
    
    if not isinstance(backend, str):
        raise ValidationError(f"Backend must be string, got {type(backend)}")
    
    if backend not in valid_backends:
        logger.warning(f"Backend '{backend}' not in known backends: {valid_backends}")
    
    # Check D-Wave specific requirements
    if backend == 'dwave':
        if not token:
            raise ValidationError("D-Wave backend requires API token")
        if not isinstance(token, str) or len(token) < 10:
            raise ValidationError("Invalid D-Wave API token format")
    
    # Validate penalty strength
    if not isinstance(penalty_strength, (int, float)):
        raise ValidationError(f"penalty_strength must be numeric, got {type(penalty_strength)}")
    
    if penalty_strength <= 0:
        raise ValidationError(f"penalty_strength must be positive, got {penalty_strength}")
    
    if penalty_strength > 100:
        logger.warning(f"Large penalty strength ({penalty_strength}) may dominate objective")


def sanitize_input(value: Any) -> Any:
    """
    Sanitize input value for security.
    
    Args:
        value: Input value to sanitize
        
    Returns:
        Sanitized value
    """
    if isinstance(value, str):
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', value)
        return sanitized.strip()
    elif isinstance(value, (int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [sanitize_input(item) for item in value]
    elif isinstance(value, dict):
        return {sanitize_input(k): sanitize_input(v) for k, v in value.items()}
    else:
        return value


def check_parameter_compatibility(params: Dict[str, Any], model_class: Type) -> None:
    """
    Check if parameters are compatible with model class.
    
    Args:
        params: Parameter dictionary
        model_class: Model class
        
    Raises:
        ValidationError: If parameters are incompatible
    """
    try:
        # Get model constructor signature
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        # Check for invalid parameters
        invalid_params = set(params.keys()) - valid_params
        if invalid_params:
            logger.warning(f"Parameters not recognized by {model_class.__name__}: {invalid_params}")
        
    except Exception as e:
        logger.debug(f"Could not validate parameter compatibility: {e}")