"""
Input validation and safety checks for quantum hyperparameter search.
"""

from typing import Any, Dict, List, Optional, Union, Type
import numpy as np
import inspect
from sklearn.base import BaseEstimator


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_search_space(search_space: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Validate and sanitize hyperparameter search space.
    
    Args:
        search_space: Dictionary mapping parameter names to possible values
        
    Returns:
        Validated and sanitized search space
        
    Raises:
        ValidationError: If search space is invalid
    """
    if not isinstance(search_space, dict):
        raise ValidationError("Search space must be a dictionary")
    
    if not search_space:
        raise ValidationError("Search space cannot be empty")
    
    validated_space = {}
    
    for param_name, param_values in search_space.items():
        # Validate parameter name
        if not isinstance(param_name, str):
            raise ValidationError(f"Parameter name must be string, got {type(param_name)}")
        
        if not param_name.isidentifier():
            raise ValidationError(f"Parameter name '{param_name}' is not a valid identifier")
        
        if param_name.startswith('_'):
            raise ValidationError(f"Parameter name '{param_name}' cannot start with underscore")
        
        # Validate parameter values
        if not isinstance(param_values, (list, tuple)):
            raise ValidationError(f"Parameter values for '{param_name}' must be list or tuple")
        
        if len(param_values) == 0:
            raise ValidationError(f"Parameter '{param_name}' must have at least one value")
        
        if len(param_values) > 1000:
            raise ValidationError(f"Parameter '{param_name}' has too many values (max 1000)")
        
        # Check for duplicates
        unique_values = []
        seen = set()
        for value in param_values:
            # Handle unhashable types
            try:
                if value not in seen:
                    unique_values.append(value)
                    seen.add(value)
            except TypeError:
                # For unhashable types, do linear search
                if value not in unique_values:
                    unique_values.append(value)
        
        if len(unique_values) != len(param_values):
            raise ValidationError(f"Parameter '{param_name}' contains duplicate values")
        
        # Validate individual values
        for i, value in enumerate(unique_values):
            if value is None and param_name != 'max_depth':  # Allow None for max_depth
                continue
            
            # Check for dangerous values
            if isinstance(value, str):
                if len(value) > 1000:
                    raise ValidationError(f"String value too long in '{param_name}' at index {i}")
                if value.strip() != value:
                    unique_values[i] = value.strip()  # Auto-strip whitespace
            
            elif isinstance(value, (int, float)):
                if not np.isfinite(value):
                    raise ValidationError(f"Invalid numeric value in '{param_name}' at index {i}")
                if abs(value) > 1e15:
                    raise ValidationError(f"Numeric value too large in '{param_name}' at index {i}")
        
        validated_space[param_name] = unique_values
    
    # Check total search space size
    total_combinations = 1
    for param_values in validated_space.values():
        total_combinations *= len(param_values)
        if total_combinations > 1e8:  # 100 million combinations max
            raise ValidationError("Search space too large (>100M combinations)")
    
    return validated_space


def validate_model_class(model_class: Type) -> None:
    """
    Validate that model class is compatible with scikit-learn.
    
    Args:
        model_class: Model class to validate
        
    Raises:
        ValidationError: If model class is invalid
    """
    if not inspect.isclass(model_class):
        raise ValidationError("model_class must be a class")
    
    # Check if it's a scikit-learn compatible estimator
    if not issubclass(model_class, BaseEstimator):
        # Check for required methods manually
        required_methods = ['fit', 'predict']
        for method in required_methods:
            if not hasattr(model_class, method):
                raise ValidationError(f"Model class must have '{method}' method")
    
    # Check constructor
    try:
        sig = inspect.signature(model_class.__init__)
        # Should be able to instantiate with no arguments (except self)
        params = [p for name, p in sig.parameters.items() if name != 'self']
        required_params = [p for p in params if p.default == inspect.Parameter.empty]
        
        if required_params:
            raise ValidationError(f"Model class constructor has required parameters: {[p.name for p in required_params]}")
    
    except Exception as e:
        raise ValidationError(f"Cannot inspect model class constructor: {e}")


def validate_data(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate training data.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Raises:
        ValidationError: If data is invalid
    """
    # Check types
    if not isinstance(X, np.ndarray):
        raise ValidationError("X must be a numpy array")
    
    if not isinstance(y, np.ndarray):
        raise ValidationError("y must be a numpy array")
    
    # Check shapes
    if X.ndim != 2:
        raise ValidationError(f"X must be 2-dimensional, got {X.ndim}")
    
    if y.ndim != 1:
        raise ValidationError(f"y must be 1-dimensional, got {y.ndim}")
    
    if X.shape[0] != y.shape[0]:
        raise ValidationError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
    
    # Check for empty data
    if X.shape[0] == 0:
        raise ValidationError("Cannot work with empty dataset")
    
    if X.shape[1] == 0:
        raise ValidationError("Cannot work with zero features")
    
    # Check for invalid values
    if not np.all(np.isfinite(X)):
        raise ValidationError("X contains invalid values (NaN or inf)")
    
    if not np.all(np.isfinite(y)):
        raise ValidationError("y contains invalid values (NaN or inf)")
    
    # Check data size limits
    if X.shape[0] > 1e6:
        raise ValidationError("Dataset too large (>1M samples)")
    
    if X.shape[1] > 10000:
        raise ValidationError("Too many features (>10K)")


def validate_optimization_params(
    n_iterations: int,
    quantum_reads: int,
    cv_folds: int
) -> None:
    """
    Validate optimization parameters.
    
    Args:
        n_iterations: Number of optimization iterations
        quantum_reads: Number of quantum reads per iteration
        cv_folds: Number of cross-validation folds
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(n_iterations, int):
        raise ValidationError("n_iterations must be an integer")
    
    if n_iterations <= 0:
        raise ValidationError("n_iterations must be positive")
    
    if n_iterations > 1000:
        raise ValidationError("n_iterations too large (max 1000)")
    
    if not isinstance(quantum_reads, int):
        raise ValidationError("quantum_reads must be an integer")
    
    if quantum_reads <= 0:
        raise ValidationError("quantum_reads must be positive")
    
    if quantum_reads > 100000:
        raise ValidationError("quantum_reads too large (max 100K)")
    
    if not isinstance(cv_folds, int):
        raise ValidationError("cv_folds must be an integer")
    
    if cv_folds < 2:
        raise ValidationError("cv_folds must be at least 2")
    
    if cv_folds > 20:
        raise ValidationError("cv_folds too large (max 20)")


def validate_constraints(
    constraints: Optional[Dict[str, Any]],
    search_space: Dict[str, List[Any]]
) -> Optional[Dict[str, Any]]:
    """
    Validate constraint specifications.
    
    Args:
        constraints: Constraint dictionary
        search_space: Parameter search space
        
    Returns:
        Validated constraints
        
    Raises:
        ValidationError: If constraints are invalid
    """
    if constraints is None:
        return None
    
    if not isinstance(constraints, dict):
        raise ValidationError("Constraints must be a dictionary")
    
    validated_constraints = {}
    
    # Validate mutual exclusion constraints
    if 'mutual_exclusion' in constraints:
        mutex_groups = constraints['mutual_exclusion']
        if not isinstance(mutex_groups, list):
            raise ValidationError("mutual_exclusion must be a list")
        
        validated_mutex = []
        for group in mutex_groups:
            if not isinstance(group, list):
                raise ValidationError("mutual_exclusion groups must be lists")
            
            if len(group) < 2:
                raise ValidationError("mutual_exclusion groups must have at least 2 items")
            
            # Validate variable names exist
            valid_group = []
            for var_name in group:
                if not isinstance(var_name, str):
                    raise ValidationError("mutual_exclusion variable names must be strings")
                
                # Check if variable exists in search space encoding
                param_found = False
                for param_name in search_space:
                    if var_name.startswith(f"{param_name}_"):
                        param_found = True
                        break
                
                if param_found:
                    valid_group.append(var_name)
            
            if len(valid_group) >= 2:
                validated_mutex.append(valid_group)
        
        if validated_mutex:
            validated_constraints['mutual_exclusion'] = validated_mutex
    
    # Validate conditional constraints
    if 'conditional' in constraints:
        conditionals = constraints['conditional']
        if not isinstance(conditionals, list):
            raise ValidationError("conditional must be a list")
        
        validated_conditionals = []
        for condition, consequence in conditionals:
            if not isinstance(condition, str) or not isinstance(consequence, str):
                raise ValidationError("conditional constraint items must be strings")
            
            validated_conditionals.append((condition, consequence))
        
        if validated_conditionals:
            validated_constraints['conditional'] = validated_conditionals
    
    return validated_constraints if validated_constraints else None