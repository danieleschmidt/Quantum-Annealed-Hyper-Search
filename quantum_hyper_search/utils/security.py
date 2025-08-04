"""
Security utilities for quantum hyperparameter search.
"""

import os
import re
import hashlib
from typing import Any, Dict, List, Optional, Set, Union
import logging


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# Dangerous parameter patterns to detect
DANGEROUS_PATTERNS = [
    r'__.*__',  # Python magic methods
    r'.*\.py$',  # File paths ending in .py
    r'/.*',     # Absolute paths
    r'\\.*',    # Windows paths
    r'.*\.\./.*',  # Path traversal
    r'exec',    # Code execution
    r'eval',    # Code evaluation
    r'import',  # Module imports
    r'subprocess',  # Process execution
    r'os\.',    # OS module usage
    r'sys\.',   # Sys module usage
    r'open\(',  # File operations
    r'file\(',  # File operations
]

# Safe parameter value types
SAFE_TYPES = (int, float, bool, str, type(None))

# Maximum safe string length
MAX_STRING_LENGTH = 1000

# Maximum safe numeric values
MAX_NUMERIC_VALUE = 1e15
MIN_NUMERIC_VALUE = -1e15


def sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize parameter dictionary to prevent security issues.
    
    Args:
        parameters: Parameter dictionary to sanitize
        
    Returns:
        Sanitized parameter dictionary
        
    Raises:
        SecurityError: If dangerous patterns are detected
    """
    if not isinstance(parameters, dict):
        raise SecurityError("Parameters must be a dictionary")
    
    sanitized = {}
    
    for key, value in parameters.items():
        # Sanitize key
        safe_key = sanitize_parameter_name(key)
        
        # Sanitize value
        safe_value = sanitize_parameter_value(value)
        
        sanitized[safe_key] = safe_value
    
    return sanitized


def sanitize_parameter_name(name: str) -> str:
    """
    Sanitize parameter name.
    
    Args:
        name: Parameter name to sanitize
        
    Returns:
        Sanitized parameter name
        
    Raises:
        SecurityError: If name contains dangerous patterns
    """
    if not isinstance(name, str):
        raise SecurityError("Parameter name must be a string")
    
    if not name:
        raise SecurityError("Parameter name cannot be empty")
    
    if len(name) > 100:
        raise SecurityError("Parameter name too long")
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            raise SecurityError(f"Dangerous pattern detected in parameter name: {name}")
    
    # Must be valid Python identifier
    if not name.isidentifier():
        raise SecurityError(f"Parameter name is not a valid identifier: {name}")
    
    # Cannot start with underscore (private)
    if name.startswith('_'):
        raise SecurityError(f"Parameter name cannot start with underscore: {name}")
    
    return name


def sanitize_parameter_value(value: Any) -> Any:
    """
    Sanitize parameter value.
    
    Args:
        value: Parameter value to sanitize
        
    Returns:
        Sanitized parameter value
        
    Raises:
        SecurityError: If value contains dangerous patterns
    """
    # Check type safety
    if not isinstance(value, SAFE_TYPES):
        # Allow lists and tuples of safe types
        if isinstance(value, (list, tuple)):
            return [sanitize_parameter_value(item) for item in value]
        else:
            raise SecurityError(f"Unsafe parameter value type: {type(value)}")
    
    if isinstance(value, str):
        return sanitize_string_value(value)
    elif isinstance(value, (int, float)):
        return sanitize_numeric_value(value)
    else:
        return value


def sanitize_string_value(value: str) -> str:
    """
    Sanitize string parameter value.
    
    Args:
        value: String value to sanitize
        
    Returns:
        Sanitized string value
        
    Raises:
        SecurityError: If string contains dangerous patterns
    """
    if len(value) > MAX_STRING_LENGTH:
        raise SecurityError(f"String value too long: {len(value)} > {MAX_STRING_LENGTH}")
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise SecurityError(f"Dangerous pattern detected in string value: {value}")
    
    # Check for control characters
    if any(ord(char) < 32 and char not in '\t\n\r' for char in value):
        raise SecurityError("String contains control characters")
    
    # Strip whitespace
    sanitized = value.strip()
    
    return sanitized


def sanitize_numeric_value(value: Union[int, float]) -> Union[int, float]:
    """
    Sanitize numeric parameter value.
    
    Args:
        value: Numeric value to sanitize
        
    Returns:
        Sanitized numeric value
        
    Raises:
        SecurityError: If value is outside safe range
    """
    import numpy as np
    
    if not np.isfinite(value):
        raise SecurityError(f"Invalid numeric value: {value}")
    
    if value > MAX_NUMERIC_VALUE:
        raise SecurityError(f"Numeric value too large: {value} > {MAX_NUMERIC_VALUE}")
    
    if value < MIN_NUMERIC_VALUE:
        raise SecurityError(f"Numeric value too small: {value} < {MIN_NUMERIC_VALUE}")
    
    return value


def check_safety(
    search_space: Dict[str, List[Any]],
    model_class: type,
    constraints: Optional[Dict] = None
) -> bool:
    """
    Perform comprehensive safety check on optimization setup.
    
    Args:
        search_space: Parameter search space
        model_class: Model class to use
        constraints: Optional constraints
        
    Returns:
        True if setup is safe
        
    Raises:
        SecurityError: If safety issues are detected
    """
    logger = logging.getLogger(__name__)
    
    # Check search space safety
    for param_name, param_values in search_space.items():
        sanitize_parameter_name(param_name)
        
        for value in param_values:
            sanitize_parameter_value(value)
    
    # Check model class safety
    check_model_class_safety(model_class)
    
    # Check constraints safety
    if constraints:
        check_constraints_safety(constraints)
    
    logger.info("Safety check passed")
    return True


def check_model_class_safety(model_class: type) -> None:
    """
    Check model class for safety issues.
    
    Args:
        model_class: Model class to check
        
    Raises:
        SecurityError: If model class is unsafe
    """
    if not hasattr(model_class, '__module__'):
        raise SecurityError("Model class has no module information")
    
    module_name = model_class.__module__
    
    # Allow well-known safe modules
    safe_modules = {
        'sklearn.',
        'xgboost.',
        'lightgbm.',
        'tensorflow.',
        'torch.',
        'keras.',
        'catboost.',
    }
    
    if not any(module_name.startswith(safe) for safe in safe_modules):
        # For custom modules, perform additional checks
        class_name = model_class.__name__
        
        # Check for dangerous patterns in class name
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, class_name, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern in model class name: {class_name}")


def check_constraints_safety(constraints: Dict[str, Any]) -> None:
    """
    Check constraints for safety issues.
    
    Args:
        constraints: Constraints dictionary
        
    Raises:
        SecurityError: If constraints contain unsafe patterns
    """
    # Convert to string for pattern checking
    constraints_str = str(constraints)
    
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, constraints_str, re.IGNORECASE):
            raise SecurityError(f"Dangerous pattern detected in constraints: {pattern}")


def generate_session_id() -> str:
    """
    Generate secure session ID for optimization run.
    
    Returns:
        Secure session ID string
    """
    import time
    import random
    
    # Use current time, process ID, and random data
    data = f"{time.time()}-{os.getpid()}-{random.random()}"
    
    # Hash to create session ID
    session_id = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    return session_id


def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mask sensitive data in logs and outputs.
    
    Args:
        data: Data dictionary to mask
        
    Returns:
        Data with sensitive values masked
    """
    masked = {}
    
    sensitive_keys = {
        'token', 'api_key', 'password', 'secret', 'key',
        'auth', 'credential', 'access_token', 'refresh_token'
    }
    
    for key, value in data.items():
        key_lower = key.lower()
        
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            masked[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked[key] = mask_sensitive_data(value)
        else:
            masked[key] = value
    
    return masked