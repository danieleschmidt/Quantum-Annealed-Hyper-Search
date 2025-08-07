"""
Utility functions and classes for quantum hyperparameter search.
"""

# Import only what exists
try:
    from .validation import validate_search_space, validate_model_class, validate_data
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

__all__ = []

if VALIDATION_AVAILABLE:
    __all__.extend([
        "validate_search_space",
        "validate_model_class", 
        "validate_data",
    ])
