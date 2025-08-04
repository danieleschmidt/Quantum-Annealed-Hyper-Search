"""
Quantum backend implementations for different hardware and simulators.
"""

from typing import Dict, Type
from .base_backend import BaseBackend
from .simulator import SimulatorBackend

# Registry of available backends
_BACKENDS: Dict[str, Type[BaseBackend]] = {
    "simulator": SimulatorBackend,
}

# Conditional imports for optional dependencies
try:
    from .dwave_backend import DWaveBackend
    _BACKENDS["dwave"] = DWaveBackend
except ImportError:
    pass

try:
    from .neal_backend import NealBackend
    _BACKENDS["neal"] = NealBackend
except ImportError:
    pass


def get_backend(backend_name: str) -> Type[BaseBackend]:
    """
    Get backend class by name.
    
    Args:
        backend_name: Name of the backend
        
    Returns:
        Backend class
        
    Raises:
        ValueError: If backend is not available
    """
    if backend_name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(f"Backend '{backend_name}' not available. Available: {available}")
    
    return _BACKENDS[backend_name]


def register_backend(name: str, backend_class: Type[BaseBackend]) -> None:
    """
    Register a new backend.
    
    Args:
        name: Backend name
        backend_class: Backend implementation class
    """
    _BACKENDS[name] = backend_class


def list_backends() -> Dict[str, bool]:
    """
    List all available backends and their availability status.
    
    Returns:
        Dictionary mapping backend names to availability status
    """
    status = {}
    for name, backend_class in _BACKENDS.items():
        try:
            backend = backend_class()
            status[name] = backend.is_available()
        except Exception:
            status[name] = False
    
    return status


__all__ = [
    "get_backend",
    "register_backend", 
    "list_backends",
    "BaseBackend",
    "SimulatorBackend",
]