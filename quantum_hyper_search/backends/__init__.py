"""
Quantum backend implementations for different hardware and simulators.
"""

from typing import Dict, Type
from .base_backend import QuantumBackend
from .backend_factory import get_backend

# Registry of available backends
_BACKENDS: Dict[str, Type[QuantumBackend]] = {}

# Conditional imports for optional dependencies
try:
    from .simulator_backend import SimulatorBackend
    _BACKENDS["simulator"] = SimulatorBackend
    _BACKENDS["neal"] = SimulatorBackend  # Neal is just SimulatorBackend
except ImportError:
    pass

try:
    from .dwave_backend import DWaveBackend
    _BACKENDS["dwave"] = DWaveBackend
except ImportError:
    pass

# Always include simple simulator
try:
    from .simple_simulator import SimpleSimulatorBackend
    _BACKENDS["simple"] = SimpleSimulatorBackend
except ImportError:
    pass


def register_backend(name: str, backend_class: Type[QuantumBackend]) -> None:
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
    "QuantumBackend",
]
