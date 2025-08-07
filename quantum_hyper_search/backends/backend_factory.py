"""
Factory for creating quantum computing backends.
"""

import logging
from typing import Optional

from .base_backend import QuantumBackend

logger = logging.getLogger(__name__)


def get_backend(backend_name: str, **kwargs) -> QuantumBackend:
    """
    Factory function to create quantum backends.
    
    Args:
        backend_name: Name of the backend ('dwave', 'simulator', 'neal', 'simple')
        **kwargs: Backend-specific configuration
        
    Returns:
        Configured quantum backend instance
    """
    backend_name = backend_name.lower()
    
    if backend_name == 'dwave':
        from .dwave_backend import DWaveBackend
        return DWaveBackend(**kwargs)
    elif backend_name in ['simulator', 'neal']:
        try:
            from .simulator_backend import SimulatorBackend
            return SimulatorBackend(**kwargs)
        except ImportError:
            logger.warning("D-Wave simulator not available, falling back to simple simulator")
            from .simple_simulator import SimpleSimulatorBackend
            return SimpleSimulatorBackend(**kwargs)
    elif backend_name == 'simple':
        from .simple_simulator import SimpleSimulatorBackend
        return SimpleSimulatorBackend(**kwargs)
    else:
        available_backends = ['dwave', 'simulator', 'neal', 'simple']
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available backends: {available_backends}"
        )