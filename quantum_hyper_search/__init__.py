```python
"""
Quantum-Annealed Hyperparameter Search Library

A hybrid quantum-classical library for hyperparameter optimization using 
D-Wave quantum annealers with seamless integration for Optuna and Ray Tune.
"""

from .main import QuantumHyperSearch
from .core.base import QuantumBackend
from .core.qubo_encoder import QUBOEncoder
from .backends import get_backend, register_backend

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

__all__ = [
    "QuantumHyperSearch",
    "QuantumBackend",
    "QUBOEncoder",
    "get_backend",
    "register_backend",
]
```
