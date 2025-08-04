"""
Quantum Annealed Hyperparameter Search

A hybrid quantum-classical library for hyperparameter optimization using 
D-Wave quantum annealers with seamless integration for Optuna and Ray Tune.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

from .main import QuantumHyperSearch
from .core.base import QuantumBackend
from .backends import get_backend, register_backend

__all__ = [
    "QuantumHyperSearch",
    "QuantumBackend", 
    "get_backend",
    "register_backend",
]