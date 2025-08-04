"""
Quantum-Annealed Hyperparameter Search Library

A hybrid quantum-classical library for hyperparameter optimization using 
D-Wave quantum annealers with seamless integration for Optuna and Ray Tune.
"""

from .core.quantum_hyper_search import QuantumHyperSearch
from .core.qubo_encoder import QUBOEncoder
from .backends.backend_factory import get_backend

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

__all__ = [
    "QuantumHyperSearch",
    "QUBOEncoder", 
    "get_backend",
]