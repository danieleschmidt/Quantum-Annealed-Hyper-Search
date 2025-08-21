"""
Quantum-Annealed Hyperparameter Search Library

A hybrid quantum-classical library for hyperparameter optimization using 
D-Wave quantum annealers with seamless integration for Optuna and Ray Tune.
"""

# Generation 3: Use optimized version for production-ready scaling
from .optimized_main import QuantumHyperSearchOptimized as QuantumHyperSearch
from .core.optimization_history import OptimizationHistory
from .core.base import QuantumBackend
from .core.qubo_encoder import QUBOEncoder
from .backends.backend_factory import get_backend

# Security Framework Integration
from .security import (
    QuantumSecurityFramework,
    SecurityPolicy,
    ComplianceManager,
    EncryptionManager,
    AuthenticationManager,
    AuthorizationManager
)

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "contact@terragonlabs.com"

__all__ = [
    "QuantumHyperSearch",
    "QuantumBackend", 
    "QUBOEncoder",
    "get_backend",
    "QuantumSecurityFramework",
    "SecurityPolicy",
    "ComplianceManager",
    "EncryptionManager",
    "AuthenticationManager",
    "AuthorizationManager",
]
