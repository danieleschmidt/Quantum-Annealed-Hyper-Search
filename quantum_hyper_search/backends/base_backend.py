"""
Base backend implementation.
"""

from quantum_hyper_search.core.base import QuantumBackend


class BaseBackend(QuantumBackend):
    """
    Base implementation of QuantumBackend with common functionality.
    """
    
    def __init__(self, token=None, **kwargs):
        """Initialize base backend."""
        super().__init__(token=token, **kwargs)
        self.name = "base"
        
    def validate_qubo(self, Q):
        """
        Validate QUBO matrix format.
        
        Args:
            Q: QUBO matrix to validate
            
        Raises:
            ValueError: If QUBO format is invalid
        """
        import numpy as np
        
        if not isinstance(Q, np.ndarray):
            raise ValueError("QUBO must be a numpy array")
        
        if len(Q.shape) != 2:
            raise ValueError("QUBO must be a 2D matrix")
        
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("QUBO must be square")
        
        if Q.shape[0] == 0:
            raise ValueError("QUBO cannot be empty")
    
    def format_samples(self, raw_samples, Q_size):
        """
        Format raw samples to standard format.
        
        Args:
            raw_samples: Raw samples from backend
            Q_size: Size of QUBO matrix
            
        Returns:
            List of formatted samples
        """
        formatted_samples = []
        
        for sample in raw_samples:
            if isinstance(sample, dict):
                # Ensure all variables are present
                formatted_sample = {}
                for i in range(Q_size):
                    formatted_sample[i] = sample.get(i, 0)
                formatted_samples.append(formatted_sample)
            else:
                # Convert array-like to dict
                formatted_sample = {}
                for i, value in enumerate(sample):
                    if i < Q_size:
                        formatted_sample[i] = int(value)
                formatted_samples.append(formatted_sample)
        
        return formatted_samples