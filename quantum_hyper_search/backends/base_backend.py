"""
Abstract base class for quantum computing backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import dimod


class QuantumBackend(ABC):
    """Abstract base class for quantum computing backends."""
    
    def __init__(self, **kwargs):
        """Initialize the backend with configuration parameters."""
        pass
    
    @abstractmethod
    def sample_qubo(
        self,
        Q: Dict[tuple, float],
        num_reads: int = 1000,
        **kwargs
    ) -> dimod.SampleSet:
        """
        Sample from a QUBO problem.
        
        Args:
            Q: QUBO matrix as dictionary of {(i,j): weight}
            num_reads: Number of samples to generate
            **kwargs: Backend-specific parameters
            
        Returns:
            SampleSet containing the results
        """
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        Get backend properties and capabilities.
        
        Returns:
            Dictionary with backend information
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if the backend is available and properly configured.
        
        Returns:
            True if backend is ready to use
        """
        try:
            properties = self.get_properties()
            return properties.get('available', False)
        except Exception:
            return False


class BaseBackend(QuantumBackend):
    """
    Base implementation of QuantumBackend with common functionality.
    """
    
    def __init__(self, token=None, **kwargs):
        """Initialize base backend."""
        super().__init__(**kwargs)
        self.token = token
        self.name = "base"
        
    def validate_qubo(self, Q):
        """
        Validate QUBO matrix format.
        
        Args:
            Q: QUBO matrix to validate
            
        Raises:
            ValueError: If QUBO format is invalid
        """
        if isinstance(Q, dict):
            # Validate dictionary format QUBO
            if not Q:
                raise ValueError("QUBO cannot be empty")
            for key in Q:
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError("QUBO dictionary keys must be tuples of length 2")
        elif isinstance(Q, np.ndarray):
            # Validate numpy array format QUBO
            if len(Q.shape) != 2:
                raise ValueError("QUBO must be a 2D matrix")
            
            if Q.shape[0] != Q.shape[1]:
                raise ValueError("QUBO must be square")
            
            if Q.shape[0] == 0:
                raise ValueError("QUBO cannot be empty")
        else:
            raise ValueError("QUBO must be a numpy array or dictionary")
    
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
