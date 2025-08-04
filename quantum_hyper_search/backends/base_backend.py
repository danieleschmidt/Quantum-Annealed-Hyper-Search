"""
Abstract base class for quantum computing backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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