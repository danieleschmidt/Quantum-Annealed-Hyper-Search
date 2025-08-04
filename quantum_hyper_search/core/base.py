"""
Base classes for quantum backends and core functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class QuantumBackend(ABC):
    """
    Abstract base class for quantum annealing backends.
    """
    
    def __init__(self, token: Optional[str] = None, **kwargs):
        """
        Initialize quantum backend.
        
        Args:
            token: Authentication token for quantum hardware
            **kwargs: Backend-specific configuration
        """
        self.token = token
        self.config = kwargs
        
    @abstractmethod
    def sample_qubo(
        self,
        Q: np.ndarray,
        num_reads: int = 1000,
        **kwargs
    ) -> List[Dict[int, int]]:
        """
        Sample from QUBO using quantum annealer.
        
        Args:
            Q: QUBO matrix
            num_reads: Number of annealing reads
            **kwargs: Backend-specific sampling parameters
            
        Returns:
            List of binary samples as dictionaries {variable_idx: value}
        """
        pass
    
    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the quantum hardware.
        
        Returns:
            Dictionary with hardware specifications
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if backend is available and accessible.
        
        Returns:
            True if backend can be used
        """
        try:
            info = self.get_hardware_info()
            return info is not None
        except Exception:
            return False


class QUBOSample:
    """
    Represents a single sample from quantum annealing.
    """
    
    def __init__(
        self,
        sample: Dict[int, int],
        energy: float,
        num_occurrences: int = 1,
        chain_break_fraction: float = 0.0
    ):
        """
        Initialize QUBO sample.
        
        Args:
            sample: Binary variable assignments
            energy: Energy of the sample
            num_occurrences: Number of times this sample occurred
            chain_break_fraction: Fraction of chains that were broken
        """
        self.sample = sample
        self.energy = energy
        self.num_occurrences = num_occurrences
        self.chain_break_fraction = chain_break_fraction
        
    def __repr__(self) -> str:
        return f"QUBOSample(energy={self.energy:.4f}, occurrences={self.num_occurrences})"
    
    def is_valid(self) -> bool:
        """
        Check if sample is valid (no chain breaks).
        
        Returns:
            True if sample has no chain breaks
        """
        return self.chain_break_fraction == 0.0