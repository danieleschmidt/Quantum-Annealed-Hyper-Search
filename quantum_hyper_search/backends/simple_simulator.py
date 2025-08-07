"""
Simple simulator backend that doesn't require D-Wave dependencies.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base_backend import QuantumBackend

logger = logging.getLogger(__name__)


class SampleResult:
    """Simple sample result class to mimic dimod.SampleSet structure."""
    
    def __init__(self, sample: Dict[int, int], energy: float):
        self.sample = sample
        self.energy = energy
        self.chain_break_fraction = 0.0


class SimpleSampleSet:
    """Simple sample set to mimic dimod.SampleSet."""
    
    def __init__(self, samples: List[SampleResult]):
        self.record = samples
        self.data_vectors = {'energy': [s.energy for s in samples]}
        self._samples = samples
    
    def samples(self):
        """Return samples as list of dictionaries."""
        return [s.sample for s in self._samples]


class SimpleSimulatorBackend(QuantumBackend):
    """
    Simple classical simulator that doesn't require D-Wave dependencies.
    
    Uses basic random sampling and simulated annealing-like optimization
    to provide quantum-like behavior for testing and development.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        **kwargs
    ):
        """
        Initialize simple simulator backend.
        
        Args:
            seed: Random seed for reproducibility
            temperature: Initial temperature for annealing
            cooling_rate: Rate at which temperature decreases
            max_iterations: Maximum iterations for annealing
        """
        super().__init__(**kwargs)
        
        self.seed = seed
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.available = True
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        logger.info(f"Initialized SimpleSimulatorBackend (temp={temperature}, cooling={cooling_rate})")
    
    def sample_qubo(
        self,
        Q: Dict[Tuple[int, int], float],
        num_reads: int = 100,
        **kwargs
    ) -> SimpleSampleSet:
        """
        Sample from QUBO using simulated annealing.
        
        Args:
            Q: QUBO matrix as dictionary of {(i,j): value}
            num_reads: Number of samples to generate
            **kwargs: Additional sampling parameters
            
        Returns:
            SimpleSampleSet with samples and energies
        """
        if not Q:
            return SimpleSampleSet([])
        
        # Get problem size
        max_var = max(max(key) for key in Q.keys())
        n_vars = max_var + 1
        
        logger.debug(f"Sampling QUBO with {n_vars} variables, {len(Q)} terms")
        
        samples = []
        
        for read in range(num_reads):
            # Generate random initial state
            state = {i: random.choice([0, 1]) for i in range(n_vars)}
            energy = self._calculate_energy(state, Q)
            
            # Simple simulated annealing
            current_temp = self.temperature
            best_state = state.copy()
            best_energy = energy
            
            for iteration in range(self.max_iterations):
                # Try flipping a random variable
                flip_var = random.randint(0, n_vars - 1)
                new_state = state.copy()
                new_state[flip_var] = 1 - new_state[flip_var]
                new_energy = self._calculate_energy(new_state, Q)
                
                # Accept or reject based on energy and temperature
                delta_e = new_energy - energy
                if delta_e < 0 or (current_temp > 0 and random.random() < np.exp(-delta_e / current_temp)):
                    state = new_state
                    energy = new_energy
                    
                    if energy < best_energy:
                        best_state = state.copy()
                        best_energy = energy
                
                # Cool down
                current_temp *= self.cooling_rate
                
                # Early termination if temperature is very low
                if current_temp < 1e-6:
                    break
            
            # Add some noise to avoid identical samples
            if read > 0 and random.random() < 0.1:
                # Occasionally use a random state instead
                best_state = {i: random.choice([0, 1]) for i in range(n_vars)}
                best_energy = self._calculate_energy(best_state, Q)
            
            samples.append(SampleResult(best_state, best_energy))
        
        # Sort by energy (best first)
        samples.sort(key=lambda x: x.energy)
        
        logger.debug(f"Generated {len(samples)} samples, best energy: {samples[0].energy:.4f}")
        
        return SimpleSampleSet(samples)
    
    def _calculate_energy(self, state: Dict[int, int], Q: Dict[Tuple[int, int], float]) -> float:
        """Calculate QUBO energy for given state."""
        energy = 0.0
        
        for (i, j), coeff in Q.items():
            if i == j:
                # Diagonal term
                energy += coeff * state.get(i, 0)
            else:
                # Off-diagonal term
                energy += coeff * state.get(i, 0) * state.get(j, 0)
        
        return energy
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        return True
    
    def get_properties(self) -> Dict[str, Any]:
        """Get backend properties."""
        return {
            'name': 'simple_simulator',
            'type': 'classical_simulator',
            'qubits': float('inf'),  # No physical limit
            'connectivity': 'all_to_all',
            'temperature': self.temperature,
            'cooling_rate': self.cooling_rate,
            'max_iterations': self.max_iterations
        }
    
    def estimate_cost(self, problem_size: int, num_reads: int) -> Dict[str, float]:
        """Estimate computational cost."""
        return {
            'time_estimate_seconds': problem_size * num_reads * 0.001,  # Very rough estimate
            'memory_mb': problem_size * 0.1,
            'cost_usd': 0.0  # Free simulator
        }