"""
Simple classical simulator backend for testing and development.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .base_backend import BaseBackend


class SimulatorBackend(BaseBackend):
    """
    Classical simulator backend using simulated annealing.
    
    This backend provides a simple classical simulation of quantum annealing
    for testing and development when quantum hardware is not available.
    """
    
    def __init__(self, token: Optional[str] = None, **kwargs):
        """
        Initialize simulator backend.
        
        Args:
            token: Not used for simulator (for compatibility)
            **kwargs: Simulator configuration options
        """
        super().__init__(token=token, **kwargs)
        self.name = "simulator"
        
        # Simulation parameters
        self.temperature_schedule = kwargs.get("temperature_schedule", "linear")
        self.initial_temperature = kwargs.get("initial_temperature", 1.0)
        self.final_temperature = kwargs.get("final_temperature", 0.01)
        self.max_iterations = kwargs.get("max_iterations", 1000)
        self.random_seed = kwargs.get("random_seed", None)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def sample_qubo(
        self,
        Q: np.ndarray,
        num_reads: int = 1000,
        **kwargs
    ) -> List[Dict[int, int]]:
        """
        Sample from QUBO using simulated annealing.
        
        Args:
            Q: QUBO matrix
            num_reads: Number of annealing runs
            **kwargs: Additional sampling parameters
            
        Returns:
            List of binary samples
        """
        self.validate_qubo(Q)
        
        n_vars = Q.shape[0]
        samples = []
        
        print(f"Running {num_reads} simulated annealing samples...")
        
        for read in range(num_reads):
            # Initialize random binary state
            state = np.random.randint(0, 2, n_vars)
            
            # Simulated annealing
            current_energy = self._compute_energy(state, Q)
            best_state = state.copy()
            best_energy = current_energy
            
            for iteration in range(self.max_iterations):
                # Temperature schedule
                progress = iteration / self.max_iterations
                temperature = self._get_temperature(progress)
                
                if temperature <= 0:
                    break
                
                # Propose random bit flip
                flip_idx = np.random.randint(0, n_vars)
                new_state = state.copy()
                new_state[flip_idx] = 1 - new_state[flip_idx]
                
                # Compute energy change
                new_energy = self._compute_energy(new_state, Q)
                delta_energy = new_energy - current_energy
                
                # Accept or reject
                if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                    state = new_state
                    current_energy = new_energy
                    
                    # Update best
                    if current_energy < best_energy:
                        best_state = state.copy()
                        best_energy = current_energy
            
            # Convert to sample format
            sample = {i: int(best_state[i]) for i in range(n_vars)}
            samples.append(sample)
        
        # Sort by energy (best first)
        energy_sample_pairs = []
        for sample in samples:
            energy = self._compute_energy_from_sample(sample, Q)
            energy_sample_pairs.append((energy, sample))
        
        energy_sample_pairs.sort(key=lambda x: x[0])
        sorted_samples = [sample for energy, sample in energy_sample_pairs]
        
        print(f"Simulated annealing complete. Best energy: {energy_sample_pairs[0][0]:.4f}")
        
        return sorted_samples
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get simulator information.
        
        Returns:
            Dictionary with simulator specifications
        """
        return {
            "name": "Classical Simulator",
            "type": "simulated_annealing",
            "qubits": "unlimited",
            "connectivity": "complete",
            "temperature_schedule": self.temperature_schedule,
            "max_iterations": self.max_iterations,
            "initial_temperature": self.initial_temperature,
            "final_temperature": self.final_temperature,
        }
    
    def _compute_energy(self, state: np.ndarray, Q: np.ndarray) -> float:
        """Compute QUBO energy for a given state."""
        return float(state.T @ Q @ state)
    
    def _compute_energy_from_sample(self, sample: Dict[int, int], Q: np.ndarray) -> float:
        """Compute QUBO energy from sample dictionary."""
        state = np.array([sample.get(i, 0) for i in range(Q.shape[0])])
        return self._compute_energy(state, Q)
    
    def _get_temperature(self, progress: float) -> float:
        """Get temperature at given progress (0 to 1)."""
        if self.temperature_schedule == "linear":
            return self.initial_temperature * (1 - progress) + self.final_temperature * progress
        elif self.temperature_schedule == "exponential":
            return self.initial_temperature * (self.final_temperature / self.initial_temperature) ** progress
        elif self.temperature_schedule == "geometric":
            ratio = (self.final_temperature / self.initial_temperature) ** (1 / self.max_iterations)
            return self.initial_temperature * (ratio ** (progress * self.max_iterations))
        else:
            # Default to linear
            return self.initial_temperature * (1 - progress) + self.final_temperature * progress