"""
Simulator backend using classical algorithms to simulate quantum annealing.
"""

import logging
from typing import Any, Dict, Optional

import dimod
import numpy as np

from .base_backend import QuantumBackend

logger = logging.getLogger(__name__)


class SimulatorBackend(QuantumBackend):
    """
    Classical simulator backend using simulated annealing.
    
    Uses the D-Wave Neal simulator for fast classical approximation
    of quantum annealing behavior.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        beta_range: Optional[tuple] = None,
        num_sweeps: int = 1000,
        **kwargs
    ):
        """
        Initialize simulator backend.
        
        Args:
            seed: Random seed for reproducibility
            beta_range: Temperature range for annealing (min_beta, max_beta)
            num_sweeps: Number of Monte Carlo sweeps
        """
        super().__init__(**kwargs)
        
        try:
            import neal
            self.sampler = neal.SimulatedAnnealingSampler()
            self.available = True
        except ImportError:
            logger.warning("Neal simulator not available. Install with: pip install dwave-neal")
            self.sampler = None
            self.available = False
        
        self.seed = seed
        self.beta_range = beta_range or (0.1, 10.0)
        self.num_sweeps = num_sweeps
        
        if seed is not None:
            np.random.seed(seed)
    
    def sample_qubo(
        self,
        Q: Dict[tuple, float],
        num_reads: int = 1000,
        **kwargs
    ) -> dimod.SampleSet:
        """
        Sample from QUBO using simulated annealing.
        
        Args:
            Q: QUBO matrix as dictionary
            num_reads: Number of samples to generate
            **kwargs: Additional parameters for the sampler
            
        Returns:
            SampleSet with optimization results
        """
        if not self.available:
            # Fallback to basic random sampling
            return self._random_sampling_fallback(Q, num_reads)
        
        # Configure sampler parameters
        sampler_kwargs = {
            'num_reads': num_reads,
            'beta_range': kwargs.get('beta_range', self.beta_range),
            'num_sweeps': kwargs.get('num_sweeps', self.num_sweeps),
        }
        
        if self.seed is not None:
            sampler_kwargs['seed'] = self.seed
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in sampler_kwargs:
                sampler_kwargs[key] = value
        
        try:
            # Sample using Neal simulator
            sampleset = self.sampler.sample_qubo(Q, **sampler_kwargs)
            
            # Ensure record attribute exists for compatibility
            if not hasattr(sampleset, 'record'):
                import collections
                Record = collections.namedtuple('Record', ['sample', 'energy', 'chain_break_fraction'])
                sampleset.record = [
                    Record(sample=dict(sample), energy=energy, chain_break_fraction=0.0)
                    for sample, energy in zip(sampleset.samples(), sampleset.data_vectors['energy'])
                ]
            
            logger.info(f"Simulator completed {num_reads} reads")
            logger.info(f"Best energy: {sampleset.first.energy:.4f}")
            
            return sampleset
            
        except Exception as e:
            logger.error(f"Simulator sampling failed: {e}")
            return self._random_sampling_fallback(Q, num_reads)
    
    def _random_sampling_fallback(
        self,
        Q: Dict[tuple, float],
        num_reads: int
    ) -> dimod.SampleSet:
        """
        Fallback to random sampling when simulator is not available.
        
        Args:
            Q: QUBO matrix
            num_reads: Number of random samples
            
        Returns:
            SampleSet with random samples
        """
        logger.warning("Using random sampling fallback")
        
        # Get all variables in the QUBO
        variables = set()
        for (i, j) in Q.keys():
            variables.add(i)
            variables.add(j)
        variables = sorted(list(variables))
        
        # Generate random binary samples
        samples = []
        energies = []
        
        for _ in range(num_reads):
            sample = {var: np.random.randint(0, 2) for var in variables}
            energy = self._calculate_qubo_energy(Q, sample)
            samples.append(sample)
            energies.append(energy)
        
        # Create SampleSet with proper record structure
        sampleset = dimod.SampleSet.from_samples(
            samples,
            energy=energies,
            vartype=dimod.BINARY
        )
        
        # Add record attribute for compatibility
        if not hasattr(sampleset, 'record'):
            # Create record-like structure
            import collections
            Record = collections.namedtuple('Record', ['sample', 'energy', 'chain_break_fraction'])
            sampleset.record = [
                Record(sample=sample, energy=energy, chain_break_fraction=0.0)
                for sample, energy in zip(samples, energies)
            ]
        
        return sampleset
    
    def _calculate_qubo_energy(
        self,
        Q: Dict[tuple, float],
        sample: Dict[int, int]
    ) -> float:
        """Calculate QUBO energy for a given sample."""
        energy = 0.0
        
        for (i, j), weight in Q.items():
            if i == j:
                # Linear term
                energy += weight * sample.get(i, 0)
            else:
                # Quadratic term
                energy += weight * sample.get(i, 0) * sample.get(j, 0)
        
        return energy
    
    def get_properties(self) -> Dict[str, Any]:
        """Get simulator backend properties."""
        return {
            'name': 'Classical Simulator',
            'type': 'simulator',
            'available': self.available,
            'qubits': 'unlimited',
            'connectivity': 'fully_connected',
            'sampler': 'simulated_annealing',
            'seed': self.seed,
            'beta_range': self.beta_range,
            'num_sweeps': self.num_sweeps
        }