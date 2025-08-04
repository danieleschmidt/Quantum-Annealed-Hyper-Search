"""
D-Wave quantum annealing backend for real quantum hardware.
"""

import logging
from typing import Any, Dict, Optional

import dimod

from .base_backend import QuantumBackend

logger = logging.getLogger(__name__)


class DWaveBackend(QuantumBackend):
    """
    D-Wave quantum annealer backend.
    
    Interfaces with D-Wave quantum computers through the Ocean SDK.
    Supports both quantum annealing and hybrid classical-quantum solvers.
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        solver: Optional[str] = None,
        chain_strength: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize D-Wave backend.
        
        Args:
            token: D-Wave API token
            endpoint: D-Wave API endpoint URL
            solver: Specific solver to use (auto-select if None)
            chain_strength: Strength of embedding chains
        """
        super().__init__(**kwargs)
        
        self.token = token
        self.endpoint = endpoint
        self.solver_name = solver
        self.chain_strength = chain_strength
        
        # Try to initialize D-Wave connection
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
            from dwave.system import LeapHybridSampler
            import dwave.inspector
            
            # Initialize quantum sampler
            sampler_kwargs = {}
            if token:
                sampler_kwargs['token'] = token
            if endpoint:
                sampler_kwargs['endpoint'] = endpoint
            if solver:
                sampler_kwargs['solver'] = solver
            
            self.quantum_sampler = DWaveSampler(**sampler_kwargs)
            self.embedded_sampler = EmbeddingComposite(self.quantum_sampler)
            
            # Initialize hybrid sampler for large problems
            hybrid_kwargs = {}
            if token:
                hybrid_kwargs['token'] = token
            self.hybrid_sampler = LeapHybridSampler(**hybrid_kwargs)
            
            self.available = True
            logger.info(f"Connected to D-Wave solver: {self.quantum_sampler.solver.name}")
            
        except ImportError:
            logger.error("D-Wave Ocean SDK not available. Install with: pip install dwave-ocean-sdk")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to connect to D-Wave: {e}")
            self.available = False
    
    def sample_qubo(
        self,
        Q: Dict[tuple, float],
        num_reads: int = 1000,
        use_hybrid: Optional[bool] = None,
        **kwargs
    ) -> dimod.SampleSet:
        """
        Sample from QUBO using D-Wave quantum annealer.
        
        Args:
            Q: QUBO matrix as dictionary
            num_reads: Number of samples to generate
            use_hybrid: Force hybrid solver usage (auto-decide if None)
            **kwargs: Additional D-Wave parameters
            
        Returns:
            SampleSet with quantum annealing results
        """
        if not self.available:
            raise RuntimeError("D-Wave backend not available")
        
        # Determine whether to use hybrid solver
        num_variables = len(set(var for edge in Q.keys() for var in edge))
        
        if use_hybrid is None:
            # Auto-decide based on problem size
            use_hybrid = num_variables > 100  # Typical embedding limit
        
        if use_hybrid:
            return self._sample_hybrid(Q, **kwargs)
        else:
            return self._sample_quantum(Q, num_reads, **kwargs)
    
    def _sample_quantum(
        self,
        Q: Dict[tuple, float],
        num_reads: int,
        **kwargs
    ) -> dimod.SampleSet:
        """Sample using quantum annealer with minor embedding."""
        sampler_kwargs = {
            'num_reads': num_reads,
        }
        
        # Set chain strength
        if self.chain_strength is not None:
            sampler_kwargs['chain_strength'] = self.chain_strength
        elif 'chain_strength' not in kwargs:
            # Auto-set chain strength based on QUBO weights
            max_weight = max(abs(w) for w in Q.values()) if Q else 1.0
            sampler_kwargs['chain_strength'] = max_weight * 2
        
        # Add custom parameters
        for key, value in kwargs.items():
            if key not in sampler_kwargs:
                sampler_kwargs[key] = value
        
        try:
            logger.info(f"Submitting QUBO to quantum annealer ({num_reads} reads)")
            sampleset = self.embedded_sampler.sample_qubo(Q, **sampler_kwargs)
            
            logger.info(f"Quantum annealing completed")
            logger.info(f"Best energy: {sampleset.first.energy:.4f}")
            logger.info(f"Chain breaks: {sampleset.data_vectors.get('chain_break_fraction', [0])[0]:.3f}")
            
            return sampleset
            
        except Exception as e:
            logger.error(f"Quantum sampling failed: {e}")
            raise
    
    def _sample_hybrid(
        self,
        Q: Dict[tuple, float],
        **kwargs
    ) -> dimod.SampleSet:
        """Sample using hybrid classical-quantum solver."""
        try:
            logger.info("Submitting QUBO to hybrid solver")
            
            # Hybrid solvers don't need num_reads parameter
            hybrid_kwargs = {k: v for k, v in kwargs.items() if k != 'num_reads'}
            
            sampleset = self.hybrid_sampler.sample_qubo(Q, **hybrid_kwargs)
            
            logger.info("Hybrid solving completed")
            logger.info(f"Best energy: {sampleset.first.energy:.4f}")
            
            return sampleset
            
        except Exception as e:
            logger.error(f"Hybrid sampling failed: {e}")
            raise
    
    def get_properties(self) -> Dict[str, Any]:
        """Get D-Wave backend properties."""
        if not self.available:
            return {
                'name': 'D-Wave',
                'type': 'quantum',
                'available': False,
                'error': 'Backend not initialized'
            }
        
        try:
            solver_props = self.quantum_sampler.solver.properties
            
            return {
                'name': 'D-Wave Quantum Annealer',
                'type': 'quantum',
                'available': True,
                'solver': self.quantum_sampler.solver.name,
                'qubits': solver_props.get('num_qubits', 'unknown'),
                'connectivity': solver_props.get('topology', {}).get('type', 'unknown'),
                'programming_thermalization': solver_props.get('default_programming_thermalization'),
                'annealing_time_range': solver_props.get('annealing_time_range'),
                'quantum_annealing': True,
                'hybrid_available': hasattr(self, 'hybrid_sampler') and self.hybrid_sampler is not None
            }
            
        except Exception as e:
            return {
                'name': 'D-Wave',
                'type': 'quantum', 
                'available': False,
                'error': str(e)
            }
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get detailed information about the current solver."""
        if not self.available:
            return {}
        
        try:
            solver = self.quantum_sampler.solver
            return {
                'name': solver.name,
                'properties': dict(solver.properties),
                'parameters': dict(solver.parameters),
            }
        except Exception as e:
            logger.error(f"Failed to get solver info: {e}")
            return {}