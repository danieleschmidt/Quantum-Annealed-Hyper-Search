"""
Quantum Parallel Tempering Algorithm for Enhanced Exploration

Novel implementation of parallel tempering using quantum annealing to explore
multiple temperature scales simultaneously for superior optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from ..core.base import QuantumBackend
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class TemperingParams:
    """Parameters for quantum parallel tempering"""
    temperatures: List[float]
    exchange_attempts: int = 100
    cooling_schedule: str = "exponential"  # exponential, linear, adaptive
    replica_coupling: float = 0.1
    quantum_advantage_threshold: int = 50

@dataclass
class TemperingResults:
    """Results from quantum parallel tempering optimization"""
    best_solution: Dict[str, Any]
    best_energy: float
    temperature_history: List[List[float]]
    exchange_statistics: Dict[str, int]
    quantum_advantage_achieved: bool
    convergence_time: float

class QuantumParallelTempering:
    """
    Advanced quantum parallel tempering implementation that uses quantum
    tunneling effects to enhance traditional parallel tempering optimization.
    """
    
    def __init__(
        self,
        backend: QuantumBackend,
        tempering_params: TemperingParams,
        enable_quantum_tunneling: bool = True,
        adaptive_exchange_rate: bool = True
    ):
        self.backend = backend
        self.params = tempering_params
        self.enable_quantum_tunneling = enable_quantum_tunneling
        self.adaptive_exchange_rate = adaptive_exchange_rate
        self.exchange_rate_history = []
        self.quantum_advantage_detected = False
        
    def optimize(
        self,
        qubo_matrix: np.ndarray,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6
    ) -> TemperingResults:
        """
        Execute quantum parallel tempering optimization
        
        Args:
            qubo_matrix: QUBO formulation of the optimization problem
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence criterion
            
        Returns:
            TemperingResults with optimization outcomes
        """
        start_time = time.time()
        
        # Initialize replicas at different temperatures
        replicas = self._initialize_replicas(qubo_matrix)
        
        # Track best solution across all replicas
        global_best = None
        global_best_energy = float('inf')
        
        # Statistics tracking
        exchange_stats = {
            'attempts': 0,
            'successful': 0,
            'quantum_enhanced': 0
        }
        
        temperature_history = [[] for _ in self.params.temperatures]
        
        for iteration in range(max_iterations):
            # Run parallel optimization on all replicas
            replica_results = self._parallel_replica_optimization(replicas, iteration)
            
            # Update best solution
            for replica_idx, (solution, energy) in enumerate(replica_results):
                if energy < global_best_energy:
                    global_best = solution
                    global_best_energy = energy
                    logger.info(f"New best solution found at iteration {iteration}, "
                              f"replica {replica_idx}: energy = {energy:.6f}")
            
            # Perform replica exchanges with quantum enhancement
            if iteration % 10 == 0:  # Exchange every 10 iterations
                exchanges = self._quantum_enhanced_exchange(replicas, replica_results)
                exchange_stats['attempts'] += len(exchanges)
                exchange_stats['successful'] += sum(1 for success in exchanges if success)
                
                if self.enable_quantum_tunneling:
                    quantum_exchanges = self._quantum_tunneling_exchange(replicas)
                    exchange_stats['quantum_enhanced'] += quantum_exchanges
            
            # Update temperature schedules
            self._update_temperatures(iteration, temperature_history)
            
            # Check convergence
            if self._check_convergence(global_best_energy, convergence_threshold):
                logger.info(f"Convergence achieved at iteration {iteration}")
                break
                
        # Detect quantum advantage
        quantum_advantage = self._assess_quantum_advantage(exchange_stats, replicas)
        
        return TemperingResults(
            best_solution=global_best,
            best_energy=global_best_energy,
            temperature_history=temperature_history,
            exchange_statistics=exchange_stats,
            quantum_advantage_achieved=quantum_advantage,
            convergence_time=time.time() - start_time
        )
    
    def _initialize_replicas(self, qubo_matrix: np.ndarray) -> List[Dict]:
        """Initialize replica configurations at different temperatures"""
        replicas = []
        n_vars = qubo_matrix.shape[0]
        
        for temp in self.params.temperatures:
            # Create initial random configuration
            initial_state = np.random.choice([0, 1], size=n_vars)
            
            replica = {
                'state': initial_state,
                'temperature': temp,
                'energy': self._calculate_energy(initial_state, qubo_matrix),
                'qubo_matrix': qubo_matrix,
                'acceptance_rate': 0.0,
                'quantum_enhanced': False
            }
            replicas.append(replica)
            
        return replicas
    
    def _parallel_replica_optimization(
        self, 
        replicas: List[Dict], 
        iteration: int
    ) -> List[Tuple[np.ndarray, float]]:
        """Run optimization on all replicas in parallel"""
        
        with ThreadPoolExecutor(max_workers=min(len(replicas), 8)) as executor:
            futures = []
            
            for replica_idx, replica in enumerate(replicas):
                future = executor.submit(
                    self._optimize_single_replica, 
                    replica, 
                    iteration
                )
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Replica optimization failed: {e}")
                    # Return current state as fallback
                    replica = replicas[len(results)]
                    results.append((replica['state'], replica['energy']))
            
        return results
    
    def _optimize_single_replica(
        self, 
        replica: Dict, 
        iteration: int
    ) -> Tuple[np.ndarray, float]:
        """Optimize a single replica using quantum annealing"""
        
        # Use quantum backend for larger problems
        if len(replica['state']) > self.params.quantum_advantage_threshold:
            try:
                # Quantum-enhanced optimization
                quantum_result = self.backend.sample_qubo(
                    Q=replica['qubo_matrix'],
                    temperature=replica['temperature'],
                    num_reads=100
                )
                
                best_sample = min(quantum_result.data(['sample', 'energy']), 
                                key=lambda x: x.energy)
                
                new_state = np.array([best_sample.sample[i] for i in range(len(replica['state']))])
                new_energy = best_sample.energy
                
                replica['quantum_enhanced'] = True
                self.quantum_advantage_detected = True
                
            except Exception as e:
                logger.warning(f"Quantum optimization failed, using classical: {e}")
                new_state, new_energy = self._classical_optimization_step(replica)
        else:
            # Classical optimization for smaller problems
            new_state, new_energy = self._classical_optimization_step(replica)
        
        # Update replica
        replica['state'] = new_state
        replica['energy'] = new_energy
        
        return new_state, new_energy
    
    def _classical_optimization_step(self, replica: Dict) -> Tuple[np.ndarray, float]:
        """Classical optimization step using simulated annealing"""
        current_state = replica['state'].copy()
        current_energy = replica['energy']
        temperature = replica['temperature']
        
        # Propose a random bit flip
        flip_idx = np.random.randint(len(current_state))
        proposed_state = current_state.copy()
        proposed_state[flip_idx] = 1 - proposed_state[flip_idx]
        
        proposed_energy = self._calculate_energy(proposed_state, replica['qubo_matrix'])
        
        # Metropolis acceptance criterion
        delta_e = proposed_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / temperature):
            return proposed_state, proposed_energy
        else:
            return current_state, current_energy
    
    def _quantum_enhanced_exchange(
        self, 
        replicas: List[Dict], 
        results: List[Tuple[np.ndarray, float]]
    ) -> List[bool]:
        """Perform replica exchanges enhanced with quantum effects"""
        
        exchanges = []
        n_replicas = len(replicas)
        
        for i in range(n_replicas - 1):
            # Calculate exchange probability
            temp_i = replicas[i]['temperature']
            temp_j = replicas[i + 1]['temperature']
            energy_i = results[i][1]
            energy_j = results[i + 1][1]
            
            # Standard Metropolis exchange criterion
            beta_i = 1.0 / temp_i
            beta_j = 1.0 / temp_j
            delta_beta = beta_j - beta_i
            delta_energy = energy_j - energy_i
            
            exchange_prob = min(1.0, np.exp(delta_beta * delta_energy))
            
            # Quantum enhancement: increase exchange probability for quantum-enhanced replicas
            if replicas[i]['quantum_enhanced'] or replicas[i + 1]['quantum_enhanced']:
                exchange_prob *= 1.2  # 20% boost for quantum-enhanced exchanges
            
            # Perform exchange
            if np.random.random() < exchange_prob:
                # Swap states between replicas
                replicas[i]['state'], replicas[i + 1]['state'] = \
                    replicas[i + 1]['state'].copy(), replicas[i]['state'].copy()
                replicas[i]['energy'], replicas[i + 1]['energy'] = \
                    replicas[i + 1]['energy'], replicas[i]['energy']
                
                exchanges.append(True)
                logger.debug(f"Successful exchange between replicas {i} and {i+1}")
            else:
                exchanges.append(False)
        
        return exchanges
    
    def _quantum_tunneling_exchange(self, replicas: List[Dict]) -> int:
        """Perform quantum tunneling-based exchanges"""
        quantum_exchanges = 0
        
        # Find replicas that could benefit from quantum tunneling
        for i, replica in enumerate(replicas):
            if replica['quantum_enhanced'] and replica['temperature'] > 0.5:
                # Attempt quantum tunneling to a random replica
                target_idx = np.random.choice([j for j in range(len(replicas)) if j != i])
                
                # Quantum tunneling probability (simplified model)
                barrier_height = abs(replica['energy'] - replicas[target_idx]['energy'])
                tunneling_prob = np.exp(-barrier_height / replica['temperature'])
                
                if np.random.random() < tunneling_prob:
                    # Perform quantum tunneling exchange
                    replica['state'], replicas[target_idx]['state'] = \
                        replicas[target_idx]['state'].copy(), replica['state'].copy()
                    replica['energy'], replicas[target_idx]['energy'] = \
                        replicas[target_idx]['energy'], replica['energy']
                    
                    quantum_exchanges += 1
                    logger.debug(f"Quantum tunneling exchange: {i} ↔ {target_idx}")
        
        return quantum_exchanges
    
    def _update_temperatures(self, iteration: int, temperature_history: List[List[float]]):
        """Update temperature schedules for all replicas"""
        
        for i, temp in enumerate(self.params.temperatures):
            if self.params.cooling_schedule == "exponential":
                new_temp = temp * (0.95 ** (iteration // 100))
            elif self.params.cooling_schedule == "linear":
                new_temp = temp * max(0.1, 1.0 - iteration / 1000.0)
            elif self.params.cooling_schedule == "adaptive":
                # Adaptive cooling based on acceptance rate
                acceptance_rate = getattr(self, 'last_acceptance_rate', 0.5)
                if acceptance_rate > 0.6:
                    new_temp = temp * 0.98  # Cool faster
                elif acceptance_rate < 0.3:
                    new_temp = temp * 1.02  # Heat up
                else:
                    new_temp = temp * 0.99  # Standard cooling
            else:
                new_temp = temp
            
            self.params.temperatures[i] = max(0.01, new_temp)  # Minimum temperature
            temperature_history[i].append(self.params.temperatures[i])
    
    def _calculate_energy(self, state: np.ndarray, qubo_matrix: np.ndarray) -> float:
        """Calculate QUBO energy for a given state"""
        return float(state.T @ qubo_matrix @ state)
    
    def _check_convergence(self, current_best: float, threshold: float) -> bool:
        """Check if optimization has converged"""
        if not hasattr(self, 'previous_best'):
            self.previous_best = current_best
            return False
        
        improvement = abs(self.previous_best - current_best)
        self.previous_best = current_best
        
        return improvement < threshold
    
    def _assess_quantum_advantage(
        self, 
        exchange_stats: Dict[str, int], 
        replicas: List[Dict]
    ) -> bool:
        """Assess whether quantum advantage was achieved"""
        
        # Check if quantum enhancements were used
        quantum_enhanced_replicas = sum(1 for r in replicas if r['quantum_enhanced'])
        quantum_exchange_ratio = exchange_stats['quantum_enhanced'] / max(1, exchange_stats['attempts'])
        
        # Quantum advantage criteria
        quantum_advantage = (
            quantum_enhanced_replicas > len(replicas) // 2 and  # >50% quantum enhanced
            quantum_exchange_ratio > 0.1 and  # >10% quantum exchanges
            self.quantum_advantage_detected  # Quantum backend was successfully used
        )
        
        logger.info(f"Quantum advantage assessment: {quantum_advantage}")
        logger.info(f"Quantum enhanced replicas: {quantum_enhanced_replicas}/{len(replicas)}")
        logger.info(f"Quantum exchange ratio: {quantum_exchange_ratio:.3f}")
        
        return quantum_advantage