"""
Quantum-specific metrics and performance monitoring.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


class QuantumMetrics:
    """
    Collect and analyze quantum-specific optimization metrics.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.quantum_samples = []
        self.classical_samples = []
        self.energies = []
        self.chain_breaks = []
        self.annealing_times = []
        self.embedding_overheads = []
        self.quantum_advantages = []
        self.start_time = None
        self.end_time = None
    
    def start_optimization(self) -> None:
        """Mark the start of optimization."""
        self.start_time = time.time()
    
    def end_optimization(self) -> None:
        """Mark the end of optimization."""
        self.end_time = time.time()
    
    def add_quantum_sample(
        self,
        sample: Dict[int, int],
        energy: float,
        chain_break_fraction: float = 0.0,
        annealing_time: Optional[float] = None,
        embedding_overhead: Optional[float] = None
    ) -> None:
        """
        Record a quantum annealing sample.
        
        Args:
            sample: Binary variable assignments
            energy: QUBO energy of the sample
            chain_break_fraction: Fraction of broken chains
            annealing_time: Time spent on quantum annealing (microseconds)
            embedding_overhead: Embedding computation time overhead
        """
        self.quantum_samples.append(sample.copy())
        self.energies.append(energy)
        self.chain_breaks.append(chain_break_fraction)
        
        if annealing_time is not None:
            self.annealing_times.append(annealing_time)
        
        if embedding_overhead is not None:
            self.embedding_overheads.append(embedding_overhead)
    
    def add_classical_sample(self, sample: Dict[int, int], energy: float) -> None:
        """
        Record a classical sample for comparison.
        
        Args:
            sample: Binary variable assignments
            energy: QUBO energy of the sample
        """
        self.classical_samples.append(sample.copy())
    
    def calculate_quantum_advantage(
        self,
        quantum_best_energy: float,
        classical_best_energy: float
    ) -> float:
        """
        Calculate quantum advantage metric.
        
        Args:
            quantum_best_energy: Best energy found by quantum method
            classical_best_energy: Best energy found by classical method
            
        Returns:
            Quantum advantage score (positive means quantum is better)
        """
        if classical_best_energy == 0:
            advantage = float('inf') if quantum_best_energy < 0 else 0
        else:
            advantage = (classical_best_energy - quantum_best_energy) / abs(classical_best_energy)
        
        self.quantum_advantages.append(advantage)
        return advantage
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of quantum performance.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            'total_quantum_samples': len(self.quantum_samples),
            'total_classical_samples': len(self.classical_samples),
            'optimization_time': self.get_optimization_time(),
        }
        
        if self.energies:
            stats.update({
                'best_energy': min(self.energies),
                'worst_energy': max(self.energies),
                'mean_energy': np.mean(self.energies),
                'energy_std': np.std(self.energies),
                'energy_improvement_ratio': self._calculate_energy_improvement()
            })
        
        if self.chain_breaks:
            stats.update({
                'mean_chain_break_fraction': np.mean(self.chain_breaks),
                'max_chain_break_fraction': max(self.chain_breaks),
                'samples_with_breaks': sum(1 for cb in self.chain_breaks if cb > 0)
            })
        
        if self.annealing_times:
            stats.update({
                'mean_annealing_time': np.mean(self.annealing_times),
                'total_annealing_time': sum(self.annealing_times)
            })
        
        if self.embedding_overheads:
            stats.update({
                'mean_embedding_overhead': np.mean(self.embedding_overheads),
                'total_embedding_overhead': sum(self.embedding_overheads)
            })
        
        if self.quantum_advantages:
            stats.update({
                'mean_quantum_advantage': np.mean(self.quantum_advantages),
                'positive_advantage_rate': sum(1 for qa in self.quantum_advantages if qa > 0) / len(self.quantum_advantages)
            })
        
        return stats
    
    def get_optimization_time(self) -> Optional[float]:
        """
        Get total optimization time in seconds.
        
        Returns:
            Optimization time or None if not available
        """
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def _calculate_energy_improvement(self) -> float:
        """Calculate energy improvement ratio over the optimization."""
        if len(self.energies) < 2:
            return 0.0
        
        first_energy = self.energies[0]
        best_energy = min(self.energies)
        
        if first_energy == 0:
            return float('inf') if best_energy < 0 else 0
        
        return (first_energy - best_energy) / abs(first_energy)
    
    def analyze_convergence(self, window_size: int = 5) -> Dict[str, Any]:
        """
        Analyze convergence properties of the optimization.
        
        Args:
            window_size: Size of moving window for convergence analysis
            
        Returns:
            Convergence analysis results
        """
        if len(self.energies) < window_size:
            return {'convergence_detected': False, 'reason': 'insufficient_data'}
        
        # Calculate best energy so far at each iteration
        best_so_far = []
        current_best = float('inf')
        
        for energy in self.energies:
            if energy < current_best:
                current_best = energy
            best_so_far.append(current_best)
        
        # Check for convergence (no improvement in last window_size iterations)
        last_improvements = np.diff(best_so_far[-window_size:])
        converged = np.all(last_improvements == 0)
        
        # Calculate convergence rate
        if len(best_so_far) > 1:
            total_improvement = best_so_far[0] - best_so_far[-1]
            convergence_rate = total_improvement / len(best_so_far)
        else:
            convergence_rate = 0.0
        
        # Find iteration where best solution was found
        best_iteration = best_so_far.index(min(best_so_far)) + 1
        
        return {
            'convergence_detected': converged,
            'convergence_rate': convergence_rate,
            'best_iteration': best_iteration,
            'iterations_since_improvement': len(best_so_far) - best_iteration,
            'improvement_ratio': self._calculate_energy_improvement(),
            'best_energy_trajectory': best_so_far
        }
    
    def get_hardware_efficiency_metrics(self) -> Dict[str, Any]:
        """
        Calculate hardware efficiency metrics.
        
        Returns:
            Hardware efficiency analysis
        """
        metrics = {}
        
        if self.chain_breaks:
            # Chain break analysis
            clean_samples = sum(1 for cb in self.chain_breaks if cb == 0)
            metrics['clean_sample_rate'] = clean_samples / len(self.chain_breaks)
            metrics['mean_chain_break_rate'] = np.mean(self.chain_breaks)
            
            # Classify chain break severity
            low_breaks = sum(1 for cb in self.chain_breaks if 0 < cb <= 0.1)
            medium_breaks = sum(1 for cb in self.chain_breaks if 0.1 < cb <= 0.3)
            high_breaks = sum(1 for cb in self.chain_breaks if cb > 0.3)
            
            metrics['chain_break_distribution'] = {
                'clean': clean_samples,
                'low_breaks': low_breaks,
                'medium_breaks': medium_breaks,
                'high_breaks': high_breaks
            }
        
        if self.annealing_times and self.embedding_overheads:
            # Time efficiency
            total_quantum_time = sum(self.annealing_times)
            total_overhead = sum(self.embedding_overheads)
            
            metrics['quantum_time_efficiency'] = total_quantum_time / (total_quantum_time + total_overhead)
            metrics['average_overhead_ratio'] = np.mean(self.embedding_overheads) / np.mean(self.annealing_times)
        
        return metrics
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export all collected data for external analysis.
        
        Returns:
            Dictionary containing all collected metrics data
        """
        return {
            'quantum_samples': self.quantum_samples,
            'classical_samples': self.classical_samples,
            'energies': self.energies,
            'chain_breaks': self.chain_breaks,
            'annealing_times': self.annealing_times,
            'embedding_overheads': self.embedding_overheads,
            'quantum_advantages': self.quantum_advantages,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'summary_statistics': self.get_summary_statistics(),
            'convergence_analysis': self.analyze_convergence(),
            'hardware_efficiency': self.get_hardware_efficiency_metrics()
        }
    
    def log_performance_summary(self) -> None:
        """Log a comprehensive performance summary."""
        stats = self.get_summary_statistics()
        convergence = self.analyze_convergence()
        efficiency = self.get_hardware_efficiency_metrics()
        
        logger.info("="*50)
        logger.info("QUANTUM PERFORMANCE SUMMARY")
        logger.info("="*50)
        
        # Basic statistics
        logger.info(f"Total quantum samples: {stats.get('total_quantum_samples', 0)}")
        logger.info(f"Optimization time: {stats.get('optimization_time', 0):.2f} seconds")
        
        if 'best_energy' in stats:
            logger.info(f"Best energy: {stats['best_energy']:.6f}")
            logger.info(f"Energy improvement: {stats['energy_improvement_ratio']:.4f}")
        
        # Convergence analysis
        if convergence['convergence_detected']:
            logger.info(f"Convergence detected at iteration {convergence['best_iteration']}")
        else:
            logger.info("No convergence detected")
        
        logger.info(f"Iterations since last improvement: {convergence.get('iterations_since_improvement', 'Unknown')}")
        
        # Hardware efficiency
        if 'clean_sample_rate' in efficiency:
            logger.info(f"Clean samples (no chain breaks): {efficiency['clean_sample_rate']:.1%}")
            logger.info(f"Average chain break rate: {efficiency['mean_chain_break_rate']:.3f}")
        
        if 'quantum_time_efficiency' in efficiency:
            logger.info(f"Quantum time efficiency: {efficiency['quantum_time_efficiency']:.1%}")
        
        logger.info("="*50)