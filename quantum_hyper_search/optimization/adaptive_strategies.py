"""
Adaptive quantum optimization strategies that learn and evolve.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging_config import get_logger

logger = get_logger('adaptive_strategies')


class AdaptiveQuantumSearch:
    """
    Adaptive quantum search that adjusts parameters based on performance.
    
    Automatically tunes quantum annealing parameters, penalty strengths,
    and search strategies based on observed performance patterns.
    """
    
    def __init__(
        self,
        initial_quantum_reads: int = 1000,
        initial_penalty_strength: float = 2.0,
        adaptation_rate: float = 0.1,
        performance_window: int = 5
    ):
        """
        Initialize adaptive quantum search.
        
        Args:
            initial_quantum_reads: Initial number of quantum reads
            initial_penalty_strength: Initial QUBO penalty strength
            adaptation_rate: Rate of parameter adaptation (0.0 to 1.0)
            performance_window: Window size for performance tracking
        """
        self.quantum_reads = initial_quantum_reads
        self.penalty_strength = initial_penalty_strength
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        
        # Performance tracking
        self.performance_history = []
        self.parameter_history = []
        self.violation_history = []
        
        # Adaptive ranges
        self.quantum_reads_range = (50, 5000)
        self.penalty_strength_range = (0.5, 10.0)
        
        logger.info(f"Initialized adaptive quantum search with {initial_quantum_reads} reads")
    
    def update_performance(
        self,
        score: float,
        constraint_violations: int,
        quantum_quality_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update performance tracking and adapt parameters.
        
        Args:
            score: Optimization score achieved
            constraint_violations: Number of constraint violations
            quantum_quality_metrics: Optional quantum-specific metrics
        """
        # Record performance
        self.performance_history.append(score)
        self.violation_history.append(constraint_violations)
        self.parameter_history.append({
            'quantum_reads': self.quantum_reads,
            'penalty_strength': self.penalty_strength
        })
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window * 2:
            self.performance_history = self.performance_history[-self.performance_window:]
            self.violation_history = self.violation_history[-self.performance_window:]
            self.parameter_history = self.parameter_history[-self.performance_window:]
        
        # Adapt parameters if we have enough history
        if len(self.performance_history) >= self.performance_window:
            self._adapt_quantum_reads()
            self._adapt_penalty_strength()
        
        logger.debug(f"Updated performance: score={score:.4f}, violations={constraint_violations}")
    
    def _adapt_quantum_reads(self) -> None:
        """Adapt quantum reads based on performance trends."""
        if len(self.performance_history) < self.performance_window:
            return
        
        recent_scores = self.performance_history[-self.performance_window:]
        recent_reads = [p['quantum_reads'] for p in self.parameter_history[-self.performance_window:]]
        
        # Calculate correlation between reads and performance
        if len(set(recent_reads)) > 1:  # Only if we have variation in reads
            correlation = np.corrcoef(recent_reads, recent_scores)[0, 1]
            
            if not np.isnan(correlation):
                # Increase reads if positive correlation, decrease if negative
                adjustment_factor = 1.0 + (self.adaptation_rate * correlation)
                new_reads = int(self.quantum_reads * adjustment_factor)
                
                # Clamp to valid range
                new_reads = max(self.quantum_reads_range[0], 
                               min(self.quantum_reads_range[1], new_reads))
                
                if new_reads != self.quantum_reads:
                    logger.info(f"Adapted quantum reads: {self.quantum_reads} -> {new_reads}")
                    self.quantum_reads = new_reads
    
    def _adapt_penalty_strength(self) -> None:
        """Adapt penalty strength based on constraint violations."""
        if len(self.violation_history) < self.performance_window:
            return
        
        recent_violations = self.violation_history[-self.performance_window:]
        avg_violations = np.mean(recent_violations)
        
        # Increase penalty if too many violations, decrease if too few
        if avg_violations > 0.1:  # More than 10% violation rate
            adjustment = 1.0 + self.adaptation_rate
        elif avg_violations < 0.05 and self.penalty_strength > self.penalty_strength_range[0]:
            adjustment = 1.0 - self.adaptation_rate
        else:
            adjustment = 1.0
        
        if adjustment != 1.0:
            new_strength = self.penalty_strength * adjustment
            new_strength = max(self.penalty_strength_range[0],
                             min(self.penalty_strength_range[1], new_strength))
            
            if abs(new_strength - self.penalty_strength) > 0.1:
                logger.info(f"Adapted penalty strength: {self.penalty_strength:.2f} -> {new_strength:.2f}")
                self.penalty_strength = new_strength
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current adaptive parameters.
        
        Returns:
            Dictionary with current parameter values
        """
        return {
            'quantum_reads': self.quantum_reads,
            'penalty_strength': self.penalty_strength,
            'adaptation_stats': {
                'performance_samples': len(self.performance_history),
                'avg_recent_score': np.mean(self.performance_history[-5:]) if len(self.performance_history) >= 5 else None,
                'avg_violations': np.mean(self.violation_history[-5:]) if len(self.violation_history) >= 5 else None
            }
        }
    
    def suggest_search_space_reduction(
        self,
        param_space: Dict[str, List[Any]],
        performance_history: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        """
        Suggest search space reduction based on performance patterns.
        
        Args:
            param_space: Current parameter search space
            performance_history: History of parameter evaluations
            
        Returns:
            Reduced search space focusing on promising regions
        """
        if len(performance_history) < 10:
            return param_space  # Not enough data
        
        # Extract top-performing parameter combinations
        sorted_history = sorted(performance_history, 
                              key=lambda x: x.get('score', 0), reverse=True)
        top_performers = sorted_history[:len(sorted_history)//4]  # Top 25%
        
        reduced_space = {}
        
        for param_name, param_values in param_space.items():
            # Find values that appear in top performers
            top_values = []
            for trial in top_performers:
                if param_name in trial.get('params', {}):
                    value = trial['params'][param_name]
                    if value in param_values and value not in top_values:
                        top_values.append(value)
            
            # If we found promising values, use them; otherwise keep original space
            if len(top_values) >= 2:
                reduced_space[param_name] = top_values
                logger.info(f"Reduced {param_name} space: {len(param_values)} -> {len(top_values)} values")
            else:
                reduced_space[param_name] = param_values
        
        return reduced_space
    
    def suggest_early_stopping(
        self,
        current_iteration: int,
        best_score: float,
        recent_scores: List[float]
    ) -> bool:
        """
        Suggest whether to stop optimization early.
        
        Args:
            current_iteration: Current optimization iteration
            best_score: Best score achieved so far
            recent_scores: Recent optimization scores
            
        Returns:
            True if early stopping is recommended
        """
        if current_iteration < 10:  # Always run at least 10 iterations
            return False
        
        if len(recent_scores) < 5:
            return False
        
        # Check if we've plateaued
        recent_best = max(recent_scores[-5:])
        improvement = recent_best - best_score
        
        # Suggest stopping if improvement is very small
        relative_improvement = improvement / abs(best_score) if best_score != 0 else improvement
        
        if abs(relative_improvement) < 0.001:  # Less than 0.1% improvement
            logger.info(f"Suggesting early stopping: minimal improvement ({relative_improvement:.4f})")
            return True
        
        return False
    
    def analyze_optimization_patterns(self) -> Dict[str, Any]:
        """
        Analyze optimization patterns and provide insights.
        
        Returns:
            Dictionary with optimization pattern analysis
        """
        analysis = {
            'convergence_detected': False,
            'parameter_sensitivity': {},
            'optimization_efficiency': None
        }
        
        if len(self.performance_history) < 5:
            return analysis
        
        # Convergence analysis
        recent_scores = self.performance_history[-5:]
        score_variance = np.var(recent_scores)
        analysis['convergence_detected'] = score_variance < 0.001
        
        # Parameter sensitivity analysis
        if len(self.parameter_history) >= 5:
            for param_name in ['quantum_reads', 'penalty_strength']:
                param_values = [p[param_name] for p in self.parameter_history[-5:]]
                if len(set(param_values)) > 1:  # Has variation
                    correlation = np.corrcoef(param_values, recent_scores)[0, 1]
                    if not np.isnan(correlation):
                        analysis['parameter_sensitivity'][param_name] = {
                            'correlation': correlation,
                            'importance': 'high' if abs(correlation) > 0.5 else 'medium' if abs(correlation) > 0.2 else 'low'
                        }
        
        # Optimization efficiency
        if len(self.performance_history) >= 3:
            improvement_rate = (self.performance_history[-1] - self.performance_history[0]) / len(self.performance_history)
            analysis['optimization_efficiency'] = {
                'improvement_per_iteration': improvement_rate,
                'total_improvement': self.performance_history[-1] - self.performance_history[0],
                'efficiency_rating': 'high' if improvement_rate > 0.01 else 'medium' if improvement_rate > 0.001 else 'low'
            }
        
        return analysis


class MultiObjectiveAdaptiveSearch(AdaptiveQuantumSearch):
    """
    Adaptive search for multi-objective optimization problems.
    """
    
    def __init__(self, objectives: List[str], **kwargs):
        """
        Initialize multi-objective adaptive search.
        
        Args:
            objectives: List of objective names
            **kwargs: Arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.objectives = objectives
        self.objective_weights = {obj: 1.0 for obj in objectives}
        self.pareto_front = []
    
    def update_multi_objective_performance(
        self,
        scores: Dict[str, float],
        constraint_violations: int
    ) -> None:
        """
        Update performance for multi-objective optimization.
        
        Args:
            scores: Dictionary of objective scores
            constraint_violations: Number of constraint violations
        """
        # Calculate weighted sum for primary adaptation
        weighted_score = sum(scores[obj] * self.objective_weights[obj] 
                           for obj in self.objectives)
        
        # Update using parent method
        self.update_performance(weighted_score, constraint_violations)
        
        # Update Pareto front
        self._update_pareto_front(scores)
        
        # Adapt objective weights if needed
        self._adapt_objective_weights(scores)
    
    def _update_pareto_front(self, scores: Dict[str, float]) -> None:
        """Update Pareto front with new solution."""
        # Simple Pareto front maintenance
        new_solution = scores.copy()
        
        # Check if new solution dominates any existing solutions
        dominated_indices = []
        is_dominated = False
        
        for i, existing in enumerate(self.pareto_front):
            if self._dominates(new_solution, existing):
                dominated_indices.append(i)
            elif self._dominates(existing, new_solution):
                is_dominated = True
                break
        
        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(dominated_indices):
                del self.pareto_front[i]
            
            # Add new solution
            self.pareto_front.append(new_solution)
            
            # Keep Pareto front size reasonable
            if len(self.pareto_front) > 50:
                # Remove some solutions (keep diverse set)
                self.pareto_front = self.pareto_front[-30:]
    
    def _dominates(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> bool:
        """Check if solution1 dominates solution2."""
        better_in_any = False
        
        for obj in self.objectives:
            if solution1[obj] < solution2[obj]:  # Assuming minimization
                return False
            elif solution1[obj] > solution2[obj]:
                better_in_any = True
        
        return better_in_any
    
    def _adapt_objective_weights(self, current_scores: Dict[str, float]) -> None:
        """Adapt objective weights based on current performance."""
        if len(self.pareto_front) < 5:
            return
        
        # Simple weight adaptation: increase weight for objectives where we're doing poorly
        for obj in self.objectives:
            current_value = current_scores[obj]
            pareto_values = [sol[obj] for sol in self.pareto_front]
            
            if pareto_values:
                percentile = np.percentile(pareto_values, 25)  # Bottom 25%
                
                if current_value < percentile:
                    # Increase weight for this objective
                    self.objective_weights[obj] *= (1.0 + self.adaptation_rate)
                else:
                    # Slightly decrease weight
                    self.objective_weights[obj] *= (1.0 - self.adaptation_rate * 0.5)
        
        # Normalize weights
        total_weight = sum(self.objective_weights.values())
        for obj in self.objectives:
            self.objective_weights[obj] /= total_weight