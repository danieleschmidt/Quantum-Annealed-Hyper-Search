"""
Adaptive QUBO Strategies - Advanced quantum annealing strategies that learn and adapt.

This module implements next-generation QUBO formulation and annealing strategies
that adapt based on problem characteristics and historical performance.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.cluster import KMeans
from scipy.optimize import differential_evolution
import logging

logger = logging.getLogger(__name__)


@dataclass
class QUBOPerformanceMetrics:
    """Performance metrics for QUBO formulations."""
    energy_gap: float
    chain_break_fraction: float
    solution_quality: float
    embedding_overhead: float
    annealing_time: float
    success_rate: float
    
    def overall_score(self) -> float:
        """Calculate overall performance score."""
        # Weighted combination of metrics (lower is better for some)
        return (
            0.3 * self.solution_quality +
            0.2 * (1.0 - self.chain_break_fraction) +
            0.2 * self.energy_gap +
            0.15 * self.success_rate +
            0.1 * (1.0 / max(self.embedding_overhead, 0.1)) +
            0.05 * (1.0 / max(self.annealing_time, 0.1))
        )


class AdaptiveQUBOStrategy(ABC):
    """Abstract base class for adaptive QUBO strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.performance_history: List[QUBOPerformanceMetrics] = []
        self.parameter_history: List[Dict] = []
        
    @abstractmethod
    def adapt_parameters(self, current_metrics: QUBOPerformanceMetrics) -> Dict[str, float]:
        """Adapt strategy parameters based on performance."""
        pass
    
    @abstractmethod
    def suggest_qubo_modifications(self, Q: Dict, history: Any) -> Dict:
        """Suggest modifications to QUBO formulation."""
        pass
    
    def update_performance(self, metrics: QUBOPerformanceMetrics, parameters: Dict):
        """Update performance history."""
        self.performance_history.append(metrics)
        self.parameter_history.append(parameters.copy())
        
        # Keep only recent history to avoid memory bloat
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
            self.parameter_history = self.parameter_history[-50:]


class AdaptivePenaltyStrategy(AdaptiveQUBOStrategy):
    """
    Adaptive penalty strength strategy that learns optimal penalty values
    based on constraint violation rates and solution quality.
    """
    
    def __init__(self):
        super().__init__("AdaptivePenalty")
        self.base_penalty = 2.0
        self.penalty_multiplier = 1.0
        self.learning_rate = 0.1
        self.target_violation_rate = 0.05  # 5% violation rate target
        
    def adapt_parameters(self, current_metrics: QUBOPerformanceMetrics) -> Dict[str, float]:
        """Adapt penalty strength based on chain break fraction and solution quality."""
        
        # Use chain break fraction as proxy for constraint violations
        violation_rate = current_metrics.chain_break_fraction
        
        # Calculate adjustment based on violation rate
        if violation_rate > self.target_violation_rate:
            # Too many violations - increase penalty
            adjustment = self.learning_rate * (violation_rate - self.target_violation_rate)
            self.penalty_multiplier *= (1 + adjustment)
        elif violation_rate < self.target_violation_rate / 2:
            # Very few violations - can decrease penalty for better solutions
            adjustment = self.learning_rate * (self.target_violation_rate - violation_rate)
            self.penalty_multiplier *= max(0.5, 1 - adjustment)
        
        # Also consider solution quality trend
        if len(self.performance_history) >= 3:
            recent_quality = [m.solution_quality for m in self.performance_history[-3:]]
            if np.mean(np.diff(recent_quality)) < 0:  # Quality declining
                self.penalty_multiplier *= 0.95  # Reduce penalty slightly
        
        new_penalty = self.base_penalty * self.penalty_multiplier
        
        logger.info(f"Adapted penalty strength: {new_penalty:.3f} (multiplier: {self.penalty_multiplier:.3f})")
        
        return {
            'penalty_strength': new_penalty,
            'penalty_multiplier': self.penalty_multiplier
        }
    
    def suggest_qubo_modifications(self, Q: Dict, history: Any) -> Dict:
        """Suggest QUBO modifications based on penalty adaptation."""
        current_penalty = self.base_penalty * self.penalty_multiplier
        
        # Identify constraint terms and adjust their weights
        modified_Q = Q.copy()
        
        # Look for terms that are likely constraint penalties (high positive diagonal terms)
        penalty_threshold = np.mean(list(Q.values())) + 2 * np.std(list(Q.values()))
        
        for (i, j), value in Q.items():
            if i == j and value > penalty_threshold:  # Diagonal penalty term
                modified_Q[(i, j)] = value * (current_penalty / self.base_penalty)
        
        return modified_Q


class AdaptiveAnnealingScheduleStrategy(AdaptiveQUBOStrategy):
    """
    Adaptive annealing schedule strategy that adjusts annealing parameters
    based on problem characteristics and performance.
    """
    
    def __init__(self):
        super().__init__("AdaptiveAnnealing")
        self.base_annealing_time = 20  # microseconds
        self.base_num_reads = 1000
        self.schedule_adaptation_factor = 1.0
        self.reads_adaptation_factor = 1.0
        
    def adapt_parameters(self, current_metrics: QUBOPerformanceMetrics) -> Dict[str, float]:
        """Adapt annealing schedule based on energy gap and success rate."""
        
        energy_gap = current_metrics.energy_gap
        success_rate = current_metrics.success_rate
        
        # Adjust annealing time based on energy gap
        if energy_gap < 0.1:  # Small energy gap - need longer annealing
            self.schedule_adaptation_factor *= 1.1
        elif energy_gap > 0.5:  # Large energy gap - can use shorter annealing
            self.schedule_adaptation_factor *= 0.95
        
        # Adjust number of reads based on success rate
        if success_rate < 0.7:  # Low success rate - need more reads
            self.reads_adaptation_factor *= 1.05
        elif success_rate > 0.9:  # High success rate - can use fewer reads
            self.reads_adaptation_factor *= 0.98
        
        # Bounds checking
        self.schedule_adaptation_factor = np.clip(self.schedule_adaptation_factor, 0.5, 3.0)
        self.reads_adaptation_factor = np.clip(self.reads_adaptation_factor, 0.5, 2.0)
        
        new_annealing_time = self.base_annealing_time * self.schedule_adaptation_factor
        new_num_reads = int(self.base_num_reads * self.reads_adaptation_factor)
        
        logger.info(f"Adapted annealing: time={new_annealing_time:.1f}μs, reads={new_num_reads}")
        
        return {
            'annealing_time': new_annealing_time,
            'num_reads': new_num_reads,
            'schedule_factor': self.schedule_adaptation_factor,
            'reads_factor': self.reads_adaptation_factor
        }
    
    def suggest_qubo_modifications(self, Q: Dict, history: Any) -> Dict:
        """No direct QUBO modifications for annealing schedule adaptation."""
        return Q


class AdaptiveEmbeddingStrategy(AdaptiveQUBOStrategy):
    """
    Adaptive embedding strategy that learns optimal minor embedding approaches
    based on problem structure and hardware performance.
    """
    
    def __init__(self):
        super().__init__("AdaptiveEmbedding")
        self.embedding_methods = ['minorminer', 'clique', 'layoutaware']
        self.method_performance = {method: [] for method in self.embedding_methods}
        self.current_method = 'minorminer'
        self.method_switch_threshold = 5  # Switch after 5 poor performances
        
    def adapt_parameters(self, current_metrics: QUBOPerformanceMetrics) -> Dict[str, float]:
        """Adapt embedding method based on chain break fraction and overhead."""
        
        # Record performance for current method
        performance_score = (
            (1.0 - current_metrics.chain_break_fraction) * 0.6 +
            (1.0 / max(current_metrics.embedding_overhead, 0.1)) * 0.4
        )
        
        self.method_performance[self.current_method].append(performance_score)
        
        # Check if we should switch methods
        if len(self.method_performance[self.current_method]) >= self.method_switch_threshold:
            current_avg = np.mean(self.method_performance[self.current_method][-self.method_switch_threshold:])
            
            # Find best alternative method
            best_method = self.current_method
            best_score = current_avg
            
            for method in self.embedding_methods:
                if method != self.current_method and self.method_performance[method]:
                    method_score = np.mean(self.method_performance[method][-self.method_switch_threshold:])
                    if method_score > best_score + 0.05:  # Significant improvement threshold
                        best_method = method
                        best_score = method_score
            
            if best_method != self.current_method:
                logger.info(f"Switching embedding method from {self.current_method} to {best_method}")
                self.current_method = best_method
        
        return {
            'embedding_method': self.current_method,
            'embedding_performance': performance_score
        }
    
    def suggest_qubo_modifications(self, Q: Dict, history: Any) -> Dict:
        """Suggest QUBO modifications to improve embedding."""
        # Analyze QUBO connectivity and suggest sparsification if needed
        total_terms = len(Q)
        diagonal_terms = sum(1 for (i, j) in Q.keys() if i == j)
        off_diagonal_terms = total_terms - diagonal_terms
        
        # If QUBO is too dense, suggest sparsification
        if off_diagonal_terms > 100:  # Arbitrary threshold
            # Remove weakest off-diagonal terms
            off_diagonal_values = [abs(value) for (i, j), value in Q.items() if i != j]
            if off_diagonal_values:
                threshold = np.percentile(off_diagonal_values, 10)  # Remove bottom 10%
                
                modified_Q = {}
                for (i, j), value in Q.items():
                    if i == j or abs(value) >= threshold:
                        modified_Q[(i, j)] = value
                
                logger.info(f"Sparsified QUBO: {len(Q)} -> {len(modified_Q)} terms")
                return modified_Q
        
        return Q


class MetaAdaptiveQUBOOptimizer:
    """
    Meta-optimizer that coordinates multiple adaptive strategies and
    learns when to apply each strategy based on problem characteristics.
    """
    
    def __init__(self):
        self.strategies = {
            'penalty': AdaptivePenaltyStrategy(),
            'annealing': AdaptiveAnnealingScheduleStrategy(),
            'embedding': AdaptiveEmbeddingStrategy()
        }
        
        self.strategy_performance = {name: [] for name in self.strategies.keys()}
        self.problem_characteristics = []
        self.coordination_history = []
        
    def analyze_problem_characteristics(self, Q: Dict, param_space: Dict) -> Dict[str, float]:
        """
        Analyze problem characteristics to guide strategy selection.
        """
        # QUBO characteristics
        qubo_size = len(set([i for (i, j) in Q.keys()] + [j for (i, j) in Q.keys()]))
        qubo_density = len(Q) / (qubo_size ** 2) if qubo_size > 0 else 0
        
        diagonal_values = [value for (i, j), value in Q.items() if i == j]
        off_diagonal_values = [value for (i, j), value in Q.items() if i != j]
        
        penalty_strength_estimate = np.mean(diagonal_values) if diagonal_values else 0
        coupling_strength_estimate = np.mean(np.abs(off_diagonal_values)) if off_diagonal_values else 0
        
        # Parameter space characteristics
        total_combinations = np.prod([len(values) for values in param_space.values()])
        avg_param_values = np.mean([len(values) for values in param_space.values()])
        
        characteristics = {
            'qubo_size': qubo_size,
            'qubo_density': qubo_density,
            'penalty_strength': penalty_strength_estimate,
            'coupling_strength': coupling_strength_estimate,
            'search_space_size': total_combinations,
            'avg_param_cardinality': avg_param_values
        }
        
        self.problem_characteristics.append(characteristics)
        return characteristics
    
    def coordinate_strategies(
        self,
        Q: Dict,
        param_space: Dict,
        current_metrics: QUBOPerformanceMetrics
    ) -> Tuple[Dict, Dict[str, Any]]:
        """
        Coordinate all adaptive strategies and return optimized QUBO and parameters.
        """
        print("⚙️ Coordinating adaptive strategies...")
        
        # Analyze problem characteristics
        characteristics = self.analyze_problem_characteristics(Q, param_space)
        
        # Apply each strategy
        modified_Q = Q.copy()
        adapted_parameters = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                # Adapt parameters
                strategy_params = strategy.adapt_parameters(current_metrics)
                adapted_parameters[strategy_name] = strategy_params
                
                # Apply QUBO modifications
                modified_Q = strategy.suggest_qubo_modifications(modified_Q, None)
                
                # Update strategy performance
                strategy.update_performance(current_metrics, strategy_params)
                
                logger.info(f"Applied {strategy_name} strategy adaptations")
                
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue
        
        # Meta-coordination: adjust strategy influences based on problem characteristics
        coordination_weights = self._calculate_coordination_weights(characteristics)
        
        # Apply weighted coordination
        final_Q = self._apply_weighted_coordination(Q, modified_Q, coordination_weights)
        
        coordination_result = {
            'characteristics': characteristics,
            'coordination_weights': coordination_weights,
            'adapted_parameters': adapted_parameters,
            'qubo_modifications': len(final_Q) != len(Q)
        }
        
        self.coordination_history.append(coordination_result)
        
        return final_Q, adapted_parameters
    
    def _calculate_coordination_weights(self, characteristics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate weights for coordinating different strategies based on problem characteristics.
        """
        weights = {'penalty': 1.0, 'annealing': 1.0, 'embedding': 1.0}
        
        # Adjust weights based on problem characteristics
        if characteristics['qubo_density'] > 0.1:  # Dense QUBO
            weights['embedding'] *= 1.2  # Embedding more important
        
        if characteristics['penalty_strength'] > 10.0:  # High penalties
            weights['penalty'] *= 1.3  # Penalty adaptation more important
        
        if characteristics['search_space_size'] > 1000:  # Large search space
            weights['annealing'] *= 1.1  # Annealing adaptation more important
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _apply_weighted_coordination(self, original_Q: Dict, modified_Q: Dict, weights: Dict) -> Dict:
        """
        Apply weighted coordination between original and modified QUBO.
        """
        # For now, use the modified QUBO directly
        # In a more advanced version, could blend multiple QUBO modifications
        return modified_Q
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of adaptive strategy performance.
        """
        summary = {
            'total_adaptations': len(self.coordination_history),
            'strategy_performance': {}
        }
        
        for strategy_name, strategy in self.strategies.items():
            if strategy.performance_history:
                scores = [m.overall_score() for m in strategy.performance_history]
                summary['strategy_performance'][strategy_name] = {
                    'mean_score': np.mean(scores),
                    'improvement_trend': np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 1 else 0,
                    'adaptations': len(scores)
                }
        
        return summary
    
    def reset_adaptation_history(self):
        """
        Reset adaptation history (useful for new problem domains).
        """
        for strategy in self.strategies.values():
            strategy.performance_history.clear()
            strategy.parameter_history.clear()
        
        self.problem_characteristics.clear()
        self.coordination_history.clear()
        
        logger.info("Reset all adaptive strategy histories")
