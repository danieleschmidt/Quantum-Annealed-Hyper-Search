"""
Advanced optimization strategies combining quantum and classical approaches.
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..core.optimization_history import OptimizationHistory


@dataclass
class StrategyConfig:
    """Configuration for optimization strategies."""
    exploration_ratio: float = 0.3
    exploitation_ratio: float = 0.7
    adaptation_rate: float = 0.1
    convergence_threshold: float = 1e-4
    max_stagnation_iterations: int = 5


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize strategy with configuration."""
        self.config = config or StrategyConfig()
        self.iteration_count = 0
        
    @abstractmethod
    def select_configurations(
        self,
        samples: List[Dict[int, int]],
        variable_map: Dict[str, int],
        param_space: Dict[str, List[Any]],
        history: OptimizationHistory,
        n_select: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Select parameter configurations from quantum samples.
        
        Args:
            samples: Quantum annealing samples
            variable_map: Variable name to index mapping
            param_space: Parameter search space
            history: Optimization history
            n_select: Number of configurations to select
            
        Returns:
            List of selected parameter configurations
        """
        pass
    
    @abstractmethod
    def update_strategy(
        self,
        current_results: List[Tuple[Dict[str, Any], float]],
        history: OptimizationHistory
    ) -> None:
        """
        Update strategy based on current results.
        
        Args:
            current_results: List of (parameters, score) tuples
            history: Optimization history
        """
        pass


class AdaptiveStrategy(OptimizationStrategy):
    """
    Adaptive strategy that balances exploration and exploitation dynamically.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize adaptive strategy."""
        super().__init__(config)
        self.stagnation_count = 0
        self.last_best_score = float('-inf')
        self.exploration_boost = 0.0
        
    def select_configurations(
        self,
        samples: List[Dict[int, int]],
        variable_map: Dict[str, int],
        param_space: Dict[str, List[Any]],
        history: OptimizationHistory,
        n_select: int = 10
    ) -> List[Dict[str, Any]]:
        """Select configurations using adaptive exploration/exploitation."""
        from ..core.qubo_formulation import QUBOEncoder
        
        # Initialize encoder for decoding
        encoder = QUBOEncoder()
        
        # Decode all samples
        decoded_configs = []
        for sample in samples:
            try:
                config = encoder.decode_sample(sample, variable_map, param_space)
                decoded_configs.append(config)
            except Exception:
                continue
        
        if not decoded_configs:
            return []
        
        # Calculate adaptive selection ratios
        current_exploration_ratio = min(
            self.config.exploration_ratio + self.exploration_boost,
            0.8
        )
        current_exploitation_ratio = 1.0 - current_exploration_ratio
        
        n_explore = int(n_select * current_exploration_ratio)
        n_exploit = n_select - n_explore
        
        selected_configs = []
        
        # Exploitation: Select configurations similar to best performers
        if n_exploit > 0 and history.n_evaluations > 0:
            exploit_configs = self._select_exploitation_configs(
                decoded_configs, history, n_exploit
            )
            selected_configs.extend(exploit_configs)
        
        # Exploration: Select diverse configurations
        if n_explore > 0:
            explore_configs = self._select_exploration_configs(
                decoded_configs, history, n_explore
            )
            selected_configs.extend(explore_configs)
        
        # Fill remaining slots with random selection
        remaining = n_select - len(selected_configs)
        if remaining > 0:
            available_configs = [
                config for config in decoded_configs
                if config not in selected_configs
            ]
            if available_configs:
                additional = np.random.choice(
                    len(available_configs),
                    size=min(remaining, len(available_configs)),
                    replace=False
                )
                selected_configs.extend([available_configs[i] for i in additional])
        
        return selected_configs[:n_select]
    
    def _select_exploitation_configs(
        self,
        decoded_configs: List[Dict[str, Any]],
        history: OptimizationHistory,
        n_select: int
    ) -> List[Dict[str, Any]]:
        """Select configurations for exploitation."""
        if history.n_evaluations == 0:
            return []
        
        # Get top performing configurations
        top_configs = [
            eval_record.parameters 
            for eval_record in history.get_top_configurations(10)
        ]
        
        if not top_configs:
            return []
        
        # Find configurations similar to top performers
        exploitation_configs = []
        for config in decoded_configs:
            similarity_scores = [
                self._calculate_similarity(config, top_config)
                for top_config in top_configs
            ]
            max_similarity = max(similarity_scores)
            
            if max_similarity > 0.5:  # Similarity threshold
                exploitation_configs.append((config, max_similarity))
        
        # Sort by similarity and select top ones
        exploitation_configs.sort(key=lambda x: x[1], reverse=True)
        return [config for config, _ in exploitation_configs[:n_select]]
    
    def _select_exploration_configs(
        self,
        decoded_configs: List[Dict[str, Any]],
        history: OptimizationHistory,
        n_select: int
    ) -> List[Dict[str, Any]]:
        """Select configurations for exploration."""
        # Get previously evaluated configurations
        evaluated_configs = [
            eval_record.parameters 
            for eval_record in history.evaluations
        ]
        
        # Calculate diversity scores
        exploration_configs = []
        for config in decoded_configs:
            # Calculate average distance from evaluated configurations
            if evaluated_configs:
                distances = [
                    1.0 - self._calculate_similarity(config, eval_config)
                    for eval_config in evaluated_configs
                ]
                avg_distance = np.mean(distances)
            else:
                avg_distance = 1.0  # Maximum exploration if no history
            
            exploration_configs.append((config, avg_distance))
        
        # Sort by diversity and select top ones
        exploration_configs.sort(key=lambda x: x[1], reverse=True)
        return [config for config, _ in exploration_configs[:n_select]]
    
    def _calculate_similarity(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two configurations."""
        if not config1 or not config2:
            return 0.0
        
        common_params = set(config1.keys()) & set(config2.keys())
        if not common_params:
            return 0.0
        
        matches = sum(
            1 for param in common_params
            if config1[param] == config2[param]
        )
        
        return matches / len(common_params)
    
    def update_strategy(
        self,
        current_results: List[Tuple[Dict[str, Any], float]],
        history: OptimizationHistory
    ) -> None:
        """Update adaptive strategy based on results."""
        if not current_results:
            return
        
        # Check for improvement
        current_best = max(score for _, score in current_results)
        
        if current_best > self.last_best_score + self.config.convergence_threshold:
            # Improvement found - reduce exploration boost
            self.stagnation_count = 0
            self.exploration_boost = max(0.0, self.exploration_boost - self.config.adaptation_rate)
            self.last_best_score = current_best
        else:
            # No significant improvement - increase exploration
            self.stagnation_count += 1
            if self.stagnation_count >= self.config.max_stagnation_iterations:
                self.exploration_boost = min(0.5, self.exploration_boost + self.config.adaptation_rate)
        
        self.iteration_count += 1


class HybridQuantumClassical(OptimizationStrategy):
    """
    Hybrid strategy combining quantum annealing with classical optimization.
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        classical_ratio: float = 0.3
    ):
        """
        Initialize hybrid strategy.
        
        Args:
            config: Strategy configuration
            classical_ratio: Ratio of classical vs quantum configurations
        """
        super().__init__(config)
        self.classical_ratio = classical_ratio
        self.classical_optimizer = None  # Could integrate scipy.optimize
        
    def select_configurations(
        self,
        samples: List[Dict[int, int]],
        variable_map: Dict[str, int],
        param_space: Dict[str, List[Any]],
        history: OptimizationHistory,
        n_select: int = 10
    ) -> List[Dict[str, Any]]:
        """Select configurations using hybrid approach."""
        from ..core.qubo_formulation import QUBOEncoder
        
        encoder = QUBOEncoder()
        
        # Quantum configurations
        n_quantum = int(n_select * (1.0 - self.classical_ratio))
        quantum_configs = []
        
        for sample in samples[:n_quantum * 2]:  # Try more samples
            try:
                config = encoder.decode_sample(sample, variable_map, param_space)
                if config not in quantum_configs:
                    quantum_configs.append(config)
                if len(quantum_configs) >= n_quantum:
                    break
            except Exception:
                continue
        
        # Classical configurations
        n_classical = n_select - len(quantum_configs)
        classical_configs = self._generate_classical_configs(
            param_space, history, n_classical
        )
        
        return quantum_configs + classical_configs
    
    def _generate_classical_configs(
        self,
        param_space: Dict[str, List[Any]],
        history: OptimizationHistory,
        n_configs: int
    ) -> List[Dict[str, Any]]:
        """Generate configurations using classical methods."""
        classical_configs = []
        
        if history.n_evaluations > 0:
            # Use gradient-based suggestions around best configurations
            top_configs = history.get_top_configurations(3)
            
            for top_config in top_configs:
                for _ in range(n_configs // len(top_configs) + 1):
                    # Generate neighborhood configuration
                    neighbor = self._generate_neighbor_config(
                        top_config.parameters, param_space
                    )
                    if neighbor and neighbor not in classical_configs:
                        classical_configs.append(neighbor)
                    
                    if len(classical_configs) >= n_configs:
                        break
                
                if len(classical_configs) >= n_configs:
                    break
        
        # Fill remaining with random configurations
        while len(classical_configs) < n_configs:
            random_config = self._generate_random_config(param_space)
            if random_config not in classical_configs:
                classical_configs.append(random_config)
        
        return classical_configs[:n_configs]
    
    def _generate_neighbor_config(
        self,
        base_config: Dict[str, Any],
        param_space: Dict[str, List[Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate neighbor configuration by modifying one parameter."""
        if not base_config:
            return None
        
        # Choose random parameter to modify
        param_names = list(base_config.keys())
        if not param_names:
            return None
        
        param_to_modify = np.random.choice(param_names)
        
        if param_to_modify not in param_space:
            return None
        
        # Create neighbor by changing one parameter value
        neighbor = base_config.copy()
        neighbor[param_to_modify] = np.random.choice(param_space[param_to_modify])
        
        return neighbor
    
    def _generate_random_config(self, param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Generate random configuration."""
        config = {}
        for param_name, param_values in param_space.items():
            config[param_name] = np.random.choice(param_values)
        return config
    
    def update_strategy(
        self,
        current_results: List[Tuple[Dict[str, Any], float]],
        history: OptimizationHistory
    ) -> None:
        """Update hybrid strategy based on results."""
        if not current_results:
            return
        
        # Analyze performance of quantum vs classical configurations
        # This could be used to dynamically adjust the classical_ratio
        
        # For now, keep the ratio fixed
        self.iteration_count += 1


class BayesianQuantumStrategy(OptimizationStrategy):
    """
    Bayesian optimization strategy enhanced with quantum sampling.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize Bayesian quantum strategy."""
        super().__init__(config)
        # Could integrate with scikit-optimize or similar libraries
        
    def select_configurations(
        self,
        samples: List[Dict[int, int]],
        variable_map: Dict[str, int],
        param_space: Dict[str, List[Any]],
        history: OptimizationHistory,
        n_select: int = 10
    ) -> List[Dict[str, Any]]:
        """Select configurations using Bayesian optimization principles."""
        # Placeholder implementation - would integrate with Gaussian processes
        from ..core.qubo_formulation import QUBOEncoder
        
        encoder = QUBOEncoder()
        configs = []
        
        for sample in samples[:n_select * 2]:
            try:
                config = encoder.decode_sample(sample, variable_map, param_space)
                if config not in configs:
                    configs.append(config)
                if len(configs) >= n_select:
                    break
            except Exception:
                continue
        
        return configs[:n_select]
    
    def update_strategy(
        self,
        current_results: List[Tuple[Dict[str, Any], float]],
        history: OptimizationHistory
    ) -> None:
        """Update Bayesian model with new observations."""
        # Placeholder - would update Gaussian process model
        self.iteration_count += 1