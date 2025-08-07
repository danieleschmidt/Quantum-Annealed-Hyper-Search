#!/usr/bin/env python3
"""
Adaptive Quantum Strategies for Hyperparameter Optimization

This module implements advanced adaptive strategies identified as research opportunities:
1. Learning-based annealing schedules
2. Dynamic topology selection
3. Feedback-driven parameter tuning
4. Reinforcement learning for quantum control
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from collections import deque
import time


@dataclass
class QuantumExperience:
    """Experience tuple for quantum optimization learning."""
    param_config: Dict[str, Any]
    qubo_properties: Dict[str, float]
    quantum_settings: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float


@dataclass
class AdaptationMetrics:
    """Metrics for evaluating adaptation performance."""
    improvement_rate: float  # Rate of improvement over iterations
    convergence_speed: float  # Iterations to convergence
    stability_score: float  # Consistency of performance
    exploration_efficiency: float  # Quality of parameter space exploration


class AdaptiveQuantumStrategy(ABC):
    """Abstract base class for adaptive quantum strategies."""
    
    @abstractmethod
    def suggest_quantum_parameters(self, problem_context: Dict[str, Any],
                                  history: List[QuantumExperience]) -> Dict[str, Any]:
        """Suggest quantum parameters based on problem context and history."""
        pass
    
    @abstractmethod
    def update_from_experience(self, experience: QuantumExperience):
        """Update strategy based on new experience."""
        pass
    
    @abstractmethod
    def get_adaptation_metrics(self) -> AdaptationMetrics:
        """Get metrics about adaptation performance."""
        pass


class LearningBasedAnnealingScheduler(AdaptiveQuantumStrategy):
    """
    Learning-based annealing scheduler that adapts the annealing schedule
    based on problem characteristics and past performance.
    """
    
    def __init__(self, learning_rate: float = 0.1, memory_size: int = 100,
                 exploration_rate: float = 0.15, schedule_types: List[str] = None):
        """
        Initialize learning-based annealing scheduler.
        
        Args:
            learning_rate: Rate of adaptation for schedule parameters
            memory_size: Number of experiences to remember
            exploration_rate: Probability of trying new schedules
            schedule_types: Types of annealing schedules to consider
        """
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.exploration_rate = exploration_rate
        self.schedule_types = schedule_types or ['linear', 'exponential', 'polynomial', 'adaptive']
        
        # Experience memory
        self.experiences = deque(maxlen=memory_size)
        
        # Schedule performance tracking
        self.schedule_performance = {schedule: [] for schedule in self.schedule_types}
        
        # Adaptive parameters for each schedule type
        self.schedule_params = {
            'linear': {'start_temp': 1.0, 'end_temp': 0.01, 'steps': 1000},
            'exponential': {'start_temp': 1.0, 'decay_rate': 0.99, 'steps': 1000},
            'polynomial': {'start_temp': 1.0, 'end_temp': 0.01, 'power': 2.0, 'steps': 1000},
            'adaptive': {'start_temp': 1.0, 'adaptation_rate': 0.1, 'steps': 1000}
        }
        
        # Problem classification model (simplified)
        self.problem_classifier = {}
        
        # Current best schedule for different problem types
        self.best_schedules = {}
    
    def suggest_quantum_parameters(self, problem_context: Dict[str, Any],
                                  history: List[QuantumExperience]) -> Dict[str, Any]:
        """Suggest quantum annealing parameters."""
        
        # Classify the problem
        problem_type = self._classify_problem(problem_context)
        
        # Choose schedule type based on exploration vs exploitation
        if np.random.random() < self.exploration_rate or problem_type not in self.best_schedules:
            # Explore: try different schedule types
            schedule_type = np.random.choice(self.schedule_types)
        else:
            # Exploit: use best known schedule for this problem type
            schedule_type = self.best_schedules[problem_type]
        
        # Generate schedule parameters
        schedule_params = self._generate_schedule_parameters(schedule_type, problem_context, history)
        
        # Convert to quantum backend parameters
        quantum_params = {
            'annealing_schedule': schedule_type,
            'schedule_params': schedule_params,
            'num_reads': self._suggest_num_reads(problem_context, history),
            'auto_scale': True,
            'programming_thermalization': self._suggest_thermalization(problem_context),
            'readout_thermalization': self._suggest_readout_thermalization(problem_context)
        }
        
        return quantum_params
    
    def _classify_problem(self, problem_context: Dict[str, Any]) -> str:
        """Classify the optimization problem to select appropriate strategy."""
        
        # Extract problem characteristics
        num_variables = problem_context.get('num_variables', 0)
        sparsity = problem_context.get('sparsity_ratio', 0)
        connectivity = problem_context.get('connectivity_degree', 0)
        param_types = problem_context.get('parameter_types', [])
        
        # Simple rule-based classification
        if num_variables < 50:
            problem_type = 'small'
        elif num_variables < 200:
            problem_type = 'medium'
        else:
            problem_type = 'large'
        
        # Refine based on other characteristics
        if sparsity > 0.8:
            problem_type += '_sparse'
        elif connectivity > 10:
            problem_type += '_dense'
        
        # Consider parameter types
        if 'continuous' in param_types and 'discrete' in param_types:
            problem_type += '_mixed'
        elif 'continuous' in param_types:
            problem_type += '_continuous'
        else:
            problem_type += '_discrete'
        
        return problem_type
    
    def _generate_schedule_parameters(self, schedule_type: str, 
                                    problem_context: Dict[str, Any],
                                    history: List[QuantumExperience]) -> Dict[str, Any]:
        """Generate parameters for the specified annealing schedule."""
        
        base_params = self.schedule_params[schedule_type].copy()
        
        # Adapt based on problem characteristics
        problem_size = problem_context.get('num_variables', 50)
        complexity_factor = np.log(max(problem_size, 1)) / 10
        
        if schedule_type == 'linear':
            # Adjust temperature range based on problem complexity
            base_params['start_temp'] = 1.0 + complexity_factor
            base_params['end_temp'] = 0.01 / (1 + complexity_factor)
            base_params['steps'] = max(1000, int(problem_size * 20))
            
        elif schedule_type == 'exponential':
            # Adjust decay rate based on problem size
            base_params['decay_rate'] = 0.99 - complexity_factor * 0.1
            base_params['steps'] = max(1000, int(problem_size * 15))
            
        elif schedule_type == 'polynomial':
            # Adjust polynomial power based on connectivity
            connectivity = problem_context.get('connectivity_degree', 5)
            base_params['power'] = 2.0 + connectivity / 10
            base_params['steps'] = max(1000, int(problem_size * 25))
            
        elif schedule_type == 'adaptive':
            # Adaptive schedule parameters
            base_params['adaptation_rate'] = 0.1 + complexity_factor * 0.05
            base_params['steps'] = max(1500, int(problem_size * 30))
        
        # Learn from history
        if history:
            self._adapt_from_history(base_params, schedule_type, history)
        
        return base_params
    
    def _adapt_from_history(self, params: Dict[str, Any], schedule_type: str,
                           history: List[QuantumExperience]):
        """Adapt parameters based on historical performance."""
        
        # Find similar experiences
        similar_experiences = []
        for exp in history[-20:]:  # Look at recent experiences
            quantum_settings = exp.quantum_settings
            if quantum_settings.get('annealing_schedule') == schedule_type:
                similar_experiences.append(exp)
        
        if not similar_experiences:
            return
        
        # Calculate performance-weighted averages for successful parameters
        success_threshold = np.percentile(
            [exp.performance_metrics.get('best_score', 0) for exp in similar_experiences], 
            70
        )
        
        successful_experiences = [
            exp for exp in similar_experiences 
            if exp.performance_metrics.get('best_score', 0) >= success_threshold
        ]
        
        if successful_experiences:
            # Weighted average of successful parameters
            weights = [exp.performance_metrics.get('best_score', 0) for exp in successful_experiences]
            total_weight = sum(weights)
            
            if total_weight > 0:
                for param_name in params.keys():
                    if param_name in successful_experiences[0].quantum_settings.get('schedule_params', {}):
                        weighted_sum = sum(
                            w * exp.quantum_settings['schedule_params'].get(param_name, params[param_name])
                            for w, exp in zip(weights, successful_experiences)
                        )
                        adapted_value = weighted_sum / total_weight
                        
                        # Blend with current value
                        params[param_name] = (
                            (1 - self.learning_rate) * params[param_name] + 
                            self.learning_rate * adapted_value
                        )
    
    def _suggest_num_reads(self, problem_context: Dict[str, Any],
                          history: List[QuantumExperience]) -> int:
        """Suggest number of quantum annealing reads."""
        
        base_reads = 1000
        problem_size = problem_context.get('num_variables', 50)
        
        # Scale with problem size
        reads = max(base_reads, int(problem_size * 20))
        
        # Cap based on computational budget
        max_reads = problem_context.get('max_quantum_reads', 10000)
        reads = min(reads, max_reads)
        
        # Adapt based on historical convergence
        if history:
            recent_convergence = [
                exp.performance_metrics.get('convergence_iterations', reads // 10)
                for exp in history[-10:]
            ]
            avg_convergence = np.mean(recent_convergence)
            
            # If converging quickly, we can use fewer reads
            if avg_convergence < reads // 20:
                reads = int(reads * 0.7)
            # If converging slowly, use more reads
            elif avg_convergence > reads // 5:
                reads = int(reads * 1.3)
        
        return max(100, min(reads, max_reads))
    
    def _suggest_thermalization(self, problem_context: Dict[str, Any]) -> int:
        """Suggest programming thermalization time."""
        
        problem_size = problem_context.get('num_variables', 50)
        connectivity = problem_context.get('connectivity_degree', 5)
        
        # Base thermalization
        base_therm = 20
        
        # Scale with problem characteristics
        therm = base_therm + int(problem_size / 10) + int(connectivity)
        
        return max(1, min(therm, 1000))
    
    def _suggest_readout_thermalization(self, problem_context: Dict[str, Any]) -> int:
        """Suggest readout thermalization time."""
        
        problem_size = problem_context.get('num_variables', 50)
        
        # Readout thermalization is typically smaller
        therm = max(1, int(problem_size / 20))
        
        return min(therm, 100)
    
    def update_from_experience(self, experience: QuantumExperience):
        """Update the scheduler based on optimization experience."""
        
        self.experiences.append(experience)
        
        # Update schedule performance tracking
        quantum_settings = experience.quantum_settings
        schedule_type = quantum_settings.get('annealing_schedule', 'linear')
        performance_score = experience.performance_metrics.get('best_score', 0)
        
        if schedule_type in self.schedule_performance:
            self.schedule_performance[schedule_type].append(performance_score)
        
        # Update best schedules for problem types
        problem_context = {
            'num_variables': experience.qubo_properties.get('num_variables', 50),
            'sparsity_ratio': experience.qubo_properties.get('sparsity_ratio', 0.5),
            'connectivity_degree': experience.qubo_properties.get('connectivity_degree', 5),
            'parameter_types': ['discrete']  # Simplified
        }
        
        problem_type = self._classify_problem(problem_context)
        
        # Update best schedule for this problem type
        if (problem_type not in self.best_schedules or 
            performance_score > self._get_best_performance(problem_type)):
            self.best_schedules[problem_type] = schedule_type
        
        # Adaptive parameter learning
        self._update_schedule_parameters(schedule_type, experience)
    
    def _get_best_performance(self, problem_type: str) -> float:
        """Get the best known performance for a problem type."""
        
        if problem_type not in self.best_schedules:
            return float('-inf')
        
        best_schedule = self.best_schedules[problem_type]
        performances = self.schedule_performance.get(best_schedule, [])
        
        return max(performances) if performances else float('-inf')
    
    def _update_schedule_parameters(self, schedule_type: str, experience: QuantumExperience):
        """Update schedule parameters based on experience."""
        
        performance_score = experience.performance_metrics.get('best_score', 0)
        schedule_params = experience.quantum_settings.get('schedule_params', {})
        
        # Only update if this was a good experience
        recent_scores = self.schedule_performance.get(schedule_type, [])
        if recent_scores:
            performance_percentile = np.percentile(recent_scores, 70)
            if performance_score >= performance_percentile:
                # Blend successful parameters with current defaults
                for param_name, param_value in schedule_params.items():
                    if param_name in self.schedule_params[schedule_type]:
                        current_value = self.schedule_params[schedule_type][param_name]
                        updated_value = (
                            (1 - self.learning_rate) * current_value +
                            self.learning_rate * param_value
                        )
                        self.schedule_params[schedule_type][param_name] = updated_value
    
    def get_adaptation_metrics(self) -> AdaptationMetrics:
        """Get metrics about adaptation performance."""
        
        if len(self.experiences) < 10:
            return AdaptationMetrics(0, 0, 0, 0)
        
        # Calculate improvement rate
        recent_scores = [exp.performance_metrics.get('best_score', 0) for exp in self.experiences[-10:]]
        early_scores = [exp.performance_metrics.get('best_score', 0) for exp in self.experiences[:10]]
        
        improvement_rate = (np.mean(recent_scores) - np.mean(early_scores)) / max(np.mean(early_scores), 0.1)
        
        # Calculate convergence speed
        convergence_times = [
            exp.performance_metrics.get('convergence_time', 1.0) for exp in self.experiences[-20:]
        ]
        convergence_speed = 1.0 / max(np.mean(convergence_times), 0.1)
        
        # Calculate stability score
        all_scores = [exp.performance_metrics.get('best_score', 0) for exp in self.experiences]
        stability_score = 1.0 / (1.0 + np.std(all_scores))
        
        # Calculate exploration efficiency
        unique_schedules = len(set(
            exp.quantum_settings.get('annealing_schedule', 'linear') 
            for exp in self.experiences
        ))
        exploration_efficiency = unique_schedules / len(self.schedule_types)
        
        return AdaptationMetrics(
            improvement_rate=improvement_rate,
            convergence_speed=convergence_speed,
            stability_score=stability_score,
            exploration_efficiency=exploration_efficiency
        )


class DynamicTopologySelector(AdaptiveQuantumStrategy):
    """
    Dynamic topology selector that chooses the best problem embedding
    and quantum hardware topology based on problem characteristics.
    """
    
    def __init__(self, available_topologies: List[str] = None,
                 embedding_methods: List[str] = None):
        """
        Initialize dynamic topology selector.
        
        Args:
            available_topologies: List of available hardware topologies
            embedding_methods: List of embedding algorithms to consider
        """
        self.available_topologies = available_topologies or ['pegasus', 'chimera', 'zephyr']
        self.embedding_methods = embedding_methods or ['minorminer', 'clique', 'fastembedding']
        
        # Topology performance tracking
        self.topology_performance = {topo: [] for topo in self.available_topologies}
        self.embedding_performance = {method: [] for method in self.embedding_methods}
        
        # Problem-topology compatibility scores
        self.compatibility_scores = {}
        
        # Experience memory
        self.experiences = deque(maxlen=200)
    
    def suggest_quantum_parameters(self, problem_context: Dict[str, Any],
                                  history: List[QuantumExperience]) -> Dict[str, Any]:
        """Suggest topology and embedding parameters."""
        
        # Analyze problem structure
        problem_structure = self._analyze_problem_structure(problem_context)
        
        # Select best topology
        best_topology = self._select_topology(problem_structure, history)
        
        # Select best embedding method
        best_embedding = self._select_embedding_method(problem_structure, best_topology, history)
        
        # Generate embedding parameters
        embedding_params = self._generate_embedding_parameters(
            best_embedding, problem_structure, best_topology
        )
        
        return {
            'topology': best_topology,
            'embedding_method': best_embedding,
            'embedding_params': embedding_params,
            'chain_strength': self._suggest_chain_strength(problem_structure),
            'auto_scale': True
        }
    
    def _analyze_problem_structure(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of the optimization problem."""
        
        num_variables = problem_context.get('num_variables', 0)
        sparsity_ratio = problem_context.get('sparsity_ratio', 0.5)
        connectivity_degree = problem_context.get('connectivity_degree', 5)
        
        # Calculate structural properties
        density = 1 - sparsity_ratio
        avg_degree = connectivity_degree
        clustering_coefficient = problem_context.get('clustering_coefficient', 0.3)
        
        # Classify problem structure
        if density < 0.1:
            structure_type = 'sparse'
        elif density > 0.8:
            structure_type = 'dense'
        else:
            structure_type = 'moderate'
        
        # Graph characteristics
        if avg_degree < 3:
            graph_type = 'tree_like'
        elif avg_degree > num_variables * 0.5:
            graph_type = 'complete_like'
        elif clustering_coefficient > 0.6:
            graph_type = 'clustered'
        else:
            graph_type = 'random'
        
        return {
            'num_variables': num_variables,
            'density': density,
            'avg_degree': avg_degree,
            'clustering_coefficient': clustering_coefficient,
            'structure_type': structure_type,
            'graph_type': graph_type
        }
    
    def _select_topology(self, problem_structure: Dict[str, Any],
                        history: List[QuantumExperience]) -> str:
        """Select the best quantum hardware topology."""
        
        num_variables = problem_structure['num_variables']
        graph_type = problem_structure['graph_type']
        density = problem_structure['density']
        
        # Rule-based topology selection
        topology_scores = {}
        
        for topology in self.available_topologies:
            score = 0.0
            
            if topology == 'pegasus':
                # Pegasus is good for medium to large problems with moderate connectivity
                if 100 <= num_variables <= 5000:
                    score += 0.4
                if graph_type in ['clustered', 'random']:
                    score += 0.3
                if 0.1 <= density <= 0.6:
                    score += 0.3
            
            elif topology == 'chimera':
                # Chimera is good for smaller problems with regular structure
                if num_variables <= 2000:
                    score += 0.4
                if graph_type in ['tree_like', 'random']:
                    score += 0.3
                if density <= 0.4:
                    score += 0.3
            
            elif topology == 'zephyr':
                # Zephyr is good for large, dense problems
                if num_variables >= 1000:
                    score += 0.4
                if graph_type in ['complete_like', 'clustered']:
                    score += 0.3
                if density >= 0.3:
                    score += 0.3
            
            # Add historical performance bonus
            if topology in self.topology_performance:
                recent_performance = self.topology_performance[topology][-10:]
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    score += avg_performance * 0.2
            
            topology_scores[topology] = score
        
        # Select topology with highest score
        best_topology = max(topology_scores, key=topology_scores.get)
        return best_topology
    
    def _select_embedding_method(self, problem_structure: Dict[str, Any],
                               topology: str, history: List[QuantumExperience]) -> str:
        """Select the best embedding method."""
        
        graph_type = problem_structure['graph_type']
        density = problem_structure['density']
        num_variables = problem_structure['num_variables']
        
        method_scores = {}
        
        for method in self.embedding_methods:
            score = 0.0
            
            if method == 'minorminer':
                # Good general-purpose method
                score += 0.3
                if graph_type in ['random', 'clustered']:
                    score += 0.2
                if num_variables <= 1000:
                    score += 0.2
            
            elif method == 'clique':
                # Good for dense, small problems
                if density >= 0.6:
                    score += 0.4
                if num_variables <= 100:
                    score += 0.3
                if graph_type == 'complete_like':
                    score += 0.3
            
            elif method == 'fastembedding':
                # Good for large, sparse problems
                if num_variables >= 500:
                    score += 0.3
                if density <= 0.3:
                    score += 0.4
                if graph_type in ['tree_like', 'sparse']:
                    score += 0.3
            
            # Add historical performance bonus
            if method in self.embedding_performance:
                recent_performance = self.embedding_performance[method][-10:]
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    score += avg_performance * 0.2
            
            method_scores[method] = score
        
        best_method = max(method_scores, key=method_scores.get)
        return best_method
    
    def _generate_embedding_parameters(self, embedding_method: str,
                                     problem_structure: Dict[str, Any],
                                     topology: str) -> Dict[str, Any]:
        """Generate parameters for the embedding method."""
        
        params = {}
        
        if embedding_method == 'minorminer':
            params = {
                'max_no_improvement': max(10, problem_structure['num_variables'] // 20),
                'random_seed': None,
                'timeout': 60.0,
                'max_beta': 100.0,
                'tries': 10
            }
        
        elif embedding_method == 'clique':
            params = {
                'use_cache': True,
                'timeout': 30.0
            }
        
        elif embedding_method == 'fastembedding':
            params = {
                'timeout': 45.0,
                'max_tries': 5,
                'verbose': False
            }
        
        # Adjust parameters based on problem size
        if problem_structure['num_variables'] > 1000:
            params['timeout'] = params.get('timeout', 60) * 2
        
        return params
    
    def _suggest_chain_strength(self, problem_structure: Dict[str, Any]) -> float:
        """Suggest chain strength for embedding."""
        
        density = problem_structure['density']
        avg_degree = problem_structure['avg_degree']
        
        # Base chain strength
        base_strength = 1.0
        
        # Adjust based on problem characteristics
        if density > 0.6:
            # Dense problems need stronger chains
            chain_strength = base_strength * (1 + density)
        elif density < 0.2:
            # Sparse problems can use weaker chains
            chain_strength = base_strength * 0.5
        else:
            chain_strength = base_strength
        
        # Further adjust based on connectivity
        if avg_degree > 10:
            chain_strength *= 1.2
        elif avg_degree < 3:
            chain_strength *= 0.8
        
        return max(0.1, min(chain_strength, 10.0))
    
    def update_from_experience(self, experience: QuantumExperience):
        """Update topology selector based on experience."""
        
        self.experiences.append(experience)
        
        # Update performance tracking
        quantum_settings = experience.quantum_settings
        topology = quantum_settings.get('topology', 'pegasus')
        embedding_method = quantum_settings.get('embedding_method', 'minorminer')
        performance_score = experience.performance_metrics.get('best_score', 0)
        
        if topology in self.topology_performance:
            self.topology_performance[topology].append(performance_score)
        
        if embedding_method in self.embedding_performance:
            self.embedding_performance[embedding_method].append(performance_score)
        
        # Update compatibility scores
        problem_key = self._get_problem_key(experience.qubo_properties)
        topology_key = f"{topology}_{embedding_method}"
        
        if problem_key not in self.compatibility_scores:
            self.compatibility_scores[problem_key] = {}
        
        if topology_key not in self.compatibility_scores[problem_key]:
            self.compatibility_scores[problem_key][topology_key] = []
        
        self.compatibility_scores[problem_key][topology_key].append(performance_score)
    
    def _get_problem_key(self, qubo_properties: Dict[str, Any]) -> str:
        """Generate a key to classify similar problems."""
        
        num_vars = qubo_properties.get('num_variables', 0)
        sparsity = qubo_properties.get('sparsity_ratio', 0.5)
        connectivity = qubo_properties.get('connectivity_degree', 5)
        
        # Discretize properties for classification
        size_class = 'small' if num_vars < 100 else 'medium' if num_vars < 1000 else 'large'
        sparsity_class = 'sparse' if sparsity > 0.7 else 'dense' if sparsity < 0.3 else 'moderate'
        conn_class = 'low' if connectivity < 5 else 'high' if connectivity > 15 else 'medium'
        
        return f"{size_class}_{sparsity_class}_{conn_class}"
    
    def get_adaptation_metrics(self) -> AdaptationMetrics:
        """Get topology adaptation metrics."""
        
        if len(self.experiences) < 10:
            return AdaptationMetrics(0, 0, 0, 0)
        
        # Calculate improvement over time
        early_scores = [exp.performance_metrics.get('best_score', 0) for exp in self.experiences[:10]]
        recent_scores = [exp.performance_metrics.get('best_score', 0) for exp in self.experiences[-10:]]
        
        improvement_rate = (np.mean(recent_scores) - np.mean(early_scores)) / max(np.mean(early_scores), 0.1)
        
        # Calculate average embedding success rate
        embedding_successes = [
            1 if exp.performance_metrics.get('embedding_success', True) else 0
            for exp in self.experiences[-50:]
        ]
        convergence_speed = np.mean(embedding_successes) if embedding_successes else 0
        
        # Calculate consistency of topology selection
        topology_selections = [
            exp.quantum_settings.get('topology', 'pegasus') for exp in self.experiences[-20:]
        ]
        unique_topologies = len(set(topology_selections))
        stability_score = 1.0 / max(unique_topologies, 1)  # More consistent = higher score
        
        # Calculate exploration efficiency
        total_topology_combinations = len(self.available_topologies) * len(self.embedding_methods)
        used_combinations = len(set(
            f"{exp.quantum_settings.get('topology', 'pegasus')}_{exp.quantum_settings.get('embedding_method', 'minorminer')}"
            for exp in self.experiences
        ))
        exploration_efficiency = used_combinations / total_topology_combinations
        
        return AdaptationMetrics(
            improvement_rate=improvement_rate,
            convergence_speed=convergence_speed,
            stability_score=stability_score,
            exploration_efficiency=exploration_efficiency
        )


class FeedbackDrivenTuner(AdaptiveQuantumStrategy):
    """
    Feedback-driven parameter tuner that adjusts all quantum parameters
    based on real-time performance feedback.
    """
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]] = None,
                 adaptation_rate: float = 0.2, momentum: float = 0.1):
        """
        Initialize feedback-driven tuner.
        
        Args:
            parameter_bounds: Min/max bounds for each parameter
            adaptation_rate: How quickly to adapt parameters
            momentum: Momentum factor for parameter updates
        """
        self.parameter_bounds = parameter_bounds or {
            'num_reads': (100, 10000),
            'chain_strength': (0.1, 20.0),
            'programming_thermalization': (1, 1000),
            'readout_thermalization': (1, 100),
            'auto_scale': (True, True)  # Boolean parameter
        }
        
        self.adaptation_rate = adaptation_rate
        self.momentum = momentum
        
        # Current parameter values
        self.current_params = {
            'num_reads': 1000,
            'chain_strength': 1.0,
            'programming_thermalization': 20,
            'readout_thermalization': 20,
            'auto_scale': True
        }
        
        # Parameter momentum (for smooth updates)
        self.parameter_momentum = {param: 0.0 for param in self.current_params}
        
        # Performance tracking
        self.performance_history = deque(maxlen=50)
        self.parameter_gradients = {param: deque(maxlen=20) for param in self.current_params}
        
        # Best known parameters
        self.best_params = self.current_params.copy()
        self.best_score = float('-inf')
    
    def suggest_quantum_parameters(self, problem_context: Dict[str, Any],
                                  history: List[QuantumExperience]) -> Dict[str, Any]:
        """Suggest quantum parameters based on feedback."""
        
        # Update parameters based on recent history
        if history:
            self._update_parameters_from_feedback(history[-5:])  # Use last 5 experiences
        
        # Add some exploration noise
        suggested_params = self._add_exploration_noise(self.current_params.copy())
        
        # Ensure parameters are within bounds
        suggested_params = self._clip_parameters(suggested_params)
        
        return suggested_params
    
    def _update_parameters_from_feedback(self, recent_experiences: List[QuantumExperience]):
        """Update parameters based on recent performance feedback."""
        
        if len(recent_experiences) < 2:
            return
        
        # Calculate performance gradient
        scores = [exp.performance_metrics.get('best_score', 0) for exp in recent_experiences]
        performance_trend = np.mean(np.diff(scores))  # Average change in performance
        
        # If performance is improving, continue in the same direction
        # If performance is declining, try to reverse recent changes
        
        for param_name in self.current_params:
            if param_name == 'auto_scale':  # Skip boolean parameters
                continue
                
            # Calculate parameter gradient
            param_values = [
                exp.quantum_settings.get(param_name, self.current_params[param_name])
                for exp in recent_experiences
            ]
            
            if len(set(param_values)) > 1:  # Only if parameter varied
                # Calculate correlation between parameter changes and performance
                param_diffs = np.diff(param_values)
                score_diffs = np.diff(scores)
                
                if len(param_diffs) > 0 and len(score_diffs) > 0:
                    correlation = np.corrcoef(param_diffs, score_diffs)[0, 1]
                    
                    if not np.isnan(correlation):
                        # Update parameter in direction of positive correlation
                        current_value = self.current_params[param_name]
                        param_range = (
                            self.parameter_bounds[param_name][1] - 
                            self.parameter_bounds[param_name][0]
                        )
                        
                        # Calculate adaptive step size
                        step_size = self.adaptation_rate * param_range * 0.1
                        
                        # Apply momentum
                        momentum_term = self.momentum * self.parameter_momentum[param_name]
                        gradient_term = correlation * step_size
                        
                        update = momentum_term + gradient_term
                        self.parameter_momentum[param_name] = update
                        
                        # Update parameter
                        new_value = current_value + update
                        self.current_params[param_name] = new_value
                        
                        # Track gradient
                        self.parameter_gradients[param_name].append(correlation)
    
    def _add_exploration_noise(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add exploration noise to parameters."""
        
        exploration_rate = 0.05  # 5% noise
        
        for param_name, value in params.items():
            if param_name == 'auto_scale':  # Skip boolean parameters
                continue
                
            if isinstance(value, (int, float)):
                param_range = (
                    self.parameter_bounds[param_name][1] - 
                    self.parameter_bounds[param_name][0]
                )
                noise = np.random.normal(0, exploration_rate * param_range)
                params[param_name] = value + noise
        
        return params
    
    def _clip_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clip parameters to their valid bounds."""
        
        for param_name, value in params.items():
            if param_name in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param_name]
                
                if param_name == 'auto_scale':
                    params[param_name] = bool(value)  # Ensure boolean
                elif isinstance(value, (int, float)):
                    params[param_name] = max(min_val, min(value, max_val))
                    
                    # Convert to int for integer parameters
                    if param_name in ['num_reads', 'programming_thermalization', 'readout_thermalization']:
                        params[param_name] = int(params[param_name])
        
        return params
    
    def update_from_experience(self, experience: QuantumExperience):
        """Update tuner based on optimization experience."""
        
        performance_score = experience.performance_metrics.get('best_score', 0)
        self.performance_history.append(performance_score)
        
        # Update best known parameters
        if performance_score > self.best_score:
            self.best_score = performance_score
            self.best_params = experience.quantum_settings.copy()
        
        # Adaptive learning rate based on performance variance
        if len(self.performance_history) >= 10:
            performance_std = np.std(list(self.performance_history)[-10:])
            # Reduce learning rate if performance is highly variable
            self.adaptation_rate = max(0.05, 0.2 / (1 + performance_std))
    
    def get_adaptation_metrics(self) -> AdaptationMetrics:
        """Get feedback-driven tuning metrics."""
        
        if len(self.performance_history) < 10:
            return AdaptationMetrics(0, 0, 0, 0)
        
        # Calculate improvement rate
        early_performance = list(self.performance_history)[:10]
        recent_performance = list(self.performance_history)[-10:]
        
        improvement_rate = (np.mean(recent_performance) - np.mean(early_performance)) / max(np.mean(early_performance), 0.1)
        
        # Calculate convergence speed (inverse of time to best performance)
        best_idx = np.argmax(self.performance_history)
        convergence_speed = 1.0 / max(best_idx + 1, 1)
        
        # Calculate stability (inverse of performance variance)
        performance_variance = np.var(list(self.performance_history)[-20:])
        stability_score = 1.0 / (1.0 + performance_variance)
        
        # Calculate exploration efficiency based on parameter diversity
        param_diversity = 0
        for param_gradients in self.parameter_gradients.values():
            if param_gradients:
                param_diversity += len(set(np.sign(param_gradients)))
        
        exploration_efficiency = param_diversity / max(len(self.parameter_gradients), 1)
        
        return AdaptationMetrics(
            improvement_rate=improvement_rate,
            convergence_speed=convergence_speed,
            stability_score=stability_score,
            exploration_efficiency=exploration_efficiency
        )