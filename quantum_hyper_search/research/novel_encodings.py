#!/usr/bin/env python3
"""
Novel QUBO Encoding Schemes for Hyperparameter Optimization

This module implements advanced encoding techniques identified as research gaps:
1. Hierarchical parameter space encoding
2. Constraint-aware QUBO formulations
3. Multi-objective optimization embeddings
4. Adaptive encoding strategies
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
import itertools
from dataclasses import dataclass


@dataclass
class EncodingMetrics:
    """Metrics for evaluating encoding quality."""
    sparsity_ratio: float  # Fraction of zero coefficients
    connectivity_degree: float  # Average node degree
    penalty_balance: float  # Ratio of constraint to objective coefficients
    embedding_efficiency: float  # Variables per parameter
    

class NovelQUBOEncoder(ABC):
    """Abstract base class for novel QUBO encoding schemes."""
    
    @abstractmethod
    def encode(self, param_space: Dict[str, List[Any]], 
               constraints: Optional[Dict] = None,
               objectives: Optional[List[str]] = None) -> Dict[Tuple[int, int], float]:
        """Encode parameter space to QUBO matrix."""
        pass
    
    @abstractmethod
    def decode(self, sample: Dict[int, int], 
               param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Decode QUBO sample back to parameter values."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> EncodingMetrics:
        """Get encoding quality metrics."""
        pass


class HierarchicalEncoder(NovelQUBOEncoder):
    """
    Hierarchical parameter space encoding that groups related parameters
    and creates multi-level optimization structures.
    """
    
    def __init__(self, hierarchy_levels: int = 3, clustering_threshold: float = 0.7):
        """
        Initialize hierarchical encoder.
        
        Args:
            hierarchy_levels: Number of hierarchy levels to create
            clustering_threshold: Similarity threshold for parameter clustering
        """
        self.hierarchy_levels = hierarchy_levels
        self.clustering_threshold = clustering_threshold
        self.parameter_groups = {}
        self.variable_mapping = {}
        self.reverse_mapping = {}
        self.last_qubo = {}
        
    def _cluster_parameters(self, param_space: Dict[str, List[Any]]) -> List[List[str]]:
        """Cluster parameters based on their characteristics."""
        params = list(param_space.keys())
        
        # Simple clustering based on parameter name similarity and value types
        clusters = []
        used_params = set()
        
        for param in params:
            if param in used_params:
                continue
                
            cluster = [param]
            used_params.add(param)
            
            for other_param in params:
                if other_param in used_params:
                    continue
                    
                # Check similarity: name pattern, value type, range similarity
                similarity = self._calculate_parameter_similarity(
                    param, other_param, param_space
                )
                
                if similarity > self.clustering_threshold:
                    cluster.append(other_param)
                    used_params.add(other_param)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_parameter_similarity(self, param1: str, param2: str, 
                                      param_space: Dict[str, List[Any]]) -> float:
        """Calculate similarity between two parameters."""
        # Name-based similarity
        name_sim = 0.0
        if any(common in param1 and common in param2 
               for common in ['depth', 'n_', 'max_', 'min_', 'learning', 'reg']):
            name_sim = 0.4
            
        # Type-based similarity
        type_sim = 0.0
        vals1, vals2 = param_space[param1], param_space[param2]
        if type(vals1[0]) == type(vals2[0]):
            type_sim = 0.3
            
        # Range-based similarity (for numeric parameters)
        range_sim = 0.0
        try:
            if all(isinstance(v, (int, float)) for v in vals1 + vals2):
                range1 = max(vals1) - min(vals1) 
                range2 = max(vals2) - min(vals2)
                if range1 > 0 and range2 > 0:
                    range_sim = 0.3 * min(range1, range2) / max(range1, range2)
        except (ValueError, TypeError):
            pass
            
        return name_sim + type_sim + range_sim
    
    def _create_hierarchy(self, clusters: List[List[str]], 
                         param_space: Dict[str, List[Any]]) -> Dict[int, List[List[str]]]:
        """Create hierarchical structure from parameter clusters."""
        hierarchy = {0: clusters}  # Level 0: original clusters
        
        current_clusters = clusters
        for level in range(1, self.hierarchy_levels):
            if len(current_clusters) <= 1:
                break
                
            # Merge clusters at each level
            new_clusters = []
            used_indices = set()
            
            for i, cluster in enumerate(current_clusters):
                if i in used_indices:
                    continue
                    
                merged_cluster = cluster.copy()
                used_indices.add(i)
                
                # Find clusters to merge with
                for j, other_cluster in enumerate(current_clusters[i+1:], i+1):
                    if j in used_indices or len(merged_cluster) + len(other_cluster) > 8:
                        continue
                        
                    # Merge if clusters are compatible
                    merged_cluster.extend(other_cluster)
                    used_indices.add(j)
                    break
                
                new_clusters.append(merged_cluster)
            
            hierarchy[level] = new_clusters
            current_clusters = new_clusters
            
        return hierarchy
    
    def encode(self, param_space: Dict[str, List[Any]], 
               constraints: Optional[Dict] = None,
               objectives: Optional[List[str]] = None) -> Dict[Tuple[int, int], float]:
        """Encode using hierarchical structure."""
        
        # Step 1: Create parameter clusters and hierarchy
        clusters = self._cluster_parameters(param_space)
        hierarchy = self._create_hierarchy(clusters, param_space)
        
        # Step 2: Create variable mapping
        var_idx = 0
        self.variable_mapping = {}
        self.reverse_mapping = {}
        
        for param, values in param_space.items():
            param_vars = []
            for i, value in enumerate(values):
                self.variable_mapping[(param, i)] = var_idx
                self.reverse_mapping[var_idx] = (param, i, value)
                param_vars.append(var_idx)
                var_idx += 1
            self.parameter_groups[param] = param_vars
        
        # Step 3: Build QUBO matrix with hierarchical constraints
        Q = {}
        penalty_strength = 2.0
        
        # Add one-hot constraints for each parameter
        for param, var_indices in self.parameter_groups.items():
            n_vars = len(var_indices)
            
            # Diagonal terms: encourage selection
            for var in var_indices:
                Q[(var, var)] = -1.0
            
            # Off-diagonal terms: penalize multiple selections  
            for i, var1 in enumerate(var_indices):
                for j, var2 in enumerate(var_indices[i+1:], i+1):
                    Q[(var1, var2)] = penalty_strength
        
        # Add hierarchical coupling terms
        for level, level_clusters in hierarchy.items():
            level_penalty = penalty_strength * (0.5 ** level)  # Weaker penalties at higher levels
            
            for cluster in level_clusters:
                if len(cluster) < 2:
                    continue
                    
                # Add coupling between parameters in the same cluster
                cluster_vars = []
                for param in cluster:
                    if param in self.parameter_groups:
                        cluster_vars.extend(self.parameter_groups[param])
                
                # Encourage coherent selections within clusters
                for i, var1 in enumerate(cluster_vars):
                    for j, var2 in enumerate(cluster_vars[i+1:], i+1):
                        if (var1, var2) in Q:
                            Q[(var1, var2)] -= level_penalty * 0.1  # Small encouragement
                        else:
                            Q[(var1, var2)] = -level_penalty * 0.1
        
        # Add constraint-aware terms
        if constraints:
            self._add_constraint_aware_terms(Q, constraints, param_space)
            
        # Add multi-objective terms
        if objectives:
            self._add_multi_objective_terms(Q, objectives, param_space)
        
        self.last_qubo = Q
        return Q
    
    def _add_constraint_aware_terms(self, Q: Dict[Tuple[int, int], float],
                                   constraints: Dict, param_space: Dict[str, List[Any]]):
        """Add constraint-aware QUBO terms."""
        constraint_penalty = 5.0
        
        for constraint_name, constraint_def in constraints.items():
            if constraint_name == 'max_depth_vs_min_samples':
                # Example: max_depth should be inversely related to min_samples_split
                if 'max_depth' in param_space and 'min_samples_split' in param_space:
                    depth_vars = self.parameter_groups.get('max_depth', [])
                    samples_vars = self.parameter_groups.get('min_samples_split', [])
                    
                    for i, depth_var in enumerate(depth_vars):
                        depth_val = param_space['max_depth'][i]
                        for j, samples_var in enumerate(samples_vars):
                            samples_val = param_space['min_samples_split'][j]
                            
                            # Penalize incompatible combinations
                            if depth_val > 10 and samples_val < 5:
                                if (depth_var, samples_var) in Q:
                                    Q[(depth_var, samples_var)] += constraint_penalty
                                else:
                                    Q[(depth_var, samples_var)] = constraint_penalty
    
    def _add_multi_objective_terms(self, Q: Dict[Tuple[int, int], float],
                                  objectives: List[str], param_space: Dict[str, List[Any]]):
        """Add multi-objective optimization terms."""
        if 'performance' in objectives and 'efficiency' in objectives:
            # Balance performance vs efficiency
            efficiency_bonus = 1.5
            
            # Encourage smaller models (fewer parameters, simpler settings)
            for param, values in param_space.items():
                if 'n_estimators' in param or 'max_depth' in param:
                    var_indices = self.parameter_groups.get(param, [])
                    
                    for i, var_idx in enumerate(var_indices):
                        value = values[i]
                        if isinstance(value, (int, float)):
                            # Bonus for smaller values (more efficient)
                            efficiency_factor = efficiency_bonus / (1 + np.log(max(1, value)))
                            if (var_idx, var_idx) in Q:
                                Q[(var_idx, var_idx)] -= efficiency_factor
                            else:
                                Q[(var_idx, var_idx)] = -efficiency_factor
    
    def decode(self, sample: Dict[int, int], 
               param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Decode QUBO sample to parameter values."""
        decoded_params = {}
        
        for param in param_space.keys():
            var_indices = self.parameter_groups.get(param, [])
            
            # Find the selected variable for this parameter
            selected_vars = [var_idx for var_idx in var_indices if sample.get(var_idx, 0) == 1]
            
            if selected_vars:
                # Use the first selected variable
                selected_var = selected_vars[0]
                _, _, value = self.reverse_mapping[selected_var]
                decoded_params[param] = value
            else:
                # Fallback: use the first option
                if var_indices:
                    _, _, value = self.reverse_mapping[var_indices[0]]
                    decoded_params[param] = value
                else:
                    decoded_params[param] = param_space[param][0]
        
        return decoded_params
    
    def get_metrics(self) -> EncodingMetrics:
        """Get encoding quality metrics."""
        if not self.last_qubo:
            return EncodingMetrics(0, 0, 0, 0)
        
        total_coeffs = len(self.last_qubo)
        zero_coeffs = sum(1 for v in self.last_qubo.values() if abs(v) < 1e-10)
        sparsity = zero_coeffs / max(total_coeffs, 1)
        
        # Calculate connectivity
        variables = set()
        for (i, j) in self.last_qubo.keys():
            variables.add(i)
            variables.add(j)
        
        num_vars = len(variables)
        connectivity = 2 * total_coeffs / max(num_vars, 1)
        
        # Calculate penalty balance
        positive_coeffs = sum(1 for v in self.last_qubo.values() if v > 0)
        negative_coeffs = sum(1 for v in self.last_qubo.values() if v < 0)
        penalty_balance = positive_coeffs / max(positive_coeffs + negative_coeffs, 1)
        
        # Calculate embedding efficiency
        num_params = len(self.parameter_groups)
        embedding_efficiency = num_vars / max(num_params, 1)
        
        return EncodingMetrics(
            sparsity_ratio=sparsity,
            connectivity_degree=connectivity,
            penalty_balance=penalty_balance,
            embedding_efficiency=embedding_efficiency
        )


class ConstraintAwareEncoder(NovelQUBOEncoder):
    """
    Constraint-aware QUBO encoder that explicitly models parameter 
    relationships and constraints in the QUBO formulation.
    """
    
    def __init__(self, constraint_strength: float = 3.0, 
                 adaptive_penalties: bool = True):
        """
        Initialize constraint-aware encoder.
        
        Args:
            constraint_strength: Base strength for constraint penalties
            adaptive_penalties: Whether to adapt penalty strengths based on problem
        """
        self.constraint_strength = constraint_strength
        self.adaptive_penalties = adaptive_penalties
        self.constraint_catalog = self._build_constraint_catalog()
        self.variable_mapping = {}
        self.reverse_mapping = {}
        self.last_qubo = {}
    
    def _build_constraint_catalog(self) -> Dict[str, Dict]:
        """Build catalog of common ML parameter constraints."""
        return {
            'tree_depth_samples': {
                'params': ['max_depth', 'min_samples_split', 'min_samples_leaf'],
                'type': 'inverse_relationship',
                'description': 'Deeper trees should have higher sample requirements'
            },
            'regularization_consistency': {
                'params': ['C', 'alpha', 'reg_alpha', 'reg_lambda'],
                'type': 'consistency',
                'description': 'Regularization parameters should be consistent'
            },
            'ensemble_size_complexity': {
                'params': ['n_estimators', 'max_depth', 'max_features'],
                'type': 'complexity_budget',
                'description': 'Total model complexity should be bounded'
            },
            'learning_rate_epochs': {
                'params': ['learning_rate', 'n_estimators', 'max_iter'],
                'type': 'compensatory',
                'description': 'Lower learning rate needs more iterations'
            }
        }
    
    def encode(self, param_space: Dict[str, List[Any]], 
               constraints: Optional[Dict] = None,
               objectives: Optional[List[str]] = None) -> Dict[Tuple[int, int], float]:
        """Encode with explicit constraint modeling."""
        
        # Step 1: Create variable mapping
        var_idx = 0
        self.variable_mapping = {}
        self.reverse_mapping = {}
        param_groups = {}
        
        for param, values in param_space.items():
            param_vars = []
            for i, value in enumerate(values):
                self.variable_mapping[(param, i)] = var_idx
                self.reverse_mapping[var_idx] = (param, i, value)
                param_vars.append(var_idx)
                var_idx += 1
            param_groups[param] = param_vars
        
        # Step 2: Initialize QUBO with one-hot constraints
        Q = {}
        base_penalty = self.constraint_strength
        
        # One-hot constraints for each parameter
        for param, var_indices in param_groups.items():
            # Encourage selection (diagonal terms)
            for var in var_indices:
                Q[(var, var)] = -1.0
            
            # Penalize multiple selections (off-diagonal terms)
            for i, var1 in enumerate(var_indices):
                for j, var2 in enumerate(var_indices[i+1:], i+1):
                    Q[(var1, var2)] = base_penalty
        
        # Step 3: Apply constraint catalog rules
        for constraint_name, constraint_def in self.constraint_catalog.items():
            self._apply_constraint_rule(Q, constraint_def, param_space, param_groups)
        
        # Step 4: Apply custom constraints if provided
        if constraints:
            for constraint_name, constraint_def in constraints.items():
                self._apply_custom_constraint(Q, constraint_name, constraint_def, 
                                            param_space, param_groups)
        
        # Step 5: Apply objective-specific terms
        if objectives:
            self._apply_objective_terms(Q, objectives, param_space, param_groups)
        
        self.last_qubo = Q
        return Q
    
    def _apply_constraint_rule(self, Q: Dict[Tuple[int, int], float],
                              constraint_def: Dict, param_space: Dict[str, List[Any]],
                              param_groups: Dict[str, List[int]]):
        """Apply a specific constraint rule from the catalog."""
        
        constraint_type = constraint_def['type']
        involved_params = [p for p in constraint_def['params'] if p in param_space]
        
        if not involved_params:
            return
            
        penalty = self.constraint_strength * 0.5  # Moderate penalty for catalog constraints
        
        if constraint_type == 'inverse_relationship':
            self._apply_inverse_relationship(Q, involved_params, param_space, param_groups, penalty)
        elif constraint_type == 'consistency':
            self._apply_consistency_constraint(Q, involved_params, param_space, param_groups, penalty)
        elif constraint_type == 'complexity_budget':
            self._apply_complexity_budget(Q, involved_params, param_space, param_groups, penalty)
        elif constraint_type == 'compensatory':
            self._apply_compensatory_constraint(Q, involved_params, param_space, param_groups, penalty)
    
    def _apply_inverse_relationship(self, Q: Dict[Tuple[int, int], float],
                                   params: List[str], param_space: Dict[str, List[Any]],
                                   param_groups: Dict[str, List[int]], penalty: float):
        """Apply inverse relationship constraint (e.g., depth vs samples)."""
        
        if 'max_depth' in params and 'min_samples_split' in params:
            depth_vars = param_groups['max_depth']
            samples_vars = param_groups['min_samples_split']
            
            for i, depth_var in enumerate(depth_vars):
                depth_val = param_space['max_depth'][i]
                for j, samples_var in enumerate(samples_vars):
                    samples_val = param_space['min_samples_split'][j]
                    
                    # Penalize combinations where high depth meets low samples
                    if isinstance(depth_val, int) and isinstance(samples_val, int):
                        violation_score = max(0, depth_val - 5) / max(1, samples_val - 2)
                        if violation_score > 2:  # Threshold for violation
                            Q[(depth_var, samples_var)] = Q.get((depth_var, samples_var), 0) + penalty * violation_score
    
    def _apply_consistency_constraint(self, Q: Dict[Tuple[int, int], float],
                                     params: List[str], param_space: Dict[str, List[Any]],
                                     param_groups: Dict[str, List[int]], penalty: float):
        """Apply consistency constraint for related parameters."""
        
        # For regularization parameters, encourage similar strength levels
        reg_params = [p for p in params if p in ['C', 'alpha', 'reg_alpha', 'reg_lambda']]
        
        if len(reg_params) >= 2:
            for i, param1 in enumerate(reg_params):
                for param2 in reg_params[i+1:]:
                    vars1 = param_groups[param1]
                    vars2 = param_groups[param2]
                    
                    for vi, var1 in enumerate(vars1):
                        val1 = param_space[param1][vi]
                        for vj, var2 in enumerate(vars2):
                            val2 = param_space[param2][vj]
                            
                            # Encourage similar regularization strengths
                            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                                # Convert to log scale for comparison
                                log_diff = abs(np.log10(max(val1, 1e-6)) - np.log10(max(val2, 1e-6)))
                                if log_diff < 1:  # Within one order of magnitude
                                    bonus = -penalty * 0.2  # Small bonus for consistency
                                    Q[(var1, var2)] = Q.get((var1, var2), 0) + bonus
    
    def _apply_complexity_budget(self, Q: Dict[Tuple[int, int], float],
                                params: List[str], param_space: Dict[str, List[Any]],
                                param_groups: Dict[str, List[int]], penalty: float):
        """Apply complexity budget constraint."""
        
        complexity_params = [p for p in params if p in param_space]
        
        if len(complexity_params) >= 2:
            # Calculate complexity scores for each combination
            all_combinations = []
            
            for param_combo in itertools.product(*[param_groups[p] for p in complexity_params]):
                complexity_score = 0
                var_combo = []
                
                for i, var in enumerate(param_combo):
                    param_name = complexity_params[i]
                    param_idx = param_groups[param_name].index(var)
                    value = param_space[param_name][param_idx]
                    var_combo.append(var)
                    
                    # Calculate complexity contribution
                    if param_name == 'n_estimators' and isinstance(value, int):
                        complexity_score += np.log(value + 1)
                    elif param_name == 'max_depth' and isinstance(value, int):
                        complexity_score += value * 0.5
                    elif param_name == 'max_features' and value == 'auto':
                        complexity_score += 2
                
                all_combinations.append((var_combo, complexity_score))
            
            # Penalize high-complexity combinations
            complexity_threshold = np.percentile([score for _, score in all_combinations], 70)
            
            for var_combo, score in all_combinations:
                if score > complexity_threshold:
                    excess_penalty = penalty * (score - complexity_threshold)
                    # Apply penalty to all pairs in the combination
                    for i, var1 in enumerate(var_combo):
                        for var2 in var_combo[i+1:]:
                            Q[(var1, var2)] = Q.get((var1, var2), 0) + excess_penalty
    
    def _apply_compensatory_constraint(self, Q: Dict[Tuple[int, int], float],
                                      params: List[str], param_space: Dict[str, List[Any]],
                                      param_groups: Dict[str, List[int]], penalty: float):
        """Apply compensatory relationship constraint."""
        
        if 'learning_rate' in params and any(p in params for p in ['n_estimators', 'max_iter']):
            lr_vars = param_groups['learning_rate']
            iter_param = 'n_estimators' if 'n_estimators' in params else 'max_iter'
            iter_vars = param_groups[iter_param]
            
            for i, lr_var in enumerate(lr_vars):
                lr_val = param_space['learning_rate'][i]
                for j, iter_var in enumerate(iter_vars):
                    iter_val = param_space[iter_param][j]
                    
                    # Encourage low LR with high iterations, high LR with low iterations
                    if isinstance(lr_val, (int, float)) and isinstance(iter_val, (int, float)):
                        # Calculate ideal relationship score
                        lr_score = -np.log10(max(lr_val, 1e-6))  # Higher for lower LR
                        iter_score = np.log10(max(iter_val, 1))   # Higher for more iterations
                        
                        compatibility = lr_score + iter_score
                        if compatibility > 3:  # Good combination
                            bonus = -penalty * 0.3
                            Q[(lr_var, iter_var)] = Q.get((lr_var, iter_var), 0) + bonus
    
    def _apply_custom_constraint(self, Q: Dict[Tuple[int, int], float],
                                constraint_name: str, constraint_def: Any,
                                param_space: Dict[str, List[Any]], 
                                param_groups: Dict[str, List[int]]):
        """Apply custom user-defined constraint."""
        
        if callable(constraint_def):
            # Constraint is a function that takes parameter values and returns penalty
            custom_penalty = self.constraint_strength * 2  # Higher penalty for custom constraints
            
            for param_combo in itertools.product(*param_space.values()):
                param_dict = dict(zip(param_space.keys(), param_combo))
                penalty_score = constraint_def(param_dict)
                
                if penalty_score > 0:
                    # Find corresponding variables and apply penalty
                    var_combo = []
                    for param_name, value in param_dict.items():
                        value_idx = param_space[param_name].index(value)
                        var_combo.append(param_groups[param_name][value_idx])
                    
                    # Apply penalty to all variable pairs
                    for i, var1 in enumerate(var_combo):
                        for var2 in var_combo[i+1:]:
                            Q[(var1, var2)] = Q.get((var1, var2), 0) + custom_penalty * penalty_score
    
    def _apply_objective_terms(self, Q: Dict[Tuple[int, int], float],
                              objectives: List[str], param_space: Dict[str, List[Any]],
                              param_groups: Dict[str, List[int]]):
        """Apply objective-specific QUBO terms."""
        
        if 'accuracy' in objectives:
            # Encourage parameters known to improve accuracy
            accuracy_bonus = 0.5
            
            for param, values in param_space.items():
                var_indices = param_groups[param]
                
                for i, var_idx in enumerate(var_indices):
                    value = values[i]
                    bonus = 0
                    
                    # Apply heuristic bonuses for accuracy-improving settings
                    if param == 'n_estimators' and isinstance(value, int) and value >= 100:
                        bonus = accuracy_bonus
                    elif param == 'max_depth' and isinstance(value, int) and 5 <= value <= 10:
                        bonus = accuracy_bonus
                    elif param == 'min_samples_split' and isinstance(value, int) and 2 <= value <= 10:
                        bonus = accuracy_bonus * 0.5
                    
                    if bonus > 0:
                        Q[(var_idx, var_idx)] = Q.get((var_idx, var_idx), 0) - bonus
        
        if 'speed' in objectives or 'efficiency' in objectives:
            # Encourage parameters that improve training/inference speed
            speed_bonus = 0.3
            
            for param, values in param_space.items():
                var_indices = param_groups[param]
                
                for i, var_idx in enumerate(var_indices):
                    value = values[i]
                    bonus = 0
                    
                    # Apply bonuses for speed-improving settings
                    if param == 'n_estimators' and isinstance(value, int) and value <= 50:
                        bonus = speed_bonus
                    elif param == 'max_depth' and isinstance(value, int) and value <= 5:
                        bonus = speed_bonus
                    elif param == 'max_features' and value in ['sqrt', 'log2']:
                        bonus = speed_bonus * 0.5
                    
                    if bonus > 0:
                        Q[(var_idx, var_idx)] = Q.get((var_idx, var_idx), 0) - bonus
    
    def decode(self, sample: Dict[int, int], 
               param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Decode QUBO sample to parameter values."""
        decoded_params = {}
        
        # Group variables by parameter
        param_vars = {}
        for var_idx, assignment in sample.items():
            if var_idx in self.reverse_mapping and assignment == 1:
                param, value_idx, value = self.reverse_mapping[var_idx]
                if param not in param_vars:
                    param_vars[param] = []
                param_vars[param].append((value_idx, value))
        
        # Select one value per parameter
        for param in param_space.keys():
            if param in param_vars and param_vars[param]:
                # If multiple values selected, choose the first one
                _, value = param_vars[param][0]
                decoded_params[param] = value
            else:
                # Fallback to first option
                decoded_params[param] = param_space[param][0]
        
        return decoded_params
    
    def get_metrics(self) -> EncodingMetrics:
        """Get constraint-aware encoding metrics."""
        if not self.last_qubo:
            return EncodingMetrics(0, 0, 0, 0)
        
        total_terms = len(self.last_qubo)
        
        # Calculate sparsity
        zero_terms = sum(1 for v in self.last_qubo.values() if abs(v) < 1e-10)
        sparsity = zero_terms / max(total_terms, 1)
        
        # Calculate connectivity
        variables = set()
        for (i, j) in self.last_qubo.keys():
            variables.add(i)
            variables.add(j)
        
        num_vars = len(variables)
        connectivity = 2 * total_terms / max(num_vars, 1)
        
        # Calculate penalty balance (constraint vs objective terms)
        positive_terms = sum(1 for v in self.last_qubo.values() if v > 0)
        negative_terms = sum(1 for v in self.last_qubo.values() if v < 0)
        penalty_balance = positive_terms / max(total_terms, 1)
        
        # Calculate embedding efficiency
        num_logical_params = len(set(param for param, _, _ in self.reverse_mapping.values()))
        embedding_efficiency = num_vars / max(num_logical_params, 1)
        
        return EncodingMetrics(
            sparsity_ratio=sparsity,
            connectivity_degree=connectivity,
            penalty_balance=penalty_balance,
            embedding_efficiency=embedding_efficiency
        )


class MultiObjectiveEncoder(NovelQUBOEncoder):
    """
    Multi-objective QUBO encoder that handles multiple optimization objectives
    with Pareto-efficient formulations.
    """
    
    def __init__(self, objective_weights: Optional[Dict[str, float]] = None,
                 pareto_sampling: bool = True, scalarization_method: str = 'weighted_sum'):
        """
        Initialize multi-objective encoder.
        
        Args:
            objective_weights: Weights for different objectives
            pareto_sampling: Whether to sample from Pareto frontier
            scalarization_method: Method for combining objectives ('weighted_sum', 'epsilon_constraint', 'tchebycheff')
        """
        self.objective_weights = objective_weights or {}
        self.pareto_sampling = pareto_sampling
        self.scalarization_method = scalarization_method
        self.variable_mapping = {}
        self.reverse_mapping = {}
        self.last_qubo = {}
        self.objective_terms = {}
    
    def encode(self, param_space: Dict[str, List[Any]], 
               constraints: Optional[Dict] = None,
               objectives: Optional[List[str]] = None) -> Dict[Tuple[int, int], float]:
        """Encode with multi-objective optimization."""
        
        if not objectives:
            objectives = ['accuracy']  # Default objective
        
        # Create variable mapping
        var_idx = 0
        self.variable_mapping = {}
        self.reverse_mapping = {}
        param_groups = {}
        
        for param, values in param_space.items():
            param_vars = []
            for i, value in enumerate(values):
                self.variable_mapping[(param, i)] = var_idx
                self.reverse_mapping[var_idx] = (param, i, value)
                param_vars.append(var_idx)
                var_idx += 1
            param_groups[param] = param_vars
        
        # Initialize QUBO with one-hot constraints
        Q = {}
        base_penalty = 2.0
        
        # One-hot constraints for each parameter
        for param, var_indices in param_groups.items():
            for var in var_indices:
                Q[(var, var)] = -0.5  # Base selection encouragement
            
            for i, var1 in enumerate(var_indices):
                for j, var2 in enumerate(var_indices[i+1:], i+1):
                    Q[(var1, var2)] = base_penalty
        
        # Build objective-specific terms
        self.objective_terms = {}
        for objective in objectives:
            self.objective_terms[objective] = self._build_objective_terms(
                objective, param_space, param_groups
            )
        
        # Combine objectives using specified method
        if self.scalarization_method == 'weighted_sum':
            self._apply_weighted_sum(Q, objectives)
        elif self.scalarization_method == 'epsilon_constraint':
            self._apply_epsilon_constraint(Q, objectives)
        elif self.scalarization_method == 'tchebycheff':
            self._apply_tchebycheff(Q, objectives)
        
        self.last_qubo = Q
        return Q
    
    def _build_objective_terms(self, objective: str, param_space: Dict[str, List[Any]],
                              param_groups: Dict[str, List[int]]) -> Dict[Tuple[int, int], float]:
        """Build QUBO terms for a specific objective."""
        
        terms = {}
        
        if objective == 'accuracy':
            # Terms that encourage accuracy-improving parameter choices
            for param, values in param_space.items():
                var_indices = param_groups[param]
                
                for i, var_idx in enumerate(var_indices):
                    value = values[i]
                    accuracy_score = self._calculate_accuracy_score(param, value)
                    
                    if accuracy_score != 0:
                        terms[(var_idx, var_idx)] = -accuracy_score  # Negative for maximization
        
        elif objective == 'speed' or objective == 'efficiency':
            # Terms that encourage speed/efficiency
            for param, values in param_space.items():
                var_indices = param_groups[param]
                
                for i, var_idx in enumerate(var_indices):
                    value = values[i]
                    speed_score = self._calculate_speed_score(param, value)
                    
                    if speed_score != 0:
                        terms[(var_idx, var_idx)] = -speed_score
        
        elif objective == 'memory':
            # Terms that encourage memory efficiency
            for param, values in param_space.items():
                var_indices = param_groups[param]
                
                for i, var_idx in enumerate(var_indices):
                    value = values[i]
                    memory_score = self._calculate_memory_score(param, value)
                    
                    if memory_score != 0:
                        terms[(var_idx, var_idx)] = -memory_score
        
        elif objective == 'interpretability':
            # Terms that encourage interpretable models
            for param, values in param_space.items():
                var_indices = param_groups[param]
                
                for i, var_idx in enumerate(var_indices):
                    value = values[i]
                    interpretability_score = self._calculate_interpretability_score(param, value)
                    
                    if interpretability_score != 0:
                        terms[(var_idx, var_idx)] = -interpretability_score
        
        return terms
    
    def _calculate_accuracy_score(self, param: str, value: Any) -> float:
        """Calculate accuracy contribution score for a parameter value."""
        score = 0.0
        
        if param == 'n_estimators' and isinstance(value, int):
            # More estimators generally improve accuracy (with diminishing returns)
            score = np.log(value + 1) / 10
        elif param == 'max_depth' and isinstance(value, int):
            # Moderate depth is often optimal
            if 5 <= value <= 15:
                score = 1.0 - abs(value - 10) / 10
        elif param == 'min_samples_split' and isinstance(value, int):
            # Lower values allow more flexibility (but risk overfitting)
            score = 0.5 / max(value, 1)
        elif param == 'learning_rate' and isinstance(value, (int, float)):
            # Moderate learning rates often work best
            if 0.01 <= value <= 0.1:
                score = 0.5
        elif param == 'C' and isinstance(value, (int, float)):
            # Regularization parameter - moderate values often good
            if 0.1 <= value <= 10:
                score = 0.3
        
        return score
    
    def _calculate_speed_score(self, param: str, value: Any) -> float:
        """Calculate speed/efficiency score for a parameter value."""
        score = 0.0
        
        if param == 'n_estimators' and isinstance(value, int):
            # Fewer estimators are faster
            score = 1.0 / max(value, 1)
        elif param == 'max_depth' and isinstance(value, int):
            # Shallower trees are faster
            score = 1.0 / max(value, 1)
        elif param == 'max_features' and isinstance(value, str):
            # Feature selection methods can speed up training
            if value in ['sqrt', 'log2']:
                score = 0.5
        elif param == 'min_samples_split' and isinstance(value, int):
            # Higher values create simpler trees (faster)
            score = np.log(value + 1) / 5
        
        return score
    
    def _calculate_memory_score(self, param: str, value: Any) -> float:
        """Calculate memory efficiency score for a parameter value."""
        score = 0.0
        
        if param == 'n_estimators' and isinstance(value, int):
            # Fewer estimators use less memory
            score = 1.0 / max(value, 1)
        elif param == 'max_depth' and isinstance(value, int):
            # Shallower trees use less memory
            score = 1.0 / max(value, 1)
        elif param == 'max_features' and isinstance(value, str):
            if value in ['sqrt', 'log2']:
                score = 0.8  # Feature selection reduces memory
        
        return score
    
    def _calculate_interpretability_score(self, param: str, value: Any) -> float:
        """Calculate interpretability score for a parameter value."""
        score = 0.0
        
        if param == 'max_depth' and isinstance(value, int):
            # Shallower trees are more interpretable
            if value <= 5:
                score = (6 - value) / 5
        elif param == 'n_estimators' and isinstance(value, int):
            # Fewer trees are more interpretable
            if value <= 10:
                score = (11 - value) / 10
        elif param == 'min_samples_leaf' and isinstance(value, int):
            # Higher values create simpler, more interpretable models
            score = np.log(value + 1) / 3
        
        return score
    
    def _apply_weighted_sum(self, Q: Dict[Tuple[int, int], float], objectives: List[str]):
        """Apply weighted sum scalarization."""
        
        # Get weights
        total_weight = sum(self.objective_weights.get(obj, 1.0) for obj in objectives)
        
        for objective in objectives:
            weight = self.objective_weights.get(objective, 1.0) / total_weight
            objective_terms = self.objective_terms[objective]
            
            for (i, j), coeff in objective_terms.items():
                Q[(i, j)] = Q.get((i, j), 0) + weight * coeff
    
    def _apply_epsilon_constraint(self, Q: Dict[Tuple[int, int], float], objectives: List[str]):
        """Apply epsilon-constraint method."""
        
        if not objectives:
            return
            
        # Use first objective as primary, others as constraints
        primary_objective = objectives[0]
        constraint_objectives = objectives[1:]
        
        # Apply primary objective
        primary_terms = self.objective_terms[primary_objective]
        for (i, j), coeff in primary_terms.items():
            Q[(i, j)] = Q.get((i, j), 0) + coeff
        
        # Apply constraint objectives with high penalties
        constraint_penalty = 10.0
        
        for constraint_obj in constraint_objectives:
            constraint_terms = self.objective_terms[constraint_obj]
            epsilon = self.objective_weights.get(f"{constraint_obj}_epsilon", 0.1)
            
            # Convert constraint to penalty terms (simplified)
            for (i, j), coeff in constraint_terms.items():
                # Penalize solutions that violate the constraint
                Q[(i, j)] = Q.get((i, j), 0) + constraint_penalty * max(0, -coeff - epsilon)
    
    def _apply_tchebycheff(self, Q: Dict[Tuple[int, int], float], objectives: List[str]):
        """Apply Tchebycheff scalarization method."""
        
        # For simplicity, implement a linearized version of Tchebycheff
        # In practice, this would require additional auxiliary variables
        
        weights = {}
        for objective in objectives:
            weights[objective] = self.objective_weights.get(objective, 1.0)
        
        max_weight = max(weights.values()) if weights else 1.0
        
        for objective in objectives:
            weight = weights[objective] / max_weight
            objective_terms = self.objective_terms[objective]
            
            # Apply weighted terms with emphasis on worst-case performance
            for (i, j), coeff in objective_terms.items():
                Q[(i, j)] = Q.get((i, j), 0) + weight * coeff * 1.5  # Amplify for min-max behavior
    
    def decode(self, sample: Dict[int, int], 
               param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Decode QUBO sample to parameter values."""
        decoded_params = {}
        
        # Group variables by parameter
        param_vars = {}
        for var_idx, assignment in sample.items():
            if var_idx in self.reverse_mapping and assignment == 1:
                param, value_idx, value = self.reverse_mapping[var_idx]
                if param not in param_vars:
                    param_vars[param] = []
                param_vars[param].append((value_idx, value))
        
        # Select one value per parameter
        for param in param_space.keys():
            if param in param_vars and param_vars[param]:
                # If multiple values selected, choose the first one
                _, value = param_vars[param][0]
                decoded_params[param] = value
            else:
                # Fallback to first option
                decoded_params[param] = param_space[param][0]
        
        return decoded_params
    
    def get_metrics(self) -> EncodingMetrics:
        """Get multi-objective encoding metrics."""
        if not self.last_qubo:
            return EncodingMetrics(0, 0, 0, 0)
        
        total_terms = len(self.last_qubo)
        
        # Calculate sparsity
        zero_terms = sum(1 for v in self.last_qubo.values() if abs(v) < 1e-10)
        sparsity = zero_terms / max(total_terms, 1)
        
        # Calculate connectivity
        variables = set()
        for (i, j) in self.last_qubo.keys():
            variables.add(i)
            variables.add(j)
        
        num_vars = len(variables)
        connectivity = 2 * total_terms / max(num_vars, 1) if num_vars > 0 else 0
        
        # Calculate penalty balance
        positive_terms = sum(1 for v in self.last_qubo.values() if v > 0)
        penalty_balance = positive_terms / max(total_terms, 1)
        
        # Calculate embedding efficiency
        num_logical_params = len(set(param for param, _, _ in self.reverse_mapping.values())) if self.reverse_mapping else 1
        embedding_efficiency = num_vars / num_logical_params if num_logical_params > 0 else 0
        
        return EncodingMetrics(
            sparsity_ratio=sparsity,
            connectivity_degree=connectivity,
            penalty_balance=penalty_balance,
            embedding_efficiency=embedding_efficiency
        )