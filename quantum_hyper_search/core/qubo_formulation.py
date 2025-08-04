"""
QUBO formulation for hyperparameter optimization problems.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from itertools import product


class QUBOEncoder:
    """
    Encodes hyperparameter optimization problems as QUBO (Quadratic Unconstrained Binary Optimization).
    """
    
    def __init__(
        self,
        encoding: str = "one_hot",
        penalty_strength: float = 2.0,
        regularization: float = 0.01
    ):
        """
        Initialize QUBO encoder.
        
        Args:
            encoding: Encoding method ('one_hot', 'binary', 'domain_wall')
            penalty_strength: Strength of constraint penalties
            regularization: L2 regularization strength for stability
        """
        self.encoding = encoding
        self.penalty_strength = penalty_strength
        self.regularization = regularization
        
        if encoding not in ["one_hot", "binary", "domain_wall"]:
            raise ValueError(f"Unknown encoding: {encoding}")
    
    def encode(
        self,
        search_space: Dict[str, List[Any]],
        objective_estimates: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict] = None
    ) -> Tuple[np.ndarray, float, Dict[str, int]]:
        """
        Encode hyperparameter search space as QUBO.
        
        Args:
            search_space: Dictionary mapping parameter names to possible values
            objective_estimates: Preliminary objective estimates for bias
            constraints: Constraint specifications
            
        Returns:
            Tuple of (Q_matrix, offset, variable_mapping)
        """
        print(f"Encoding search space with {self.encoding} encoding...")
        
        # Create variable mapping
        variable_map = self._create_variable_mapping(search_space)
        n_vars = len(variable_map)
        
        # Initialize QUBO matrix
        Q = np.zeros((n_vars, n_vars))
        offset = 0.0
        
        # Add objective bias
        if objective_estimates:
            Q, offset = self._add_objective_bias(Q, offset, variable_map, search_space, objective_estimates)
        
        # Add constraint penalties
        if constraints:
            Q, offset = self._add_constraint_penalties(Q, offset, variable_map, search_space, constraints)
        
        # Add one-hot constraints for categorical parameters
        Q, offset = self._add_one_hot_constraints(Q, offset, variable_map, search_space)
        
        # Add regularization for numerical stability
        Q += self.regularization * np.eye(n_vars)
        
        print(f"QUBO encoding complete: {n_vars} variables, density: {np.count_nonzero(Q) / (n_vars**2):.3f}")
        
        return Q, offset, variable_map
    
    def _create_variable_mapping(self, search_space: Dict[str, List[Any]]) -> Dict[str, int]:
        """Create mapping from variable names to indices."""
        variable_map = {}
        var_idx = 0
        
        for param_name, param_values in search_space.items():
            if self.encoding == "one_hot":
                for i, value in enumerate(param_values):
                    var_name = f"{param_name}_{i}"
                    variable_map[var_name] = var_idx
                    var_idx += 1
            elif self.encoding == "binary":
                n_bits = int(np.ceil(np.log2(len(param_values))))
                for bit in range(n_bits):
                    var_name = f"{param_name}_bit_{bit}"
                    variable_map[var_name] = var_idx
                    var_idx += 1
            else:  # domain_wall
                for i in range(len(param_values)):
                    var_name = f"{param_name}_wall_{i}"
                    variable_map[var_name] = var_idx
                    var_idx += 1
        
        return variable_map
    
    def _add_objective_bias(
        self,
        Q: np.ndarray,
        offset: float,
        variable_map: Dict[str, int],
        search_space: Dict[str, List[Any]],
        objective_estimates: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """Add objective function bias to QUBO."""
        if not objective_estimates:
            return Q, offset
            
        # Convert objective estimates to bias terms
        for param_config_str, score in objective_estimates.items():
            try:
                # Parse parameter configuration
                param_config = dict(eval(param_config_str))
                
                # Add bias for this configuration
                bias_strength = -score  # Negative because we want to maximize
                
                for param_name, param_value in param_config.items():
                    if param_name in search_space:
                        param_idx = search_space[param_name].index(param_value)
                        var_name = f"{param_name}_{param_idx}"
                        if var_name in variable_map:
                            var_idx = variable_map[var_name]
                            Q[var_idx, var_idx] += bias_strength * 0.1
                            
            except Exception as e:
                print(f"Warning: Could not parse objective estimate - {e}")
                continue
        
        return Q, offset
    
    def _add_constraint_penalties(
        self,
        Q: np.ndarray,
        offset: float,
        variable_map: Dict[str, int],
        search_space: Dict[str, List[Any]],
        constraints: Dict
    ) -> Tuple[np.ndarray, float]:
        """Add constraint penalties to QUBO."""
        penalty = self.penalty_strength
        
        # Mutual exclusion constraints
        if "mutual_exclusion" in constraints:
            for exclusion_group in constraints["mutual_exclusion"]:
                for i, var1 in enumerate(exclusion_group):
                    for j, var2 in enumerate(exclusion_group):
                        if i < j and var1 in variable_map and var2 in variable_map:
                            idx1, idx2 = variable_map[var1], variable_map[var2]
                            Q[idx1, idx2] += penalty
                            Q[idx2, idx1] += penalty
        
        # Conditional constraints (if A then B)
        if "conditional" in constraints:
            for condition, consequence in constraints["conditional"]:
                if condition in variable_map and consequence in variable_map:
                    cond_idx = variable_map[condition]
                    cons_idx = variable_map[consequence]
                    # Penalty for A=1, B=0
                    Q[cond_idx, cond_idx] += penalty
                    Q[cons_idx, cons_idx] += penalty
                    Q[cond_idx, cons_idx] -= penalty
                    Q[cons_idx, cond_idx] -= penalty
        
        return Q, offset
    
    def _add_one_hot_constraints(
        self,
        Q: np.ndarray,
        offset: float,
        variable_map: Dict[str, int],
        search_space: Dict[str, List[Any]]
    ) -> Tuple[np.ndarray, float]:
        """Add one-hot constraints for categorical parameters."""
        if self.encoding != "one_hot":
            return Q, offset
            
        penalty = self.penalty_strength
        
        for param_name, param_values in search_space.items():
            if len(param_values) <= 1:
                continue
                
            # Get variable indices for this parameter
            var_indices = []
            for i, value in enumerate(param_values):
                var_name = f"{param_name}_{i}"
                if var_name in variable_map:
                    var_indices.append(variable_map[var_name])
            
            if len(var_indices) <= 1:
                continue
            
            # Add penalty for not selecting exactly one
            # (sum_i x_i - 1)^2 = sum_i x_i^2 + sum_{i!=j} x_i*x_j - 2*sum_i x_i + 1
            
            # Diagonal terms: x_i^2 = x_i (since x_i is binary)
            for idx in var_indices:
                Q[idx, idx] += penalty - 2 * penalty
            
            # Off-diagonal terms: x_i * x_j
            for i, idx1 in enumerate(var_indices):
                for j, idx2 in enumerate(var_indices):
                    if i < j:
                        Q[idx1, idx2] += 2 * penalty
                        Q[idx2, idx1] += 2 * penalty
            
            # Constant term
            offset += penalty
        
        return Q, offset
    
    def decode_sample(
        self,
        sample: Dict[int, int],
        variable_map: Dict[str, int],
        search_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Decode binary sample to parameter configuration.
        
        Args:
            sample: Binary variable assignments
            variable_map: Mapping from variable names to indices
            search_space: Original search space
            
        Returns:
            Parameter configuration dictionary
        """
        # Reverse variable mapping
        idx_to_var = {idx: var for var, idx in variable_map.items()}
        
        params = {}
        
        for param_name, param_values in search_space.items():
            if self.encoding == "one_hot":
                selected_idx = None
                max_value = -1
                
                for i, value in enumerate(param_values):
                    var_name = f"{param_name}_{i}"
                    if var_name in variable_map:
                        var_idx = variable_map[var_name]
                        if var_idx in sample and sample[var_idx] > max_value:
                            max_value = sample[var_idx]
                            selected_idx = i
                
                if selected_idx is not None:
                    params[param_name] = param_values[selected_idx]
                else:
                    # Fallback to random selection
                    params[param_name] = np.random.choice(param_values)
                    
            elif self.encoding == "binary":
                # Decode binary representation
                binary_value = 0
                n_bits = int(np.ceil(np.log2(len(param_values))))
                
                for bit in range(n_bits):
                    var_name = f"{param_name}_bit_{bit}"
                    if var_name in variable_map:
                        var_idx = variable_map[var_name]
                        if var_idx in sample and sample[var_idx] == 1:
                            binary_value += 2 ** bit
                
                # Clamp to valid range
                binary_value = min(binary_value, len(param_values) - 1)
                params[param_name] = param_values[binary_value]
                
            else:  # domain_wall
                # Find domain wall position
                wall_pos = 0
                for i in range(len(param_values)):
                    var_name = f"{param_name}_wall_{i}"
                    if var_name in variable_map:
                        var_idx = variable_map[var_name]
                        if var_idx in sample and sample[var_idx] == 1:
                            wall_pos = i + 1
                            break
                
                wall_pos = min(wall_pos, len(param_values) - 1)
                params[param_name] = param_values[wall_pos]
        
        return params