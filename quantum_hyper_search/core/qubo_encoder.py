"""
QUBO (Quadratic Unconstrained Binary Optimization) encoder for hyperparameter spaces.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QUBOEncoder:
    """
    Encodes hyperparameter optimization problems as QUBO matrices.
    
    Transforms discrete hyperparameter search spaces into binary quadratic
    optimization problems suitable for quantum annealing.
    """
    
    def __init__(
        self,
        encoding: str = 'one_hot',
        penalty_strength: float = 2.0,
        use_performance_bias: bool = True
    ):
        """
        Initialize QUBO encoder.
        
        Args:
            encoding: Encoding method ('one_hot', 'binary', 'domain_wall')
            penalty_strength: Strength of constraint penalty terms
            use_performance_bias: Use historical performance to bias QUBO
        """
        self.encoding = encoding
        self.penalty_strength = penalty_strength
        self.use_performance_bias = use_performance_bias
        
        if encoding not in ['one_hot', 'binary', 'domain_wall']:
            raise ValueError(f"Unsupported encoding: {encoding}")
    
    def encode_search_space(
        self,
        param_space: Dict[str, List[Any]],
        history: Optional[Any] = None
    ) -> Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]]:
        """
        Encode hyperparameter search space as QUBO matrix.
        
        Args:
            param_space: Dictionary of parameter names and possible values
            history: Optional optimization history for bias
            
        Returns:
            Tuple of (QUBO_matrix, offset, variable_mapping)
        """
        if self.encoding == 'one_hot':
            return self._encode_one_hot(param_space, history)
        elif self.encoding == 'binary':
            return self._encode_binary(param_space, history)
        elif self.encoding == 'domain_wall':
            return self._encode_domain_wall(param_space, history)
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")
    
    def _encode_one_hot(
        self,
        param_space: Dict[str, List[Any]],
        history: Optional[Any] = None
    ) -> Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]]:
        """
        One-hot encoding: each parameter value gets one binary variable.
        """
        Q = {}
        variable_map = {}
        var_idx = 0
        
        # Create variables for each parameter value
        for param_name, param_values in param_space.items():
            param_vars = []
            
            for i, value in enumerate(param_values):
                variable_map[var_idx] = (param_name, value)
                param_vars.append(var_idx)
                var_idx += 1
            
            # Add constraint: exactly one value per parameter
            # Constraint: (sum_i x_i - 1)^2 = sum_i x_i^2 + 2*sum_{i<j} x_i*x_j - 2*sum_i x_i + 1
            # QUBO terms: diagonal gets (1 - 2), off-diagonal gets 2*penalty
            
            for i in range(len(param_vars)):
                var_i = param_vars[i]
                Q[(var_i, var_i)] = Q.get((var_i, var_i), 0.0) + self.penalty_strength * (1 - 2)
                
                for j in range(i + 1, len(param_vars)):
                    var_j = param_vars[j]
                    key = (min(var_i, var_j), max(var_i, var_j))
                    Q[key] = Q.get(key, 0.0) + 2 * self.penalty_strength
        
        # Add performance bias if available
        if self.use_performance_bias and history and hasattr(history, 'trials'):
            self._add_performance_bias(Q, variable_map, history)
        
        offset = len(param_space) * self.penalty_strength  # Constant term from constraints
        return Q, offset, variable_map
    
    def _encode_binary(
        self,
        param_space: Dict[str, List[Any]],
        history: Optional[Any] = None
    ) -> Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]]:
        """
        Binary encoding: use log2(n) binary variables per parameter.
        """
        Q = {}
        variable_map = {}
        var_idx = 0
        
        for param_name, param_values in param_space.items():
            n_values = len(param_values)
            n_bits = int(np.ceil(np.log2(max(2, n_values))))
            
            param_vars = []
            for bit in range(n_bits):
                variable_map[var_idx] = (param_name, f'bit_{bit}')
                param_vars.append(var_idx)
                var_idx += 1
            
            # Add constraints to ensure valid encodings
            # This is a simplified version - full implementation would need
            # more sophisticated constraint handling
            
            for i, var_i in enumerate(param_vars):
                Q[(var_i, var_i)] = Q.get((var_i, var_i), 0.0) - 0.1  # Small bias
        
        return Q, 0.0, variable_map
    
    def _encode_domain_wall(
        self,
        param_space: Dict[str, List[Any]],
        history: Optional[Any] = None
    ) -> Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]]:
        """
        Domain wall encoding: use n-1 binary variables per parameter.
        """
        Q = {}
        variable_map = {}
        var_idx = 0
        
        for param_name, param_values in param_space.items():
            n_values = len(param_values)
            param_vars = []
            
            # Create n-1 variables for n values
            for i in range(n_values - 1):
                variable_map[var_idx] = (param_name, f'wall_{i}')
                param_vars.append(var_idx)
                var_idx += 1
            
            # Add ordering constraints: x_i >= x_{i+1}
            for i in range(len(param_vars) - 1):
                var_i = param_vars[i]
                var_j = param_vars[i + 1]
                
                # Constraint: x_i - x_j >= 0, implemented as penalty for x_j > x_i
                Q[(var_j, var_j)] = Q.get((var_j, var_j), 0.0) + self.penalty_strength
                key = (min(var_i, var_j), max(var_i, var_j))
                Q[key] = Q.get(key, 0.0) - self.penalty_strength
        
        return Q, 0.0, variable_map
    
    def _add_performance_bias(
        self,
        Q: Dict[Tuple[int, int], float],
        variable_map: Dict[int, Tuple[str, Any]],
        history: Any
    ) -> None:
        """
        Add bias based on historical performance.
        """
        if not hasattr(history, 'trials') or not history.trials:
            return
        
        # Calculate average scores for each parameter value
        param_scores = {}
        
        for trial, score in zip(history.trials, history.scores):
            for param_name, param_value in trial.items():
                key = (param_name, param_value)
                if key not in param_scores:
                    param_scores[key] = []
                param_scores[key].append(score)
        
        # Calculate average scores
        param_avg_scores = {}
        for key, scores in param_scores.items():
            param_avg_scores[key] = np.mean(scores)
        
        # Apply bias to QUBO diagonal terms
        bias_strength = 0.1  # Small bias to not override constraints
        
        for var_idx, (param_name, param_value) in variable_map.items():
            key = (param_name, param_value)
            if key in param_avg_scores:
                # Higher scores get negative bias (encourage selection)
                bias = -bias_strength * param_avg_scores[key]
                Q[(var_idx, var_idx)] = Q.get((var_idx, var_idx), 0.0) + bias
    
    def decode_sample(
        self,
        sample: Dict[int, int],
        variable_map: Dict[int, Tuple[str, Any]],
        param_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Decode binary sample back to parameter values.
        
        Args:
            sample: Binary variable assignments
            variable_map: Mapping from variable indices to (param_name, param_value)
            param_space: Original parameter space
            
        Returns:
            Dictionary of parameter assignments
        """
        if self.encoding == 'one_hot':
            return self._decode_one_hot(sample, variable_map, param_space)
        elif self.encoding == 'binary':
            return self._decode_binary(sample, variable_map, param_space)
        elif self.encoding == 'domain_wall':
            return self._decode_domain_wall(sample, variable_map, param_space)
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")
    
    def _decode_one_hot(
        self,
        sample: Dict[int, int],
        variable_map: Dict[int, Tuple[str, Any]],
        param_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Decode one-hot encoded sample.
        """
        params = {}
        
        # Group variables by parameter
        param_vars = {}
        for var_idx, (param_name, param_value) in variable_map.items():
            if param_name not in param_vars:
                param_vars[param_name] = []
            param_vars[param_name].append((var_idx, param_value))
        
        # Select the activated variable for each parameter
        for param_name, var_list in param_vars.items():
            best_score = -1
            best_value = None
            
            for var_idx, param_value in var_list:
                activation = sample.get(var_idx, 0)
                if activation > best_score:
                    best_score = activation
                    best_value = param_value
            
            # Fallback to first value if no variable is activated
            if best_value is None and param_name in param_space:
                best_value = param_space[param_name][0]
            
            if best_value is not None:
                params[param_name] = best_value
        
        return params
    
    def _decode_binary(
        self,
        sample: Dict[int, int],
        variable_map: Dict[int, Tuple[str, Any]],
        param_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Decode binary encoded sample.
        """
        params = {}
        
        # Group variables by parameter
        param_vars = {}
        for var_idx, (param_name, bit_name) in variable_map.items():
            if param_name not in param_vars:
                param_vars[param_name] = []
            param_vars[param_name].append((var_idx, bit_name))
        
        for param_name, var_list in param_vars.items():
            # Convert binary representation to index
            binary_value = 0
            for var_idx, bit_name in sorted(var_list, key=lambda x: x[1]):
                bit_value = sample.get(var_idx, 0)
                bit_pos = int(bit_name.split('_')[1])
                binary_value += bit_value * (2 ** bit_pos)
            
            # Map to parameter value
            if param_name in param_space:
                param_values = param_space[param_name]
                if binary_value < len(param_values):
                    params[param_name] = param_values[binary_value]
                else:
                    # Fallback to first value
                    params[param_name] = param_values[0]
        
        return params
    
    def _decode_domain_wall(
        self,
        sample: Dict[int, int],
        variable_map: Dict[int, Tuple[str, Any]],
        param_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Decode domain wall encoded sample.
        """
        params = {}
        
        # Group variables by parameter
        param_vars = {}
        for var_idx, (param_name, wall_name) in variable_map.items():
            if param_name not in param_vars:
                param_vars[param_name] = []
            param_vars[param_name].append((var_idx, wall_name))
        
        for param_name, var_list in param_vars.items():
            # Count number of active walls
            active_walls = 0
            for var_idx, wall_name in var_list:
                if sample.get(var_idx, 0) == 1:
                    active_walls += 1
            
            # Map to parameter value
            if param_name in param_space:
                param_values = param_space[param_name]
                value_idx = min(active_walls, len(param_values) - 1)
                params[param_name] = param_values[value_idx]
        
        return params
    
    def estimate_qubo_size(self, param_space: Dict[str, List[Any]]) -> int:
        """
        Estimate the size of QUBO matrix for given parameter space.
        
        Args:
            param_space: Dictionary of parameter names and possible values
            
        Returns:
            Estimated number of binary variables
        """
        total_vars = 0
        
        for param_name, param_values in param_space.items():
            n_values = len(param_values)
            
            if self.encoding == 'one_hot':
                total_vars += n_values
            elif self.encoding == 'binary':
                total_vars += int(np.ceil(np.log2(max(2, n_values))))
            elif self.encoding == 'domain_wall':
                total_vars += max(1, n_values - 1)
        
        return total_vars
    
    def _encode_one_hot(
        self,
        param_space: Dict[str, List[Any]],
        history: Optional[Any] = None
    ) -> Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]]:
        """
        One-hot encoding: each parameter value gets a binary variable.
        Exactly one variable per parameter must be active.
        """
        Q = {}
        offset = 0.0
        param_mapping = {}
        var_idx = 0
        
        # Create binary variables for each parameter value
        param_var_ranges = {}
        
        for param_name, param_values in param_space.items():
            start_idx = var_idx
            param_var_ranges[param_name] = []
            
            for param_value in param_values:
                param_mapping[var_idx] = (param_name, param_value)
                param_var_ranges[param_name].append(var_idx)
                var_idx += 1
        
        # Add objective terms (bias toward better historical performance)
        if history and self.use_performance_bias and len(history.trials) > 0:
            self._add_performance_bias(Q, param_mapping, history)
        
        # Add one-hot constraints for each parameter
        for param_name, var_indices in param_var_ranges.items():
            self._add_one_hot_constraint(Q, var_indices)
        
        logger.info(f"Created QUBO with {len(param_mapping)} variables")
        logger.info(f"QUBO has {len(Q)} non-zero entries")
        
        return Q, offset, param_mapping
    
    def _add_performance_bias(
        self,
        Q: Dict[Tuple[int, int], float],
        param_mapping: Dict[int, Tuple[str, Any]],
        history: Any
    ) -> None:
        """Add bias terms based on historical performance."""
        if len(history.trials) < 2:
            return
            
        # Calculate performance statistics for each parameter value
        param_value_scores = {}
        
        for trial, score in zip(history.trials, history.scores):
            for param_name, param_value in trial.items():
                key = (param_name, param_value)
                if key not in param_value_scores:
                    param_value_scores[key] = []
                param_value_scores[key].append(score)
        
        # Add bias terms to encourage better performing parameter values
        max_score = max(history.scores)
        min_score = min(history.scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        for var_idx, (param_name, param_value) in param_mapping.items():
            key = (param_name, param_value)
            if key in param_value_scores:
                avg_score = np.mean(param_value_scores[key])
                # Normalize to [-1, 1] range and scale
                normalized_score = 2 * (avg_score - min_score) / score_range - 1
                bias = -0.5 * normalized_score  # Negative because we minimize QUBO
                Q[(var_idx, var_idx)] = Q.get((var_idx, var_idx), 0.0) + bias
    
    def _add_one_hot_constraint(
        self,
        Q: Dict[Tuple[int, int], float],
        var_indices: List[int]
    ) -> None:
        """
        Add one-hot constraint: exactly one variable must be 1.
        Penalty form: P * (1 - sum(x_i))^2 = P * (1 - 2*sum(x_i) + sum_i(x_i) + 2*sum_{i<j}(x_i * x_j))
        """
        n_vars = len(var_indices)
        
        if n_vars <= 1:
            return
        
        # Linear terms: -2P for each variable
        for var_idx in var_indices:
            Q[(var_idx, var_idx)] = Q.get((var_idx, var_idx), 0.0) + self.penalty_strength
        
        # Quadratic terms: 2P for each pair
        for i, var_i in enumerate(var_indices):
            for j, var_j in enumerate(var_indices):
                if i < j:
                    Q[(var_i, var_j)] = Q.get((var_i, var_j), 0.0) + 2 * self.penalty_strength
    
    def decode_solution(
        self,
        sample: Dict[int, int],
        param_mapping: Dict[int, Tuple[str, Any]]
    ) -> Dict[str, Any]:
        """
        Decode binary solution back to parameter values.
        
        Args:
            sample: Binary variable assignments
            param_mapping: Mapping from variable indices to (param_name, param_value)
            
        Returns:
            Dictionary of parameter assignments
        """
        params = {}
        
        # Group variables by parameter name
        param_groups = {}
        for var_idx, (param_name, param_value) in param_mapping.items():
            if param_name not in param_groups:
                param_groups[param_name] = []
            param_groups[param_name].append((var_idx, param_value))
        
        # For each parameter, find the active variable (should be exactly one)
        for param_name, var_list in param_groups.items():
            active_vars = [
                (var_idx, param_value) 
                for var_idx, param_value in var_list 
                if sample.get(var_idx, 0) == 1
            ]
            
            if len(active_vars) == 1:
                params[param_name] = active_vars[0][1]
            elif len(active_vars) == 0:
                # No variable active - choose first option as default
                params[param_name] = var_list[0][1]
                logger.warning(f"No active variable for {param_name}, using default")
            else:
                # Multiple variables active - choose first one
                params[param_name] = active_vars[0][1]
                logger.warning(f"Multiple active variables for {param_name}, using first")
        
        return params
    
    def estimate_qubo_size(self, param_space: Dict[str, List[Any]]) -> int:
        """
        Estimate the number of binary variables needed for the QUBO.
        
        Args:
            param_space: Dictionary mapping parameter names to possible values
            
        Returns:
            Estimated number of binary variables
        """
        if self.encoding == 'one_hot':
            return sum(len(values) for values in param_space.values())
        else:
            raise NotImplementedError(f"Size estimation for {self.encoding} not implemented")
    
    def validate_solution(
        self,
        sample: Dict[int, int],
        param_mapping: Dict[int, Tuple[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that a solution satisfies all constraints.
        
        Args:
            sample: Binary variable assignments
            param_mapping: Mapping from variable indices to (param_name, param_value)
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'violations': [],
            'parameter_counts': {}
        }
        
        # Group variables by parameter name
        param_groups = {}
        for var_idx, (param_name, param_value) in param_mapping.items():
            if param_name not in param_groups:
                param_groups[param_name] = []
            param_groups[param_name].append(var_idx)
        
        # Check one-hot constraints
        for param_name, var_indices in param_groups.items():
            active_count = sum(sample.get(var_idx, 0) for var_idx in var_indices)
            validation['parameter_counts'][param_name] = active_count
            
            if active_count != 1:
                validation['valid'] = False
                validation['violations'].append({
                    'type': 'one_hot_violation',
                    'parameter': param_name,
                    'active_count': active_count,
                    'expected': 1
                })
        
        return validation