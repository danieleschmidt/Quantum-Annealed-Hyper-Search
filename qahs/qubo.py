"""
QUBO (Quadratic Unconstrained Binary Optimization) formulation for hyperparameter encoding.
"""

import math
import numpy as np
from typing import Dict, List, Tuple


class QUBOFormulation:
    """
    Encodes hyperparameters as binary variables for QUBO optimization.
    Supports integer and continuous parameter types.
    """

    def encode_integer(self, name: str, low: int, high: int) -> Dict[str, int]:
        """
        Encode an integer parameter as binary variables.

        Args:
            name: Parameter name
            low: Minimum integer value (inclusive)
            high: Maximum integer value (inclusive)

        Returns:
            Dict mapping '{name}_bit_{i}' -> bit index
        """
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
        n_values = high - low + 1
        n_bits = max(1, math.ceil(math.log2(n_values))) if n_values > 1 else 1
        return {f"{name}_bit_{i}": i for i in range(n_bits)}

    def encode_continuous(
        self, name: str, low: float, high: float, n_bits: int = 4
    ) -> Dict[str, int]:
        """
        Encode a continuous parameter by discretization into n_bits binary variables.

        Args:
            name: Parameter name
            low: Minimum float value
            high: Maximum float value
            n_bits: Number of bits (discretization levels = 2^n_bits)

        Returns:
            Dict mapping '{name}_bit_{i}' -> bit index
        """
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
        if n_bits < 1:
            raise ValueError(f"n_bits ({n_bits}) must be >= 1")
        return {f"{name}_bit_{i}": i for i in range(n_bits)}

    def build_qubo_matrix(
        self, params_config: List[dict]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a QUBO matrix Q for the given parameter configuration.

        The matrix is diagonal with -1 entries (encoding encourages each bit to be
        set independently). Off-diagonal terms can encode constraints.

        Args:
            params_config: List of dicts with keys:
                - 'name': str
                - 'type': 'int' or 'float'
                - 'low': numeric
                - 'high': numeric
                - 'n_bits': int (optional, for float type, default 4)

        Returns:
            (Q, var_names): QUBO matrix (n x n) and list of variable names
        """
        var_names = []
        for cfg in params_config:
            name = cfg["name"]
            ptype = cfg["type"]
            low = cfg["low"]
            high = cfg["high"]
            if ptype == "int":
                mapping = self.encode_integer(name, int(low), int(high))
            elif ptype == "float":
                n_bits = cfg.get("n_bits", 4)
                mapping = self.encode_continuous(name, low, high, n_bits)
            else:
                raise ValueError(f"Unknown type: {ptype}. Use 'int' or 'float'.")
            var_names.extend(mapping.keys())

        n = len(var_names)
        Q = np.zeros((n, n), dtype=float)

        # Diagonal: slight negative bias to allow free bit assignment
        # (neutral objective — optimization is driven by the objective function)
        np.fill_diagonal(Q, 0.0)

        return Q, var_names

    def decode_solution(self, bits: np.ndarray, params_config: List[dict]) -> dict:
        """
        Decode a binary bit string back to parameter values.

        Args:
            bits: Binary array of shape (n_vars,)
            params_config: Same config as used for build_qubo_matrix

        Returns:
            Dict mapping parameter name -> decoded value
        """
        result = {}
        bit_idx = 0

        for cfg in params_config:
            name = cfg["name"]
            ptype = cfg["type"]
            low = cfg["low"]
            high = cfg["high"]

            if ptype == "int":
                low_i = int(low)
                high_i = int(high)
                n_values = high_i - low_i + 1
                n_bits = max(1, math.ceil(math.log2(n_values))) if n_values > 1 else 1
                param_bits = bits[bit_idx: bit_idx + n_bits]
                bit_idx += n_bits
                # Decode as binary integer
                decoded = sum(int(b) * (2 ** i) for i, b in enumerate(param_bits))
                # Clamp to valid range
                decoded = min(decoded, n_values - 1)
                result[name] = decoded + low_i

            elif ptype == "float":
                n_bits = cfg.get("n_bits", 4)
                param_bits = bits[bit_idx: bit_idx + n_bits]
                bit_idx += n_bits
                # Decode as binary integer then map to [low, high]
                max_val = (2 ** n_bits) - 1
                decoded_int = sum(int(b) * (2 ** i) for i, b in enumerate(param_bits))
                if max_val == 0:
                    result[name] = float(low)
                else:
                    result[name] = low + (high - low) * decoded_int / max_val

            else:
                raise ValueError(f"Unknown type: {ptype}")

        return result
