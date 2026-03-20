"""
Optuna-compatible interface for QAHS.
Does not require optuna to be installed.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .qubo import QUBOFormulation
from .sa_solver import SimulatedAnnealingSearcher


class QAHSOptunaSampler:
    """
    Optuna-compatible hyperparameter sampler using QUBO/SA.

    This sampler builds a QUBO encoding for each trial's parameter space,
    solves it with Simulated Annealing to decode candidate parameters,
    evaluates the objective, and returns the best found configuration.

    Args:
        sa_solver: Optional pre-configured SimulatedAnnealingSearcher
        seed: Random seed
    """

    def __init__(
        self,
        sa_solver: Optional[SimulatedAnnealingSearcher] = None,
        seed: Optional[int] = None,
    ):
        self.params_config: List[dict] = []
        self.sa_solver = sa_solver or SimulatedAnnealingSearcher(seed=seed)
        self._qubo = QUBOFormulation()
        self._seed = seed

    def _get_or_add_param(self, name: str, ptype: str, low, high, n_bits: int = 4):
        """Register a parameter if not already registered."""
        for cfg in self.params_config:
            if cfg["name"] == name:
                return cfg
        cfg = {"name": name, "type": ptype, "low": low, "high": high}
        if ptype == "float":
            cfg["n_bits"] = n_bits
        self.params_config.append(cfg)
        return cfg

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """
        Suggest an integer hyperparameter.

        Args:
            name: Parameter name
            low: Minimum value (inclusive)
            high: Maximum value (inclusive)

        Returns:
            Suggested integer value
        """
        cfg = self._get_or_add_param(name, "int", low, high)
        Q, _ = self._qubo.build_qubo_matrix([cfg])
        # Add random noise to encourage exploration
        n = Q.shape[0]
        rng = np.random.default_rng(self._seed)
        Q_noisy = Q + rng.uniform(-0.5, 0.5, size=(n, n))
        Q_noisy = (Q_noisy + Q_noisy.T) / 2  # symmetrize
        bits = self.sa_solver.solve(Q_noisy)
        result = self._qubo.decode_solution(bits, [cfg])
        return result[name]

    def suggest_float(self, name: str, low: float, high: float, n_bits: int = 4) -> float:
        """
        Suggest a float hyperparameter.

        Args:
            name: Parameter name
            low: Minimum value
            high: Maximum value
            n_bits: Discretization bits

        Returns:
            Suggested float value
        """
        cfg = self._get_or_add_param(name, "float", low, high, n_bits)
        Q, _ = self._qubo.build_qubo_matrix([cfg])
        n = Q.shape[0]
        rng = np.random.default_rng(self._seed)
        Q_noisy = Q + rng.uniform(-0.5, 0.5, size=(n, n))
        Q_noisy = (Q_noisy + Q_noisy.T) / 2
        bits = self.sa_solver.solve(Q_noisy)
        result = self._qubo.decode_solution(bits, [cfg])
        return result[name]

    def sample(
        self,
        objective_fn: Callable[[dict], float],
        n_trials: int = 20,
    ) -> Tuple[dict, float]:
        """
        Run SA-based HPO.

        For each trial:
          1. Build QUBO matrix from registered params_config (or uses all registered params)
          2. Solve with SA to get a bit string
          3. Decode bits to parameter dict
          4. Evaluate objective_fn(params)
          5. Track best result

        Args:
            objective_fn: Function that takes a dict of params and returns a float score
                          (lower is better)
            n_trials: Number of optimization trials

        Returns:
            (best_params, best_value): Best parameters found and their objective value
        """
        best_params: Optional[dict] = None
        best_value = float("inf")

        for trial in range(n_trials):
            if not self.params_config:
                # No params registered — return empty
                return {}, float("inf")

            # Build QUBO with random perturbation for exploration
            Q, var_names = self._qubo.build_qubo_matrix(self.params_config)
            n = Q.shape[0]

            if n > 0:
                rng = np.random.default_rng(
                    None if self._seed is None else self._seed + trial
                )
                # Add random diagonal noise to encourage diverse bit patterns
                noise = rng.uniform(-1.0, 1.0, size=n)
                Q_trial = Q.copy()
                Q_trial += np.diag(noise)
            else:
                Q_trial = Q

            bits = self.sa_solver.solve(Q_trial)
            params = self._qubo.decode_solution(bits, self.params_config)

            try:
                value = objective_fn(params)
            except Exception:
                continue

            if value < best_value:
                best_value = value
                best_params = params.copy()

        if best_params is None:
            best_params = {}

        return best_params, best_value
