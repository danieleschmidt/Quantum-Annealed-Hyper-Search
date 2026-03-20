"""
Simulated Annealing solver for QUBO problems.
"""

import math
import numpy as np
from typing import Optional


class SimulatedAnnealingSearcher:
    """
    Minimizes x^T Q x over binary vectors x using Simulated Annealing.

    Args:
        n_steps: Number of SA iterations (default 1000)
        T_init: Initial temperature (default 10.0)
        T_final: Final temperature (default 0.01)
        cooling: Cooling schedule — 'exponential' or 'linear' (default 'exponential')
        seed: Random seed for reproducibility (default None)
    """

    def __init__(
        self,
        n_steps: int = 1000,
        T_init: float = 10.0,
        T_final: float = 0.01,
        cooling: str = "exponential",
        seed: Optional[int] = None,
    ):
        self.n_steps = n_steps
        self.T_init = T_init
        self.T_final = T_final
        self.cooling = cooling
        self._rng = np.random.default_rng(seed)

    def temperature_schedule(self, step: int) -> float:
        """
        Compute temperature at a given step.

        Args:
            step: Current step (0-indexed)

        Returns:
            Temperature value
        """
        if self.n_steps <= 1:
            return self.T_final

        progress = step / (self.n_steps - 1)  # 0 -> 1

        if self.cooling == "exponential":
            # T(t) = T_init * (T_final/T_init)^(step/(n_steps-1))
            ratio = self.T_final / max(self.T_init, 1e-10)
            return self.T_init * (ratio ** progress)
        elif self.cooling == "linear":
            return self.T_init + (self.T_final - self.T_init) * progress
        else:
            raise ValueError(f"Unknown cooling schedule: {self.cooling}. Use 'exponential' or 'linear'.")

    def _energy(self, x: np.ndarray, Q: np.ndarray) -> float:
        """Compute QUBO energy: E = x^T Q x"""
        return float(x @ Q @ x)

    def solve(self, Q: np.ndarray) -> np.ndarray:
        """
        Minimize x^T Q x over binary x using Simulated Annealing.

        Args:
            Q: QUBO matrix of shape (n, n)

        Returns:
            Binary array x of shape (n,) that approximately minimizes the objective
        """
        n = Q.shape[0]
        if n == 0:
            return np.array([], dtype=int)

        # Start with random binary vector
        x = self._rng.integers(0, 2, size=n).astype(float)
        energy = self._energy(x, Q)

        best_x = x.copy()
        best_energy = energy

        for step in range(self.n_steps):
            T = self.temperature_schedule(step)

            # Pick a random bit to flip
            flip_idx = int(self._rng.integers(0, n))
            x_new = x.copy()
            x_new[flip_idx] = 1.0 - x_new[flip_idx]

            new_energy = self._energy(x_new, Q)
            delta = new_energy - energy

            # Metropolis acceptance criterion
            if delta < 0 or (T > 1e-10 and self._rng.random() < math.exp(-delta / T)):
                x = x_new
                energy = new_energy

                if energy < best_energy:
                    best_energy = energy
                    best_x = x.copy()

        return best_x.astype(int)
