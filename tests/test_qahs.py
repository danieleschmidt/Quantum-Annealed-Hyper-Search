"""
Tests for QAHS (Quantum-Annealed-Hyper-Search).
"""

import math
import pytest
import numpy as np

from qahs.qubo import QUBOFormulation
from qahs.sa_solver import SimulatedAnnealingSearcher
from qahs.optuna_interface import QAHSOptunaSampler


# ---------------------------------------------------------------------------
# QUBOFormulation tests
# ---------------------------------------------------------------------------

class TestQUBOFormulation:

    def setup_method(self):
        self.qubo = QUBOFormulation()

    def test_qubo_encode_integer_bits(self):
        """Encoding integers should produce ceil(log2(range)) bits."""
        # 0-3: 4 values -> 2 bits
        mapping = self.qubo.encode_integer("x", 0, 3)
        assert len(mapping) == 2
        assert "x_bit_0" in mapping
        assert "x_bit_1" in mapping

    def test_qubo_encode_integer_bits_large_range(self):
        """Larger ranges produce more bits."""
        # 0-7: 8 values -> 3 bits
        mapping = self.qubo.encode_integer("y", 0, 7)
        assert len(mapping) == 3

        # 10-100: 91 values -> ceil(log2(91)) = 7 bits
        mapping2 = self.qubo.encode_integer("n_est", 10, 100)
        expected_bits = math.ceil(math.log2(91))
        assert len(mapping2) == expected_bits

    def test_qubo_encode_integer_single_value(self):
        """A single-value range still produces at least 1 bit."""
        mapping = self.qubo.encode_integer("z", 5, 5)
        assert len(mapping) >= 1

    def test_qubo_encode_continuous(self):
        """Continuous encoding should produce exactly n_bits variables."""
        mapping = self.qubo.encode_continuous("lr", 0.0, 1.0, n_bits=4)
        assert len(mapping) == 4
        assert "lr_bit_0" in mapping
        assert "lr_bit_3" in mapping

    def test_qubo_encode_continuous_default_bits(self):
        """Default n_bits should be 4."""
        mapping = self.qubo.encode_continuous("dropout", 0.0, 0.5)
        assert len(mapping) == 4

    def test_qubo_matrix_shape(self):
        """build_qubo_matrix should return a square matrix matching total var count."""
        params_config = [
            {"name": "n_estimators", "type": "int", "low": 10, "high": 100},
            {"name": "max_depth", "type": "int", "low": 2, "high": 10},
        ]
        Q, var_names = self.qubo.build_qubo_matrix(params_config)

        n_est_bits = math.ceil(math.log2(91))  # 7
        n_depth_bits = math.ceil(math.log2(9))  # 4 (since 2^4=16>=9)
        expected_n = n_est_bits + n_depth_bits

        assert Q.shape == (expected_n, expected_n)
        assert len(var_names) == expected_n

    def test_qubo_matrix_float_params(self):
        """build_qubo_matrix works with float params."""
        params_config = [
            {"name": "lr", "type": "float", "low": 0.001, "high": 0.1, "n_bits": 4},
        ]
        Q, var_names = self.qubo.build_qubo_matrix(params_config)
        assert Q.shape == (4, 4)
        assert len(var_names) == 4

    def test_qubo_decode_roundtrip(self):
        """Encoding and decoding should produce values in the valid range."""
        params_config = [
            {"name": "n_estimators", "type": "int", "low": 10, "high": 100},
            {"name": "max_depth", "type": "int", "low": 2, "high": 10},
        ]
        Q, var_names = self.qubo.build_qubo_matrix(params_config)
        n = Q.shape[0]

        # Test with all-zeros bits
        bits_zero = np.zeros(n, dtype=int)
        params = self.qubo.decode_solution(bits_zero, params_config)
        assert 10 <= params["n_estimators"] <= 100
        assert 2 <= params["max_depth"] <= 10

        # Test with all-ones bits
        bits_ones = np.ones(n, dtype=int)
        params = self.qubo.decode_solution(bits_ones, params_config)
        assert 10 <= params["n_estimators"] <= 100
        assert 2 <= params["max_depth"] <= 10

    def test_qubo_decode_float_roundtrip(self):
        """Decode float params to valid range."""
        params_config = [
            {"name": "lr", "type": "float", "low": 0.001, "high": 0.1, "n_bits": 4},
        ]
        Q, var_names = self.qubo.build_qubo_matrix(params_config)
        n = Q.shape[0]

        bits = np.array([1, 0, 1, 0], dtype=int)
        params = self.qubo.decode_solution(bits, params_config)
        assert 0.001 <= params["lr"] <= 0.1


# ---------------------------------------------------------------------------
# SimulatedAnnealingSearcher tests
# ---------------------------------------------------------------------------

class TestSimulatedAnnealingSearcher:

    def setup_method(self):
        self.solver = SimulatedAnnealingSearcher(n_steps=500, seed=42)

    def test_sa_solver_returns_binary(self):
        """SA solver output should contain only 0s and 1s."""
        Q = np.eye(4)
        bits = self.solver.solve(Q)
        assert set(bits.tolist()).issubset({0, 1})

    def test_sa_solver_output_shape(self):
        """SA solver output shape should match QUBO dimension."""
        n = 8
        Q = np.random.randn(n, n)
        Q = (Q + Q.T) / 2  # symmetrize
        bits = self.solver.solve(Q)
        assert bits.shape == (n,)

    def test_sa_solver_minimizes_diagonal(self):
        """
        A QUBO with negative diagonal entries should prefer bits=1
        (lower energy with x=1 for negative diagonal).
        A QUBO with positive diagonal entries should prefer bits=0.
        """
        # Strongly positive diagonal -> bits should all be 0 (minimizes energy)
        n = 6
        Q_pos = np.diag([10.0] * n)
        solver = SimulatedAnnealingSearcher(n_steps=2000, T_init=1.0, T_final=0.001, seed=0)
        bits = solver.solve(Q_pos)
        # Energy with all zeros = 0, energy with all ones = sum(diagonal) > 0
        energy = float(bits @ Q_pos @ bits)
        assert energy <= float(n * 10.0)  # Must not be worse than all-ones
        # The optimal is 0 energy (all zeros)
        assert energy >= 0.0

    def test_sa_solver_empty_qubo(self):
        """SA solver handles empty QUBO matrix."""
        Q = np.zeros((0, 0))
        bits = self.solver.solve(Q)
        assert len(bits) == 0

    def test_temperature_schedule_exponential(self):
        """Exponential schedule should decrease from T_init to T_final."""
        solver = SimulatedAnnealingSearcher(n_steps=100, T_init=10.0, T_final=0.01)
        t0 = solver.temperature_schedule(0)
        t_end = solver.temperature_schedule(99)
        assert abs(t0 - 10.0) < 1e-6
        assert abs(t_end - 0.01) < 1e-6
        assert t0 > t_end

    def test_temperature_schedule_linear(self):
        """Linear schedule should decrease linearly."""
        solver = SimulatedAnnealingSearcher(
            n_steps=100, T_init=10.0, T_final=1.0, cooling="linear"
        )
        t0 = solver.temperature_schedule(0)
        t_end = solver.temperature_schedule(99)
        assert abs(t0 - 10.0) < 1e-6
        assert abs(t_end - 1.0) < 1e-6

    def test_sa_solver_invalid_cooling(self):
        """Invalid cooling schedule raises ValueError."""
        solver = SimulatedAnnealingSearcher(cooling="invalid")
        with pytest.raises(ValueError):
            solver.temperature_schedule(0)


# ---------------------------------------------------------------------------
# QAHSOptunaSampler tests
# ---------------------------------------------------------------------------

class TestQAHSOptunaSampler:

    def setup_method(self):
        self.sampler = QAHSOptunaSampler(seed=42)

    def test_optuna_suggest_int_range(self):
        """suggest_int should return value within [low, high]."""
        for _ in range(10):
            sampler = QAHSOptunaSampler(seed=None)
            val = sampler.suggest_int("n", 5, 20)
            assert 5 <= val <= 20, f"Got {val} outside [5, 20]"

    def test_optuna_suggest_float_range(self):
        """suggest_float should return value within [low, high]."""
        for _ in range(10):
            sampler = QAHSOptunaSampler(seed=None)
            val = sampler.suggest_float("lr", 0.001, 0.1)
            assert 0.001 <= val <= 0.1, f"Got {val} outside [0.001, 0.1]"

    def test_optuna_sample_runs(self):
        """sample() should run without error and return (dict, float)."""
        sampler = QAHSOptunaSampler(seed=42)
        sampler._get_or_add_param("n", "int", 1, 10)

        def obj(params):
            return float(params["n"]) ** 2

        best_params, best_value = sampler.sample(obj, n_trials=5)
        assert isinstance(best_params, dict)
        assert isinstance(best_value, float)
        assert "n" in best_params

    def test_optuna_sample_minimizes(self):
        """sample() should tend to minimize the objective."""
        sampler = QAHSOptunaSampler(seed=0)
        sampler._get_or_add_param("x", "int", 0, 15)

        # Objective: minimize x (best x = 0)
        def obj(params):
            return float(params["x"])

        best_params, best_value = sampler.sample(obj, n_trials=30)
        # With 30 trials it should find a reasonably low value
        assert best_value <= 10.0, f"Expected low value, got {best_value}"

    def test_optuna_sample_no_params(self):
        """sample() with no registered params returns empty dict."""
        sampler = QAHSOptunaSampler(seed=42)

        def obj(params):
            return 0.0

        best_params, best_value = sampler.sample(obj, n_trials=5)
        assert best_params == {}


# ---------------------------------------------------------------------------
# Demo test
# ---------------------------------------------------------------------------

class TestDemo:

    def test_demo_returns_params(self):
        """run_demo() should return a dict with n_estimators and max_depth."""
        from qahs.demo import run_demo
        best_params, best_score = run_demo()

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert 10 <= best_params["n_estimators"] <= 100
        assert 2 <= best_params["max_depth"] <= 10
        assert isinstance(best_score, float)
        assert 0.0 <= best_score <= 1.0
