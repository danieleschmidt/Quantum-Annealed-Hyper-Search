"""
Basic functionality tests for quantum hyperparameter search.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from quantum_hyper_search import QuantumHyperSearch
from quantum_hyper_search.core.qubo_formulation import QUBOEncoder
from quantum_hyper_search.core.optimization_history import OptimizationHistory
from quantum_hyper_search.backends import get_backend, list_backends


class TestQuantumHyperSearch:
    """Test main QuantumHyperSearch functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def sample_search_space(self):
        """Create sample search space."""
        return {
            'n_estimators': [10, 20],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    
    def test_initialization(self):
        """Test QuantumHyperSearch initialization."""
        qhs = QuantumHyperSearch(backend='simulator')
        assert qhs.backend_name == 'simulator'
        assert qhs.encoder is not None
        assert qhs.history is not None
    
    def test_basic_optimization(self, sample_data, sample_search_space):
        """Test basic optimization functionality."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(backend='simulator')
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=sample_search_space,
            X=X,
            y=y,
            n_iterations=2,
            quantum_reads=10,
            cv_folds=2,
            random_state=42
        )
        
        # Check results
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert all(key in sample_search_space for key in best_params.keys())
        assert history.n_evaluations > 0
        assert history.best_score > 0


class TestQUBOEncoder:
    """Test QUBO encoding functionality."""
    
    @pytest.fixture
    def sample_search_space(self):
        """Create sample search space."""
        return {
            'param1': ['a', 'b', 'c'],
            'param2': [1, 2, 3, 4]
        }
    
    def test_initialization(self):
        """Test QUBOEncoder initialization."""
        encoder = QUBOEncoder()
        assert encoder.encoding == 'one_hot'
        assert encoder.penalty_strength == 2.0
    
    def test_variable_mapping(self, sample_search_space):
        """Test variable mapping creation."""
        encoder = QUBOEncoder()
        var_map = encoder._create_variable_mapping(sample_search_space)
        
        # Check one-hot encoding variables
        expected_vars = ['param1_0', 'param1_1', 'param1_2', 'param2_0', 'param2_1', 'param2_2', 'param2_3']
        assert len(var_map) == len(expected_vars)
        for var in expected_vars:
            assert var in var_map
    
    def test_qubo_encoding(self, sample_search_space):
        """Test QUBO matrix encoding."""
        encoder = QUBOEncoder()
        Q, offset, var_map = encoder.encode(sample_search_space)
        
        # Check matrix properties
        assert Q.shape[0] == Q.shape[1]  # Square matrix
        assert Q.shape[0] == len(var_map)  # Correct size
        assert offset >= 0  # Non-negative offset
    
    def test_sample_decoding(self, sample_search_space):
        """Test sample decoding."""
        encoder = QUBOEncoder()
        Q, offset, var_map = encoder.encode(sample_search_space)
        
        # Create a valid sample (one-hot)
        sample = {i: 0 for i in range(len(var_map))}
        sample[0] = 1  # param1_0
        sample[3] = 1  # param2_0
        
        params = encoder.decode_sample(sample, var_map, sample_search_space)
        
        assert 'param1' in params
        assert 'param2' in params
        assert params['param1'] in sample_search_space['param1']
        assert params['param2'] in sample_search_space['param2']


class TestOptimizationHistory:
    """Test optimization history functionality."""
    
    def test_initialization(self):
        """Test OptimizationHistory initialization."""
        history = OptimizationHistory()
        assert history.n_evaluations == 0
        assert history.best_score == float('-inf')
        assert history.best_params is None
    
    def test_add_evaluation(self):
        """Test adding evaluations."""
        history = OptimizationHistory()
        
        params = {'param1': 'a', 'param2': 1}
        score = 0.85
        iteration = 1
        
        history.add_evaluation(params, score, iteration)
        
        assert history.n_evaluations == 1
        assert history.best_score == score
        assert history.best_params == params
    
    def test_best_tracking(self):
        """Test best parameter tracking."""
        history = OptimizationHistory()
        
        # Add evaluations with increasing scores
        evaluations = [
            ({'param1': 'a'}, 0.7, 1),
            ({'param1': 'b'}, 0.9, 1),
            ({'param1': 'c'}, 0.8, 2)
        ]
        
        for params, score, iteration in evaluations:
            history.add_evaluation(params, score, iteration)
        
        assert history.best_score == 0.9
        assert history.best_params == {'param1': 'b'}
        assert history.n_evaluations == 3
    
    def test_convergence_data(self):
        """Test convergence data extraction."""
        history = OptimizationHistory()
        
        evaluations = [
            ({'param1': 'a'}, 0.7, 1),
            ({'param1': 'b'}, 0.9, 1),
            ({'param1': 'c'}, 0.8, 2)
        ]
        
        for params, score, iteration in evaluations:
            history.add_evaluation(params, score, iteration)
        
        iterations, best_scores = history.get_convergence_data()
        
        assert len(iterations) == 3
        assert len(best_scores) == 3
        assert best_scores == [0.7, 0.9, 0.9]  # Best so far


class TestBackends:
    """Test backend functionality."""
    
    def test_list_backends(self):
        """Test backend listing."""
        backends = list_backends()
        assert isinstance(backends, dict)
        assert 'simulator' in backends
    
    def test_get_backend(self):
        """Test backend retrieval."""
        backend_class = get_backend('simulator')
        backend = backend_class()
        assert backend.name == 'simulator'
        assert backend.is_available()
    
    def test_simulator_hardware_info(self):
        """Test simulator hardware info."""
        backend_class = get_backend('simulator')
        backend = backend_class()
        info = backend.get_hardware_info()
        
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'type' in info
        assert info['type'] == 'simulated_annealing'
    
    def test_simulator_sampling(self):
        """Test simulator QUBO sampling."""
        backend_class = get_backend('simulator')
        backend = backend_class(max_iterations=10)  # Fast test
        
        # Simple 2x2 QUBO
        Q = np.array([[1, -1], [-1, 1]])
        
        samples = backend.sample_qubo(Q, num_reads=5)
        
        assert len(samples) == 5
        assert all(isinstance(sample, dict) for sample in samples)
        assert all(len(sample) == 2 for sample in samples)
        assert all(val in [0, 1] for sample in samples for val in sample.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])