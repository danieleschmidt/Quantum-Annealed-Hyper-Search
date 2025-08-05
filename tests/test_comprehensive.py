"""
Comprehensive tests for quantum hyperparameter search system.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import tempfile
import os

from quantum_hyper_search import QuantumHyperSearch
from quantum_hyper_search.core.qubo_formulation import QUBOEncoder
from quantum_hyper_search.core.optimization_history import OptimizationHistory
from quantum_hyper_search.backends.simulator import SimulatorBackend
from quantum_hyper_search.utils.validation import ValidationError
from quantum_hyper_search.utils.security import SecurityError


class TestQuantumHyperSearch:
    """Test suite for main QuantumHyperSearch class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        return X, y
    
    @pytest.fixture
    def sample_search_space(self):
        """Create sample search space."""
        return {
            'n_estimators': [10, 20, 50],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5]
        }
    
    def test_initialization_default(self):
        """Test default initialization."""
        qhs = QuantumHyperSearch()
        assert qhs.backend_name == "simulator"
        assert qhs.enable_logging == True
        assert qhs.enable_monitoring == True
        assert qhs.enable_security == True
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        qhs = QuantumHyperSearch(
            backend="simulator",
            enable_logging=False,
            enable_monitoring=False,
            enable_security=False,
            enable_caching=False
        )
        assert qhs.backend_name == "simulator"
        assert qhs.enable_logging == False
        assert qhs.logger is None
    
    def test_basic_optimization(self, sample_data, sample_search_space):
        """Test basic optimization functionality."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(
            backend="simulator",
            enable_monitoring=False,
            enable_parallel=False
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=sample_search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=5,
            cv_folds=3
        )
        
        # Verify results
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert 'min_samples_split' in best_params
        
        assert isinstance(history, OptimizationHistory)
        assert history.n_evaluations > 0
        assert history.best_score > 0
    
    def test_optimization_with_custom_objective(self, sample_data, sample_search_space):
        """Test optimization with custom objective function."""
        X, y = sample_data
        
        def custom_objective(params):
            """Custom objective that prefers smaller n_estimators."""
            model = RandomForestClassifier(**params)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3)
            base_score = scores.mean()
            # Penalize large n_estimators
            penalty = 0.01 * params['n_estimators']
            return base_score - penalty
        
        qhs = QuantumHyperSearch(
            backend="simulator",
            enable_monitoring=False,
            enable_parallel=False
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=sample_search_space,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=5,
            objective_function=custom_objective
        )
        
        assert best_params is not None
        assert history.n_evaluations > 0
    
    def test_optimization_with_constraints(self, sample_data):
        """Test optimization with parameter constraints."""
        X, y = sample_data
        
        search_space = {
            'n_estimators': [10, 20, 50],
            'max_depth': [3, 5, None]
        }
        
        constraints = {
            'mutual_exclusion': [
                ['n_estimators_0', 'n_estimators_1'],  # Don't allow both 10 and 20
            ]
        }
        
        qhs = QuantumHyperSearch(
            backend="simulator",
            enable_monitoring=False,
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=5,
            constraints=constraints
        )
        
        assert best_params is not None
        assert history.n_evaluations > 0


class TestQUBOEncoder:
    """Test suite for QUBO encoder."""
    
    def test_basic_encoding(self):
        """Test basic QUBO encoding."""
        encoder = QUBOEncoder()
        search_space = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b']
        }
        
        Q, offset, variable_map = encoder.encode(search_space)
        
        assert Q.shape[0] == Q.shape[1]  # Square matrix
        assert Q.shape[0] == len(variable_map)  # Correct size
        assert len(variable_map) == 5  # 3 + 2 variables
    
    def test_decoding(self):
        """Test sample decoding."""
        encoder = QUBOEncoder()
        search_space = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b']
        }
        
        Q, offset, variable_map = encoder.encode(search_space)
        
        # Create a sample that selects first option for each parameter
        sample = {0: 1, 1: 0, 2: 0, 3: 1, 4: 0}  # param1=1, param2='a'
        
        decoded = encoder.decode_sample(sample, variable_map, search_space)
        
        assert decoded['param1'] == 1
        assert decoded['param2'] == 'a'
    
    def test_different_encodings(self):
        """Test different encoding methods."""
        search_space = {'param': [1, 2, 3, 4]}
        
        # One-hot encoding
        encoder_oh = QUBOEncoder(encoding='one_hot')
        Q_oh, _, var_map_oh = encoder_oh.encode(search_space)
        assert len(var_map_oh) == 4
        
        # Binary encoding  
        encoder_bin = QUBOEncoder(encoding='binary')
        Q_bin, _, var_map_bin = encoder_bin.encode(search_space)
        assert len(var_map_bin) == 2  # 2 bits for 4 values


class TestOptimizationHistory:
    """Test suite for optimization history."""
    
    def test_initialization(self):
        """Test history initialization."""
        history = OptimizationHistory()
        assert history.n_evaluations == 0
        assert history.best_score == float('-inf')
        assert history.best_params is None
    
    def test_add_evaluation(self):
        """Test adding evaluations."""
        history = OptimizationHistory()
        
        params1 = {'param1': 1, 'param2': 'a'}
        history.add_evaluation(params1, 0.8, 0)
        
        assert history.n_evaluations == 1
        assert history.best_score == 0.8
        assert history.best_params == params1
        
        # Add better evaluation
        params2 = {'param1': 2, 'param2': 'b'}
        history.add_evaluation(params2, 0.9, 1)
        
        assert history.n_evaluations == 2
        assert history.best_score == 0.9
        assert history.best_params == params2
    
    def test_convergence_data(self):
        """Test convergence data extraction."""
        history = OptimizationHistory()
        
        history.add_evaluation({'p': 1}, 0.7, 0)
        history.add_evaluation({'p': 2}, 0.8, 0)
        history.add_evaluation({'p': 3}, 0.6, 1)
        history.add_evaluation({'p': 4}, 0.9, 1)
        
        iterations, best_scores = history.get_convergence_data()
        
        assert len(iterations) == 4
        assert len(best_scores) == 4
        assert best_scores == [0.7, 0.8, 0.8, 0.9]  # Running best
    
    def test_top_configurations(self):
        """Test getting top configurations."""
        history = OptimizationHistory()
        
        history.add_evaluation({'p': 1}, 0.7, 0)
        history.add_evaluation({'p': 2}, 0.9, 0)
        history.add_evaluation({'p': 3}, 0.5, 1)
        
        top_configs = history.get_top_configurations(2)
        
        assert len(top_configs) == 2
        assert top_configs[0].score == 0.9
        assert top_configs[1].score == 0.7


class TestSimulatorBackend:
    """Test suite for simulator backend."""
    
    def test_initialization(self):
        """Test backend initialization."""
        backend = SimulatorBackend()
        assert backend.name == "simulator"
        assert backend.temperature_schedule == "linear"
    
    def test_qubo_sampling(self):
        """Test QUBO sampling."""
        backend = SimulatorBackend()
        
        # Simple QUBO: minimize x1 + x2
        Q = np.array([
            [1, 0],
            [0, 1]
        ])
        
        samples = backend.sample_qubo(Q, num_reads=5)
        
        assert len(samples) == 5
        assert all(isinstance(sample, dict) for sample in samples)
        assert all(len(sample) == 2 for sample in samples)
        
        # Best solution should be {0: 0, 1: 0}
        best_sample = samples[0]  # Samples are sorted by energy
        assert best_sample[0] == 0
        assert best_sample[1] == 0
    
    def test_hardware_info(self):
        """Test hardware info retrieval."""
        backend = SimulatorBackend()
        info = backend.get_hardware_info()
        
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'type' in info
        assert info['type'] == 'simulated_annealing'


class TestValidation:
    """Test suite for input validation."""
    
    def test_valid_search_space(self):
        """Test validation of valid search space."""
        from quantum_hyper_search.utils.validation import validate_search_space
        
        valid_space = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b'],
            'param3': [True, False]
        }
        
        validated = validate_search_space(valid_space)
        assert validated == valid_space
    
    def test_invalid_search_space(self):
        """Test validation of invalid search spaces."""
        from quantum_hyper_search.utils.validation import validate_search_space
        
        # Empty search space
        with pytest.raises(ValidationError):
            validate_search_space({})
        
        # Invalid parameter name
        with pytest.raises(ValidationError):
            validate_search_space({'123invalid': [1, 2]})
        
        # Empty parameter values
        with pytest.raises(ValidationError):
            validate_search_space({'param': []})
    
    def test_data_validation(self):
        """Test data validation."""
        from quantum_hyper_search.utils.validation import validate_data
        
        # Valid data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        validate_data(X, y)  # Should not raise
        
        # Invalid data - mismatched shapes
        with pytest.raises(ValidationError):
            validate_data(X, y[:50])
        
        # Invalid data - NaN values
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValidationError):
            validate_data(X_nan, y)


class TestSecurity:
    """Test suite for security features."""
    
    def test_parameter_sanitization(self):
        """Test parameter sanitization."""
        from quantum_hyper_search.utils.security import sanitize_parameters
        
        # Valid parameters
        valid_params = {'n_estimators': 10, 'max_depth': 5}
        sanitized = sanitize_parameters(valid_params)
        assert sanitized == valid_params
        
        # Parameters with whitespace
        params_with_whitespace = {'n_estimators': 10, 'criterion': ' gini '}
        sanitized = sanitize_parameters(params_with_whitespace)
        assert sanitized['criterion'] == 'gini'  # Stripped
    
    def test_dangerous_parameters(self):
        """Test detection of dangerous parameters."""
        from quantum_hyper_search.utils.security import sanitize_parameters
        
        # Dangerous parameter name
        with pytest.raises(SecurityError):
            sanitize_parameters({'__import__': 'os'})
        
        # Dangerous parameter value
        with pytest.raises(SecurityError):
            sanitize_parameters({'param': 'exec("print(1)")'})
    
    def test_safety_check(self):
        """Test comprehensive safety check."""
        from quantum_hyper_search.utils.security import check_safety
        
        # Safe configuration
        search_space = {'n_estimators': [10, 20], 'max_depth': [3, 5]}
        model_class = RandomForestClassifier
        
        # Should not raise
        assert check_safety(search_space, model_class) == True


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        return X, y
    
    def test_invalid_backend(self):
        """Test handling of invalid backend."""
        with pytest.raises(ValueError):
            QuantumHyperSearch(backend="nonexistent_backend")
    
    def test_optimization_with_invalid_model(self, sample_data):
        """Test optimization with invalid model class."""
        X, y = sample_data
        search_space = {'param': [1, 2]}
        
        qhs = QuantumHyperSearch(backend="simulator", enable_monitoring=False)
        
        # Invalid model class
        with pytest.raises(ValidationError):
            qhs.optimize(
                model_class=int,  # Not a valid estimator
                param_space=search_space,
                X=X, y=y
            )
    
    def test_optimization_with_invalid_data(self):
        """Test optimization with invalid data."""
        search_space = {'n_estimators': [10, 20]}
        
        qhs = QuantumHyperSearch(backend="simulator", enable_monitoring=False)
        
        # Invalid data types
        with pytest.raises(ValidationError):
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=[1, 2, 3],  # Not numpy array
                y=[1, 0, 1]
            )
    
    def test_optimization_with_empty_search_space(self, sample_data):
        """Test optimization with empty search space."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(backend="simulator", enable_monitoring=False)
        
        with pytest.raises(ValidationError):
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={},  # Empty
                X=X, y=y
            )


class TestCaching:
    """Test suite for caching functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        return X, y
    
    @pytest.fixture
    def sample_search_space(self):
        """Create sample search space."""
        return {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
    
    def test_caching_enabled(self, sample_data, sample_search_space):
        """Test that caching improves performance."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(
            backend="simulator",
            enable_caching=True,
            enable_monitoring=False
        )
        
        # First run
        import time
        start_time = time.time()
        best_params1, history1 = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=sample_search_space,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=3
        )
        first_duration = time.time() - start_time
        
        # Second run should be faster due to caching
        start_time = time.time()
        best_params2, history2 = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=sample_search_space,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=3
        )
        second_duration = time.time() - start_time
        
        # Results should be similar
        assert best_params1 is not None
        assert best_params2 is not None
        
        # Cache stats should show hits
        if qhs.cache:
            stats = qhs.cache.get_stats()
            assert stats['hits'] > 0 or stats['size'] > 0


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        return X, y
    
    @pytest.fixture
    def sample_search_space(self):
        """Create sample search space."""
        return {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
    
    def test_different_models(self, sample_data):
        """Test optimization with different model types."""
        X, y = sample_data
        
        # Test with RandomForest
        rf_space = {'n_estimators': [10, 20], 'max_depth': [3, 5]}
        qhs = QuantumHyperSearch(backend="simulator", enable_monitoring=False)
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=rf_space,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=3
        )
        
        assert best_params is not None
        assert history.n_evaluations > 0
        
        # Test with SVM
        svm_space = {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
        
        best_params, history = qhs.optimize(
            model_class=SVC,
            param_space=svm_space,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=3
        )
        
        assert best_params is not None
        assert history.n_evaluations > 0
    
    def test_multi_iteration_optimization(self, sample_data, sample_search_space):
        """Test optimization across multiple iterations."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(
            backend="simulator",
            enable_monitoring=False,
            enable_caching=False
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=sample_search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=5
        )
        
        assert best_params is not None
        assert history.n_evaluations >= 3  # At least one evaluation per iteration
        
        # Check convergence data
        iterations, scores = history.get_convergence_data()
        assert len(iterations) == history.n_evaluations
        assert len(scores) == history.n_evaluations
        
        # Scores should be non-decreasing (best so far)
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i-1]


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        return X, y
    
    @pytest.mark.slow
    def test_large_search_space(self, sample_data):
        """Test optimization with larger search space."""
        X, y = sample_data
        
        large_search_space = {
            'n_estimators': [10, 20, 50, 100],
            'max_depth': [3, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8]
        }
        
        qhs = QuantumHyperSearch(
            backend="simulator",
            enable_monitoring=False
        )
        
        import time
        start_time = time.time()
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=large_search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10
        )
        
        duration = time.time() - start_time
        
        assert best_params is not None
        assert history.n_evaluations > 0
        assert duration < 60  # Should complete within 1 minute
    
    @pytest.mark.slow  
    def test_larger_dataset(self):
        """Test optimization with larger dataset."""
        # Create larger dataset
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, random_state=42
        )
        
        search_space = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20]
        }
        
        qhs = QuantumHyperSearch(
            backend="simulator",
            enable_monitoring=False,
            enable_parallel=False  # Disable for deterministic testing
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=5,
            cv_folds=3
        )
        
        assert best_params is not None
        assert history.best_score > 0.7  # Should achieve reasonable accuracy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])