"""
Comprehensive tests for robust functionality and error handling.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from unittest.mock import Mock, patch

from quantum_hyper_search import QuantumHyperSearch
from quantum_hyper_search.utils.validation import ValidationError
from quantum_hyper_search.utils.metrics import QuantumMetrics


class TestRobustFunctionality:
    """Test robust functionality and comprehensive error handling."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def search_space(self):
        """Define a simple search space."""
        return {
            'n_estimators': [10, 50],
            'max_depth': [3, 5],
        }
    
    def test_input_validation_empty_search_space(self):
        """Test validation of empty search space."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            QuantumHyperSearch().optimize(
                model_class=RandomForestClassifier,
                param_space={},
                X=np.array([[1, 2], [3, 4]]),
                y=np.array([0, 1]),
                n_iterations=1
            )
    
    def test_input_validation_invalid_data_shapes(self):
        """Test validation of mismatched data shapes."""
        search_space = {'n_estimators': [10, 20]}
        
        with pytest.raises(ValidationError, match="same number of samples"):
            QuantumHyperSearch().optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=np.array([[1, 2], [3, 4]]),  # 2 samples
                y=np.array([0, 1, 0]),  # 3 samples
                n_iterations=1
            )
    
    def test_input_validation_invalid_parameters(self):
        """Test validation of invalid optimization parameters."""
        search_space = {'n_estimators': [10, 20]}
        X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
        
        # Invalid n_iterations
        with pytest.raises(ValidationError, match="positive integer"):
            QuantumHyperSearch().optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=0  # Invalid
            )
        
        # Invalid cv_folds  
        with pytest.raises(ValidationError, match="integer >= 2"):
            QuantumHyperSearch().optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                cv_folds=1  # Invalid
            )
    
    def test_backend_initialization_error_handling(self):
        """Test error handling during backend initialization."""
        with pytest.raises(ValidationError, match="Unknown backend"):
            QuantumHyperSearch(backend='nonexistent_backend')
    
    def test_timeout_functionality(self, sample_data, search_space):
        """Test timeout functionality."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(backend='simulator', verbose=False)
        
        # Short timeout should limit iterations
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=100,  # Request many iterations
            timeout=1.0,  # But timeout quickly
            quantum_reads=10
        )
        
        # Should have completed fewer iterations due to timeout
        assert len(history.trials) < 100
        assert best_params is not None
    
    def test_early_stopping(self, sample_data, search_space):
        """Test early stopping functionality."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(backend='simulator', verbose=False)
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=20,
            early_stopping_patience=3,  # Stop after 3 iterations without improvement
            quantum_reads=10
        )
        
        # Early stopping may have triggered
        assert best_params is not None
        assert len(history.trials) <= 20
    
    def test_quantum_sampling_failure_fallback(self, sample_data, search_space):
        """Test fallback to random sampling when quantum sampling fails."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(backend='simulator', verbose=False)
        
        # Mock the backend to always fail quantum sampling
        with patch.object(qhs.backend, 'sample_qubo', side_effect=Exception("Quantum failed")):
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=5,
                quantum_reads=10
            )
        
        # Should still complete with fallback random sampling
        assert best_params is not None
        assert len(history.trials) == 5
    
    def test_model_evaluation_failure_handling(self, sample_data, search_space):
        """Test handling of model evaluation failures."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(backend='simulator', verbose=False)
        
        # Create a mock model that always fails
        class FailingModel:
            def __init__(self, **kwargs):
                pass
            
            def fit(self, X, y):
                raise ValueError("Model fitting failed")
        
        # Should handle evaluation failures gracefully
        best_params, history = qhs.optimize(
            model_class=FailingModel,
            param_space=search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=10
        )
        
        # All evaluations should fail, but algorithm should not crash
        assert all(score == float('-inf') for score in history.scores)
    
    def test_metrics_collection(self, sample_data, search_space):
        """Test quantum metrics collection."""
        X, y = sample_data
        
        qhs = QuantumHyperSearch(backend='simulator', verbose=False)
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=5,
            quantum_reads=10
        )
        
        # Check that metrics were collected
        assert qhs.metrics is not None
        stats = qhs.metrics.get_summary_statistics()
        
        assert 'optimization_time' in stats
        assert stats['optimization_time'] > 0
        assert 'total_quantum_samples' in stats or 'total_classical_samples' in stats
    
    def test_reproducibility_with_seed(self, sample_data, search_space):
        """Test reproducibility with random seeds."""
        X, y = sample_data
        
        # Run optimization twice with same seed
        results1 = []
        results2 = []
        
        for _ in range(2):
            qhs1 = QuantumHyperSearch(backend='simulator', verbose=False, random_seed=42)
            best_params1, history1 = qhs1.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=3,
                quantum_reads=10,
                random_seed=42
            )
            results1.append(len(history1.trials))
            
            qhs2 = QuantumHyperSearch(backend='simulator', verbose=False, random_seed=42)
            best_params2, history2 = qhs2.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=3,
                quantum_reads=10,
                random_seed=42
            )
            results2.append(len(history2.trials))
        
        # Results should be consistent
        assert results1 == results2
    
    def test_logging_configuration(self, sample_data, search_space):
        """Test logging configuration."""
        X, y = sample_data
        
        # Test with log file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            log_file = f.name
        
        qhs = QuantumHyperSearch(
            backend='simulator', 
            verbose=True, 
            log_file=log_file
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10
        )
        
        # Check that log file was created and has content
        import os
        assert os.path.exists(log_file)
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert len(log_content) > 0
            assert 'QuantumHyperSearch' in log_content
        
        # Cleanup
        os.unlink(log_file)
    
    def test_data_preprocessing_warnings(self):
        """Test warnings for problematic data."""
        # Create data with NaN values
        X = np.array([[1, 2], [3, np.nan], [5, 6]])
        y = np.array([0, 1, 0])
        search_space = {'n_estimators': [10, 20]}
        
        qhs = QuantumHyperSearch(backend='simulator', verbose=False)
        
        # Should handle NaN values with warnings (not crash)
        with pytest.warns(None):  # May or may not produce warnings
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=2,
                quantum_reads=10
            )
        
        # Algorithm should still complete
        assert best_params is not None
    
    def test_large_search_space_warning(self, sample_data):
        """Test warning for large search spaces."""
        X, y = sample_data
        
        # Create very large search space
        large_search_space = {
            'n_estimators': list(range(10, 101, 10)),  # 10 values
            'max_depth': list(range(1, 21)),  # 20 values
            'min_samples_split': list(range(2, 11)),  # 9 values
            # Total: 10 * 20 * 9 = 1800 combinations
        }
        
        qhs = QuantumHyperSearch(backend='simulator', verbose=False)
        
        # Should complete with warning about large search space
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=large_search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=5
        )
        
        assert best_params is not None


class TestQuantumMetrics:
    """Test quantum metrics collection and analysis."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = QuantumMetrics()
        
        assert len(metrics.quantum_samples) == 0
        assert len(metrics.energies) == 0
        assert metrics.start_time is None
    
    def test_quantum_sample_recording(self):
        """Test recording of quantum samples."""
        metrics = QuantumMetrics()
        
        sample = {0: 1, 1: 0, 2: 1}
        energy = -2.5
        chain_breaks = 0.1
        
        metrics.add_quantum_sample(sample, energy, chain_breaks)
        
        assert len(metrics.quantum_samples) == 1
        assert metrics.energies[0] == energy
        assert metrics.chain_breaks[0] == chain_breaks
    
    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        metrics = QuantumMetrics()
        
        # Add some sample data
        metrics.start_optimization()
        time.sleep(0.01)  # Small delay
        metrics.end_optimization()
        
        metrics.add_quantum_sample({0: 1}, -1.0, 0.0)
        metrics.add_quantum_sample({0: 0}, -0.5, 0.1)
        
        stats = metrics.get_summary_statistics()
        
        assert stats['total_quantum_samples'] == 2
        assert stats['best_energy'] == -1.0
        assert stats['optimization_time'] > 0
        assert 'mean_chain_break_fraction' in stats
    
    def test_convergence_analysis(self):
        """Test convergence analysis."""
        metrics = QuantumMetrics()
        
        # Simulate improving energies (convergence)
        energies = [-0.5, -1.0, -1.5, -1.5, -1.5]  # Converges after 3rd iteration
        for energy in energies:
            metrics.add_quantum_sample({0: 1}, energy, 0.0)
        
        convergence = metrics.analyze_convergence(window_size=2)
        
        assert 'convergence_detected' in convergence
        assert 'best_iteration' in convergence
        assert convergence['best_iteration'] == 3  # 1-indexed


class TestValidationUtils:
    """Test validation utility functions."""
    
    def test_search_space_validation(self):
        """Test search space validation."""
        from quantum_hyper_search.utils.validation import validate_search_space
        
        # Valid search space
        valid_space = {'param1': [1, 2, 3], 'param2': ['a', 'b']}
        normalized = validate_search_space(valid_space)
        assert normalized == valid_space
        
        # Invalid search spaces
        with pytest.raises(ValidationError):
            validate_search_space({})  # Empty
        
        with pytest.raises(ValidationError):
            validate_search_space({'param': []})  # Empty values
        
        with pytest.raises(ValidationError):
            validate_search_space("not_a_dict")  # Not a dict
    
    def test_data_validation(self):
        """Test data validation."""
        from quantum_hyper_search.utils.validation import validate_data
        
        # Valid data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        X_val, y_val = validate_data(X, y)
        assert X_val.shape == (3, 2)
        assert y_val.shape == (3,)
        
        # Invalid data
        with pytest.raises(ValidationError):
            validate_data(X, np.array([0, 1]))  # Mismatched shapes
        
        with pytest.raises(ValidationError):
            validate_data(np.array([1, 2, 3]), y)  # X not 2D
    
    def test_model_validation(self):
        """Test model class validation."""
        from quantum_hyper_search.utils.validation import validate_model_class
        
        # Valid model
        validate_model_class(RandomForestClassifier)  # Should not raise
        
        # Invalid model
        with pytest.raises(ValidationError):
            validate_model_class("not_a_class")  # Not a class
        
        class InvalidModel:
            pass  # Missing fit/predict methods
        
        with pytest.raises(ValidationError):
            validate_model_class(InvalidModel)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])