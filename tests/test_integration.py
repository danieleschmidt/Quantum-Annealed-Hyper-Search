"""
Integration tests for quantum hyperparameter search.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
import time

from quantum_hyper_search import QuantumHyperSearch
from quantum_hyper_search.utils.validation import ValidationError
from quantum_hyper_search.utils.security import SecurityError


class TestQuantumHyperSearchIntegration:
    """Integration tests for the complete quantum hyperparameter search system."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def rf_search_space(self):
        """Random Forest search space."""
        return {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    def test_basic_optimization_flow(self, classification_data, rf_search_space):
        """Test basic optimization flow end-to-end."""
        X, y = classification_data
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_logging=False,
            enable_monitoring=False
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=rf_search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=20,
            cv_folds=3,
            random_state=42
        )
        
        # Verify results
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert all(key in rf_search_space for key in best_params.keys())
        assert history.n_evaluations > 0
        assert history.best_score > 0
        
        # Verify best model actually works
        model = RandomForestClassifier(**best_params, random_state=42)
        scores = cross_val_score(model, X, y, cv=3)
        assert scores.mean() > 0.5  # Reasonable performance
    
    def test_multiple_models(self, classification_data):
        """Test optimization with different model types."""
        X, y = classification_data
        
        model_configs = [
            (RandomForestClassifier, {
                'n_estimators': [10, 20],
                'max_depth': [3, 5]
            }),
            (GradientBoostingClassifier, {
                'n_estimators': [10, 20],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2]
            })
        ]
        
        for model_class, search_space in model_configs:
            qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
            
            best_params, history = qhs.optimize(
                model_class=model_class,
                param_space=search_space,
                X=X, y=y,
                n_iterations=2,
                quantum_reads=10,
                cv_folds=2
            )
            
            assert best_params is not None
            assert history.best_score > 0
    
    def test_custom_objective_function(self, classification_data, rf_search_space):
        """Test optimization with custom objective function."""
        X, y = classification_data
        
        def custom_objective(params):
            """Custom objective that penalizes large models."""
            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=2)
            base_score = scores.mean()
            
            # Penalty for large n_estimators
            penalty = params['n_estimators'] * 0.001
            return base_score - penalty
        
        qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=rf_search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10,
            objective_function=custom_objective
        )
        
        assert best_params is not None
        assert history.best_score > 0
    
    def test_optimization_with_constraints(self, classification_data):
        """Test optimization with parameter constraints."""
        X, y = classification_data
        
        search_space = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10]
        }
        
        # Define constraints
        constraints = {
            'conditional': [
                ('n_estimators_2', 'max_depth_2')  # If n_estimators=100, then max_depth=10
            ]
        }
        
        qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10,
            cv_folds=2,
            constraints=constraints
        )
        
        assert best_params is not None
        assert history.best_score > 0
    
    def test_robustness_features(self, classification_data, rf_search_space):
        """Test robustness features (Generation 2)."""
        X, y = classification_data
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_logging=True,
            enable_monitoring=True,
            enable_security=True,
            log_level='INFO'
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=rf_search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10,
            cv_folds=2
        )
        
        assert best_params is not None
        assert qhs.session_id is not None
        assert len(qhs.session_id) > 0
    
    def test_optimization_features(self, classification_data, rf_search_space):
        """Test optimization features (Generation 3)."""
        X, y = classification_data
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_caching=True,
            enable_parallel=False,  # Disable for testing
            optimization_strategy='adaptive',
            cache_size=100
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=rf_search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=10,
            cv_folds=2
        )
        
        assert best_params is not None
        
        # Check caching worked
        if qhs.cache:
            cache_stats = qhs.cache.get_stats()
            assert cache_stats['size'] > 0
            # Should have cache hits due to repeated evaluations
            assert cache_stats['hits'] > 0
    
    def test_validation_errors(self, classification_data):
        """Test input validation errors."""
        X, y = classification_data
        
        qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
        
        # Test invalid search space
        with pytest.raises(ValidationError):
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={},  # Empty search space
                X=X, y=y,
                n_iterations=1,
                quantum_reads=10
            )
        
        # Test invalid model class
        with pytest.raises(ValidationError):
            qhs.optimize(
                model_class=dict,  # Not a valid model class
                param_space={'param': [1, 2]},
                X=X, y=y,
                n_iterations=1,
                quantum_reads=10
            )
        
        # Test mismatched data
        with pytest.raises(ValidationError):
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={'n_estimators': [10, 20]},
                X=X,
                y=y[:50],  # Mismatched length
                n_iterations=1,
                quantum_reads=10
            )
    
    def test_security_features(self, classification_data):
        """Test security features."""
        X, y = classification_data
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_security=True,
            enable_logging=False
        )
        
        # Test with potentially dangerous parameter names
        with pytest.raises(SecurityError):
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={'__dangerous__': [1, 2]},  # Dangerous parameter name
                X=X, y=y,
                n_iterations=1,
                quantum_reads=10
            )
    
    def test_convergence_behavior(self, classification_data, rf_search_space):
        """Test convergence behavior over multiple iterations."""
        X, y = classification_data
        
        qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=rf_search_space,
            X=X, y=y,
            n_iterations=5,
            quantum_reads=20,
            cv_folds=3
        )
        
        # Check convergence properties
        iterations, best_scores = history.get_convergence_data()
        assert len(iterations) > 0
        assert len(best_scores) > 0
        
        # Best scores should be non-decreasing
        for i in range(1, len(best_scores)):
            assert best_scores[i] >= best_scores[i-1]
    
    def test_performance_monitoring(self, classification_data, rf_search_space):
        """Test performance monitoring capabilities."""
        X, y = classification_data
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_monitoring=True,
            enable_logging=False
        )
        
        start_time = time.time()
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=rf_search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10,
            cv_folds=2
        )
        
        end_time = time.time()
        
        assert best_params is not None
        
        # Check monitoring data was collected
        if qhs.monitor and qhs.monitor.performance_monitor:
            metrics = qhs.monitor.performance_monitor.get_metrics()
            assert metrics.total_evaluations > 0
            assert metrics.total_duration > 0
            assert metrics.total_duration < (end_time - start_time) + 1  # Reasonable timing
    
    def test_different_backends(self, classification_data):
        """Test different quantum backends."""
        X, y = classification_data
        
        search_space = {'n_estimators': [10, 20], 'max_depth': [3, 5]}
        
        # Test simulator backend
        qhs_sim = QuantumHyperSearch(backend='simulator', enable_logging=False)
        best_params_sim, history_sim = qhs_sim.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10,
            cv_folds=2
        )
        
        assert best_params_sim is not None
        assert history_sim.best_score > 0
    
    def test_history_analysis(self, classification_data, rf_search_space):
        """Test optimization history analysis."""
        X, y = classification_data
        
        qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=rf_search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=15,
            cv_folds=2
        )
        
        # Test history analysis methods
        stats = history.get_statistics()
        assert 'n_evaluations' in stats
        assert 'best_score' in stats
        assert 'mean_score' in stats
        assert stats['n_evaluations'] > 0
        
        # Test parameter importance
        if history.n_evaluations >= 10:
            importance = history.get_parameter_importance()
            assert isinstance(importance, dict)
            assert len(importance) > 0
        
        # Test top configurations
        top_configs = history.get_top_configurations(5)
        assert len(top_configs) > 0
        assert all(hasattr(config, 'parameters') for config in top_configs)
        assert all(hasattr(config, 'score') for config in top_configs)
    
    def test_regression_optimization(self, regression_data):
        """Test optimization on regression tasks."""
        X, y = regression_data
        
        search_space = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5, 10]
        }
        
        from sklearn.ensemble import RandomForestRegressor
        
        qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
        
        best_params, history = qhs.optimize(
            model_class=RandomForestRegressor,
            param_space=search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10,
            cv_folds=2,
            scoring='neg_mean_squared_error'
        )
        
        assert best_params is not None
        assert history.best_score < 0  # MSE is negative in sklearn
        
        # Verify model works
        model = RandomForestRegressor(**best_params, random_state=42)
        scores = cross_val_score(model, X, y, cv=2, scoring='neg_mean_squared_error')
        assert all(score < 0 for score in scores)  # Valid MSE scores


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_small_scale_performance(self):
        """Test performance on small-scale problems."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        search_space = {
            'n_estimators': [10, 20, 50],
            'max_depth': [3, 5]
        }
        
        qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
        
        start_time = time.time()
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=10,
            cv_folds=2
        )
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Performance assertions
        assert elapsed_time < 30  # Should complete within 30 seconds
        assert history.n_evaluations > 0
        assert best_params is not None
        
        print(f"Small-scale benchmark: {elapsed_time:.2f}s, {history.n_evaluations} evaluations")
    
    def test_medium_scale_performance(self):
        """Test performance on medium-scale problems."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        
        search_space = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5]
        }
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_caching=True,
            enable_logging=False
        )
        
        start_time = time.time()
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=4,
            quantum_reads=20,
            cv_folds=3
        )
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Performance assertions
        assert elapsed_time < 120  # Should complete within 2 minutes
        assert history.n_evaluations > 5
        assert best_params is not None
        
        # Cache should provide benefits
        if qhs.cache:
            cache_stats = qhs.cache.get_stats()
            assert cache_stats['hit_rate'] > 0.1  # At least some cache hits
        
        print(f"Medium-scale benchmark: {elapsed_time:.2f}s, {history.n_evaluations} evaluations")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])