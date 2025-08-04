"""
Quality gates and compliance tests for quantum hyperparameter search.
"""

import pytest
import numpy as np
import time
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from quantum_hyper_search import QuantumHyperSearch


class TestQualityGates:
    """Quality gate tests to ensure production readiness."""
    
    @pytest.fixture
    def test_data(self):
        """Standard test dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def standard_search_space(self):
        """Standard search space for testing."""
        return {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5]
        }
    
    def test_code_quality_gate(self):
        """Test code quality requirements."""
        # Test import structure
        from quantum_hyper_search import QuantumHyperSearch
        from quantum_hyper_search.core import QUBOEncoder, OptimizationHistory
        from quantum_hyper_search.backends import get_backend, list_backends
        from quantum_hyper_search.utils import validate_search_space, sanitize_parameters
        
        # Test that all core classes are properly defined
        assert hasattr(QuantumHyperSearch, 'optimize')
        assert hasattr(QUBOEncoder, 'encode')
        assert hasattr(OptimizationHistory, 'add_evaluation')
        
        # Test backend availability
        backends = list_backends()
        assert 'simulator' in backends
        assert backends['simulator'] is True
        
        print("✅ Code quality gate passed")
    
    def test_performance_gate(self, test_data, standard_search_space):
        """Test performance requirements."""
        X, y = test_data
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_caching=True,
            enable_logging=False
        )
        
        # Performance requirement: Complete within reasonable time
        start_time = time.time()
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=standard_search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=15,
            cv_folds=3
        )
        
        elapsed_time = time.time() - start_time
        
        # Quality gates
        assert elapsed_time < 60, f"Optimization took too long: {elapsed_time:.2f}s"
        assert history.n_evaluations >= 5, f"Too few evaluations: {history.n_evaluations}"
        assert history.best_score > 0.7, f"Poor optimization result: {history.best_score:.4f}"
        
        # Cache efficiency
        if qhs.cache:
            cache_stats = qhs.cache.get_stats()
            assert cache_stats['hit_rate'] > 0.0, "Cache not being utilized"
        
        print(f"✅ Performance gate passed: {elapsed_time:.2f}s, score: {history.best_score:.4f}")
    
    def test_memory_usage_gate(self, test_data, standard_search_space):
        """Test memory usage requirements."""
        X, y = test_data
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_monitoring=True,
            enable_logging=False
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=standard_search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=10,
            cv_folds=2
        )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory gate: Should not use excessive memory
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f}MB"
        
        print(f"✅ Memory usage gate passed: {memory_increase:.1f}MB increase")
    
    def test_accuracy_gate(self, test_data, standard_search_space):
        """Test optimization accuracy requirements."""
        X, y = test_data
        
        # Run multiple optimization runs for consistency
        scores = []
        
        for run in range(3):
            qhs = QuantumHyperSearch(
                backend='simulator',
                enable_logging=False
            )
            
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=standard_search_space,
                X=X, y=y,
                n_iterations=3,
                quantum_reads=15,
                cv_folds=3,
                random_state=42 + run
            )
            
            scores.append(history.best_score)
        
        # Accuracy gates
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        assert mean_score > 0.75, f"Poor average performance: {mean_score:.4f}"
        assert std_score < 0.1, f"Inconsistent results: std={std_score:.4f}"
        assert all(score > 0.7 for score in scores), f"Some runs performed poorly: {scores}"
        
        print(f"✅ Accuracy gate passed: mean={mean_score:.4f}, std={std_score:.4f}")
    
    def test_robustness_gate(self, test_data):
        """Test system robustness requirements."""
        X, y = test_data
        
        # Test with various challenging configurations
        test_cases = [
            # Small search space
            {'n_estimators': [10]},
            
            # Large search space
            {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [3, 5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8]
            },
            
            # Edge case parameters
            {
                'n_estimators': [1, 500],
                'max_depth': [1, 50]
            }
        ]
        
        for i, search_space in enumerate(test_cases):
            try:
                qhs = QuantumHyperSearch(
                    backend='simulator',
                    enable_security=True,
                    enable_logging=False
                )
                
                best_params, history = qhs.optimize(
                    model_class=RandomForestClassifier,
                    param_space=search_space,
                    X=X, y=y,
                    n_iterations=2,
                    quantum_reads=10,
                    cv_folds=2
                )
                
                assert best_params is not None, f"Test case {i} failed to return parameters"
                assert history.best_score > 0, f"Test case {i} returned invalid score"
                
            except Exception as e:
                pytest.fail(f"Robustness test case {i} failed: {e}")
        
        print("✅ Robustness gate passed")
    
    def test_security_gate(self, test_data):
        """Test security requirements."""
        X, y = test_data
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_security=True,
            enable_logging=False
        )
        
        # Test input sanitization
        safe_params = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        # This should work
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=safe_params,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=5,
            cv_folds=2
        )
        
        assert best_params is not None
        
        # Test that dangerous inputs are rejected
        dangerous_cases = [
            {'__import__': ['os']},
            {'eval_param': ['malicious_code']},
            {'../_dangerous': [1, 2]}
        ]
        
        for dangerous_space in dangerous_cases:
            with pytest.raises(Exception):  # Should raise SecurityError or ValidationError
                qhs.optimize(
                    model_class=RandomForestClassifier,
                    param_space=dangerous_space,
                    X=X, y=y,
                    n_iterations=1,
                    quantum_reads=5
                )
        
        print("✅ Security gate passed")
    
    def test_scalability_gate(self):
        """Test scalability requirements."""
        # Test with increasing problem sizes
        problem_sizes = [
            (100, 5),   # Small
            (500, 10),  # Medium
            (1000, 15)  # Large
        ]
        
        times = []
        
        for n_samples, n_features in problem_sizes:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                random_state=42
            )
            
            search_space = {
                'n_estimators': [10, 50],
                'max_depth': [3, 5]
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
                n_iterations=2,
                quantum_reads=10,
                cv_folds=2
            )
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            assert best_params is not None
        
        # Check that scaling is reasonable (not exponential)
        # Time should increase sub-quadratically with problem size
        time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        max_ratio = max(time_ratios)
        
        assert max_ratio < 5.0, f"Poor scalability: max time ratio {max_ratio:.2f}"
        
        print(f"✅ Scalability gate passed: time ratios {time_ratios}")
    
    def test_error_handling_gate(self, test_data):
        """Test error handling requirements."""
        X, y = test_data
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_logging=False
        )
        
        # Test various error conditions
        error_cases = [
            # Empty search space
            ({}, "Empty search space should be rejected"),
            
            # Invalid parameter values
            ({'n_estimators': []}, "Empty parameter list should be rejected"),
            
            # Non-existent parameter
            ({'nonexistent_param': [1, 2]}, "Invalid parameter should be handled"),
        ]
        
        for search_space, description in error_cases:
            with pytest.raises(Exception) as exc_info:
                qhs.optimize(
                    model_class=RandomForestClassifier,
                    param_space=search_space,
                    X=X, y=y,
                    n_iterations=1,
                    quantum_reads=5
                )
            
            # Should raise a meaningful exception
            assert exc_info.value is not None, description
        
        print("✅ Error handling gate passed")
    
    def test_documentation_gate(self):
        """Test documentation requirements."""
        from quantum_hyper_search import QuantumHyperSearch
        from quantum_hyper_search.core.qubo_formulation import QUBOEncoder
        
        # Test that key classes have docstrings
        assert QuantumHyperSearch.__doc__ is not None, "Main class missing docstring"
        assert QuantumHyperSearch.optimize.__doc__ is not None, "Main method missing docstring"
        assert QUBOEncoder.__doc__ is not None, "Core class missing docstring"
        assert QUBOEncoder.encode.__doc__ is not None, "Core method missing docstring"
        
        # Test that docstrings are meaningful (not just placeholders)
        main_doc = QuantumHyperSearch.__doc__.strip()
        assert len(main_doc) > 50, "Main docstring too short"
        assert "quantum" in main_doc.lower(), "Main docstring should mention quantum"
        
        optimize_doc = QuantumHyperSearch.optimize.__doc__.strip()
        assert len(optimize_doc) > 100, "Optimize docstring too short"
        assert "Args:" in optimize_doc, "Optimize docstring should document arguments"
        assert "Returns:" in optimize_doc, "Optimize docstring should document returns"
        
        print("✅ Documentation gate passed")
    
    def test_compatibility_gate(self, test_data):
        """Test compatibility requirements."""
        X, y = test_data
        
        # Test compatibility with different scikit-learn models
        compatible_models = [
            RandomForestClassifier,
            # Could add more models here
        ]
        
        for model_class in compatible_models:
            qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
            
            # Use model-appropriate search space
            if model_class == RandomForestClassifier:
                search_space = {'n_estimators': [10, 20], 'max_depth': [3, 5]}
            else:
                search_space = {'param': [1, 2]}  # Generic fallback
            
            try:
                best_params, history = qhs.optimize(
                    model_class=model_class,
                    param_space=search_space,
                    X=X, y=y,
                    n_iterations=1,
                    quantum_reads=5,
                    cv_folds=2
                )
                
                assert best_params is not None
                
            except Exception as e:
                pytest.fail(f"Compatibility failed for {model_class.__name__}: {e}")
        
        print("✅ Compatibility gate passed")


class TestComprehensiveQualityGates:
    """Comprehensive quality gate test suite."""
    
    def test_all_quality_gates(self):
        """Run all quality gates in sequence."""
        print("\n🚦 Running comprehensive quality gate tests...")
        
        # Create test data
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=8,
            n_classes=2,
            random_state=42
        )
        
        search_space = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5]
        }
        
        # Initialize quality gate tracking
        passed_gates = []
        failed_gates = []
        
        gates = [
            ("Performance", self._test_performance_comprehensive),
            ("Accuracy", self._test_accuracy_comprehensive),
            ("Robustness", self._test_robustness_comprehensive),
            ("Security", self._test_security_comprehensive),
            ("Memory", self._test_memory_comprehensive),
        ]
        
        for gate_name, gate_test in gates:
            try:
                gate_test(X, y, search_space)
                passed_gates.append(gate_name)
                print(f"✅ {gate_name} gate: PASSED")
            except Exception as e:
                failed_gates.append((gate_name, str(e)))
                print(f"❌ {gate_name} gate: FAILED - {e}")
        
        # Overall assessment
        total_gates = len(gates)
        passed_count = len(passed_gates)
        pass_rate = passed_count / total_gates
        
        print(f"\n📊 Quality Gate Summary:")
        print(f"   Passed: {passed_count}/{total_gates} ({pass_rate:.1%})")
        print(f"   Failed: {len(failed_gates)}")
        
        if failed_gates:
            print(f"   Failed gates: {[name for name, _ in failed_gates]}")
        
        # Require minimum pass rate
        assert pass_rate >= 0.8, f"Quality gate pass rate too low: {pass_rate:.1%}"
        
        print("🎉 Comprehensive quality gates PASSED!")
    
    def _test_performance_comprehensive(self, X, y, search_space):
        """Comprehensive performance test."""
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_caching=True,
            enable_monitoring=True,
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
        elapsed_time = time.time() - start_time
        
        # Performance requirements
        assert elapsed_time < 90, f"Too slow: {elapsed_time:.2f}s"
        assert history.n_evaluations >= 8, f"Too few evaluations: {history.n_evaluations}"
        assert best_params is not None, "No best parameters found"
    
    def _test_accuracy_comprehensive(self, X, y, search_space):
        """Comprehensive accuracy test."""
        scores = []
        
        for seed in [42, 123, 456]:
            qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=3,
                quantum_reads=15,
                cv_folds=3,
                random_state=seed
            )
            scores.append(history.best_score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        assert mean_score > 0.78, f"Poor mean accuracy: {mean_score:.4f}"
        assert std_score < 0.08, f"High variance: {std_score:.4f}"
    
    def _test_robustness_comprehensive(self, X, y, search_space):
        """Comprehensive robustness test."""
        # Test edge cases
        edge_cases = [
            {'n_iterations': 1, 'quantum_reads': 5},
            {'n_iterations': 5, 'quantum_reads': 50},
            {'cv_folds': 2},
            {'cv_folds': 5},
        ]
        
        for case in edge_cases:
            qhs = QuantumHyperSearch(backend='simulator', enable_logging=False)
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                **case
            )
            assert best_params is not None, f"Failed with case: {case}"
    
    def _test_security_comprehensive(self, X, y, search_space):
        """Comprehensive security test."""
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_security=True,
            enable_logging=False
        )
        
        # Should work with safe parameters
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=1,
            quantum_reads=5,
            cv_folds=2
        )
        assert best_params is not None
    
    def _test_memory_comprehensive(self, X, y, search_space):
        """Comprehensive memory test."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            enable_caching=True,
            enable_monitoring=True,
            enable_logging=False
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=4,
            quantum_reads=15,
            cv_folds=3
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 300, f"Excessive memory: {memory_increase:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])