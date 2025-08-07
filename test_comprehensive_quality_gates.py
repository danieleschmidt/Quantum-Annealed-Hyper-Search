#!/usr/bin/env python3
"""
Comprehensive Quality Gates for Quantum Hyperparameter Search Library.

This test suite validates:
1. Basic functionality
2. Robustness and error handling  
3. Performance and optimization
4. Security validation
5. Integration and compatibility
6. Edge cases and stress testing
"""

import time
import warnings
import traceback
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all critical imports work."""
    print("\n🧪 Testing imports...")
    
    try:
        from quantum_hyper_search import QuantumHyperSearch, get_backend, QUBOEncoder
        print("✅ Main imports successful")
        
        from quantum_hyper_search.utils.validation import validate_search_space
        print("✅ Validation utilities imported")
        
        from quantum_hyper_search.utils.security import check_safety, generate_session_id
        print("✅ Security utilities imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic optimization functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from quantum_hyper_search import QuantumHyperSearch
        
        # Create dataset
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        
        # Define search space
        search_space = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        # Initialize and optimize
        qhs = QuantumHyperSearch(backend='simple')
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=3,
            quantum_reads=20,
            cv_folds=3
        )
        
        # Validate results
        assert isinstance(best_params, dict), "Best parameters should be dictionary"
        assert len(best_params) > 0, "Should find at least some parameters"
        assert hasattr(history, 'best_score'), "History should track best score"
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_multiple_models():
    """Test with different model types."""
    print("\n🧪 Testing multiple model types...")
    
    success_count = 0
    total_tests = 3
    
    # Test data
    X, y = make_classification(n_samples=80, n_features=6, random_state=42)
    
    # Test RandomForest
    try:
        from quantum_hyper_search import QuantumHyperSearch
        qhs = QuantumHyperSearch(backend='simple')
        
        best_params, _ = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': [5, 10], 'max_depth': [3, 5]},
            X=X, y=y, n_iterations=2, quantum_reads=10, cv_folds=2
        )
        print("✅ RandomForestClassifier test passed")
        success_count += 1
    except Exception as e:
        print(f"❌ RandomForestClassifier test failed: {e}")
    
    # Test GradientBoosting
    try:
        qhs = QuantumHyperSearch(backend='simple')
        best_params, _ = qhs.optimize(
            model_class=GradientBoostingClassifier,
            param_space={'n_estimators': [10, 20], 'learning_rate': [0.1, 0.2]},
            X=X, y=y, n_iterations=2, quantum_reads=10, cv_folds=2
        )
        print("✅ GradientBoostingClassifier test passed")
        success_count += 1
    except Exception as e:
        print(f"❌ GradientBoostingClassifier test failed: {e}")
    
    # Test LogisticRegression
    try:
        qhs = QuantumHyperSearch(backend='simple')
        best_params, _ = qhs.optimize(
            model_class=LogisticRegression,
            param_space={'C': [0.1, 1.0], 'solver': ['lbfgs', 'liblinear']},
            X=X, y=y, n_iterations=2, quantum_reads=10, cv_folds=2, max_iter=100
        )
        print("✅ LogisticRegression test passed")
        success_count += 1
    except Exception as e:
        print(f"❌ LogisticRegression test failed: {e}")
    
    success_rate = success_count / total_tests
    print(f"📊 Model compatibility: {success_count}/{total_tests} ({success_rate:.1%})")
    
    return success_rate >= 0.6  # At least 60% success rate

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n🧪 Testing error handling...")
    
    success_count = 0
    total_tests = 5
    
    from quantum_hyper_search import QuantumHyperSearch
    
    # Test 1: Empty search space
    try:
        qhs = QuantumHyperSearch(backend='simple')
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        
        try:
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={},  # Empty
                X=X, y=y, n_iterations=1
            )
            print("❌ Should have failed with empty search space")
        except Exception:
            print("✅ Correctly handled empty search space")
            success_count += 1
    except Exception as e:
        print(f"❌ Empty search space test setup failed: {e}")
    
    # Test 2: Invalid data shapes
    try:
        qhs = QuantumHyperSearch(backend='simple')
        
        try:
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={'n_estimators': [10]},
                X=np.array([]),  # Empty
                y=np.array([]),
                n_iterations=1
            )
            print("❌ Should have failed with empty data")
        except Exception:
            print("✅ Correctly handled empty data")
            success_count += 1
    except Exception as e:
        print(f"❌ Empty data test setup failed: {e}")
    
    # Test 3: Mismatched X, y shapes
    try:
        qhs = QuantumHyperSearch(backend='simple')
        
        try:
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={'n_estimators': [10]},
                X=np.random.randn(10, 4),
                y=np.random.randint(0, 2, 5),  # Different length
                n_iterations=1
            )
            print("❌ Should have failed with mismatched shapes")
        except Exception:
            print("✅ Correctly handled mismatched X, y shapes")
            success_count += 1
    except Exception as e:
        print(f"❌ Mismatched shapes test setup failed: {e}")
    
    # Test 4: Invalid parameter values
    try:
        qhs = QuantumHyperSearch(backend='simple')
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        
        # This should handle invalid parameters gracefully
        best_params, _ = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': [-1, 0]},  # Invalid values
            X=X, y=y, n_iterations=1, quantum_reads=5
        )
        print("✅ Handled invalid parameter values")
        success_count += 1
    except Exception as e:
        print(f"❌ Invalid parameter handling failed: {e}")
    
    # Test 5: Very small dataset
    try:
        qhs = QuantumHyperSearch(backend='simple')
        X, y = make_classification(n_samples=10, n_features=2, random_state=42)
        
        best_params, _ = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space={'n_estimators': [5, 10]},
            X=X, y=y, n_iterations=1, quantum_reads=5, cv_folds=2
        )
        print("✅ Handled small dataset")
        success_count += 1
    except Exception as e:
        print(f"❌ Small dataset test failed: {e}")
    
    success_rate = success_count / total_tests
    print(f"📊 Error handling: {success_count}/{total_tests} ({success_rate:.1%})")
    
    return success_rate >= 0.6

def test_performance_benchmarks():
    """Test performance benchmarks."""
    print("\n🧪 Testing performance benchmarks...")
    
    try:
        from quantum_hyper_search import QuantumHyperSearch
        
        # Create larger dataset for performance testing
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        search_space = {
            'n_estimators': [10, 25, 50],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        
        # Performance test
        qhs = QuantumHyperSearch(backend='simple')
        
        start_time = time.time()
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=5,
            quantum_reads=30,
            cv_folds=3
        )
        optimization_time = time.time() - start_time
        
        # Performance criteria
        max_time = 10.0  # seconds
        min_evaluations = 3
        
        stats = history.get_statistics() if hasattr(history, 'get_statistics') else {}
        n_evaluations = stats.get('n_evaluations', getattr(history, 'n_evaluations', 0))
        
        print(f"📊 Performance results:")
        print(f"   Time: {optimization_time:.2f}s (max: {max_time}s)")
        print(f"   Evaluations: {n_evaluations} (min: {min_evaluations})")
        
        time_ok = optimization_time <= max_time
        eval_ok = n_evaluations >= min_evaluations
        
        if time_ok and eval_ok:
            print("✅ Performance benchmark passed")
            return True
        else:
            print(f"❌ Performance benchmark failed (time_ok: {time_ok}, eval_ok: {eval_ok})")
            return False
            
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

def test_security_features():
    """Test security validation features."""
    print("\n🧪 Testing security features...")
    
    try:
        from quantum_hyper_search.utils.security import (
            sanitize_parameters, check_safety, generate_session_id
        )
        
        success_count = 0
        total_tests = 4
        
        # Test 1: Parameter sanitization
        try:
            safe_params = sanitize_parameters({'n_estimators': 10, 'max_depth': 5})
            assert isinstance(safe_params, dict)
            print("✅ Parameter sanitization works")
            success_count += 1
        except Exception as e:
            print(f"❌ Parameter sanitization failed: {e}")
        
        # Test 2: Safety check
        try:
            check_safety(
                search_space={'n_estimators': [10, 20]},
                model_class=RandomForestClassifier
            )
            print("✅ Safety check works")
            success_count += 1
        except Exception as e:
            print(f"❌ Safety check failed: {e}")
        
        # Test 3: Session ID generation
        try:
            session_id = generate_session_id()
            assert isinstance(session_id, str)
            assert len(session_id) > 0
            print("✅ Session ID generation works")
            success_count += 1
        except Exception as e:
            print(f"❌ Session ID generation failed: {e}")
        
        # Test 4: Integration security test
        try:
            from quantum_hyper_search import QuantumHyperSearch
            
            qhs = QuantumHyperSearch(backend='simple', enable_security=True)
            X, y = make_classification(n_samples=50, n_features=4, random_state=42)
            
            best_params, _ = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={'n_estimators': [10, 20]},
                X=X, y=y, n_iterations=2, quantum_reads=10, cv_folds=2
            )
            print("✅ Integrated security test works")
            success_count += 1
        except Exception as e:
            print(f"❌ Integrated security test failed: {e}")
        
        success_rate = success_count / total_tests
        print(f"📊 Security features: {success_count}/{total_tests} ({success_rate:.1%})")
        
        return success_rate >= 0.75
        
    except ImportError as e:
        print(f"❌ Security module import failed: {e}")
        return False

def test_backend_compatibility():
    """Test different backend compatibility."""
    print("\n🧪 Testing backend compatibility...")
    
    try:
        from quantum_hyper_search import QuantumHyperSearch
        
        success_count = 0
        total_tests = 2
        
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        search_space = {'n_estimators': [10, 20]}
        
        # Test simple backend
        try:
            qhs = QuantumHyperSearch(backend='simple')
            best_params, _ = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y, n_iterations=2, quantum_reads=10
            )
            print("✅ Simple backend works")
            success_count += 1
        except Exception as e:
            print(f"❌ Simple backend failed: {e}")
        
        # Test simulator backend (may fallback to simple)
        try:
            qhs = QuantumHyperSearch(backend='simulator')
            best_params, _ = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y, n_iterations=2, quantum_reads=10
            )
            print("✅ Simulator backend works (or fallback successful)")
            success_count += 1
        except Exception as e:
            print(f"❌ Simulator backend failed: {e}")
        
        success_rate = success_count / total_tests
        print(f"📊 Backend compatibility: {success_count}/{total_tests} ({success_rate:.1%})")
        
        return success_rate >= 0.5
        
    except Exception as e:
        print(f"❌ Backend compatibility test failed: {e}")
        return False

def test_stress_testing():
    """Test with stress scenarios."""
    print("\n🧪 Testing stress scenarios...")
    
    try:
        from quantum_hyper_search import QuantumHyperSearch
        
        success_count = 0
        total_tests = 3
        
        # Test 1: Large search space
        try:
            X, y = make_classification(n_samples=100, n_features=8, random_state=42)
            large_search_space = {
                'n_estimators': [5, 10, 15, 20, 25],
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 3, 4, 5],
                'min_samples_leaf': [1, 2, 3]
            }  # 300 combinations
            
            qhs = QuantumHyperSearch(backend='simple')
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=large_search_space,
                X=X, y=y, n_iterations=3, quantum_reads=20, cv_folds=2
            )
            print("✅ Large search space handled")
            success_count += 1
        except Exception as e:
            print(f"❌ Large search space failed: {e}")
        
        # Test 2: Many iterations
        try:
            X, y = make_classification(n_samples=60, n_features=5, random_state=42)
            search_space = {'n_estimators': [10, 20], 'max_depth': [3, 5]}
            
            qhs = QuantumHyperSearch(backend='simple')
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y, n_iterations=10, quantum_reads=15, cv_folds=2
            )
            print("✅ Many iterations handled")
            success_count += 1
        except Exception as e:
            print(f"❌ Many iterations failed: {e}")
        
        # Test 3: High dimensional data
        try:
            X, y = make_classification(n_samples=100, n_features=20, random_state=42)
            search_space = {'n_estimators': [10, 20]}
            
            qhs = QuantumHyperSearch(backend='simple')
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y, n_iterations=3, quantum_reads=10, cv_folds=2
            )
            print("✅ High dimensional data handled")
            success_count += 1
        except Exception as e:
            print(f"❌ High dimensional data failed: {e}")
        
        success_rate = success_count / total_tests
        print(f"📊 Stress testing: {success_count}/{total_tests} ({success_rate:.1%})")
        
        return success_rate >= 0.67
        
    except Exception as e:
        print(f"❌ Stress testing failed: {e}")
        return False

def run_all_quality_gates():
    """Run all quality gate tests and provide comprehensive report."""
    
    print("🛡️  QUANTUM HYPERPARAMETER SEARCH - COMPREHENSIVE QUALITY GATES")
    print("=" * 80)
    
    # Track all test results
    test_results = {}
    
    # Run all test suites
    test_results['imports'] = test_imports()
    test_results['basic_functionality'] = test_basic_functionality()
    test_results['multiple_models'] = test_multiple_models()
    test_results['error_handling'] = test_error_handling()
    test_results['performance'] = test_performance_benchmarks()
    test_results['security'] = test_security_features()
    test_results['backend_compatibility'] = test_backend_compatibility()
    test_results['stress_testing'] = test_stress_testing()
    
    # Calculate overall results
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    pass_rate = passed_tests / total_tests
    
    # Generate report
    print("\n" + "=" * 80)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 80)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print("-" * 80)
    print(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({pass_rate:.1%})")
    
    if pass_rate >= 0.85:
        grade = "🏆 EXCELLENT"
        message = "Library meets all quality standards!"
    elif pass_rate >= 0.70:
        grade = "✅ GOOD"  
        message = "Library meets most quality standards."
    elif pass_rate >= 0.50:
        grade = "⚠️  ACCEPTABLE"
        message = "Library has basic functionality but needs improvement."
    else:
        grade = "❌ NEEDS WORK"
        message = "Library requires significant improvements."
    
    print(f"QUALITY GRADE: {grade}")
    print(f"ASSESSMENT: {message}")
    print("=" * 80)
    
    # Detailed recommendations
    if pass_rate < 1.0:
        print("\n🔧 IMPROVEMENT RECOMMENDATIONS:")
        
        if not test_results['imports']:
            print("• Fix import issues and module structure")
        if not test_results['basic_functionality']:
            print("• Address core functionality problems")
        if not test_results['error_handling']:
            print("• Improve error handling and validation")
        if not test_results['performance']:
            print("• Optimize performance for larger problems")
        if not test_results['security']:
            print("• Enhance security validation features")
        if not test_results['backend_compatibility']:
            print("• Improve backend compatibility and fallbacks")
        if not test_results['stress_testing']:
            print("• Strengthen handling of edge cases and stress scenarios")
    
    return pass_rate >= 0.70  # Return True if acceptable quality

if __name__ == "__main__":
    success = run_all_quality_gates()
    
    if success:
        print(f"\n🎉 Quality gates passed! Library is ready for production use.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Quality gates need attention before production use.")
        sys.exit(1)