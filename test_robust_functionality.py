#!/usr/bin/env python3
"""
Test robust functionality of the quantum hyperparameter search library.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Test the robust implementation
try:
    from quantum_hyper_search.robust_main import QuantumHyperSearchRobust
    print("✅ Successfully imported QuantumHyperSearchRobust")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def test_robust_optimization():
    """Test robust quantum hyperparameter optimization."""
    print("\n🧪 Testing robust quantum hyperparameter optimization...")
    
    # Create a dataset
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=2,
        n_informative=8,
        random_state=42
    )
    print(f"✅ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define a search space
    search_space = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    print(f"✅ Search space: {len(search_space)} parameters, {np.prod([len(v) for v in search_space.values()])} combinations")
    
    try:
        # Initialize robust quantum optimizer
        qhs = QuantumHyperSearchRobust(
            backend='simple',
            encoding='one_hot',
            penalty_strength=2.0,
            enable_security=True,
            max_retries=2,
            timeout_per_iteration=60.0,
            fallback_to_random=True
        )
        print("✅ Initialized robust QuantumHyperSearchRobust")
        
        # Run optimization with various parameters
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X,
            y=y,
            n_iterations=5,
            quantum_reads=50,
            cv_folds=3,
            scoring='accuracy',
            early_stopping_patience=3,
            random_state=42  # Additional model parameter
        )
        
        print(f"✅ Robust optimization completed!")
        
        # Print detailed results
        print(f"\n📊 Results:")
        print(f"🏆 Best parameters: {best_params}")
        if history.best_score != float('-inf'):
            print(f"📈 Best score: {history.best_score:.4f}")
        
        # Print statistics
        stats = history.get_statistics()
        print(f"\n📊 Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        # Verify the parameters work if we got any
        if best_params and history.best_score != float('-inf'):
            model = RandomForestClassifier(**best_params, random_state=42)
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            print(f"✅ Verification: CV score = {scores.mean():.4f} ± {scores.std():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling capabilities."""
    print("\n🧪 Testing error handling...")
    
    # Test with invalid parameters
    try:
        qhs = QuantumHyperSearchRobust(backend='simple')
        
        # Invalid search space
        try:
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={},  # Empty search space
                X=np.random.randn(10, 5),
                y=np.random.randint(0, 2, 10),
                n_iterations=1
            )
            print("❌ Should have failed with empty search space")
            return False
        except Exception as e:
            print(f"✅ Correctly caught empty search space: {type(e).__name__}")
        
        # Invalid data
        try:
            qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={'n_estimators': [10, 20]},
                X=np.array([]),  # Empty data
                y=np.array([]),
                n_iterations=1
            )
            print("❌ Should have failed with empty data")
            return False
        except Exception as e:
            print(f"✅ Correctly caught empty data: {type(e).__name__}")
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_security_features():
    """Test security validation features."""
    print("\n🧪 Testing security features...")
    
    try:
        qhs = QuantumHyperSearchRobust(backend='simple', enable_security=True)
        
        # Test with safe parameters
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        search_space = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X,
            y=y,
            n_iterations=2,
            quantum_reads=20,
            cv_folds=2,
            random_state=42  # This should be sanitized
        )
        
        print("✅ Security validation passed with safe parameters")
        
        # Test session ID generation
        session_id = qhs.session_id
        if session_id and len(session_id) > 0:
            print(f"✅ Session ID generated: {session_id}")
        else:
            print("❌ Session ID not generated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_behavior():
    """Test fallback and recovery behavior."""
    print("\n🧪 Testing fallback behavior...")
    
    try:
        # Initialize with a backend that might fail
        qhs = QuantumHyperSearchRobust(
            backend='simple',
            fallback_to_random=True,
            max_retries=2
        )
        
        X, y = make_classification(n_samples=30, n_features=4, random_state=42)
        search_space = {'n_estimators': [5, 10]}
        
        # This should work even if some iterations fail
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X,
            y=y,
            n_iterations=3,
            quantum_reads=10,
            cv_folds=2
        )
        
        print(f"✅ Fallback test completed")
        print(f"   Evaluations: {history.n_evaluations}")
        print(f"   Errors: {history.n_errors}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

if __name__ == "__main__":
    print("🌌 Quantum Hyperparameter Search - Robust Functionality Test")
    print("=" * 70)
    
    success = True
    
    # Run tests
    success &= test_robust_optimization()
    success &= test_error_handling()
    success &= test_security_features()
    success &= test_fallback_behavior()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 All robust functionality tests passed! The library is reliable.")
    else:
        print("❌ Some tests failed. The library needs more robustness work.")
    
    print("=" * 70)