#!/usr/bin/env python3
"""
Test basic functionality of the quantum hyperparameter search library.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Test if we can import the library
try:
    from quantum_hyper_search import QuantumHyperSearch
    print("✅ Successfully imported QuantumHyperSearch")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def test_basic_optimization():
    """Test basic quantum hyperparameter optimization."""
    print("\n🧪 Testing basic quantum hyperparameter optimization...")
    
    # Create a simple dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        n_informative=8,
        random_state=42
    )
    print(f"✅ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define a simple search space
    search_space = {
        'n_estimators': [10, 25, 50],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    print(f"✅ Search space: {search_space}")
    
    try:
        # Initialize quantum optimizer with simple simulator backend
        qhs = QuantumHyperSearch(
            backend='simple',
            encoding='one_hot',
            penalty_strength=2.0
        )
        print("✅ Initialized QuantumHyperSearch with simulator backend")
        
        # Run optimization
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X,
            y=y,
            n_iterations=3,  # Keep small for testing
            quantum_reads=100,  # Keep small for testing
            cv_folds=3,  # Keep small for testing
            scoring='accuracy'
        )
        
        print(f"✅ Optimization completed!")
        print(f"🏆 Best parameters: {best_params}")
        print(f"📊 Best score: {history.best_score:.4f}")
        
        # Verify the parameters work
        if best_params:
            model = RandomForestClassifier(**best_params)
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            print(f"✅ Verification: CV score = {scores.mean():.4f} ± {scores.std():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qubo_encoder():
    """Test QUBO encoder functionality."""
    print("\n🧪 Testing QUBO encoder...")
    
    try:
        from quantum_hyper_search.core.qubo_encoder import QUBOEncoder
        
        encoder = QUBOEncoder(encoding='one_hot', penalty_strength=2.0)
        print("✅ Created QUBO encoder")
        
        # Test encoding
        param_space = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b']
        }
        
        Q, offset, var_map = encoder.encode_search_space(param_space)
        print(f"✅ Encoded parameter space")
        print(f"   QUBO size: {len([k for k in Q.keys() if k[0] == k[1]])} variables")
        print(f"   Variable mapping: {var_map}")
        
        # Test decoding
        sample = {i: 1 if i == 0 else 0 for i in range(len(var_map))}
        decoded = encoder.decode_sample(sample, var_map, param_space)
        print(f"✅ Decoded sample: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ QUBO encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation():
    """Test validation functions."""
    print("\n🧪 Testing validation functions...")
    
    try:
        from quantum_hyper_search.utils.validation import (
            validate_search_space, validate_model_class, validate_data,
            validate_optimization_params, ValidationError
        )
        
        # Test search space validation
        search_space = {'param1': [1, 2, 3], 'param2': ['a', 'b']}
        validated = validate_search_space(search_space)
        print("✅ Search space validation passed")
        
        # Test model class validation
        validate_model_class(RandomForestClassifier)
        print("✅ Model class validation passed")
        
        # Test data validation
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        X_val, y_val = validate_data(X, y)
        print("✅ Data validation passed")
        
        # Test optimization params validation
        validate_optimization_params(10, 100, 3, 'accuracy')
        print("✅ Optimization params validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌌 Quantum Hyperparameter Search - Basic Functionality Test")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_validation()
    success &= test_qubo_encoder()
    success &= test_basic_optimization()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All basic tests passed! The library is functional.")
    else:
        print("❌ Some tests failed. The library needs more work.")
    
    print("=" * 60)