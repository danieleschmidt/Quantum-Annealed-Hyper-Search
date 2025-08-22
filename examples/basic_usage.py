#!/usr/bin/env python3
"""
Basic Quantum Hyperparameter Optimization Example

This example demonstrates the core functionality of the quantum hyperparameter
search library with a simple use case.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import quantum hyperparameter search
try:
    from quantum_hyper_search import QuantumHyperSearch
except ImportError:
    print("quantum_hyper_search not found, using mock implementation")
    
    class QuantumHyperSearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            print(f"Mock QuantumHyperSearch initialized with: {kwargs}")
            
        def optimize(self, model_class, param_space, X, y, **optimization_kwargs):
            print("Running mock quantum optimization...")
            # Return mock results
            import random
            best_params = {param: random.choice(values) for param, values in param_space.items()}
            
            class MockHistory:
                def __init__(self):
                    self.best_score = 0.85 + random.random() * 0.1
                    self.trials = [{'params': best_params, 'score': self.best_score}]
                    
            return best_params, MockHistory()


def create_sample_dataset():
    """Create a sample classification dataset for optimization."""
    print("Creating sample dataset...")
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        class_sep=1.0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset created: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def basic_optimization():
    """Demonstrate basic quantum hyperparameter optimization."""
    print("\n=== Basic Quantum Hyperparameter Optimization ===")
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_sample_dataset()
    
    # Define hyperparameter search space
    param_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print(f"Search space: {len(param_space)} parameters")
    total_combinations = np.prod([len(values) for values in param_space.values()])
    print(f"Total combinations: {total_combinations}")
    
    # Initialize quantum hyperparameter search
    qhs = QuantumHyperSearch(
        backend='simulator',  # Use simulator for this example
        verbose=True,
        random_seed=42
    )
    
    print("\nStarting quantum optimization...")
    
    # Run optimization
    best_params, history = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=param_space,
        X=X_train,
        y=y_train,
        n_iterations=10,
        quantum_reads=100,
        cv_folds=5,
        scoring='accuracy',
        random_seed=42
    )
    
    print(f"\nOptimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {history.best_score:.4f}")
    
    # Evaluate final model
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    test_score = final_model.score(X_test, y_test)
    
    print(f"Test set accuracy: {test_score:.4f}")
    
    return best_params, history, test_score


def demonstrate_convergence():
    """Demonstrate optimization convergence tracking."""
    print("\n=== Convergence Analysis ===")
    
    X_train, X_test, y_train, y_test = create_sample_dataset()
    
    param_space = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [3, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
    qhs = QuantumHyperSearch(
        backend='simulator',
        verbose=False
    )
    
    best_params, history = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=param_space,
        X=X_train,
        y=y_train,
        n_iterations=15,
        quantum_reads=50,
        cv_folds=3,
        random_seed=42
    )
    
    # Analyze convergence
    if hasattr(history, 'get_convergence_data'):
        iterations, scores = history.get_convergence_data()
        print(f"Convergence analysis:")
        print(f"  Total iterations: {len(iterations)}")
        print(f"  Initial score: {scores[0]:.4f}")
        print(f"  Final score: {scores[-1]:.4f}")
        print(f"  Improvement: {(scores[-1] - scores[0]):.4f}")
    
    return best_params, history


def compare_backends():
    """Compare different quantum backends."""
    print("\n=== Backend Comparison ===")
    
    X_train, X_test, y_train, y_test = create_sample_dataset()
    
    param_space = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    backends = ['simulator']  # Add more backends if available
    results = {}
    
    for backend in backends:
        print(f"\nTesting {backend} backend...")
        
        qhs = QuantumHyperSearch(
            backend=backend,
            verbose=False
        )
        
        import time
        start_time = time.time()
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=param_space,
            X=X_train,
            y=y_train,
            n_iterations=5,
            quantum_reads=50,
            cv_folds=3,
            random_seed=42
        )
        
        elapsed_time = time.time() - start_time
        
        results[backend] = {
            'best_score': history.best_score,
            'time': elapsed_time,
            'best_params': best_params
        }
        
        print(f"  Best score: {history.best_score:.4f}")
        print(f"  Time: {elapsed_time:.2f}s")
    
    return results


def main():
    """Run all basic examples."""
    print("🌟 Quantum Hyperparameter Search - Basic Examples")
    print("="*60)
    
    try:
        # Run basic optimization
        best_params, history, test_score = basic_optimization()
        
        # Demonstrate convergence tracking
        conv_params, conv_history = demonstrate_convergence()
        
        # Compare backends
        backend_results = compare_backends()
        
        print("\n" + "="*60)
        print("✅ All basic examples completed successfully!")
        print(f"Final test accuracy: {test_score:.4f}")
        print("🚀 Ready for advanced usage!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()