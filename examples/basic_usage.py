"""
Basic usage example for Quantum Annealed Hyperparameter Search.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from quantum_hyper_search import QuantumHyperSearch


def main():
    """Run basic quantum hyperparameter optimization example."""
    print("🌌 Quantum Annealed Hyperparameter Search - Basic Example")
    print("=" * 60)
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Define hyperparameter search space
    print("\n🔍 Defining search space...")
    search_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    total_combinations = np.prod([len(v) for v in search_space.values()])
    print(f"Total parameter combinations: {total_combinations}")
    
    # Initialize quantum optimizer
    print("\n⚛️  Initializing quantum optimizer...")
    qhs = QuantumHyperSearch(
        backend='simulator',  # Use simulator for this example
        encoding='one_hot',
        penalty_strength=2.0
    )
    
    # Run quantum-enhanced optimization
    print("\n🚀 Starting quantum optimization...")
    try:
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X_train,
            y=y_train,
            n_iterations=5,  # Keep small for demo
            quantum_reads=100,  # Keep small for demo
            cv_folds=3,
            scoring='accuracy',
            random_state=42
        )
        
        print("\n✅ Optimization complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {history.best_score:.4f}")
        print(f"Total evaluations: {history.n_evaluations}")
        
        # Train final model and evaluate on test set
        print("\n📈 Evaluating final model...")
        final_model = RandomForestClassifier(**best_params, random_state=42)
        final_model.fit(X_train, y_train)
        test_score = final_model.score(X_test, y_test)
        
        print(f"Test set accuracy: {test_score:.4f}")
        
        # Show optimization statistics
        print("\n📊 Optimization Statistics:")
        stats = history.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Show parameter importance if enough evaluations
        if history.n_evaluations >= 10:
            print("\n🎯 Parameter Importance:")
            importance = history.get_parameter_importance()
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {param}: {imp:.3f}")
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Example completed successfully!")
    else:
        print("\n💥 Example failed!")
        exit(1)