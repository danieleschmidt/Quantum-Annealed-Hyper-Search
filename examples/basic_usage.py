#!/usr/bin/env python3
"""
Basic usage example of Quantum-Annealed Hyperparameter Search.

This example demonstrates how to use the quantum hyperparameter optimization
for a simple machine learning classification task.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from quantum_hyper_search import QuantumHyperSearch


def main():
    """Run basic quantum hyperparameter optimization example."""
    print("🌌 Quantum-Annealed Hyperparameter Search - Basic Example")
    print("=" * 60)
    
    # Create sample dataset
    print("📊 Creating sample classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Define hyperparameter search space
    print("\n🔍 Defining hyperparameter search space...")
    search_space = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    print("Search space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    
    # Initialize quantum optimizer
    print("\n⚛️  Initializing quantum hyperparameter search...")
    qhs = QuantumHyperSearch(
        backend='simulator',  # Use simulator for this example
        penalty_strength=2.0,
        verbose=True
    )
    
    print(f"Backend: {qhs.backend.get_properties()['name']}")
    
    # Run quantum-enhanced optimization
    print("\n🚀 Running quantum-enhanced optimization...")
    print("This may take a few minutes...")
    
    best_params, history = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=search_space,
        X=X_train,
        y=y_train,
        n_iterations=20,
        quantum_reads=100,
        cv_folds=5,
        scoring='accuracy',
        random_seed=42
    )
    
    # Display results
    print("\n📈 Optimization Results")
    print("=" * 30)
    print(f"Best CV Score: {history.best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Train final model with best parameters
    print("\n🎯 Training final model...")
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    test_score = final_model.score(X_test, y_test)
    print(f"Test Set Accuracy: {test_score:.4f}")
    
    # Show optimization history
    print("\n📊 Optimization History")
    print("=" * 25)
    print("Iteration | CV Score | Parameters")
    print("-" * 50)
    
    for i, (trial, score) in enumerate(zip(history.trials, history.scores)):
        params_str = ", ".join([f"{k}={v}" for k, v in trial.items()])
        if len(params_str) > 30:
            params_str = params_str[:27] + "..."
        print(f"{i+1:9d} | {score:8.4f} | {params_str}")
    
    # Plot convergence if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        iterations, best_scores = history.get_convergence_data()
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, best_scores, 'b-', linewidth=2, label='Best Score')
        plt.scatter(range(len(history.scores)), history.scores, 
                   alpha=0.6, c='red', s=30, label='All Trials')
        plt.xlabel('Iteration')
        plt.ylabel('Cross-Validation Score')
        plt.title('Quantum Hyperparameter Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('optimization_convergence.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Convergence plot saved as 'optimization_convergence.png'")
        
    except ImportError:
        print("\n📊 Install matplotlib to see convergence plots: pip install matplotlib")
    
    print("\n✅ Quantum hyperparameter optimization complete!")
    
    # Summary of quantum advantage
    print("\n🌟 Quantum Advantage Analysis")
    print("=" * 30)
    print(f"• Explored {len(history.trials)} parameter configurations")
    print(f"• Search space size: {np.prod([len(v) for v in search_space.values()]):,} combinations")
    print(f"• Coverage: {len(history.trials) / np.prod([len(v) for v in search_space.values()]) * 100:.2f}%")
    print(f"• Best result found in iteration {history.scores.index(history.best_score) + 1}")
    
    improvement = (history.best_score - min(history.scores)) / min(history.scores) * 100
    print(f"• Performance improvement: {improvement:.1f}% over worst trial")


if __name__ == "__main__":
    main()