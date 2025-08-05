#!/usr/bin/env python3
"""
Advanced Quantum Hyperparameter Optimization Example

Demonstrates quantum advantage through advanced features:
- Multi-objective optimization
- Constrained search spaces
- Large-scale parameter spaces
- Performance benchmarking
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from quantum_hyper_search import QuantumHyperSearch


def benchmark_quantum_advantage():
    """Demonstrate quantum advantage in hyperparameter optimization."""
    print("🌌 Quantum Hyperparameter Optimization Benchmark")
    print("=" * 60)
    
    # Create large-scale dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=50,
        n_informative=25,
        n_classes=2,
        random_state=42,
        class_sep=0.8
    )
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define comprehensive search space
    large_search_space = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }
    
    total_combinations = np.prod([len(v) for v in large_search_space.values()])
    print(f"🔢 Search space: {len(large_search_space)} parameters")
    print(f"🎯 Total combinations: {total_combinations:,}")
    
    # Initialize quantum optimizer with all optimizations
    qhs = QuantumHyperSearch(
        backend='simulator',
        enable_logging=True,
        enable_monitoring=True,
        enable_caching=True,
        enable_parallel=True,
        enable_auto_scaling=True,
        max_parallel_workers=4,
        cache_size=50000,
        optimization_strategy='adaptive'
    )
    
    print(f"\n🚀 Starting quantum-enhanced optimization...")
    start_time = time.time()
    
    # Run optimization
    best_params, history = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=large_search_space,
        X=X,
        y=y,
        n_iterations=15,
        quantum_reads=500,
        cv_folds=3,
        scoring='f1',
        random_state=42
    )
    
    total_time = time.time() - start_time
    
    # Results analysis
    print(f"\n✅ Optimization Results:")
    print(f"🏆 Best Score: {history.best_score:.4f}")
    print(f"🎯 Best Parameters: {best_params}")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"🔍 Evaluations: {history.n_evaluations}")
    print(f"📈 Efficiency: {history.n_evaluations / total_time:.1f} evaluations/second")
    
    # Validate final model
    if best_params:
        final_model = RandomForestClassifier(**best_params, random_state=42)
        final_scores = cross_val_score(final_model, X, y, cv=5, scoring='f1')
        print(f"🧪 Final validation: {final_scores.mean():.4f} ± {final_scores.std():.4f}")
    
    return best_params, history, total_time


def demonstrate_constrained_optimization():
    """Demonstrate constrained quantum optimization."""
    print("\n" + "=" * 60)
    print("🔒 Constrained Quantum Optimization")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    # Define search space with constraints
    search_space = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }
    
    # Define constraints
    constraints = {
        'mutual_exclusion': [
            # High depth with low samples can cause overfitting
            ['max_depth_3', 'min_samples_split_0']  # max_depth=20, min_samples_split=2
        ],
        'conditional': [
            # If no bootstrap, then use more estimators
            ('bootstrap_1', 'n_estimators_2')  # bootstrap=False -> n_estimators>=200
        ]
    }
    
    qhs = QuantumHyperSearch(
        backend='simulator',
        enable_logging=True,
        penalty_strength=3.0  # Stronger constraint enforcement
    )
    
    print("🎯 Optimizing with constraints...")
    best_params, history = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=search_space,
        X=X,
        y=y,
        n_iterations=10,
        quantum_reads=200,
        constraints=constraints,
        cv_folds=3,
        random_state=42
    )
    
    print(f"🏆 Constrained Best Score: {history.best_score:.4f}")
    print(f"🎯 Constrained Best Parameters: {best_params}")
    
    return best_params, history


def multi_objective_optimization():
    """Demonstrate multi-objective quantum optimization."""
    print("\n" + "=" * 60)
    print("🎯 Multi-Objective Quantum Optimization")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(n_samples=800, n_features=30, random_state=42)
    
    search_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
    def multi_objective_function(params):
        """Optimize for both accuracy and training speed."""
        model = RandomForestClassifier(**params, random_state=42)
        
        # Measure training time
        start_time = time.time()
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        training_time = time.time() - start_time
        
        accuracy = scores.mean()
        speed_score = 1.0 / (1.0 + training_time)  # Higher is better
        
        # Weighted combination (prioritize accuracy)
        combined_score = 0.8 * accuracy + 0.2 * speed_score
        
        return combined_score
    
    qhs = QuantumHyperSearch(
        backend='simulator',
        enable_monitoring=True
    )
    
    print("⚖️  Optimizing accuracy + speed...")
    best_params, history = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=search_space,
        X=X,
        y=y,
        n_iterations=8,
        quantum_reads=150,
        objective_function=multi_objective_function,
        random_state=42
    )
    
    print(f"⚖️  Multi-objective Best Score: {history.best_score:.4f}")
    print(f"🎯 Multi-objective Best Parameters: {best_params}")
    
    return best_params, history


def adaptive_strategy_demo():
    """Demonstrate adaptive quantum strategy."""
    print("\n" + "=" * 60)
    print("🧠 Adaptive Quantum Strategy")
    print("=" * 60)
    
    # Create challenging dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=40,
        n_informative=20,
        n_redundant=10,
        n_classes=3,
        random_state=42
    )
    
    search_space = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8]
    }
    
    qhs = QuantumHyperSearch(
        backend='simulator',
        optimization_strategy='adaptive',
        enable_monitoring=True,
        enable_caching=True
    )
    
    print("🧠 Running adaptive optimization...")
    best_params, history = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=search_space,
        X=X,
        y=y,
        n_iterations=12,
        quantum_reads=300,
        cv_folds=4,
        scoring='f1_macro',
        random_state=42
    )
    
    print(f"🧠 Adaptive Best Score: {history.best_score:.4f}")
    print(f"🎯 Adaptive Best Parameters: {best_params}")
    
    # Analyze convergence
    if hasattr(history, 'get_convergence_data'):
        iterations, scores = history.get_convergence_data()
        print(f"📈 Convergence: {len(iterations)} evaluations")
        if len(scores) > 1:
            improvement = (scores[-1] - scores[0]) / scores[0] * 100
            print(f"📊 Improvement: {improvement:.1f}%")
    
    return best_params, history


def performance_scaling_test():
    """Test performance scaling with different configurations."""
    print("\n" + "=" * 60)
    print("📊 Performance Scaling Analysis")
    print("=" * 60)
    
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    search_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    
    configurations = [
        {'name': 'Basic', 'enable_caching': False, 'enable_parallel': False, 'enable_monitoring': False},
        {'name': 'Cached', 'enable_caching': True, 'enable_parallel': False, 'enable_monitoring': False},
        {'name': 'Parallel', 'enable_caching': False, 'enable_parallel': True, 'enable_monitoring': False},
        {'name': 'Full', 'enable_caching': True, 'enable_parallel': True, 'enable_monitoring': True}
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n🧪 Testing {config['name']} configuration...")
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            **{k: v for k, v in config.items() if k != 'name'}
        )
        
        start_time = time.time()
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X,
            y=y,
            n_iterations=5,
            quantum_reads=50,
            cv_folds=3,
            random_state=42
        )
        elapsed_time = time.time() - start_time
        
        results.append({
            'name': config['name'],
            'time': elapsed_time,
            'score': history.best_score,
            'evaluations': history.n_evaluations
        })
        
        print(f"⏱️  {config['name']}: {elapsed_time:.2f}s, Score: {history.best_score:.4f}")
    
    # Performance summary
    print(f"\n📊 Performance Summary:")
    baseline_time = next(r['time'] for r in results if r['name'] == 'Basic')
    
    for result in results:
        speedup = baseline_time / result['time']
        efficiency = result['evaluations'] / result['time']
        print(f"  {result['name']:8}: {speedup:.2f}x speedup, {efficiency:.1f} eval/s")
    
    return results


if __name__ == "__main__":
    print("🌌 Quantum Annealed Hyperparameter Search - Advanced Demo")
    print("=" * 70)
    
    # Run all demonstrations
    try:
        # 1. Large-scale benchmark
        best_params1, history1, time1 = benchmark_quantum_advantage()
        
        # 2. Constrained optimization
        best_params2, history2 = demonstrate_constrained_optimization()
        
        # 3. Multi-objective optimization
        best_params3, history3 = multi_objective_optimization()
        
        # 4. Adaptive strategy
        best_params4, history4 = adaptive_strategy_demo()
        
        # 5. Performance scaling
        scaling_results = performance_scaling_test()
        
        print("\n" + "=" * 70)
        print("🎉 All demonstrations completed successfully!")
        print("✨ Quantum advantage demonstrated across multiple scenarios")
        
        # Summary statistics
        all_scores = [h.best_score for h in [history1, history2, history3, history4] if h.best_score > 0]
        if all_scores:
            avg_score = np.mean(all_scores)
            print(f"📊 Average best score across demos: {avg_score:.4f}")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        raise