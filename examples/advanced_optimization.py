#!/usr/bin/env python3
"""
Advanced quantum hyperparameter optimization example.

Demonstrates Generation 3 capabilities: scaling, caching, adaptive strategies,
and parallel optimization.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from quantum_hyper_search import QuantumHyperSearch
from quantum_hyper_search.optimization.caching import get_global_cache, configure_global_cache
from quantum_hyper_search.optimization.adaptive_strategies import AdaptiveQuantumSearch
from quantum_hyper_search.optimization.parallel_optimization import parallel_hyperparameter_search


def create_advanced_dataset():
    """Create a more complex dataset for testing.""" 
    print("📊 Creating advanced multi-class dataset...")
    
    X, y = make_classification(
        n_samples=2000,
        n_features=50,
        n_classes=3,
        n_informative=30,
        n_redundant=10,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    return X_train, X_test, y_train, y_test


def demonstrate_caching():
    """Demonstrate intelligent caching capabilities."""
    print("\n🗄️  Testing Intelligent Caching System")
    print("=" * 50)
    
    # Configure global cache
    cache = configure_global_cache(
        cache_dir='.advanced_quantum_cache',
        max_cache_size_mb=100,
        ttl_hours=1.0
    )
    
    X_train, X_test, y_train, y_test = create_advanced_dataset()
    
    search_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("🔥 First optimization run (populating cache)...")
    start_time = time.time()
    
    qhs = QuantumHyperSearch(
        backend='simulator',
        verbose=False,
        random_seed=42
    )
    
    best_params1, history1 = qhs.optimize(
        model_class=RandomForestClassifier,
        param_space=search_space,
        X=X_train,
        y=y_train,
        n_iterations=8,
        quantum_reads=50,
        cv_folds=3
    )
    
    first_run_time = time.time() - start_time
    
    print("⚡ Second optimization run (using cache)...")
    start_time = time.time()
    
    qhs2 = QuantumHyperSearch(
        backend='simulator',
        verbose=False,
        random_seed=42  # Same seed to trigger cache hits
    )
    
    best_params2, history2 = qhs2.optimize(
        model_class=RandomForestClassifier,
        param_space=search_space,
        X=X_train,
        y=y_train,
        n_iterations=8,
        quantum_reads=50,
        cv_folds=3
    )
    
    second_run_time = time.time() - start_time
    
    # Display caching results
    cache_stats = cache.get_cache_statistics()
    speedup = first_run_time / second_run_time if second_run_time > 0 else 1.0
    
    print(f"\n📈 Caching Performance:")
    print(f"   First run: {first_run_time:.2f}s")
    print(f"   Second run: {second_run_time:.2f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Cache hit rate: {cache_stats['performance']['hit_rate']:.1%}")
    print(f"   Cache entries: {cache_stats['memory_cache']['eval_entries']}")


def demonstrate_adaptive_strategies():
    """Demonstrate adaptive quantum search strategies."""
    print("\n🧠 Testing Adaptive Quantum Strategies")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = create_advanced_dataset()
    
    # Initialize adaptive search
    adaptive_search = AdaptiveQuantumSearch(
        initial_quantum_reads=100,
        initial_penalty_strength=2.0,
        adaptation_rate=0.15
    )
    
    search_space = {
        'n_estimators': [100, 200, 500],
        'max_depth': [15, 25, None],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    print("🔄 Running adaptive optimization...")
    
    qhs = QuantumHyperSearch(
        backend='simulator',
        verbose=False
    )
    
    # Track adaptation over iterations
    adaptation_history = []
    
    for iteration in range(12):
        # Get current adaptive parameters
        current_params = adaptive_search.get_current_parameters()
        adaptation_history.append(current_params.copy())
        
        # Run single iteration with adaptive parameters
        best_params, history = qhs.optimize(
            model_class=GradientBoostingClassifier,
            param_space=search_space,
            X=X_train,
            y=y_train,
            n_iterations=1,
            quantum_reads=current_params['quantum_reads'],
            cv_folds=3
        )
        
        # Update adaptive search with results
        score = history.best_score
        violations = 0  # Would normally check constraint violations
        adaptive_search.update_performance(score, violations)
        
        print(f"   Iteration {iteration+1}: Score={score:.4f}, "
              f"Reads={current_params['quantum_reads']}, "
              f"Penalty={current_params['penalty_strength']:.2f}")
    
    # Analyze adaptation patterns
    analysis = adaptive_search.analyze_optimization_patterns()
    print(f"\n🔍 Adaptation Analysis:")
    print(f"   Convergence detected: {analysis['convergence_detected']}")
    print(f"   Parameter sensitivity: {analysis['parameter_sensitivity']}")
    if analysis['optimization_efficiency']:
        print(f"   Efficiency rating: {analysis['optimization_efficiency']['efficiency_rating']}")


def demonstrate_parallel_optimization():
    """Demonstrate parallel and distributed optimization."""
    print("\n⚡ Testing Parallel Quantum Optimization")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = create_advanced_dataset()
    
    # Define evaluation function for parallel use
    def evaluate_params(params):
        """Evaluate model parameters."""
        try:
            model = RandomForestClassifier(**params, random_state=42)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return float(np.mean(scores))
        except Exception:
            return 0.0
    
    search_space = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("🚀 Running parallel hyperparameter search...")
    start_time = time.time()
    
    # Run parallel search
    results = parallel_hyperparameter_search(
        search_space=search_space,
        evaluation_function=evaluate_params,
        n_parallel_evaluations=20,
        n_workers=4
    )
    
    parallel_time = time.time() - start_time
    
    # Sequential comparison
    print("🐌 Running sequential search for comparison...")
    start_time = time.time()
    
    sequential_results = []
    for _ in range(20):
        params = {
            param: np.random.choice(values)
            for param, values in search_space.items()
        }
        score = evaluate_params(params)
        sequential_results.append((params, score))
    
    sequential_time = time.time() - start_time
    
    # Analyze results
    best_parallel = max(results, key=lambda x: x[1])
    best_sequential = max(sequential_results, key=lambda x: x[1])
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    
    print(f"\n📊 Parallel Optimization Results:")
    print(f"   Parallel time: {parallel_time:.2f}s")
    print(f"   Sequential time: {sequential_time:.2f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Best parallel score: {best_parallel[1]:.4f}")
    print(f"   Best sequential score: {best_sequential[1]:.4f}")
    print(f"   Parallel advantage: {(best_parallel[1] - best_sequential[1]):.4f}")


def demonstrate_multi_model_optimization():
    """Demonstrate optimization across multiple model types."""
    print("\n🎯 Testing Multi-Model Optimization")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = create_advanced_dataset()
    
    # Define different models and their search spaces
    model_configs = {
        'RandomForest': {
            'model_class': RandomForestClassifier,
            'search_space': {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'model_class': GradientBoostingClassifier,
            'search_space': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'SVM': {
            'model_class': SVC,
            'search_space': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
    }
    
    print("🔄 Optimizing multiple model types...")
    
    results = {}
    
    for model_name, config in model_configs.items():
        print(f"   Optimizing {model_name}...")
        
        qhs = QuantumHyperSearch(
            backend='simulator',
            verbose=False
        )
        
        start_time = time.time()
        best_params, history = qhs.optimize(
            model_class=config['model_class'],
            param_space=config['search_space'],
            X=X_train,
            y=y_train,
            n_iterations=6,
            quantum_reads=50,
            cv_folds=3
        )
        optimization_time = time.time() - start_time
        
        # Test best model
        final_model = config['model_class'](**best_params, random_state=42)
        final_model.fit(X_train, y_train)
        test_score = final_model.score(X_test, y_test)
        
        results[model_name] = {
            'best_params': best_params,
            'cv_score': history.best_score,
            'test_score': test_score,
            'optimization_time': optimization_time,
            'trials': len(history.trials)
        }
        
        print(f"     CV Score: {history.best_score:.4f}")
        print(f"     Test Score: {test_score:.4f}")
        print(f"     Time: {optimization_time:.2f}s")
    
    # Find best overall model
    best_model = max(results.items(), key=lambda x: x[1]['test_score'])
    
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   Test Score: {best_model[1]['test_score']:.4f}")
    print(f"   Parameters: {best_model[1]['best_params']}")


def main():
    """Run all advanced optimization demonstrations."""
    print("🌟 Advanced Quantum Hyperparameter Optimization Demo")
    print("=" * 60)
    print("Demonstrating Generation 3 capabilities:")
    print("  • Intelligent caching and memoization")
    print("  • Adaptive quantum strategies")
    print("  • Parallel and distributed optimization")
    print("  • Multi-model comparison")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_caching()
        demonstrate_adaptive_strategies()
        demonstrate_parallel_optimization()
        demonstrate_multi_model_optimization()
        
        print("\n✅ All advanced demonstrations completed successfully!")
        print("\n🚀 Quantum Hyperparameter Search is ready for production!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()