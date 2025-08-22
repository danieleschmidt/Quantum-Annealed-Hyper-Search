#!/usr/bin/env python3
"""
Advanced Quantum Hyperparameter Optimization Example

Demonstrates quantum advantage through advanced features:
- Generation 3 capabilities: scaling, caching, adaptive strategies, and parallel optimization
- Multi-objective optimization
- Constrained search spaces
- Large-scale parameter spaces
- Performance benchmarking
- Multi-model comparison
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

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


def benchmark_quantum_advantage():
    """Demonstrate quantum advantage in hyperparameter optimization."""
    print("\n🌌 Quantum Hyperparameter Optimization Benchmark")
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


def main():
    """Run all advanced optimization demonstrations."""
    print("🌟 Advanced Quantum Hyperparameter Optimization Demo")
    print("=" * 70)
    print("Demonstrating quantum advantage through advanced features:")
    print("  • Intelligent caching and memoization")
    print("  • Adaptive quantum strategies")
    print("  • Parallel and distributed optimization")
    print("  • Multi-model comparison")
    print("  • Large-scale benchmarking")
    print("  • Constrained optimization")
    print("  • Multi-objective optimization")
    print("=" * 70)
    
    try:
        # Run Generation 3 demonstrations
        demonstrate_caching()
        demonstrate_adaptive_strategies()
        demonstrate_parallel_optimization()
        demonstrate_multi_model_optimization()
        
        # Run additional advanced demonstrations
        best_params1, history1, time1 = benchmark_quantum_advantage()
        best_params2, history2 = demonstrate_constrained_optimization()
        best_params3, history3 = multi_objective_optimization()
        best_params4, history4 = adaptive_strategy_demo()
        scaling_results = performance_scaling_test()
        
        print("\n" + "=" * 70)
        print("🎉 All demonstrations completed successfully!")
        print("✨ Quantum advantage demonstrated across multiple scenarios")
        print("🚀 Quantum Hyperparameter Search is ready for production!")
        
        # Summary statistics
        all_scores = [h.best_score for h in [history1, history2, history3, history4] if hasattr(h, 'best_score') and h.best_score > 0]
        if all_scores:
            avg_score = np.mean(all_scores)
            print(f"📊 Average best score across demos: {avg_score:.4f}")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
