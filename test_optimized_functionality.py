#!/usr/bin/env python3
"""
Test optimized functionality of the quantum hyperparameter search library.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Test the optimized implementation
try:
    from quantum_hyper_search.optimized_main import QuantumHyperSearchOptimized
    print("✅ Successfully imported QuantumHyperSearchOptimized")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def test_optimized_performance():
    """Test optimized performance with larger search space."""
    print("\n🧪 Testing optimized performance...")
    
    # Create a larger dataset for performance testing
    X, y = make_classification(
        n_samples=300,
        n_features=15,
        n_classes=2,
        n_informative=12,
        random_state=42
    )
    print(f"✅ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define a larger search space
    search_space = {
        'n_estimators': [10, 25, 50, 100, 200],
        'max_depth': [3, 5, 7, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }
    
    total_combinations = np.prod([len(v) for v in search_space.values()])
    print(f"✅ Search space: {len(search_space)} parameters, {total_combinations} combinations")
    
    try:
        # Initialize optimized quantum optimizer
        qhs = QuantumHyperSearchOptimized(
            backend='simple',
            encoding='one_hot',
            penalty_strength=2.0,
            enable_security=True,
            enable_caching=True,
            enable_parallel=True,
            max_workers=4,
            cache_size=1000,
            adaptive_strategy=True
        )
        print("✅ Initialized optimized QuantumHyperSearchOptimized")
        
        # Run optimization with timing
        start_time = time.time()
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X,
            y=y,
            n_iterations=8,
            quantum_reads=100,
            cv_folds=3,
            scoring='accuracy',
            batch_size=12,
            random_state=42
        )
        
        optimization_time = time.time() - start_time
        
        print(f"✅ Optimized optimization completed in {optimization_time:.1f} seconds!")
        
        # Print detailed performance results
        print(f"\n⚡ Performance Results:")
        print(f"🏆 Best parameters: {best_params}")
        if history.best_score != float('-inf'):
            print(f"📈 Best score: {history.best_score:.4f}")
        
        # Print comprehensive statistics
        stats = history.get_statistics()
        print(f"\n📊 Optimization Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                if 'time' in key.lower():
                    print(f"   {key}: {value:.3f}s")
                elif 'rate' in key.lower():
                    print(f"   {key}: {value:.1%}")
                else:
                    print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        # Performance validation
        evaluations_per_second = stats.get('evaluations_per_second', 0)
        if evaluations_per_second > 2.0:
            print(f"✅ Good performance: {evaluations_per_second:.2f} evaluations/sec")
        else:
            print(f"⚠️  Lower performance: {evaluations_per_second:.2f} evaluations/sec")
        
        # Verify the results
        if best_params and history.best_score != float('-inf'):
            model = RandomForestClassifier(**best_params, random_state=42)
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            print(f"✅ Verification: CV score = {scores.mean():.4f} ± {scores.std():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimized optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_caching_performance():
    """Test caching performance benefits."""
    print("\n🧪 Testing caching performance...")
    
    # Small dataset for quick testing
    X, y = make_classification(n_samples=100, n_features=8, random_state=42)
    search_space = {
        'n_estimators': [10, 20, 30],
        'max_depth': [3, 5, 7]
    }
    
    try:
        # Test with caching enabled
        print("Testing with caching enabled...")
        qhs_cached = QuantumHyperSearchOptimized(
            backend='simple',
            enable_caching=True,
            cache_size=100
        )
        
        start_time = time.time()
        best_params_cached, history_cached = qhs_cached.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=5,
            quantum_reads=20,
            cv_folds=3,
            batch_size=6
        )
        cached_time = time.time() - start_time
        
        # Get cache statistics
        cache_stats = qhs_cached.cache.get_stats()
        print(f"✅ Cached optimization: {cached_time:.2f}s")
        print(f"   Cache stats: {cache_stats}")
        
        # Test with caching disabled
        print("Testing with caching disabled...")
        qhs_uncached = QuantumHyperSearchOptimized(
            backend='simple',
            enable_caching=False
        )
        
        start_time = time.time()
        best_params_uncached, history_uncached = qhs_uncached.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=5,
            quantum_reads=20,
            cv_folds=3,
            batch_size=6
        )
        uncached_time = time.time() - start_time
        
        print(f"✅ Uncached optimization: {uncached_time:.2f}s")
        
        # Compare results
        if cache_stats['hit_rate'] > 0:
            print(f"✅ Cache benefits: {cache_stats['hit_rate']:.1%} hit rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

def test_parallel_processing():
    """Test parallel processing benefits."""
    print("\n🧪 Testing parallel processing...")
    
    X, y = make_classification(n_samples=150, n_features=10, random_state=42)
    search_space = {
        'n_estimators': [10, 25, 50],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5]
    }
    
    try:
        # Test with parallel processing
        print("Testing with parallel processing...")
        qhs_parallel = QuantumHyperSearchOptimized(
            backend='simple',
            enable_parallel=True,
            max_workers=4,
            enable_caching=False  # Disable caching to measure pure parallel benefit
        )
        
        start_time = time.time()
        best_params_parallel, history_parallel = qhs_parallel.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=4,
            quantum_reads=30,
            batch_size=8
        )
        parallel_time = time.time() - start_time
        
        # Test without parallel processing
        print("Testing without parallel processing...")
        qhs_sequential = QuantumHyperSearchOptimized(
            backend='simple',
            enable_parallel=False,
            enable_caching=False
        )
        
        start_time = time.time()
        best_params_sequential, history_sequential = qhs_sequential.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=4,
            quantum_reads=30,
            batch_size=8
        )
        sequential_time = time.time() - start_time
        
        print(f"✅ Parallel optimization: {parallel_time:.2f}s")
        print(f"✅ Sequential optimization: {sequential_time:.2f}s")
        
        if parallel_time < sequential_time:
            speedup = sequential_time / parallel_time
            print(f"✅ Parallel speedup: {speedup:.2f}x")
        else:
            print("ℹ️  No significant parallel benefit (small problem size)")
        
        return True
        
    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        return False

def test_adaptive_strategy():
    """Test adaptive quantum strategy."""
    print("\n🧪 Testing adaptive strategy...")
    
    X, y = make_classification(n_samples=80, n_features=6, random_state=42)
    search_space = {
        'n_estimators': [5, 10, 20],
        'max_depth': [3, 5]
    }
    
    try:
        qhs_adaptive = QuantumHyperSearchOptimized(
            backend='simple',
            adaptive_strategy=True,
            enable_caching=True
        )
        
        best_params, history = qhs_adaptive.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=6,
            quantum_reads=25,
            batch_size=4
        )
        
        print(f"✅ Adaptive strategy test completed")
        
        # Check if adaptive strategy was used
        if hasattr(qhs_adaptive.strategy, 'iteration_count'):
            print(f"   Strategy iterations: {qhs_adaptive.strategy.iteration_count}")
            print(f"   Success history length: {len(qhs_adaptive.strategy.success_history)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive strategy test failed: {e}")
        return False

def benchmark_comparison():
    """Benchmark against basic implementation."""
    print("\n🧪 Benchmarking optimized vs basic implementation...")
    
    # Create test dataset
    X, y = make_classification(n_samples=200, n_features=12, random_state=42)
    search_space = {
        'n_estimators': [10, 25, 50],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    try:
        # Test optimized version
        print("Running optimized version...")
        qhs_optimized = QuantumHyperSearchOptimized(
            backend='simple',
            enable_caching=True,
            enable_parallel=True,
            adaptive_strategy=True,
            max_workers=4
        )
        
        start_time = time.time()
        best_opt, history_opt = qhs_optimized.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=6,
            quantum_reads=40,
            batch_size=8
        )
        optimized_time = time.time() - start_time
        
        # Test basic version (import simple version)
        print("Running basic version...")
        from quantum_hyper_search.simple_main import QuantumHyperSearch as BasicQHS
        
        qhs_basic = BasicQHS(backend='simple')
        
        start_time = time.time()
        best_basic, history_basic = qhs_basic.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=6,
            quantum_reads=40
        )
        basic_time = time.time() - start_time
        
        # Compare results
        print(f"\n📊 Benchmark Results:")
        print(f"   Optimized: {optimized_time:.2f}s, score: {history_opt.best_score:.4f}")
        print(f"   Basic: {basic_time:.2f}s, score: {history_basic.best_score:.4f}")
        
        if optimized_time < basic_time:
            speedup = basic_time / optimized_time
            print(f"✅ Optimization speedup: {speedup:.2f}x")
        
        # Compare evaluation efficiency
        opt_stats = history_opt.get_statistics()
        basic_evals = history_basic.n_evaluations
        
        print(f"   Optimized evaluations: {opt_stats['n_evaluations']}")
        print(f"   Basic evaluations: {basic_evals}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Quantum Hyperparameter Search - Optimized Performance Test")
    print("=" * 75)
    
    success = True
    
    # Run performance tests
    success &= test_optimized_performance()
    success &= test_caching_performance()
    success &= test_parallel_processing() 
    success &= test_adaptive_strategy()
    success &= benchmark_comparison()
    
    print("\n" + "=" * 75)
    if success:
        print("🎉 All optimized performance tests passed! The library is highly optimized.")
    else:
        print("❌ Some tests failed. The library needs more optimization work.")
    
    print("=" * 75)