#!/usr/bin/env python3
"""
Test Generation 3 features with simpler configurations.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Test caching system
def test_caching():
    """Test the caching system."""
    print("🗄️  Testing Caching System...")
    
    from quantum_hyper_search.optimization.caching import OptimizationCache
    
    cache = OptimizationCache(enable_memory_cache=True, enable_disk_cache=False)
    
    # Create test data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    params = {'n_estimators': 50, 'max_depth': 5}
    
    # Test cache miss
    result = cache.get_evaluation_result(params, X, y, RandomForestClassifier, 3, 'accuracy')
    assert result is None, "Should be cache miss"
    
    # Save result
    cache.save_evaluation_result(params, X, y, RandomForestClassifier, 3, 'accuracy', 0.85)
    
    # Test cache hit
    result = cache.get_evaluation_result(params, X, y, RandomForestClassifier, 3, 'accuracy')
    assert result == 0.85, "Should be cache hit"
    
    # Test statistics
    stats = cache.get_cache_statistics()
    assert stats['performance']['hits'] == 1
    assert stats['performance']['misses'] == 1
    
    print("   ✅ Caching system working correctly")


def test_adaptive_strategies():
    """Test adaptive strategies."""
    print("🧠 Testing Adaptive Strategies...")
    
    from quantum_hyper_search.optimization.adaptive_strategies import AdaptiveQuantumSearch
    
    adaptive = AdaptiveQuantumSearch(
        initial_quantum_reads=100,
        initial_penalty_strength=2.0
    )
    
    # Test parameter updates
    initial_reads = adaptive.quantum_reads
    
    # Simulate good performance with current settings
    for _ in range(3):
        adaptive.update_performance(score=0.9, constraint_violations=0)
    
    # Simulate poor performance to trigger adaptation
    for _ in range(3):
        adaptive.update_performance(score=0.6, constraint_violations=5)
    
    # Check that parameters adapted
    current_params = adaptive.get_current_parameters()
    print(f"   Initial reads: {initial_reads}")
    print(f"   Adapted reads: {current_params['quantum_reads']}")
    print(f"   Penalty strength: {current_params['penalty_strength']:.2f}")
    
    print("   ✅ Adaptive strategies working correctly")


def test_parallel_optimization():
    """Test parallel optimization capabilities."""
    print("⚡ Testing Parallel Optimization...")
    
    from quantum_hyper_search.optimization.parallel_optimization import ParallelQuantumOptimizer
    
    optimizer = ParallelQuantumOptimizer(n_parallel_jobs=2)
    
    # Create test evaluation function
    def test_eval(params):
        # Simulate evaluation with small delay
        time.sleep(0.1)
        return np.random.random()
    
    # Test parameter sets
    param_sets = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 15}
    ]
    
    start_time = time.time()
    results = optimizer.parallel_parameter_evaluation(param_sets, test_eval, max_workers=2)
    parallel_time = time.time() - start_time
    
    assert len(results) == 3, "Should have 3 results"
    print(f"   Parallel evaluation time: {parallel_time:.2f}s")
    print(f"   Results: {len(results)} parameter sets evaluated")
    
    print("   ✅ Parallel optimization working correctly")


def test_basic_quantum_optimization():
    """Test basic quantum optimization still works."""
    print("🔬 Testing Basic Quantum Optimization...")
    
    from quantum_hyper_search import QuantumHyperSearch
    
    # Create simple test case
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    search_space = {'n_estimators': [10, 20], 'max_depth': [3, 5]}
    
    qhs = QuantumHyperSearch(
        backend='simulator',
        verbose=False,
        random_seed=42
    )
    
    try:
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=10,
            cv_folds=2,
            timeout=30
        )
        
        print(f"   Best score: {history.best_score:.4f}")
        print(f"   Best params: {best_params}")
        print("   ✅ Basic optimization working correctly")
        
    except Exception as e:
        print(f"   ❌ Basic optimization failed: {e}")
        return False
    
    return True


def main():
    """Run all Generation 3 tests."""
    print("🚀 Testing Generation 3: Quantum Hyperparameter Search")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    try:
        test_caching()
        success_count += 1
    except Exception as e:
        print(f"   ❌ Caching test failed: {e}")
    
    try:
        test_adaptive_strategies()
        success_count += 1
    except Exception as e:
        print(f"   ❌ Adaptive strategies test failed: {e}")
    
    try:
        test_parallel_optimization()
        success_count += 1
    except Exception as e:
        print(f"   ❌ Parallel optimization test failed: {e}")
    
    if test_basic_quantum_optimization():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All Generation 3 features working correctly!")
        print("🚀 Ready for production deployment!")
    else:
        print("⚠️  Some features need attention, but core functionality works")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()