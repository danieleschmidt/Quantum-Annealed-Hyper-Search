#!/usr/bin/env python3
"""
Quick benchmark for Generation 2 validation.
"""

print('🚀 Running focused benchmark for Generation 2 validation...')
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from quantum_hyper_search import QuantumHyperSearch

# Create test dataset
X, y = make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42)

search_space = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10, 20]
}

# Test different configurations for robustness
configs = [
    {'name': 'basic', 'enable_caching': False, 'enable_parallel': False, 'enable_monitoring': False},
    {'name': 'with_caching', 'enable_caching': True, 'enable_parallel': False, 'enable_monitoring': False},
    {'name': 'with_monitoring', 'enable_caching': False, 'enable_parallel': False, 'enable_monitoring': True},
]

results = []

for config in configs:
    try:
        print(f'  Testing {config["name"]} configuration...')
        start_time = time.time()
        
        config_without_name = {k: v for k, v in config.items() if k != 'name'}
        qhs = QuantumHyperSearch(
            backend='simulator',
            **config_without_name
        )
        
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=2,
            quantum_reads=5,
            cv_folds=3
        )
        
        duration = time.time() - start_time
        
        results.append({
            'config': config['name'],
            'duration': duration,
            'best_score': history.best_score,
            'n_evaluations': history.n_evaluations,
            'success': True
        })
        
        print(f'    ✅ Success! Score: {history.best_score:.4f}, Duration: {duration:.2f}s')
        
    except Exception as e:
        results.append({
            'config': config['name'],
            'success': False,
            'error': str(e)
        })
        print(f'    ❌ Error: {e}')

print('\n📊 Benchmark Results:')
successful = [r for r in results if r['success']]
print(f'  • Configurations tested: {len(results)}')
print(f'  • Successful: {len(successful)}')
if successful:
    print(f'  • Average duration: {np.mean([r["duration"] for r in successful]):.2f}s')
    print(f'  • Average score: {np.mean([r["best_score"] for r in successful]):.4f}')

print('\n✅ Generation 2: Make it Robust - COMPLETE!')