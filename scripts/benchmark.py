#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for quantum hyperparameter search.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from quantum_hyper_search import QuantumHyperSearch


class QuantumBenchmark:
    """Comprehensive benchmarking suite for quantum hyperparameter optimization."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = output_dir
        self.results = []
        
        # Ensure output directory exists
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def run_all_benchmarks(self) -> pd.DataFrame:
        """Run complete benchmark suite."""
        print("🚀 Starting Quantum Hyperparameter Search Benchmark Suite")
        print("=" * 60)
        
        # Performance benchmarks
        print("\n📊 Running Performance Benchmarks...")
        self.benchmark_scalability()
        self.benchmark_search_space_sizes()
        self.benchmark_different_models()
        
        # Quality benchmarks  
        print("\n🎯 Running Quality Benchmarks...")
        self.benchmark_optimization_quality()
        self.benchmark_convergence_speed()
        
        # Feature benchmarks
        print("\n⚡ Running Feature Benchmarks...")
        self.benchmark_caching_performance()
        self.benchmark_parallel_processing()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results
        results_path = f"{self.output_dir}/benchmark_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n📁 Results saved to: {results_path}")
        
        # Generate report
        self.generate_benchmark_report(results_df)
        
        return results_df
    
    def benchmark_scalability(self) -> None:
        """Benchmark scalability with different dataset sizes."""
        print("  • Testing dataset scalability...")
        
        dataset_sizes = [50, 100, 500, 1000]
        features = [5, 10, 20, 50]
        
        for n_samples in dataset_sizes:
            for n_features in features[:3]:  # Limit features for speed
                try:
                    # Create dataset
                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_classes=2,
                        random_state=42
                    )
                    
                    # Simple search space
                    search_space = {
                        'n_estimators': [10, 50],
                        'max_depth': [5, 10]
                    }
                    
                    # Run optimization
                    start_time = time.time()
                    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    qhs = QuantumHyperSearch(
                        backend="simulator",
                        enable_monitoring=False,
                        enable_caching=False,
                        enable_parallel=False
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
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_used = memory_after - memory_before
                    
                    # Record result
                    self.results.append({
                        'benchmark_type': 'scalability',
                        'dataset_size': n_samples,
                        'n_features': n_features,
                        'duration_seconds': duration,
                        'memory_mb': memory_used,
                        'best_score': history.best_score,
                        'n_evaluations': history.n_evaluations,
                        'success': True
                    })
                    
                except Exception as e:
                    self.results.append({
                        'benchmark_type': 'scalability',
                        'dataset_size': n_samples,
                        'n_features': n_features,
                        'duration_seconds': -1,
                        'memory_mb': -1,
                        'best_score': -1,
                        'n_evaluations': -1,
                        'success': False,
                        'error': str(e)
                    })
    
    def benchmark_search_space_sizes(self) -> None:
        """Benchmark performance with different search space sizes."""
        print("  • Testing search space scalability...")
        
        # Create base dataset
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        # Different search space complexities
        search_spaces = {
            'small': {
                'n_estimators': [10, 50],
                'max_depth': [5, 10]
            },
            'medium': {
                'n_estimators': [10, 50, 100],
                'max_depth': [3, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'large': {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [3, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8]
            }
        }
        
        for space_name, search_space in search_spaces.items():
            total_combinations = 1
            for param_values in search_space.values():
                total_combinations *= len(param_values)
            
            try:
                start_time = time.time()
                
                qhs = QuantumHyperSearch(
                    backend="simulator",
                    enable_monitoring=False
                )
                
                best_params, history = qhs.optimize(
                    model_class=RandomForestClassifier,
                    param_space=search_space,
                    X=X, y=y,
                    n_iterations=2,
                    quantum_reads=10
                )
                
                duration = time.time() - start_time
                
                self.results.append({
                    'benchmark_type': 'search_space',
                    'space_size': space_name,
                    'total_combinations': total_combinations,
                    'duration_seconds': duration,
                    'best_score': history.best_score,
                    'n_evaluations': history.n_evaluations,
                    'success': True
                })
                
            except Exception as e:
                self.results.append({
                    'benchmark_type': 'search_space',
                    'space_size': space_name,
                    'total_combinations': total_combinations,
                    'duration_seconds': -1,
                    'best_score': -1,
                    'n_evaluations': -1,
                    'success': False,
                    'error': str(e)
                })
    
    def benchmark_different_models(self) -> None:
        """Benchmark performance with different ML models."""
        print("  • Testing different ML models...")
        
        # Create datasets
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
        
        models_and_spaces = [
            {
                'name': 'RandomForestClassifier',
                'model_class': RandomForestClassifier,
                'data': (X_clf, y_clf),
                'search_space': {
                    'n_estimators': [10, 50],
                    'max_depth': [5, 10]
                },
                'scoring': 'accuracy'
            },
            {
                'name': 'RandomForestRegressor', 
                'model_class': RandomForestRegressor,
                'data': (X_reg, y_reg),
                'search_space': {
                    'n_estimators': [10, 50],
                    'max_depth': [5, 10]
                },
                'scoring': 'r2'
            },
            {
                'name': 'SVC',
                'model_class': SVC,
                'data': (X_clf, y_clf),
                'search_space': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear']
                },
                'scoring': 'accuracy'
            }
        ]
        
        for model_config in models_and_spaces:
            try:
                X, y = model_config['data']
                start_time = time.time()
                
                qhs = QuantumHyperSearch(
                    backend="simulator",
                    enable_monitoring=False
                )
                
                best_params, history = qhs.optimize(
                    model_class=model_config['model_class'],
                    param_space=model_config['search_space'],
                    X=X, y=y,
                    n_iterations=2,
                    quantum_reads=5,
                    scoring=model_config['scoring']
                )
                
                duration = time.time() - start_time
                
                self.results.append({
                    'benchmark_type': 'model_types',
                    'model_name': model_config['name'],
                    'duration_seconds': duration,
                    'best_score': history.best_score,
                    'n_evaluations': history.n_evaluations,
                    'success': True
                })
                
            except Exception as e:
                self.results.append({
                    'benchmark_type': 'model_types',
                    'model_name': model_config['name'],
                    'duration_seconds': -1,
                    'best_score': -1,
                    'n_evaluations': -1,
                    'success': False,
                    'error': str(e)
                })
    
    def benchmark_optimization_quality(self) -> None:
        """Benchmark optimization quality vs baseline methods."""
        print("  • Testing optimization quality...")
        
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        search_space = {
            'n_estimators': [10, 50, 100],
            'max_depth': [5, 10, 20]
        }
        
        methods = {
            'quantum': lambda: QuantumHyperSearch(backend="simulator", enable_monitoring=False),
            'random': None  # Placeholder for random search comparison
        }
        
        # Test quantum method
        try:
            start_time = time.time()
            
            qhs = methods['quantum']()
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=3,
                quantum_reads=10
            )
            
            duration = time.time() - start_time
            
            # Test final model performance
            final_model = RandomForestClassifier(**best_params)
            final_model.fit(X, y)
            final_score = accuracy_score(y, final_model.predict(X))
            
            self.results.append({
                'benchmark_type': 'quality',
                'method': 'quantum',
                'duration_seconds': duration,
                'best_score': history.best_score,
                'final_score': final_score,
                'n_evaluations': history.n_evaluations,
                'convergence_iteration': getattr(history, 'convergence_iteration', -1),
                'success': True
            })
            
        except Exception as e:
            self.results.append({
                'benchmark_type': 'quality',
                'method': 'quantum',
                'duration_seconds': -1,
                'best_score': -1,
                'final_score': -1,
                'n_evaluations': -1,
                'success': False,
                'error': str(e)
            })
    
    def benchmark_convergence_speed(self) -> None:
        """Benchmark convergence speed over iterations."""
        print("  • Testing convergence speed...")
        
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        search_space = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10]
        }
        
        try:
            qhs = QuantumHyperSearch(
                backend="simulator",
                enable_monitoring=False
            )
            
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=5,
                quantum_reads=8
            )
            
            # Analyze convergence
            iterations, scores = history.get_convergence_data()
            
            # Find when 95% of final performance was reached
            final_score = max(scores)
            target_score = 0.95 * final_score
            convergence_iteration = -1
            
            for i, score in enumerate(scores):
                if score >= target_score:
                    convergence_iteration = i
                    break
            
            self.results.append({
                'benchmark_type': 'convergence',
                'final_score': final_score,
                'convergence_iteration': convergence_iteration,
                'total_iterations': len(iterations),
                'n_evaluations': history.n_evaluations,
                'improvement_rate': (final_score - scores[0]) / len(scores) if len(scores) > 1 else 0,
                'success': True
            })
            
        except Exception as e:
            self.results.append({
                'benchmark_type': 'convergence',
                'final_score': -1,
                'convergence_iteration': -1,
                'total_iterations': -1,
                'n_evaluations': -1,
                'success': False,
                'error': str(e)
            })
    
    def benchmark_caching_performance(self) -> None:
        """Benchmark caching performance improvements."""
        print("  • Testing caching performance...")
        
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        search_space = {
            'n_estimators': [10, 50],
            'max_depth': [5, 10]
        }
        
        # Test without caching
        try:
            start_time = time.time()
            qhs_no_cache = QuantumHyperSearch(
                backend="simulator",
                enable_caching=False,
                enable_monitoring=False
            )
            
            _, history_no_cache = qhs_no_cache.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=2,
                quantum_reads=5
            )
            
            duration_no_cache = time.time() - start_time
            
            # Test with caching (run twice to see caching effect)
            start_time = time.time()
            qhs_cache = QuantumHyperSearch(
                backend="simulator", 
                enable_caching=True,
                enable_monitoring=False
            )
            
            # First run
            _, history_cache1 = qhs_cache.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=2,
                quantum_reads=5
            )
            
            # Second run (should benefit from caching)
            _, history_cache2 = qhs_cache.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=2,
                quantum_reads=5
            )
            
            duration_with_cache = time.time() - start_time
            
            # Get cache stats
            cache_stats = qhs_cache.cache.get_stats() if qhs_cache.cache else {}
            
            self.results.append({
                'benchmark_type': 'caching',
                'duration_no_cache': duration_no_cache,
                'duration_with_cache': duration_with_cache,
                'speedup_ratio': duration_no_cache / duration_with_cache if duration_with_cache > 0 else 0,
                'cache_hits': cache_stats.get('hits', 0),
                'cache_hit_rate': cache_stats.get('hit_rate', 0),
                'success': True
            })
            
        except Exception as e:
            self.results.append({
                'benchmark_type': 'caching',
                'duration_no_cache': -1,
                'duration_with_cache': -1,
                'speedup_ratio': -1,
                'cache_hits': -1,
                'cache_hit_rate': -1,
                'success': False,
                'error': str(e)
            })
    
    def benchmark_parallel_processing(self) -> None:
        """Benchmark parallel processing performance."""
        print("  • Testing parallel processing...")
        
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        search_space = {
            'n_estimators': [10, 20, 50],
            'max_depth': [3, 5, 10]
        }
        
        # Test sequential processing
        try:
            start_time = time.time()
            qhs_sequential = QuantumHyperSearch(
                backend="simulator",
                enable_parallel=False,
                enable_monitoring=False
            )
            
            _, history_sequential = qhs_sequential.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=2,
                quantum_reads=5
            )
            
            duration_sequential = time.time() - start_time
            
            # Test parallel processing
            start_time = time.time()
            qhs_parallel = QuantumHyperSearch(
                backend="simulator",
                enable_parallel=True,
                max_parallel_workers=2,
                enable_monitoring=False
            )
            
            _, history_parallel = qhs_parallel.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=2,
                quantum_reads=5
            )
            
            duration_parallel = time.time() - start_time
            
            self.results.append({
                'benchmark_type': 'parallel',
                'duration_sequential': duration_sequential,
                'duration_parallel': duration_parallel,
                'speedup_ratio': duration_sequential / duration_parallel if duration_parallel > 0 else 0,
                'n_evaluations_sequential': history_sequential.n_evaluations,
                'n_evaluations_parallel': history_parallel.n_evaluations,
                'success': True
            })
            
        except Exception as e:
            self.results.append({
                'benchmark_type': 'parallel',
                'duration_sequential': -1,
                'duration_parallel': -1,
                'speedup_ratio': -1,
                'n_evaluations_sequential': -1,
                'n_evaluations_parallel': -1,
                'success': False,
                'error': str(e)
            })
    
    def generate_benchmark_report(self, results_df: pd.DataFrame) -> None:
        """Generate comprehensive benchmark report."""
        print("\n📊 Generating Benchmark Report...")
        
        report_path = f"{self.output_dir}/benchmark_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Quantum Hyperparameter Search - Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            total_tests = len(results_df)
            successful_tests = len(results_df[results_df['success'] == True])
            
            f.write(f"📊 Overall Statistics:\n")
            f.write(f"  • Total benchmarks: {total_tests}\n")
            f.write(f"  • Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)\n")
            f.write(f"  • Failed: {total_tests - successful_tests}\n\n")
            
            # Performance summary by benchmark type
            for benchmark_type in results_df['benchmark_type'].unique():
                subset = results_df[results_df['benchmark_type'] == benchmark_type]
                successful = subset[subset['success'] == True]
                
                f.write(f"🎯 {benchmark_type.upper()} Results:\n")
                f.write(f"  • Tests: {len(subset)}\n")
                f.write(f"  • Success rate: {len(successful)/len(subset)*100:.1f}%\n")
                
                if 'duration_seconds' in successful.columns and len(successful) > 0:
                    avg_duration = successful['duration_seconds'].mean()
                    f.write(f"  • Average duration: {avg_duration:.2f}s\n")
                
                if 'best_score' in successful.columns and len(successful) > 0:
                    avg_score = successful['best_score'].mean()
                    f.write(f"  • Average best score: {avg_score:.4f}\n")
                
                f.write("\n")
            
            # Performance recommendations
            f.write("💡 Performance Recommendations:\n")
            
            # Scalability insights
            scalability_results = results_df[results_df['benchmark_type'] == 'scalability']
            if len(scalability_results) > 0:
                successful_scalability = scalability_results[scalability_results['success'] == True]
                if len(successful_scalability) > 0:
                    max_dataset_size = successful_scalability['dataset_size'].max()
                    f.write(f"  • Tested up to {max_dataset_size} samples successfully\n")
                    
                    # Memory usage analysis
                    avg_memory = successful_scalability['memory_mb'].mean()
                    f.write(f"  • Average memory usage: {avg_memory:.1f} MB\n")
            
            # Quality insights
            quality_results = results_df[results_df['benchmark_type'] == 'quality']
            if len(quality_results) > 0:
                successful_quality = quality_results[quality_results['success'] == True]
                if len(successful_quality) > 0:
                    avg_final_score = successful_quality['final_score'].mean()
                    f.write(f"  • Average final model score: {avg_final_score:.4f}\n")
            
            # Caching insights
            caching_results = results_df[results_df['benchmark_type'] == 'caching']
            if len(caching_results) > 0:
                successful_caching = caching_results[caching_results['success'] == True]
                if len(successful_caching) > 0:
                    avg_speedup = successful_caching['speedup_ratio'].mean()
                    f.write(f"  • Caching provides {avg_speedup:.1f}x average speedup\n")
            
            f.write("\n")
            f.write("📊 Detailed results available in benchmark_results.csv\n")
        
        print(f"📁 Report saved to: {report_path}")
        
        # Generate plots if matplotlib is available
        try:
            self.generate_benchmark_plots(results_df)
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
    
    def generate_benchmark_plots(self, results_df: pd.DataFrame) -> None:
        """Generate benchmark visualization plots."""
        plt.style.use('default')
        
        # Success rate by benchmark type
        plt.figure(figsize=(10, 6))
        success_rates = results_df.groupby('benchmark_type')['success'].mean()
        success_rates.plot(kind='bar')
        plt.title('Success Rate by Benchmark Type')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/success_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance by dataset size (if available)
        scalability_data = results_df[
            (results_df['benchmark_type'] == 'scalability') & 
            (results_df['success'] == True)
        ]
        
        if len(scalability_data) > 0:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            sns.scatterplot(data=scalability_data, x='dataset_size', y='duration_seconds')
            plt.title('Duration vs Dataset Size')
            plt.xlabel('Dataset Size')
            plt.ylabel('Duration (seconds)')
            
            plt.subplot(1, 2, 2)
            sns.scatterplot(data=scalability_data, x='dataset_size', y='memory_mb')
            plt.title('Memory Usage vs Dataset Size')
            plt.xlabel('Dataset Size') 
            plt.ylabel('Memory Usage (MB)')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/scalability_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print("📊 Benchmark plots generated")


def main():
    """Run comprehensive benchmark suite."""
    benchmark = QuantumBenchmark()
    results = benchmark.run_all_benchmarks()
    
    print("\n🎉 Benchmark Suite Complete!")
    print(f"📊 {len(results)} benchmarks executed")
    print(f"✅ {len(results[results['success'] == True])} successful")
    
    return results


if __name__ == "__main__":
    results = main()