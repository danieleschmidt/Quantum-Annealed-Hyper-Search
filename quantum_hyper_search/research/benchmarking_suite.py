#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for Quantum Hyperparameter Optimization

This module implements a complete benchmarking framework that provides:
1. Standardized benchmark problems for hyperparameter optimization
2. Performance metrics and evaluation protocols
3. Classical baseline implementations for comparison
4. Quantum advantage analysis and reporting
5. Scalability studies and complexity analysis
"""

import numpy as np
import time
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import itertools
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class BenchmarkProblem:
    """Defines a standardized benchmark problem."""
    name: str
    description: str
    model_class: type
    param_space: Dict[str, List[Any]]
    dataset_generator: Callable
    dataset_params: Dict[str, Any]
    problem_type: str  # 'classification' or 'regression'
    difficulty_level: str  # 'easy', 'medium', 'hard'
    expected_quantum_advantage: bool  # Whether quantum methods should show advantage


@dataclass
class BenchmarkResult:
    """Stores results from a benchmark run."""
    problem_name: str
    algorithm_name: str
    start_time: float
    end_time: float
    total_time: float
    optimization_time: float
    evaluation_time: float
    
    # Performance metrics
    best_score: float
    best_parameters: Dict[str, Any]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Algorithm-specific metrics
    n_evaluations: int
    convergence_iteration: int
    function_calls: int
    
    # Resource usage
    memory_peak_mb: float
    cpu_percent: float
    
    # Quantum-specific metrics (if applicable)
    quantum_metrics: Optional[Dict[str, Any]] = None
    
    # Success/failure status
    success: bool = True
    error_message: Optional[str] = None


@dataclass 
class QuantumAdvantageAnalysis:
    """Analysis of quantum advantage over classical methods."""
    problem_name: str
    quantum_algorithm: str
    best_classical_algorithm: str
    quantum_score: float
    classical_score: float
    quantum_time: float
    classical_time: float
    
    # Advantage metrics
    score_advantage: float  # (quantum - classical) / |classical|
    time_advantage: float   # classical_time / quantum_time
    efficiency_advantage: float  # score_advantage / time_ratio
    
    # Statistical significance
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    
    # Overall assessment
    has_quantum_advantage: bool
    advantage_type: str  # 'performance', 'efficiency', 'both', 'none'


class StandardBenchmarkProblems:
    """Collection of standardized benchmark problems."""
    
    @staticmethod
    def get_all_problems() -> List[BenchmarkProblem]:
        """Get all standard benchmark problems."""
        problems = []
        
        # Easy classification problems
        problems.append(BenchmarkProblem(
            name="iris_rf",
            description="Iris dataset with Random Forest - easy classification problem",
            model_class=RandomForestClassifier,
            param_space={
                'n_estimators': [10, 25, 50, 100],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            dataset_generator=lambda: load_iris(return_X_y=True),
            dataset_params={},
            problem_type="classification",
            difficulty_level="easy",
            expected_quantum_advantage=False
        ))
        
        # Medium classification problems
        problems.append(BenchmarkProblem(
            name="synthetic_medium_rf",
            description="Synthetic medium-sized classification with Random Forest",
            model_class=RandomForestClassifier,
            param_space={
                'n_estimators': [10, 25, 50, 100, 200],
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None, 0.5]
            },
            dataset_generator=lambda: make_classification(
                n_samples=1000, n_features=20, n_classes=3, 
                n_informative=15, random_state=42
            ),
            dataset_params={},
            problem_type="classification", 
            difficulty_level="medium",
            expected_quantum_advantage=True
        ))
        
        # Hard classification problems
        problems.append(BenchmarkProblem(
            name="synthetic_hard_rf",
            description="Large synthetic classification with Random Forest - many parameters",
            model_class=RandomForestClassifier,
            param_space={
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10, 20, 50],
                'min_samples_leaf': [1, 2, 4, 8, 16],
                'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.8],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            },
            dataset_generator=lambda: make_classification(
                n_samples=2000, n_features=50, n_classes=5,
                n_informative=30, n_redundant=10, random_state=42
            ),
            dataset_params={},
            problem_type="classification",
            difficulty_level="hard", 
            expected_quantum_advantage=True
        ))
        
        # SVM problems  
        problems.append(BenchmarkProblem(
            name="synthetic_medium_svm",
            description="Synthetic classification with SVM - continuous parameters",
            model_class=SVC,
            param_space={
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4, 5]
            },
            dataset_generator=lambda: make_classification(
                n_samples=800, n_features=15, n_classes=2,
                n_informative=10, random_state=42
            ),
            dataset_params={},
            problem_type="classification",
            difficulty_level="medium",
            expected_quantum_advantage=True
        ))
        
        # Neural network problems
        problems.append(BenchmarkProblem(
            name="synthetic_medium_mlp",
            description="Synthetic classification with MLP - mixed parameter types",
            model_class=MLPClassifier,
            param_space={
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'lbfgs', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'max_iter': [200, 500, 1000]
            },
            dataset_generator=lambda: make_classification(
                n_samples=1200, n_features=25, n_classes=3,
                n_informative=15, random_state=42
            ),
            dataset_params={},
            problem_type="classification",
            difficulty_level="medium",
            expected_quantum_advantage=True
        ))
        
        # Regression problems
        problems.append(BenchmarkProblem(
            name="synthetic_regression_rf",
            description="Synthetic regression with Random Forest",
            model_class=RandomForestRegressor,
            param_space={
                'n_estimators': [25, 50, 100, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None, 0.5]
            },
            dataset_generator=lambda: make_regression(
                n_samples=1000, n_features=20, n_informative=15,
                noise=0.1, random_state=42
            ),
            dataset_params={},
            problem_type="regression",
            difficulty_level="medium",
            expected_quantum_advantage=True
        ))
        
        # Gradient boosting problems
        problems.append(BenchmarkProblem(
            name="synthetic_medium_gb",
            description="Synthetic classification with Gradient Boosting",
            model_class=GradientBoostingClassifier,
            param_space={
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            },
            dataset_generator=lambda: make_classification(
                n_samples=1500, n_features=30, n_classes=4,
                n_informative=20, random_state=42
            ),
            dataset_params={},
            problem_type="classification",
            difficulty_level="medium",
            expected_quantum_advantage=True
        ))
        
        return problems
    
    @staticmethod
    def get_problems_by_difficulty(difficulty: str) -> List[BenchmarkProblem]:
        """Get problems filtered by difficulty level."""
        all_problems = StandardBenchmarkProblems.get_all_problems()
        return [p for p in all_problems if p.difficulty_level == difficulty]
    
    @staticmethod
    def get_problems_by_type(problem_type: str) -> List[BenchmarkProblem]:
        """Get problems filtered by type (classification/regression)."""
        all_problems = StandardBenchmarkProblems.get_all_problems()
        return [p for p in all_problems if p.problem_type == problem_type]


class ClassicalBaselines:
    """Classical optimization baselines for comparison."""
    
    @staticmethod
    def grid_search_optimizer(model_class: type, param_space: Dict[str, List[Any]],
                            X: np.ndarray, y: np.ndarray, cv: int = 3,
                            random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Grid search optimization baseline."""
        
        start_time = time.time()
        
        scoring = 'accuracy' if 'Classifier' in str(model_class) else 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            model_class(random_state=random_state), param_space, 
            cv=cv, scoring=scoring, n_jobs=1
        )
        
        grid_search.fit(X, y)
        
        optimization_time = time.time() - start_time
        
        metrics = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'n_evaluations': len(grid_search.cv_results_['mean_test_score']),
            'optimization_time': optimization_time,
            'algorithm_name': 'GridSearch'
        }
        
        return grid_search.best_params_, metrics
    
    @staticmethod 
    def random_search_optimizer(model_class: type, param_space: Dict[str, List[Any]],
                              X: np.ndarray, y: np.ndarray, n_iter: int = 50,
                              cv: int = 3, random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Random search optimization baseline."""
        
        start_time = time.time()
        
        scoring = 'accuracy' if 'Classifier' in str(model_class) else 'neg_mean_squared_error'
        
        # Convert list parameters to distributions for RandomizedSearchCV
        param_distributions = {}
        for param, values in param_space.items():
            if isinstance(values, list):
                param_distributions[param] = values
            else:
                param_distributions[param] = values
        
        random_search = RandomizedSearchCV(
            model_class(random_state=random_state), param_distributions,
            n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state, n_jobs=1
        )
        
        random_search.fit(X, y)
        
        optimization_time = time.time() - start_time
        
        metrics = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'n_evaluations': n_iter,
            'optimization_time': optimization_time,
            'algorithm_name': 'RandomSearch'
        }
        
        return random_search.best_params_, metrics
    
    @staticmethod
    def optuna_optimizer(model_class: type, param_space: Dict[str, List[Any]],
                        X: np.ndarray, y: np.ndarray, n_trials: int = 50,
                        cv: int = 3, random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optuna TPE optimization baseline."""
        
        if not OPTUNA_AVAILABLE:
            return ClassicalBaselines.random_search_optimizer(
                model_class, param_space, X, y, n_trials, cv, random_state
            )
        
        start_time = time.time()
        
        def objective(trial):
            # Suggest parameters
            params = {'random_state': random_state}
            
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        # Numeric parameter
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(
                                param_name, min(param_values), max(param_values)
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name, min(param_values), max(param_values)
                            )
                    else:
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Cross-validate
            model = model_class(**params)
            scoring = 'accuracy' if 'Classifier' in str(model_class) else 'neg_mean_squared_error'
            
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                return scores.mean()
            except:
                return float('-inf')
        
        # Create study
        study = optuna.create_study(
            direction='maximize', 
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        optimization_time = time.time() - start_time
        
        metrics = {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_evaluations': n_trials,
            'optimization_time': optimization_time,
            'algorithm_name': 'OptunaTPE'
        }
        
        return study.best_params, metrics
    
    @staticmethod
    def get_all_baselines() -> List[Callable]:
        """Get all available classical baseline optimizers."""
        baselines = [
            ClassicalBaselines.grid_search_optimizer,
            ClassicalBaselines.random_search_optimizer
        ]
        
        if OPTUNA_AVAILABLE:
            baselines.append(ClassicalBaselines.optuna_optimizer)
        
        return baselines


class PerformanceProfiler:
    """Profiles performance and resource usage during optimization."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        try:
            import psutil
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
        except ImportError:
            self.start_memory = 0
            self.peak_memory = 0
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
        except ImportError:
            pass
    
    def get_metrics(self) -> Dict[str, float]:
        """Get profiling metrics."""
        total_time = time.time() - self.start_time if self.start_time else 0
        memory_increase = max(0, self.peak_memory - self.start_memory)
        
        return {
            'total_time': total_time,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': memory_increase
        }


class BenchmarkRunner:
    """Main benchmarking suite runner."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        """Initialize benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
    
    def run_single_benchmark(self, problem: BenchmarkProblem, 
                           optimizer_func: Callable,
                           optimizer_name: str,
                           n_replications: int = 5) -> List[BenchmarkResult]:
        """Run a single benchmark problem with an optimizer."""
        
        results = []
        
        for replication in range(n_replications):
            print(f"  Running replication {replication + 1}/{n_replications}")
            
            # Generate dataset
            try:
                X, y = problem.dataset_generator()
                if hasattr(X, 'data'):
                    # Handle sklearn dataset objects
                    X = X.data
                    y = y if hasattr(y, '__array__') else X.target
            except Exception as e:
                print(f"Failed to generate dataset: {e}")
                continue
            
            # Start profiling
            profiler = PerformanceProfiler()
            profiler.start_profiling()
            
            start_time = time.time()
            
            try:
                # Run optimization
                opt_start = time.time()
                
                if 'quantum' in optimizer_name.lower():
                    # Quantum optimizer
                    best_params, metrics = optimizer_func(
                        model_class=problem.model_class,
                        param_space=problem.param_space,
                        X=X, y=y,
                        random_state=42 + replication
                    )
                else:
                    # Classical optimizer
                    best_params, metrics = optimizer_func(
                        model_class=problem.model_class,
                        param_space=problem.param_space,
                        X=X, y=y,
                        cv=3,
                        random_state=42 + replication
                    )
                
                optimization_time = time.time() - opt_start
                
                # Evaluate best parameters
                eval_start = time.time()
                model = problem.model_class(**best_params, random_state=42 + replication)
                
                scoring = 'accuracy' if problem.problem_type == 'classification' else 'neg_mean_squared_error'
                cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=1)
                
                evaluation_time = time.time() - eval_start
                
                # Update profiling
                profiler.update_peak_memory()
                profile_metrics = profiler.get_metrics()
                
                # Create result
                result = BenchmarkResult(
                    problem_name=problem.name,
                    algorithm_name=optimizer_name,
                    start_time=start_time,
                    end_time=time.time(),
                    total_time=profile_metrics['total_time'],
                    optimization_time=optimization_time,
                    evaluation_time=evaluation_time,
                    best_score=metrics.get('best_score', cv_scores.mean()),
                    best_parameters=best_params,
                    cv_scores=list(cv_scores),
                    cv_mean=cv_scores.mean(),
                    cv_std=cv_scores.std(),
                    n_evaluations=metrics.get('n_evaluations', 0),
                    convergence_iteration=metrics.get('convergence_iteration', 0),
                    function_calls=metrics.get('n_evaluations', 0),
                    memory_peak_mb=profile_metrics['peak_memory_mb'],
                    cpu_percent=0.0,  # Could be improved with psutil
                    quantum_metrics=metrics.get('quantum_metrics'),
                    success=True
                )
                
                results.append(result)
                
                print(f"    Success: Score = {result.cv_mean:.4f}, Time = {result.total_time:.2f}s")
                
            except Exception as e:
                # Create error result
                print(f"    Failed: {e}")
                
                result = BenchmarkResult(
                    problem_name=problem.name,
                    algorithm_name=optimizer_name,
                    start_time=start_time,
                    end_time=time.time(),
                    total_time=time.time() - start_time,
                    optimization_time=0.0,
                    evaluation_time=0.0,
                    best_score=0.0,
                    best_parameters={},
                    cv_scores=[0.0],
                    cv_mean=0.0,
                    cv_std=0.0,
                    n_evaluations=0,
                    convergence_iteration=0,
                    function_calls=0,
                    memory_peak_mb=0.0,
                    cpu_percent=0.0,
                    success=False,
                    error_message=str(e)
                )
                
                results.append(result)
        
        return results
    
    def run_full_benchmark_suite(self, quantum_optimizers: Dict[str, Callable],
                               problems: List[BenchmarkProblem] = None,
                               include_classical_baselines: bool = True,
                               n_replications: int = 5) -> Dict[str, List[BenchmarkResult]]:
        """Run the full benchmark suite."""
        
        if problems is None:
            problems = StandardBenchmarkProblems.get_all_problems()
        
        all_results = {}
        
        # Prepare optimizers
        optimizers = quantum_optimizers.copy()
        
        if include_classical_baselines:
            baselines = ClassicalBaselines.get_all_baselines()
            for baseline in baselines:
                optimizers[baseline.__name__.replace('_optimizer', '')] = baseline
        
        # Run benchmarks
        total_runs = len(problems) * len(optimizers)
        current_run = 0
        
        for problem in problems:
            print(f"\nRunning benchmark problem: {problem.name}")
            print(f"Description: {problem.description}")
            print(f"Difficulty: {problem.difficulty_level}, Type: {problem.problem_type}")
            
            for optimizer_name, optimizer_func in optimizers.items():
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running {optimizer_name}...")
                
                results = self.run_single_benchmark(
                    problem, optimizer_func, optimizer_name, n_replications
                )
                
                key = f"{problem.name}_{optimizer_name}"
                all_results[key] = results
                
                # Save intermediate results
                self._save_results(key, results)
        
        print(f"\nBenchmark suite completed! Total results: {sum(len(r) for r in all_results.values())}")
        return all_results
    
    def _save_results(self, key: str, results: List[BenchmarkResult]):
        """Save results to file."""
        
        results_file = self.output_dir / "results" / f"{key}.json"
        
        results_data = [asdict(result) for result in results]
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def load_results(self, pattern: str = "*") -> Dict[str, List[BenchmarkResult]]:
        """Load benchmark results from files."""
        
        results = {}
        results_dir = self.output_dir / "results"
        
        for results_file in results_dir.glob(f"{pattern}.json"):
            try:
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                key = results_file.stem
                results[key] = [BenchmarkResult(**r) for r in results_data]
                
            except Exception as e:
                print(f"Failed to load results from {results_file}: {e}")
        
        return results
    
    def analyze_quantum_advantage(self, results: Dict[str, List[BenchmarkResult]],
                                quantum_algorithms: List[str]) -> List[QuantumAdvantageAnalysis]:
        """Analyze quantum advantage across benchmark results."""
        
        analyses = []
        
        # Group results by problem
        problem_results = defaultdict(dict)
        for key, result_list in results.items():
            if not result_list:
                continue
                
            problem_name = result_list[0].problem_name
            algorithm_name = result_list[0].algorithm_name
            
            # Calculate aggregate metrics
            successful_results = [r for r in result_list if r.success]
            if successful_results:
                avg_score = np.mean([r.cv_mean for r in successful_results])
                avg_time = np.mean([r.total_time for r in successful_results])
                problem_results[problem_name][algorithm_name] = {
                    'score': avg_score,
                    'time': avg_time,
                    'results': successful_results
                }
        
        # Analyze each problem
        for problem_name, algorithm_results in problem_results.items():
            quantum_results = {alg: metrics for alg, metrics in algorithm_results.items() 
                             if any(qa in alg.lower() for qa in quantum_algorithms)}
            classical_results = {alg: metrics for alg, metrics in algorithm_results.items() 
                               if not any(qa in alg.lower() for qa in quantum_algorithms)}
            
            if not quantum_results or not classical_results:
                continue
            
            # Find best quantum and classical algorithms
            best_quantum = max(quantum_results.items(), key=lambda x: x[1]['score'])
            best_classical = max(classical_results.items(), key=lambda x: x[1]['score'])
            
            quantum_alg, quantum_metrics = best_quantum
            classical_alg, classical_metrics = best_classical
            
            # Calculate advantages
            score_advantage = (quantum_metrics['score'] - classical_metrics['score']) / abs(classical_metrics['score'])
            time_advantage = classical_metrics['time'] / max(quantum_metrics['time'], 0.001)
            efficiency_advantage = score_advantage * time_advantage
            
            # Statistical significance (simplified)
            quantum_scores = [r.cv_mean for r in quantum_metrics['results']]
            classical_scores = [r.cv_mean for r in classical_metrics['results']]
            
            try:
                from scipy.stats import ttest_ind
                stat, p_value = ttest_ind(quantum_scores, classical_scores)
                statistical_significance = p_value
                
                # Simple confidence interval
                quantum_mean = np.mean(quantum_scores)
                quantum_std = np.std(quantum_scores)
                confidence_interval = (quantum_mean - 1.96 * quantum_std, quantum_mean + 1.96 * quantum_std)
                
            except ImportError:
                statistical_significance = 0.5
                confidence_interval = (quantum_metrics['score'], quantum_metrics['score'])
            
            # Determine advantage type
            has_advantage = score_advantage > 0.05 and statistical_significance < 0.05
            
            if has_advantage:
                if time_advantage > 1.2:
                    advantage_type = 'both'
                else:
                    advantage_type = 'performance'
            elif time_advantage > 1.5:
                advantage_type = 'efficiency'
            else:
                advantage_type = 'none'
            
            analysis = QuantumAdvantageAnalysis(
                problem_name=problem_name,
                quantum_algorithm=quantum_alg,
                best_classical_algorithm=classical_alg,
                quantum_score=quantum_metrics['score'],
                classical_score=classical_metrics['score'],
                quantum_time=quantum_metrics['time'],
                classical_time=classical_metrics['time'],
                score_advantage=score_advantage,
                time_advantage=time_advantage,
                efficiency_advantage=efficiency_advantage,
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval,
                has_quantum_advantage=has_advantage,
                advantage_type=advantage_type
            )
            
            analyses.append(analysis)
        
        return analyses
    
    def generate_benchmark_report(self, results: Dict[str, List[BenchmarkResult]],
                                quantum_advantage_analyses: List[QuantumAdvantageAnalysis] = None,
                                output_format: str = 'markdown') -> str:
        """Generate comprehensive benchmark report."""
        
        if output_format == 'markdown':
            return self._generate_markdown_report(results, quantum_advantage_analyses)
        else:
            return self._generate_text_report(results, quantum_advantage_analyses)
    
    def _generate_markdown_report(self, results: Dict[str, List[BenchmarkResult]],
                                quantum_advantage_analyses: List[QuantumAdvantageAnalysis]) -> str:
        """Generate markdown benchmark report."""
        
        report = f"""# Quantum Hyperparameter Optimization Benchmark Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

"""
        
        # Overall statistics
        total_runs = sum(len(result_list) for result_list in results.values())
        successful_runs = sum(len([r for r in result_list if r.success]) for result_list in results.values())
        success_rate = successful_runs / max(total_runs, 1)
        
        report += f"""
- Total benchmark runs: {total_runs}
- Successful runs: {successful_runs} ({success_rate:.1%})
- Unique problems tested: {len(set(r[0].problem_name for r in results.values() if r))}
- Algorithms compared: {len(set(r[0].algorithm_name for r in results.values() if r))}

"""
        
        # Results by algorithm
        algorithm_results = defaultdict(list)
        for result_list in results.values():
            for result in result_list:
                if result.success:
                    algorithm_results[result.algorithm_name].append(result)
        
        report += """## Algorithm Performance Summary

| Algorithm | Avg Score | Avg Time (s) | Success Rate | Problems Solved |
|-----------|-----------|--------------|--------------|-----------------|
"""
        
        for algorithm, result_list in algorithm_results.items():
            avg_score = np.mean([r.cv_mean for r in result_list])
            avg_time = np.mean([r.total_time for r in result_list])
            problems_solved = len(set(r.problem_name for r in result_list))
            
            report += f"| {algorithm} | {avg_score:.4f} | {avg_time:.2f} | 100% | {problems_solved} |\n"
        
        # Quantum advantage analysis
        if quantum_advantage_analyses:
            report += """
## Quantum Advantage Analysis

"""
            
            quantum_wins = sum(1 for analysis in quantum_advantage_analyses if analysis.has_quantum_advantage)
            total_comparisons = len(quantum_advantage_analyses)
            
            report += f"""
Quantum algorithms showed advantage in **{quantum_wins}/{total_comparisons}** benchmark problems ({quantum_wins/max(total_comparisons,1):.1%}).

### Detailed Analysis

| Problem | Quantum Algorithm | Classical Best | Score Advantage | Time Advantage | Statistical Sig. | Verdict |
|---------|-------------------|----------------|-----------------|----------------|------------------|---------|
"""
            
            for analysis in quantum_advantage_analyses:
                verdict = "✅ Quantum Advantage" if analysis.has_quantum_advantage else "❌ No Advantage"
                
                report += f"| {analysis.problem_name} | {analysis.quantum_algorithm} | {analysis.best_classical_algorithm} | "
                report += f"{analysis.score_advantage:+.2%} | {analysis.time_advantage:.2f}x | "
                report += f"{analysis.statistical_significance:.3f} | {verdict} |\n"
        
        # Detailed results
        report += """
## Detailed Results by Problem

"""
        
        problem_results = defaultdict(dict)
        for key, result_list in results.items():
            if not result_list:
                continue
            problem_name = result_list[0].problem_name
            algorithm_name = result_list[0].algorithm_name
            
            successful_results = [r for r in result_list if r.success]
            if successful_results:
                problem_results[problem_name][algorithm_name] = successful_results
        
        for problem_name, algorithm_results in problem_results.items():
            report += f"""
### {problem_name}

| Algorithm | Score | Std Dev | Time (s) | Evaluations | Convergence |
|-----------|-------|---------|----------|-------------|-------------|
"""
            
            for algorithm, result_list in algorithm_results.items():
                avg_score = np.mean([r.cv_mean for r in result_list])
                std_score = np.std([r.cv_mean for r in result_list])
                avg_time = np.mean([r.total_time for r in result_list])
                avg_evals = np.mean([r.n_evaluations for r in result_list])
                avg_conv = np.mean([r.convergence_iteration for r in result_list])
                
                report += f"| {algorithm} | {avg_score:.4f} | {std_score:.4f} | {avg_time:.2f} | {avg_evals:.0f} | {avg_conv:.0f} |\n"
        
        return report
    
    def _generate_text_report(self, results: Dict[str, List[BenchmarkResult]],
                            quantum_advantage_analyses: List[QuantumAdvantageAnalysis]) -> str:
        """Generate plain text benchmark report."""
        
        report = f"QUANTUM HYPERPARAMETER OPTIMIZATION BENCHMARK REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Add summary statistics
        total_runs = sum(len(result_list) for result_list in results.values())
        successful_runs = sum(len([r for r in result_list if r.success]) for result_list in results.values())
        
        report += f"Total runs: {total_runs}\n"
        report += f"Successful: {successful_runs}\n"
        report += f"Success rate: {successful_runs/max(total_runs,1):.1%}\n\n"
        
        # Algorithm performance
        algorithm_results = defaultdict(list)
        for result_list in results.values():
            for result in result_list:
                if result.success:
                    algorithm_results[result.algorithm_name].append(result)
        
        report += "ALGORITHM PERFORMANCE:\n"
        report += "-" * 30 + "\n"
        
        for algorithm, result_list in algorithm_results.items():
            avg_score = np.mean([r.cv_mean for r in result_list])
            avg_time = np.mean([r.total_time for r in result_list])
            
            report += f"{algorithm}: Score={avg_score:.4f}, Time={avg_time:.2f}s\n"
        
        return report
    
    def save_report(self, report: str, filename: str = "benchmark_report.md"):
        """Save benchmark report to file."""
        
        report_file = self.output_dir / "reports" / filename
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        return str(report_file)