#!/usr/bin/env python3
"""
Reproducible Experimental Framework for Quantum Hyperparameter Optimization

This module provides a comprehensive experimental framework for conducting
reproducible quantum machine learning research, including:
1. Controlled experimental design
2. Statistical significance testing
3. Reproducible result tracking
4. Cross-validation and validation protocols
5. Systematic comparison methodologies
"""

import numpy as np
import json
import pickle
import hashlib
import time
import os
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict
import pandas as pd
from pathlib import Path

# ML imports
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# Statistical testing
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare


@dataclass
class ExperimentalCondition:
    """Defines a single experimental condition."""
    algorithm_name: str
    algorithm_params: Dict[str, Any]
    dataset_params: Dict[str, Any]
    random_seed: int
    replications: int
    condition_id: str = None
    
    def __post_init__(self):
        if self.condition_id is None:
            # Generate unique condition ID based on parameters
            condition_str = json.dumps({
                'algorithm': self.algorithm_name,
                'params': self.algorithm_params,
                'dataset': self.dataset_params,
                'seed': self.random_seed
            }, sort_keys=True)
            self.condition_id = hashlib.md5(condition_str.encode()).hexdigest()[:12]


@dataclass
class ExperimentResult:
    """Stores the results of a single experimental run."""
    condition_id: str
    replication_id: int
    start_time: float
    end_time: float
    algorithm_name: str
    dataset_name: str
    
    # Performance metrics
    best_parameters: Dict[str, Any]
    best_score: float
    cross_validation_scores: List[float]
    training_time: float
    optimization_time: float
    
    # Algorithm-specific metrics
    n_evaluations: int
    convergence_iteration: int
    quantum_metrics: Optional[Dict[str, Any]] = None
    
    # Statistical measures
    cv_mean: float = None
    cv_std: float = None
    
    def __post_init__(self):
        if self.cv_mean is None:
            self.cv_mean = np.mean(self.cross_validation_scores)
        if self.cv_std is None:
            self.cv_std = np.std(self.cross_validation_scores)


@dataclass
class ExperimentSuite:
    """Defines a complete experimental suite with multiple conditions."""
    suite_name: str
    description: str
    conditions: List[ExperimentalCondition]
    evaluation_protocol: str
    significance_level: float = 0.05
    suite_id: str = None
    
    def __post_init__(self):
        if self.suite_id is None:
            self.suite_id = str(uuid.uuid4())[:8]


class DatasetGenerator:
    """Generates controlled datasets for experimental evaluation."""
    
    @staticmethod
    def generate_classification_dataset(n_samples: int = 1000, n_features: int = 20,
                                      n_classes: int = 2, n_informative: int = None,
                                      class_sep: float = 1.0, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, str]:
        """Generate a controlled classification dataset."""
        
        if n_informative is None:
            n_informative = max(2, n_features // 2)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=max(0, n_features - n_informative),
            class_sep=class_sep,
            random_state=random_state
        )
        
        dataset_name = f"synth_clf_{n_samples}s_{n_features}f_{n_classes}c"
        
        return X, y, dataset_name
    
    @staticmethod
    def generate_regression_dataset(n_samples: int = 1000, n_features: int = 20,
                                  n_informative: int = None, noise: float = 0.1,
                                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, str]:
        """Generate a controlled regression dataset."""
        
        if n_informative is None:
            n_informative = max(2, n_features // 2)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=random_state
        )
        
        dataset_name = f"synth_reg_{n_samples}s_{n_features}f"
        
        return X, y, dataset_name
    
    @staticmethod
    def generate_challenging_datasets() -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Generate a suite of challenging datasets for comprehensive evaluation."""
        
        datasets = []
        
        # Small dataset
        X, y, name = DatasetGenerator.generate_classification_dataset(
            n_samples=200, n_features=10, n_classes=2, class_sep=0.8, random_state=42
        )
        datasets.append((X, y, f"{name}_small_challenging"))
        
        # High-dimensional dataset
        X, y, name = DatasetGenerator.generate_classification_dataset(
            n_samples=500, n_features=50, n_classes=3, class_sep=1.2, random_state=43
        )
        datasets.append((X, y, f"{name}_high_dim"))
        
        # Imbalanced dataset
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1],
            class_sep=1.0, random_state=44
        )
        datasets.append((X, y, "synth_clf_imbalanced"))
        
        # Noisy dataset
        X, y, name = DatasetGenerator.generate_classification_dataset(
            n_samples=800, n_features=15, n_classes=2, class_sep=0.5, random_state=45
        )
        datasets.append((X, y, f"{name}_noisy"))
        
        # Regression datasets
        X, y, name = DatasetGenerator.generate_regression_dataset(
            n_samples=600, n_features=12, noise=0.2, random_state=46
        )
        datasets.append((X, y, f"{name}_moderate_noise"))
        
        return datasets


class EvaluationProtocol:
    """Defines evaluation protocols for experimental validation."""
    
    @staticmethod
    def stratified_k_fold_cv(X: np.ndarray, y: np.ndarray, model: Any, 
                           k: int = 5, scoring: str = 'accuracy',
                           random_state: int = 42) -> Tuple[List[float], float, float]:
        """Perform stratified k-fold cross-validation."""
        
        if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.floating):
            # Regression case
            cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
        else:
            # Classification case
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            return list(scores), np.mean(scores), np.std(scores)
        except Exception as e:
            # Fallback for problematic cases
            print(f"CV failed with error: {e}, using simple train-test split")
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if scoring == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif scoring == 'neg_mean_squared_error':
                score = -mean_squared_error(y_test, y_pred)
            else:
                score = 0.5  # Default fallback
            
            return [score], score, 0.0
    
    @staticmethod
    def time_series_split_cv(X: np.ndarray, y: np.ndarray, model: Any,
                           n_splits: int = 5, scoring: str = 'accuracy') -> Tuple[List[float], float, float]:
        """Perform time series split cross-validation."""
        
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
        
        return list(scores), np.mean(scores), np.std(scores)
    
    @staticmethod
    def nested_cv(X: np.ndarray, y: np.ndarray, model_class: type,
                 param_space: Dict[str, List[Any]], outer_k: int = 5,
                 inner_k: int = 3, scoring: str = 'accuracy',
                 random_state: int = 42) -> Tuple[List[float], Dict[str, Any]]:
        """Perform nested cross-validation for unbiased performance estimation."""
        
        from sklearn.model_selection import GridSearchCV
        
        if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.floating):
            outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
            inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=random_state + 1)
        else:
            outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=random_state)
            inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=random_state + 1)
        
        nested_scores = []
        best_params_list = []
        
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                model_class(), param_space, cv=inner_cv, 
                scoring=scoring, n_jobs=1
            )
            
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Evaluate on outer test set
            y_pred = grid_search.predict(X_test_outer)
            
            if scoring == 'accuracy':
                score = accuracy_score(y_test_outer, y_pred)
            elif scoring == 'neg_mean_squared_error':
                score = -mean_squared_error(y_test_outer, y_pred)
            else:
                score = grid_search.score(X_test_outer, y_test_outer)
            
            nested_scores.append(score)
            best_params_list.append(grid_search.best_params_)
        
        # Aggregate best parameters (mode for categorical, mean for numerical)
        aggregated_params = {}
        if best_params_list:
            for param in best_params_list[0].keys():
                values = [params[param] for params in best_params_list]
                if isinstance(values[0], (int, float)):
                    aggregated_params[param] = np.mean(values)
                else:
                    # Use most common value
                    from collections import Counter
                    aggregated_params[param] = Counter(values).most_common(1)[0][0]
        
        return nested_scores, aggregated_params


class StatisticalTesting:
    """Statistical testing utilities for experimental comparisons."""
    
    @staticmethod
    def compare_two_algorithms(results1: List[float], results2: List[float],
                             test_type: str = 'wilcoxon',
                             alpha: float = 0.05) -> Dict[str, Any]:
        """Compare two algorithms using statistical tests."""
        
        results = {
            'algorithm1_mean': np.mean(results1),
            'algorithm1_std': np.std(results1),
            'algorithm2_mean': np.mean(results2),
            'algorithm2_std': np.std(results2),
            'effect_size': abs(np.mean(results1) - np.mean(results2)) / np.sqrt((np.var(results1) + np.var(results2)) / 2),
            'test_type': test_type,
            'alpha': alpha
        }
        
        if test_type == 'wilcoxon':
            try:
                statistic, p_value = wilcoxon(results1, results2, alternative='two-sided')
                results['statistic'] = statistic
                results['p_value'] = p_value
            except ValueError:
                # Fall back to Mann-Whitney U test if Wilcoxon fails
                statistic, p_value = mannwhitneyu(results1, results2, alternative='two-sided')
                results['statistic'] = statistic
                results['p_value'] = p_value
                results['test_type'] = 'mannwhitneyu'
                
        elif test_type == 'mannwhitneyu':
            statistic, p_value = mannwhitneyu(results1, results2, alternative='two-sided')
            results['statistic'] = statistic
            results['p_value'] = p_value
            
        elif test_type == 'ttest':
            statistic, p_value = stats.ttest_rel(results1, results2)
            results['statistic'] = statistic
            results['p_value'] = p_value
        
        results['significant'] = p_value < alpha
        results['interpretation'] = StatisticalTesting._interpret_comparison(results)
        
        return results
    
    @staticmethod
    def compare_multiple_algorithms(algorithm_results: Dict[str, List[float]],
                                  alpha: float = 0.05) -> Dict[str, Any]:
        """Compare multiple algorithms using Friedman test and post-hoc analysis."""
        
        algorithm_names = list(algorithm_results.keys())
        results_matrix = np.array([algorithm_results[name] for name in algorithm_names])
        
        # Friedman test
        try:
            statistic, p_value = friedmanchisquare(*results_matrix)
            
            results = {
                'test_type': 'friedman',
                'statistic': statistic,
                'p_value': p_value,
                'alpha': alpha,
                'significant': p_value < alpha,
                'algorithm_names': algorithm_names
            }
            
            # Calculate means and rankings
            means = {name: np.mean(algorithm_results[name]) for name in algorithm_names}
            results['algorithm_means'] = means
            
            # Rank algorithms by mean performance
            ranked_algorithms = sorted(algorithm_names, key=lambda x: means[x], reverse=True)
            results['ranking'] = ranked_algorithms
            
            # Post-hoc pairwise comparisons if significant
            if results['significant']:
                pairwise_results = {}
                for i, alg1 in enumerate(algorithm_names):
                    for j, alg2 in enumerate(algorithm_names[i+1:], i+1):
                        comparison = StatisticalTesting.compare_two_algorithms(
                            algorithm_results[alg1], algorithm_results[alg2], 
                            test_type='wilcoxon', alpha=alpha
                        )
                        pairwise_results[f"{alg1}_vs_{alg2}"] = comparison
                
                results['pairwise_comparisons'] = pairwise_results
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'test_type': 'friedman',
                'algorithm_names': algorithm_names,
                'algorithm_means': {name: np.mean(algorithm_results[name]) for name in algorithm_names}
            }
    
    @staticmethod
    def _interpret_comparison(comparison_results: Dict[str, Any]) -> str:
        """Interpret statistical comparison results."""
        
        p_value = comparison_results['p_value']
        alpha = comparison_results['alpha']
        effect_size = comparison_results.get('effect_size', 0)
        
        mean1 = comparison_results['algorithm1_mean']
        mean2 = comparison_results['algorithm2_mean']
        
        if p_value >= alpha:
            return f"No significant difference (p={p_value:.4f} >= α={alpha})"
        
        better_algorithm = "Algorithm 1" if mean1 > mean2 else "Algorithm 2"
        
        if effect_size < 0.2:
            magnitude = "small"
        elif effect_size < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return f"Significant difference (p={p_value:.4f} < α={alpha}): {better_algorithm} is better with {magnitude} effect size ({effect_size:.3f})"


class ExperimentTracker:
    """Tracks and persists experimental results."""
    
    def __init__(self, experiment_dir: str = "./experiments"):
        """Initialize experiment tracker."""
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "results").mkdir(exist_ok=True)
        (self.experiment_dir / "configs").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
    
    def save_experiment_config(self, experiment_suite: ExperimentSuite) -> str:
        """Save experiment configuration."""
        
        config_file = self.experiment_dir / "configs" / f"{experiment_suite.suite_id}_config.json"
        
        config_data = {
            'suite_name': experiment_suite.suite_name,
            'description': experiment_suite.description,
            'suite_id': experiment_suite.suite_id,
            'evaluation_protocol': experiment_suite.evaluation_protocol,
            'significance_level': experiment_suite.significance_level,
            'conditions': [asdict(condition) for condition in experiment_suite.conditions],
            'created_at': time.time()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return str(config_file)
    
    def save_result(self, result: ExperimentResult) -> str:
        """Save a single experimental result."""
        
        result_file = self.experiment_dir / "results" / f"{result.condition_id}_{result.replication_id}_result.json"
        
        result_data = asdict(result)
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        return str(result_file)
    
    def load_results(self, suite_id: str) -> List[ExperimentResult]:
        """Load all results for a given experiment suite."""
        
        results = []
        results_dir = self.experiment_dir / "results"
        
        for result_file in results_dir.glob("*_result.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                result = ExperimentResult(**result_data)
                results.append(result)
                
            except Exception as e:
                print(f"Failed to load result from {result_file}: {e}")
        
        return results
    
    def generate_report(self, suite_id: str, output_format: str = 'markdown') -> str:
        """Generate a comprehensive experimental report."""
        
        results = self.load_results(suite_id)
        
        if not results:
            return "No results found for the given suite ID."
        
        # Group results by condition
        condition_results = defaultdict(list)
        for result in results:
            condition_results[result.condition_id].append(result)
        
        # Generate report
        if output_format == 'markdown':
            return self._generate_markdown_report(condition_results, suite_id)
        elif output_format == 'latex':
            return self._generate_latex_report(condition_results, suite_id)
        else:
            return self._generate_text_report(condition_results, suite_id)
    
    def _generate_markdown_report(self, condition_results: Dict, suite_id: str) -> str:
        """Generate a markdown report."""
        
        report = f"""# Experimental Report - Suite {suite_id}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Total Conditions: {len(condition_results)}
Total Replications: {sum(len(results) for results in condition_results.values())}

## Results by Condition

"""
        
        for condition_id, results in condition_results.items():
            if not results:
                continue
                
            # Calculate statistics
            cv_scores = [r.cv_mean for r in results]
            best_scores = [r.best_score for r in results]
            training_times = [r.training_time for r in results]
            optimization_times = [r.optimization_time for r in results]
            
            report += f"""### Condition: {condition_id}

Algorithm: {results[0].algorithm_name}
Dataset: {results[0].dataset_name}
Replications: {len(results)}

**Performance Metrics:**
- CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}
- Best Score: {np.mean(best_scores):.4f} ± {np.std(best_scores):.4f}
- Training Time: {np.mean(training_times):.2f}s ± {np.std(training_times):.2f}s
- Optimization Time: {np.mean(optimization_times):.2f}s ± {np.std(optimization_times):.2f}s

**Best Parameters (most frequent):**
"""
            
            # Find most common best parameters
            all_params = [result.best_parameters for result in results]
            param_keys = set()
            for params in all_params:
                param_keys.update(params.keys())
            
            for param in param_keys:
                values = [params.get(param) for params in all_params if param in params]
                if values:
                    if isinstance(values[0], (int, float)):
                        report += f"- {param}: {np.mean(values):.4f} ± {np.std(values):.4f}\n"
                    else:
                        from collections import Counter
                        most_common = Counter(values).most_common(1)[0]
                        report += f"- {param}: {most_common[0]} (frequency: {most_common[1]}/{len(values)})\n"
            
            report += "\n---\n\n"
        
        return report
    
    def _generate_text_report(self, condition_results: Dict, suite_id: str) -> str:
        """Generate a simple text report."""
        
        report = f"Experimental Report - Suite {suite_id}\n"
        report += "=" * 50 + "\n\n"
        
        for condition_id, results in condition_results.items():
            if not results:
                continue
                
            cv_scores = [r.cv_mean for r in results]
            report += f"Condition {condition_id}:\n"
            report += f"  Algorithm: {results[0].algorithm_name}\n"
            report += f"  Mean CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n"
            report += f"  Replications: {len(results)}\n\n"
        
        return report
    
    def _generate_latex_report(self, condition_results: Dict, suite_id: str) -> str:
        """Generate a LaTeX report."""
        
        report = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}

\\title{{Experimental Report - Suite {suite_id}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section{{Summary}}

Total Conditions: {len(condition_results)}

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lcccc}}
\\toprule
Condition & Algorithm & CV Score & Training Time & Replications \\\\
\\midrule
"""
        
        for condition_id, results in condition_results.items():
            if not results:
                continue
                
            cv_scores = [r.cv_mean for r in results]
            training_times = [r.training_time for r in results]
            
            report += f"{condition_id[:8]} & {results[0].algorithm_name} & "
            report += f"{np.mean(cv_scores):.3f} $\\pm$ {np.std(cv_scores):.3f} & "
            report += f"{np.mean(training_times):.2f}s & {len(results)} \\\\\n"
        
        report += """\\bottomrule
\\end{tabular}
\\caption{Experimental Results Summary}
\\end{table}

\\end{document}"""
        
        return report


class ExperimentRunner:
    """Main class for running reproducible experiments."""
    
    def __init__(self, experiment_dir: str = "./experiments"):
        """Initialize experiment runner."""
        self.tracker = ExperimentTracker(experiment_dir)
        self.dataset_generator = DatasetGenerator()
        
    def run_experiment_suite(self, experiment_suite: ExperimentSuite,
                           algorithms: Dict[str, Callable],
                           verbose: bool = True) -> List[ExperimentResult]:
        """Run a complete experiment suite."""
        
        # Save experiment configuration
        config_file = self.tracker.save_experiment_config(experiment_suite)
        if verbose:
            print(f"Saved experiment configuration to: {config_file}")
        
        all_results = []
        
        for condition_idx, condition in enumerate(experiment_suite.conditions):
            if verbose:
                print(f"\nRunning condition {condition_idx + 1}/{len(experiment_suite.conditions)}: {condition.condition_id}")
                print(f"Algorithm: {condition.algorithm_name}")
                print(f"Dataset params: {condition.dataset_params}")
            
            # Generate dataset
            dataset_params = condition.dataset_params.copy()
            dataset_params['random_state'] = condition.random_seed
            
            if dataset_params.get('problem_type', 'classification') == 'classification':
                X, y, dataset_name = self.dataset_generator.generate_classification_dataset(**dataset_params)
            else:
                X, y, dataset_name = self.dataset_generator.generate_regression_dataset(**dataset_params)
            
            # Run replications
            for replication in range(condition.replications):
                if verbose:
                    print(f"  Replication {replication + 1}/{condition.replications}")
                
                result = self._run_single_experiment(
                    condition, replication, X, y, dataset_name,
                    algorithms[condition.algorithm_name], 
                    experiment_suite.evaluation_protocol
                )
                
                all_results.append(result)
                
                # Save result immediately
                self.tracker.save_result(result)
                
                if verbose:
                    print(f"    CV Score: {result.cv_mean:.4f} ± {result.cv_std:.4f}")
                    print(f"    Best Score: {result.best_score:.4f}")
                    print(f"    Optimization Time: {result.optimization_time:.2f}s")
        
        if verbose:
            print(f"\nExperiment suite completed. Total results: {len(all_results)}")
        
        return all_results
    
    def _run_single_experiment(self, condition: ExperimentalCondition,
                             replication_id: int, X: np.ndarray, y: np.ndarray,
                             dataset_name: str, algorithm_factory: Callable,
                             evaluation_protocol: str) -> ExperimentResult:
        """Run a single experimental replication."""
        
        start_time = time.time()
        
        # Create algorithm instance
        algorithm_params = condition.algorithm_params.copy()
        algorithm_params['random_state'] = condition.random_seed + replication_id
        
        try:
            # Time the algorithm initialization and optimization
            opt_start = time.time()
            
            if 'QuantumHyperSearch' in condition.algorithm_name:
                # Quantum hyperparameter optimization
                algorithm = algorithm_factory(**algorithm_params)
                
                # Get model class and param space from algorithm params
                model_class = algorithm_params.get('model_class', RandomForestClassifier)
                param_space = algorithm_params.get('param_space', {
                    'n_estimators': [10, 20, 50],
                    'max_depth': [3, 5, 7]
                })
                
                best_params, history = algorithm.optimize(
                    model_class=model_class,
                    param_space=param_space,
                    X=X, y=y,
                    n_iterations=algorithm_params.get('n_iterations', 5),
                    quantum_reads=algorithm_params.get('quantum_reads', 100),
                    cv_folds=3,
                    random_state=condition.random_seed + replication_id
                )
                
                # Create model with best parameters
                model = model_class(**best_params)
                
                optimization_time = time.time() - opt_start
                n_evaluations = getattr(history, 'n_evaluations', 5)
                convergence_iteration = getattr(history, 'convergence_iteration', n_evaluations)
                best_score = getattr(history, 'best_score', 0.0)
                
                # Quantum-specific metrics
                quantum_metrics = {
                    'backend': algorithm_params.get('backend', 'simple'),
                    'encoding': algorithm_params.get('encoding', 'one_hot'),
                    'quantum_reads': algorithm_params.get('quantum_reads', 100)
                }
                
            else:
                # Classical hyperparameter optimization (e.g., GridSearch, RandomSearch)
                from sklearn.model_selection import GridSearchCV
                
                model_class = algorithm_params.get('model_class', RandomForestClassifier)
                param_space = algorithm_params.get('param_space', {
                    'n_estimators': [10, 20, 50],
                    'max_depth': [3, 5, 7]
                })
                
                grid_search = GridSearchCV(
                    model_class(), param_space, cv=3, 
                    scoring='accuracy' if 'Classifier' in str(model_class) else 'neg_mean_squared_error',
                    n_jobs=1
                )
                
                grid_search.fit(X, y)
                
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                optimization_time = time.time() - opt_start
                n_evaluations = len(grid_search.cv_results_['mean_test_score'])
                convergence_iteration = n_evaluations
                quantum_metrics = None
            
            # Evaluate using specified protocol
            train_start = time.time()
            
            if evaluation_protocol == 'stratified_k_fold':
                cv_scores, cv_mean, cv_std = EvaluationProtocol.stratified_k_fold_cv(
                    X, y, model, k=5, 
                    scoring='accuracy' if 'Classifier' in str(type(model)) else 'neg_mean_squared_error',
                    random_state=condition.random_seed + replication_id
                )
            else:
                # Default to stratified k-fold
                cv_scores, cv_mean, cv_std = EvaluationProtocol.stratified_k_fold_cv(
                    X, y, model, k=5,
                    scoring='accuracy' if 'Classifier' in str(type(model)) else 'neg_mean_squared_error',
                    random_state=condition.random_seed + replication_id
                )
            
            training_time = time.time() - train_start
            
            # Create result object
            result = ExperimentResult(
                condition_id=condition.condition_id,
                replication_id=replication_id,
                start_time=start_time,
                end_time=time.time(),
                algorithm_name=condition.algorithm_name,
                dataset_name=dataset_name,
                best_parameters=best_params,
                best_score=best_score,
                cross_validation_scores=cv_scores,
                training_time=training_time,
                optimization_time=optimization_time,
                n_evaluations=n_evaluations,
                convergence_iteration=convergence_iteration,
                quantum_metrics=quantum_metrics,
                cv_mean=cv_mean,
                cv_std=cv_std
            )
            
            return result
            
        except Exception as e:
            # Create error result
            print(f"Experiment failed: {e}")
            
            return ExperimentResult(
                condition_id=condition.condition_id,
                replication_id=replication_id,
                start_time=start_time,
                end_time=time.time(),
                algorithm_name=condition.algorithm_name,
                dataset_name=dataset_name,
                best_parameters={},
                best_score=0.0,
                cross_validation_scores=[0.0],
                training_time=0.0,
                optimization_time=time.time() - start_time,
                n_evaluations=0,
                convergence_iteration=0,
                quantum_metrics=None,
                cv_mean=0.0,
                cv_std=0.0
            )
    
    def compare_algorithms(self, suite_id: str, algorithm_names: List[str] = None,
                          significance_level: float = 0.05) -> Dict[str, Any]:
        """Compare algorithms from experimental results."""
        
        results = self.tracker.load_results(suite_id)
        
        if not results:
            return {'error': 'No results found for the given suite ID'}
        
        # Group results by algorithm
        if algorithm_names is None:
            algorithm_names = list(set(r.algorithm_name for r in results))
        
        algorithm_results = defaultdict(list)
        for result in results:
            if result.algorithm_name in algorithm_names:
                algorithm_results[result.algorithm_name].append(result.cv_mean)
        
        # Remove algorithms with insufficient data
        algorithm_results = {
            name: scores for name, scores in algorithm_results.items() 
            if len(scores) >= 3  # Minimum for statistical testing
        }
        
        if len(algorithm_results) < 2:
            return {'error': 'Need at least 2 algorithms with sufficient data for comparison'}
        
        # Perform statistical comparison
        comparison = StatisticalTesting.compare_multiple_algorithms(
            algorithm_results, alpha=significance_level
        )
        
        return comparison
    
    def generate_experiment_report(self, suite_id: str, output_format: str = 'markdown') -> str:
        """Generate a comprehensive experiment report."""
        return self.tracker.generate_report(suite_id, output_format)