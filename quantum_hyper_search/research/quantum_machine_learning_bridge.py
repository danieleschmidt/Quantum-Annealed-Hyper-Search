#!/usr/bin/env python3
"""
Quantum-Classical Machine Learning Bridge
Advanced integration between quantum optimization and classical ML workflows.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class QuantumMLMetrics:
    """Metrics for quantum-enhanced machine learning."""
    classical_baseline_score: float
    quantum_enhanced_score: float
    improvement_ratio: float
    training_time_ratio: float
    convergence_speed: float
    model_complexity_reduction: float


class QuantumFeatureEncoder(ABC):
    """Abstract base class for quantum feature encoding."""
    
    @abstractmethod
    def encode_classical_features(self, X: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Encode classical features into quantum representation."""
        pass
    
    @abstractmethod
    def decode_quantum_solution(self, quantum_solution: Dict[int, int]) -> np.ndarray:
        """Decode quantum solution back to classical feature space."""
        pass


class AmplitudeEncodingTransformer(QuantumFeatureEncoder):
    """
    Amplitude Encoding for Classical Features
    
    Encodes classical feature vectors into quantum amplitude patterns
    suitable for quantum optimization.
    """
    
    def __init__(self, n_qubits: int = 8, normalization: str = 'l2'):
        self.n_qubits = n_qubits
        self.normalization = normalization
        self.feature_mapping = {}
        self.inverse_mapping = {}
        
    def encode_classical_features(self, X: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Encode feature matrix into QUBO representation."""
        
        if X.shape[0] == 0:
            return {}
        
        # Normalize features
        if self.normalization == 'l2':
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        elif self.normalization == 'minmax':
            X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
        else:
            X_norm = X
        
        # Create QUBO encoding that preserves feature relationships
        Q = {}
        n_features = X_norm.shape[1]
        n_samples = X_norm.shape[0]
        
        # Create feature correlation matrix
        correlation_matrix = np.corrcoef(X_norm.T)
        
        # Map features to qubits
        qubit_idx = 0
        for feat_idx in range(min(n_features, self.n_qubits)):
            self.feature_mapping[feat_idx] = qubit_idx
            self.inverse_mapping[qubit_idx] = feat_idx
            qubit_idx += 1
        
        # Encode correlations as QUBO terms
        for i in range(len(self.feature_mapping)):
            for j in range(i, len(self.feature_mapping)):
                qi, qj = self.feature_mapping[i], self.feature_mapping[j]
                
                if i == j:
                    # Diagonal terms: average feature activation
                    activation = np.mean(X_norm[:, i])
                    Q[(qi, qj)] = -activation  # Negative to encourage activation
                else:
                    # Off-diagonal terms: feature correlations
                    if abs(correlation_matrix[i, j]) > 0.1:
                        Q[(qi, qj)] = correlation_matrix[i, j]
        
        return Q
    
    def decode_quantum_solution(self, quantum_solution: Dict[int, int]) -> np.ndarray:
        """Decode quantum solution to feature importance weights."""
        
        weights = np.zeros(len(self.inverse_mapping))
        
        for qubit, value in quantum_solution.items():
            if qubit in self.inverse_mapping:
                feature_idx = self.inverse_mapping[qubit]
                weights[feature_idx] = value
        
        return weights / (np.sum(weights) + 1e-8)  # Normalize to probabilities


class QuantumFeatureSelection:
    """
    Quantum-Enhanced Feature Selection
    
    Uses quantum optimization to select optimal feature subsets
    for machine learning models.
    """
    
    def __init__(self, max_features: int = 10, 
                 selection_criterion: str = 'mutual_information'):
        self.max_features = max_features
        self.selection_criterion = selection_criterion
        self.selected_features = []
        self.feature_scores = {}
        
    def select_features(self, X: np.ndarray, y: np.ndarray,
                       quantum_sampler: Optional[Callable] = None) -> Tuple[np.ndarray, List[int]]:
        """Select features using quantum optimization."""
        
        n_samples, n_features = X.shape
        
        # Create QUBO for feature selection
        Q = self._create_feature_selection_qubo(X, y)
        
        # Solve using quantum sampler
        if quantum_sampler:
            try:
                samples = quantum_sampler(Q, num_reads=100)
                best_sample = max(samples, key=lambda s: self._evaluate_feature_subset(X, y, s))
                selected_indices = [i for i, val in best_sample.items() if val == 1]
            except Exception as e:
                logger.error(f"Quantum feature selection failed: {e}")
                selected_indices = self._fallback_feature_selection(X, y)
        else:
            selected_indices = self._fallback_feature_selection(X, y)
        
        # Limit to max_features
        if len(selected_indices) > self.max_features:
            # Rank by individual feature importance
            feature_importance = []
            for idx in selected_indices:
                importance = self._calculate_feature_importance(X[:, [idx]], y)
                feature_importance.append((idx, importance))
            
            # Select top features
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in feature_importance[:self.max_features]]
        
        self.selected_features = selected_indices
        return X[:, selected_indices], selected_indices
    
    def _create_feature_selection_qubo(self, X: np.ndarray, y: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Create QUBO formulation for feature selection."""
        
        Q = {}
        n_features = X.shape[1]
        
        # Calculate individual feature importances
        for i in range(n_features):
            importance = self._calculate_feature_importance(X[:, [i]], y)
            Q[(i, i)] = -importance  # Negative to encourage selection of important features
        
        # Add penalty for selecting too many features
        penalty_strength = 1.0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                Q[(i, j)] = penalty_strength / n_features
        
        # Add correlation-based terms
        correlation_matrix = np.corrcoef(X.T)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                correlation = abs(correlation_matrix[i, j])
                if correlation > 0.7:  # Penalize highly correlated features
                    Q[(i, j)] = Q.get((i, j), 0) + correlation * penalty_strength
        
        return Q
    
    def _calculate_feature_importance(self, X_feat: np.ndarray, y: np.ndarray) -> float:
        """Calculate importance of a single feature."""
        
        if self.selection_criterion == 'mutual_information':
            return self._mutual_information(X_feat.flatten(), y)
        elif self.selection_criterion == 'correlation':
            return abs(np.corrcoef(X_feat.flatten(), y)[0, 1])
        elif self.selection_criterion == 'variance':
            return np.var(X_feat.flatten())
        else:
            # Default: correlation
            return abs(np.corrcoef(X_feat.flatten(), y)[0, 1])
    
    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between x and y."""
        
        # Discretize continuous variables
        x_discrete = np.digitize(x, np.percentile(x, [25, 50, 75]))
        
        if len(np.unique(y)) > 10:  # Continuous target
            y_discrete = np.digitize(y, np.percentile(y, [25, 50, 75]))
        else:  # Discrete target
            y_discrete = y.astype(int)
        
        # Calculate mutual information
        unique_x = np.unique(x_discrete)
        unique_y = np.unique(y_discrete)
        
        mi = 0.0
        n_total = len(x_discrete)
        
        for xi in unique_x:
            for yi in unique_y:
                p_xy = np.sum((x_discrete == xi) & (y_discrete == yi)) / n_total
                p_x = np.sum(x_discrete == xi) / n_total
                p_y = np.sum(y_discrete == yi) / n_total
                
                if p_xy > 0 and p_x > 0 and p_y > 0:
                    mi += p_xy * np.log2(p_xy / (p_x * p_y))
        
        return max(0, mi)
    
    def _evaluate_feature_subset(self, X: np.ndarray, y: np.ndarray, 
                                feature_subset: Dict[int, int]) -> float:
        """Evaluate quality of a feature subset."""
        
        selected_features = [i for i, val in feature_subset.items() if val == 1]
        
        if not selected_features:
            return 0.0
        
        X_subset = X[:, selected_features]
        
        # Use simple correlation-based evaluation
        if len(np.unique(y)) > 10:  # Regression
            correlations = [abs(np.corrcoef(X_subset[:, i], y)[0, 1]) 
                          for i in range(X_subset.shape[1])]
            return np.mean(correlations)
        else:  # Classification
            # Use class separation as criterion
            class_means = []
            for class_val in np.unique(y):
                class_mask = y == class_val
                if np.sum(class_mask) > 0:
                    class_mean = np.mean(X_subset[class_mask], axis=0)
                    class_means.append(class_mean)
            
            if len(class_means) > 1:
                # Calculate separation between class means
                separation = 0.0
                for i in range(len(class_means)):
                    for j in range(i + 1, len(class_means)):
                        separation += np.linalg.norm(class_means[i] - class_means[j])
                return separation / (len(class_means) * (len(class_means) - 1) / 2)
            
        return 0.0
    
    def _fallback_feature_selection(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Fallback feature selection using classical methods."""
        
        n_features = X.shape[1]
        feature_scores = []
        
        for i in range(n_features):
            score = self._calculate_feature_importance(X[:, [i]], y)
            feature_scores.append((i, score))
        
        # Sort by importance and select top features
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in feature_scores[:self.max_features]]


class QuantumHyperparameterOptimizer:
    """
    Quantum-Enhanced Hyperparameter Optimization
    
    Optimizes ML model hyperparameters using quantum algorithms.
    """
    
    def __init__(self, model_class: Any, param_space: Dict[str, List[Any]]):
        self.model_class = model_class
        self.param_space = param_space
        self.optimization_history = []
        
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                quantum_optimizer: Optional[Callable] = None,
                                cv_folds: int = 3,
                                scoring: str = 'accuracy') -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using quantum optimization."""
        
        # Create QUBO formulation for hyperparameter optimization
        Q = self._create_hyperparameter_qubo(X, y, cv_folds, scoring)
        
        # Use quantum optimizer if available
        if quantum_optimizer:
            try:
                quantum_samples = quantum_optimizer(Q, num_reads=50)
                best_sample = max(quantum_samples, 
                                key=lambda s: self._evaluate_hyperparameter_config(X, y, s, cv_folds, scoring))
                best_params = self._sample_to_hyperparameters(best_sample)
            except Exception as e:
                logger.error(f"Quantum hyperparameter optimization failed: {e}")
                best_params = self._fallback_hyperparameter_optimization(X, y, cv_folds, scoring)
        else:
            best_params = self._fallback_hyperparameter_optimization(X, y, cv_folds, scoring)
        
        # Evaluate best configuration
        best_score = self._evaluate_hyperparameter_config(X, y, 
                                                         self._hyperparameters_to_sample(best_params),
                                                         cv_folds, scoring)
        
        # Store optimization result
        self.optimization_history.append({
            'params': best_params,
            'score': best_score,
            'timestamp': time.time()
        })
        
        return best_params, best_score
    
    def _create_hyperparameter_qubo(self, X: np.ndarray, y: np.ndarray,
                                   cv_folds: int, scoring: str) -> Dict[Tuple[int, int], float]:
        """Create QUBO formulation for hyperparameter optimization."""
        
        Q = {}
        var_idx = 0
        param_to_vars = {}
        
        # Create binary variables for each hyperparameter choice
        for param_name, param_values in self.param_space.items():
            param_vars = []
            for value in param_values:
                param_vars.append(var_idx)
                var_idx += 1
            param_to_vars[param_name] = param_vars
            
            # One-hot constraint: exactly one value per parameter
            for i, var1 in enumerate(param_vars):
                Q[(var1, var1)] = -1.0  # Encourage selection
                for j, var2 in enumerate(param_vars[i+1:], i+1):
                    Q[(var1, var2)] = 2.0  # Penalize multiple selections
        
        # Sample configurations to estimate objective landscape
        sample_configs = []
        sample_scores = []
        
        for _ in range(min(20, np.prod([len(values) for values in self.param_space.values()]))):
            random_config = {param: np.random.choice(values) 
                           for param, values in self.param_space.items()}
            score = self._evaluate_hyperparameter_config(X, y, 
                                                        self._hyperparameters_to_sample(random_config),
                                                        cv_folds, scoring)
            sample_configs.append(random_config)
            sample_scores.append(score)
        
        # Add objective terms based on sampled performance
        if sample_scores:
            score_range = max(sample_scores) - min(sample_scores)
            if score_range > 0:
                for config, score in zip(sample_configs, sample_scores):
                    normalized_score = (score - min(sample_scores)) / score_range
                    
                    # Add bonus for high-performing configurations
                    for param_name, param_value in config.items():
                        if param_name in param_to_vars:
                            try:
                                value_idx = self.param_space[param_name].index(param_value)
                                var = param_to_vars[param_name][value_idx]
                                Q[(var, var)] = Q.get((var, var), 0) - normalized_score * 0.5
                            except ValueError:
                                pass
        
        return Q
    
    def _evaluate_hyperparameter_config(self, X: np.ndarray, y: np.ndarray,
                                       config_sample: Dict[int, int],
                                       cv_folds: int, scoring: str) -> float:
        """Evaluate a hyperparameter configuration."""
        
        try:
            # Convert sample to hyperparameters
            params = self._sample_to_hyperparameters(config_sample)
            
            # Create and evaluate model
            model = self.model_class(**params)
            
            if SKLEARN_AVAILABLE and hasattr(model, 'fit'):
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                return np.mean(scores)
            else:
                # Fallback evaluation
                model.fit(X, y)
                predictions = model.predict(X)
                
                if scoring == 'accuracy':
                    return accuracy_score(y, predictions)
                elif scoring == 'f1':
                    return f1_score(y, predictions, average='weighted')
                else:
                    return -mean_squared_error(y, predictions)
                    
        except Exception as e:
            logger.error(f"Hyperparameter evaluation failed: {e}")
            return 0.0
    
    def _sample_to_hyperparameters(self, sample: Dict[int, int]) -> Dict[str, Any]:
        """Convert QUBO sample to hyperparameter dictionary."""
        
        params = {}
        var_idx = 0
        
        for param_name, param_values in self.param_space.items():
            # Find which variable is set to 1
            selected_idx = 0  # Default to first value
            for i, value in enumerate(param_values):
                if sample.get(var_idx + i, 0) == 1:
                    selected_idx = i
                    break
            
            params[param_name] = param_values[selected_idx]
            var_idx += len(param_values)
        
        return params
    
    def _hyperparameters_to_sample(self, params: Dict[str, Any]) -> Dict[int, int]:
        """Convert hyperparameter dictionary to QUBO sample."""
        
        sample = {}
        var_idx = 0
        
        for param_name, param_values in self.param_space.items():
            param_value = params.get(param_name, param_values[0])
            
            # Set corresponding variable to 1
            try:
                value_idx = param_values.index(param_value)
                sample[var_idx + value_idx] = 1
                # Set other variables to 0
                for i in range(len(param_values)):
                    if i != value_idx:
                        sample[var_idx + i] = 0
            except ValueError:
                # Value not in list, use first value
                sample[var_idx] = 1
                for i in range(1, len(param_values)):
                    sample[var_idx + i] = 0
            
            var_idx += len(param_values)
        
        return sample
    
    def _fallback_hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray,
                                            cv_folds: int, scoring: str) -> Dict[str, Any]:
        """Fallback random search for hyperparameters."""
        
        best_params = None
        best_score = -np.inf
        
        for _ in range(20):  # Random search iterations
            random_params = {param: np.random.choice(values) 
                           for param, values in self.param_space.items()}
            
            score = self._evaluate_hyperparameter_config(X, y,
                                                        self._hyperparameters_to_sample(random_params),
                                                        cv_folds, scoring)
            
            if score > best_score:
                best_score = score
                best_params = random_params
        
        return best_params or {param: values[0] for param, values in self.param_space.items()}


class QuantumMLBridge:
    """
    Main Quantum-Classical ML Bridge
    
    Orchestrates quantum-enhanced machine learning workflows.
    """
    
    def __init__(self, quantum_backend: str = 'simulated'):
        self.quantum_backend = quantum_backend
        self.feature_encoder = AmplitudeEncodingTransformer()
        self.feature_selector = QuantumFeatureSelection()
        self.optimization_history = []
        
    def quantum_enhanced_pipeline(self, X: np.ndarray, y: np.ndarray,
                                 model_class: Any,
                                 hyperparameter_space: Dict[str, List[Any]],
                                 quantum_sampler: Optional[Callable] = None) -> Tuple[Any, QuantumMLMetrics]:
        """Run complete quantum-enhanced ML pipeline."""
        
        start_time = time.time()
        
        # Step 1: Quantum feature selection
        logger.info("Starting quantum feature selection...")
        X_selected, selected_features = self.feature_selector.select_features(
            X, y, quantum_sampler
        )
        
        # Step 2: Quantum hyperparameter optimization
        logger.info("Starting quantum hyperparameter optimization...")
        hyperopt = QuantumHyperparameterOptimizer(model_class, hyperparameter_space)
        best_params, best_score = hyperopt.optimize_hyperparameters(
            X_selected, y, quantum_sampler
        )
        
        # Step 3: Train final model
        logger.info("Training final quantum-optimized model...")
        final_model = model_class(**best_params)
        final_model.fit(X_selected, y)
        
        # Step 4: Calculate metrics
        classical_baseline = self._get_classical_baseline(X, y, model_class, hyperparameter_space)
        
        quantum_time = time.time() - start_time
        classical_time = classical_baseline['time']
        
        metrics = QuantumMLMetrics(
            classical_baseline_score=classical_baseline['score'],
            quantum_enhanced_score=best_score,
            improvement_ratio=best_score / max(classical_baseline['score'], 1e-6),
            training_time_ratio=classical_time / max(quantum_time, 1e-6),
            convergence_speed=1.0 / quantum_time,
            model_complexity_reduction=1.0 - (len(selected_features) / X.shape[1])
        )
        
        # Store results
        result_record = {
            'model': final_model,
            'selected_features': selected_features,
            'best_params': best_params,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.optimization_history.append(result_record)
        
        return final_model, metrics
    
    def _get_classical_baseline(self, X: np.ndarray, y: np.ndarray,
                              model_class: Any, hyperparameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Get classical baseline performance."""
        
        start_time = time.time()
        
        # Use default hyperparameters
        default_params = {param: values[0] for param, values in hyperparameter_space.items()}
        
        try:
            baseline_model = model_class(**default_params)
            baseline_model.fit(X, y)
            
            if SKLEARN_AVAILABLE:
                scores = cross_val_score(baseline_model, X, y, cv=3)
                baseline_score = np.mean(scores)
            else:
                predictions = baseline_model.predict(X)
                if len(np.unique(y)) > 10:  # Regression
                    baseline_score = -mean_squared_error(y, predictions)
                else:  # Classification
                    baseline_score = accuracy_score(y, predictions)
        except Exception as e:
            logger.error(f"Classical baseline evaluation failed: {e}")
            baseline_score = 0.0
        
        baseline_time = time.time() - start_time
        
        return {
            'score': baseline_score,
            'time': baseline_time
        }
    
    def get_quantum_ml_report(self) -> str:
        """Generate quantum ML enhancement report."""
        
        if not self.optimization_history:
            return "No quantum ML optimization data available."
        
        latest_result = self.optimization_history[-1]
        metrics = latest_result['metrics']
        
        report = f"""
# Quantum-Enhanced Machine Learning Report

## Performance Improvements
- **Accuracy Improvement**: {(metrics.improvement_ratio - 1) * 100:.1f}%
- **Training Time Speedup**: {metrics.training_time_ratio:.2f}x
- **Model Complexity Reduction**: {metrics.model_complexity_reduction * 100:.1f}%
- **Convergence Speed**: {metrics.convergence_speed:.3f} /s

## Model Configuration
- **Selected Features**: {len(latest_result['selected_features'])}
- **Optimized Parameters**: {latest_result['best_params']}

## Quantum Advantage Assessment
"""
        
        if metrics.improvement_ratio > 1.2:
            report += "🟢 **Significant quantum advantage** - Substantial improvements over classical methods"
        elif metrics.improvement_ratio > 1.05:
            report += "🟡 **Moderate quantum advantage** - Measurable improvements achieved"
        elif metrics.improvement_ratio > 0.95:
            report += "🟠 **Marginal quantum advantage** - Comparable performance with potential benefits"
        else:
            report += "🔴 **No clear quantum advantage** - Classical methods currently superior"
        
        return report