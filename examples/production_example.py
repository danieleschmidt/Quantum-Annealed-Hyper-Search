#!/usr/bin/env python3
"""
Production-Ready Quantum Hyperparameter Optimization Example

This example demonstrates production-ready usage with:
- Error handling and monitoring
- Security best practices  
- Performance optimization
- Logging and metrics
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

from quantum_hyper_search import QuantumHyperSearch


class ProductionQuantumOptimizer:
    """Production-ready quantum hyperparameter optimizer."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """Initialize production optimizer with configuration."""
        self.setup_logging(log_level)
        self.config = self.load_config(config_path)
        
        # Initialize quantum optimizer with production settings
        self.optimizer = QuantumHyperSearch(
            backend=self.config.get('backend', 'simulator'),
            enable_logging=True,
            enable_monitoring=True,
            enable_security=True,
            enable_caching=True,
            enable_parallel=True,
            enable_auto_scaling=True,
            max_parallel_workers=self.config.get('max_workers'),
            cache_size=self.config.get('cache_size', 50000),
            optimization_strategy=self.config.get('strategy', 'adaptive'),
            log_level=log_level
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Production quantum optimizer initialized")
    
    def setup_logging(self, log_level: str):
        """Setup production logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('quantum_optimizer.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'backend': 'simulator',
            'max_workers': None,  # Auto-detect
            'cache_size': 50000,
            'strategy': 'adaptive',
            'security': {
                'input_validation': 'strict',
                'parameter_sanitization': True
            },
            'performance': {
                'timeout': 3600,  # 1 hour max per optimization
                'max_iterations': 50,
                'early_stopping': True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def validate_inputs(
        self,
        model_class: type,
        param_space: Dict[str, list],
        X: np.ndarray,
        y: np.ndarray
    ) -> bool:
        """Validate inputs for production safety."""
        try:
            # Validate model class
            if not hasattr(model_class, 'fit') or not hasattr(model_class, 'predict'):
                raise ValueError("Model class must have fit() and predict() methods")
            
            # Validate parameter space
            if not param_space or not isinstance(param_space, dict):
                raise ValueError("Parameter space must be a non-empty dictionary")
            
            total_combinations = np.prod([len(v) for v in param_space.values()])
            if total_combinations > 1e6:
                self.logger.warning(f"Large search space: {total_combinations:,} combinations")
            
            # Validate data
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have same number of samples")
            
            if X.shape[0] < 10:
                raise ValueError("Insufficient training data (minimum 10 samples)")
            
            # Check for common issues
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                raise ValueError("Data contains NaN values")
            
            if len(np.unique(y)) < 2:
                raise ValueError("Target must have at least 2 classes")
            
            self.logger.info("Input validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            raise
    
    def optimize_with_monitoring(
        self,
        model_class: type,
        param_space: Dict[str, list],
        X: np.ndarray,
        y: np.ndarray,
        **optimization_kwargs
    ) -> Dict[str, Any]:
        """Run optimization with comprehensive monitoring."""
        
        # Validate inputs
        self.validate_inputs(model_class, param_space, X, y)
        
        # Setup monitoring
        start_time = time.time()
        optimization_id = f"opt_{int(start_time)}"
        
        self.logger.info(f"Starting optimization {optimization_id}")
        self.logger.info(f"Dataset shape: {X.shape}, Parameters: {len(param_space)}")
        
        try:
            # Run optimization with timeout protection
            timeout = self.config['performance']['timeout']
            max_iterations = min(
                optimization_kwargs.get('n_iterations', 20),
                self.config['performance']['max_iterations']
            )
            
            # Set production defaults
            optimization_params = {
                'n_iterations': max_iterations,
                'quantum_reads': optimization_kwargs.get('quantum_reads', 500),
                'cv_folds': optimization_kwargs.get('cv_folds', 5),
                'scoring': optimization_kwargs.get('scoring', 'f1_macro'),
                **{k: v for k, v in optimization_kwargs.items() 
                   if k not in ['n_iterations', 'quantum_reads', 'cv_folds', 'scoring']}
            }
            
            best_params, history = self.optimizer.optimize(
                model_class=model_class,
                param_space=param_space,
                X=X,
                y=y,
                **optimization_params
            )
            
            total_time = time.time() - start_time
            
            # Comprehensive result validation
            if best_params is None:
                raise RuntimeError("Optimization failed to find valid parameters")
            
            if history.best_score <= 0:
                self.logger.warning(f"Low optimization score: {history.best_score}")
            
            # Create comprehensive results
            results = {
                'optimization_id': optimization_id,
                'best_parameters': best_params,
                'best_score': history.best_score,
                'total_time': total_time,
                'n_evaluations': history.n_evaluations,
                'efficiency': history.n_evaluations / total_time,
                'dataset_info': {
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'n_classes': len(np.unique(y))
                },
                'search_space_info': {
                    'n_parameters': len(param_space),
                    'total_combinations': np.prod([len(v) for v in param_space.values()])
                },
                'performance_metrics': {
                    'optimization_time': total_time,
                    'evaluations_per_second': history.n_evaluations / total_time,
                    'convergence_iterations': history.n_evaluations
                }
            }
            
            self.logger.info(f"Optimization {optimization_id} completed successfully")
            self.logger.info(f"Best score: {history.best_score:.4f}")
            self.logger.info(f"Total time: {total_time:.2f}s")
            self.logger.info(f"Efficiency: {results['efficiency']:.1f} eval/s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimization {optimization_id} failed: {e}")
            raise
    
    def validate_final_model(
        self,
        model_class: type,
        best_params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Validate final model with comprehensive testing."""
        try:
            self.logger.info("Validating final model...")
            
            # Create and train final model
            final_model = model_class(**best_params, random_state=42)
            final_model.fit(X, y)
            
            # Generate predictions
            y_pred = final_model.predict(X)
            
            # Create classification report
            report = classification_report(y, y_pred, output_dict=True)
            
            validation_results = {
                'model_params': best_params,
                'training_accuracy': report['accuracy'],
                'macro_avg_f1': report['macro avg']['f1-score'],
                'weighted_avg_f1': report['weighted avg']['f1-score'],
                'per_class_metrics': {
                    str(k): v for k, v in report.items() 
                    if k not in ['accuracy', 'macro avg', 'weighted avg']
                },
                'validation_passed': report['accuracy'] > 0.5
            }
            
            self.logger.info(f"Final model validation: {validation_results['validation_passed']}")
            self.logger.info(f"Training accuracy: {validation_results['training_accuracy']:.4f}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            raise
    
    def save_results(
        self,
        results: Dict[str, Any],
        filepath: str = "optimization_results.json"
    ):
        """Save optimization results to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise


def production_workflow_example():
    """Complete production workflow example."""
    print("🌌 Production Quantum Hyperparameter Optimization")
    print("=" * 60)
    
    # Initialize production optimizer
    optimizer = ProductionQuantumOptimizer(log_level="INFO")
    
    # Create realistic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        class_sep=0.8,
        random_state=42
    )
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Define comprehensive search space
    param_space = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    total_combinations = np.prod([len(v) for v in param_space.values()])
    print(f"🔍 Search space: {len(param_space)} parameters, {total_combinations:,} combinations")
    
    try:
        # Run production optimization
        print("\n🚀 Starting production optimization...")
        results = optimizer.optimize_with_monitoring(
            model_class=RandomForestClassifier,
            param_space=param_space,
            X=X,
            y=y,
            n_iterations=15,
            quantum_reads=300,
            cv_folds=5,
            scoring='f1_macro'
        )
        
        # Validate final model
        print("\n🧪 Validating final model...")
        validation = optimizer.validate_final_model(
            RandomForestClassifier,
            results['best_parameters'],
            X,
            y
        )
        
        # Combine results
        final_results = {
            'optimization': results,
            'validation': validation,
            'timestamp': time.time()
        }
        
        # Save results
        optimizer.save_results(final_results, "production_optimization_results.json")
        
        # Summary
        print(f"\n✅ Production optimization completed!")
        print(f"🏆 Best Score: {results['best_score']:.4f}")
        print(f"🎯 Best Parameters: {results['best_parameters']}")
        print(f"⏱️  Total Time: {results['total_time']:.2f} seconds")
        print(f"📊 Efficiency: {results['efficiency']:.1f} evaluations/second")
        print(f"🧪 Validation Accuracy: {validation['training_accuracy']:.4f}")
        print(f"✅ Validation Passed: {validation['validation_passed']}")
        
        return final_results
        
    except Exception as e:
        print(f"❌ Production optimization failed: {e}")
        raise


if __name__ == "__main__":
    # Run production workflow
    results = production_workflow_example()
    
    print(f"\n🎉 Production workflow completed successfully!")
    print(f"📄 Results saved to production_optimization_results.json")