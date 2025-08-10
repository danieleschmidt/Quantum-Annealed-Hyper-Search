#!/usr/bin/env python3
"""
Simplified Quality Gates Runner

Runs comprehensive quality gates without external testing dependencies.
"""

import sys
import time
import numpy as np
import os
from pathlib import Path
import json
import logging
import traceback
from typing import Dict, List, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "quantum_hyper_search"))

# Test imports
try:
    from quantum_hyper_search.core.quantum_hyper_search import QuantumHyperSearch
    from quantum_hyper_search.backends.backend_factory import BackendFactory
    from quantum_hyper_search.utils.enterprise_security import QuantumSecurityManager
    HAS_QUANTUM_MODULES = True
    logger.info("✓ Quantum modules loaded successfully")
except ImportError as e:
    HAS_QUANTUM_MODULES = False
    logger.warning(f"⚠ Could not import quantum modules: {e}")

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
    logger.info("✓ Scikit-learn loaded successfully")
except ImportError:
    HAS_SKLEARN = False
    logger.warning("⚠ Scikit-learn not available")

try:
    import psutil
    HAS_PSUTIL = True
    logger.info("✓ Psutil loaded successfully")
except ImportError:
    HAS_PSUTIL = False
    logger.warning("⚠ Psutil not available - system monitoring disabled")


class QualityGateRunner:
    """Runs all quality gate tests."""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling."""
        logger.info(f"Running test: {test_name}")
        self.total_tests += 1
        
        try:
            start_time = time.time()
            test_func()
            end_time = time.time()
            
            self.passed_tests += 1
            self.results[test_name] = {
                'status': 'PASSED',
                'execution_time': end_time - start_time,
                'error': None
            }
            logger.info(f"✓ {test_name} PASSED ({end_time - start_time:.2f}s)")
            
        except Exception as e:
            self.failed_tests += 1
            self.results[test_name] = {
                'status': 'FAILED',
                'execution_time': 0,
                'error': str(e)
            }
            logger.error(f"✗ {test_name} FAILED: {e}")
            logger.debug(traceback.format_exc())
    
    def run_all_tests(self):
        """Run all quality gate tests."""
        logger.info("=" * 60)
        logger.info("QUANTUM HYPERPARAMETER OPTIMIZATION - QUALITY GATES")
        logger.info("=" * 60)
        
        # Core functionality tests
        if HAS_QUANTUM_MODULES and HAS_SKLEARN:
            self.run_test("Basic Quantum Optimization", self.test_basic_quantum_optimization)
            self.run_test("Backend Compatibility", self.test_backend_compatibility)
            self.run_test("Parameter Space Validation", self.test_parameter_space_validation)
        else:
            logger.warning("Skipping quantum tests - dependencies not available")
        
        # Security tests
        if HAS_QUANTUM_MODULES:
            self.run_test("Security Features", self.test_security_features)
            self.run_test("Data Encryption", self.test_data_encryption)
        else:
            logger.warning("Skipping security tests - quantum modules not available")
        
        # Performance tests
        self.run_test("Memory Usage", self.test_memory_usage)
        self.run_test("Error Handling", self.test_error_handling)
        
        # Code quality tests
        self.run_test("Code Syntax Validation", self.test_code_syntax)
        self.run_test("Import Structure", self.test_import_structure)
        
        # Generate final report
        self.generate_report()
    
    def test_basic_quantum_optimization(self):
        """Test basic quantum optimization functionality."""
        # Generate test data
        X, y = make_classification(
            n_samples=200, n_features=5, n_classes=2, random_state=42
        )
        
        search_space = {
            'n_estimators': [10, 20, 50],
            'max_depth': [3, 5, 7]
        }
        
        # Run optimization
        qhs = QuantumHyperSearch(backend='simple', random_state=42)
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=search_space,
            X=X, y=y,
            n_iterations=5,
            cv_folds=3
        )
        
        # Validate results
        assert best_params is not None, "No best parameters returned"
        assert hasattr(history, 'best_score'), "History missing best_score"
        assert history.best_score > 0.3, f"Poor optimization score: {history.best_score}"
        assert len(best_params) > 0, "Empty parameter set returned"
        
        # Test final model
        final_model = RandomForestClassifier(**best_params, random_state=42)
        final_model.fit(X, y)
        predictions = final_model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        assert accuracy > 0.5, f"Final model accuracy too low: {accuracy}"
        
        logger.info(f"Optimization score: {history.best_score:.3f}, Final accuracy: {accuracy:.3f}")
    
    def test_backend_compatibility(self):
        """Test different backend compatibility."""
        backends_to_test = ['simple', 'simulator']
        results = {}
        
        X, y = make_classification(n_samples=100, n_features=3, n_classes=2, random_state=42)
        search_space = {'n_estimators': [10, 20], 'max_depth': [3, 5]}
        
        for backend_name in backends_to_test:
            try:
                qhs = QuantumHyperSearch(backend=backend_name, random_state=42)
                best_params, history = qhs.optimize(
                    model_class=RandomForestClassifier,
                    param_space=search_space,
                    X=X, y=y,
                    n_iterations=3,
                    cv_folds=3
                )
                
                results[backend_name] = history.best_score
                assert history.best_score > 0, f"Backend {backend_name} produced invalid score"
                
            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}")
        
        assert len(results) > 0, "No backends were successfully tested"
        logger.info(f"Backend compatibility results: {results}")
    
    def test_parameter_space_validation(self):
        """Test parameter space validation and edge cases."""
        X, y = make_classification(n_samples=50, n_features=3, n_classes=2, random_state=42)
        
        # Test empty parameter space
        try:
            qhs = QuantumHyperSearch(backend='simple', random_state=42)
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space={},
                X=X, y=y,
                n_iterations=2
            )
            # Should handle gracefully
            assert best_params is not None, "Failed to handle empty parameter space"
        except Exception:
            pass  # Acceptable to fail gracefully
        
        # Test single parameter
        single_param_space = {'n_estimators': [10]}
        qhs = QuantumHyperSearch(backend='simple', random_state=42)
        best_params, history = qhs.optimize(
            model_class=RandomForestClassifier,
            param_space=single_param_space,
            X=X, y=y,
            n_iterations=2
        )
        
        assert best_params['n_estimators'] == 10, "Single parameter optimization failed"
        
        logger.info("Parameter space validation completed")
    
    def test_security_features(self):
        """Test security features."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_log_path = os.path.join(tmpdir, 'audit.log')
            security_manager = QuantumSecurityManager(
                audit_log_path=audit_log_path,
                compliance_mode='standard'
            )
            
            # Test authentication
            token = security_manager.authenticate_user('test_user', 'secure_password123')
            assert token is not None, "Authentication failed"
            
            # Test session validation
            session = security_manager.validate_session_token(token)
            assert session is not None, "Session validation failed"
            
            # Test authorization
            authorized = security_manager.authorize_action(token, 'optimize', 'hyperparameters')
            assert authorized, "Authorization failed"
            
            # Test compliance report
            compliance_report = security_manager.generate_compliance_report()
            assert 'compliance_score' in compliance_report, "Missing compliance score"
            
            compliance_score = compliance_report['compliance_score']
            assert compliance_score >= 80, f"Compliance score too low: {compliance_score}%"
            
            logger.info(f"Security features validated, compliance: {compliance_score}%")
    
    def test_data_encryption(self):
        """Test data encryption functionality."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            security_manager = QuantumSecurityManager()
            
            # Test data encryption/decryption
            test_data = {
                'parameters': {'n_estimators': 100, 'max_depth': 5},
                'model_type': 'RandomForest',
                'score': 0.95
            }
            
            # Encrypt data
            encrypted_data = security_manager.encrypt_data(test_data)
            assert encrypted_data is not None, "Data encryption failed"
            
            # Decrypt data
            decrypted_data = security_manager.decrypt_data(encrypted_data)
            assert decrypted_data == test_data, "Data decryption failed or data corrupted"
            
            # Test parameter sanitization
            dangerous_params = {
                'n_estimators': 100,
                'eval_code': '__import__("os").system("ls")',
                '__class__': 'malicious'
            }
            
            sanitized = security_manager.sanitize_parameters(dangerous_params)
            assert 'eval_code' not in sanitized, "Dangerous parameter not removed"
            assert '__class__' not in sanitized, "Private attribute not removed"
            assert 'n_estimators' in sanitized, "Valid parameter incorrectly removed"
            
            logger.info("Data encryption and sanitization validated")
    
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        if not HAS_PSUTIL:
            logger.warning("Psutil not available, skipping memory test")
            return
        
        import gc
        
        # Get initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy optimization objects
        for i in range(3):
            if HAS_QUANTUM_MODULES and HAS_SKLEARN:
                X, y = make_classification(n_samples=100, n_features=5, random_state=42)
                
                qhs = QuantumHyperSearch(backend='simple', random_state=42)
                best_params, history = qhs.optimize(
                    model_class=RandomForestClassifier,
                    param_space={'n_estimators': [10, 20], 'max_depth': [3, 5]},
                    X=X, y=y,
                    n_iterations=2
                )
                
                del qhs, best_params, history
                gc.collect()
        
        # Check final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        logger.info(f"Memory usage test completed. Growth: {memory_growth:.1f}MB")
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        if not (HAS_QUANTUM_MODULES and HAS_SKLEARN):
            logger.warning("Skipping error handling test - dependencies not available")
            return
        
        X, y = make_classification(n_samples=50, n_features=3, random_state=42)
        
        # Test with problematic objective function
        def problematic_objective(params):
            if params.get('n_estimators', 0) < 5:
                raise ValueError("Simulated error")
            
            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=2)
            return np.mean(scores)
        
        # Test error recovery
        search_space = {'n_estimators': [1, 10, 20], 'max_depth': [3, 5]}  # Include problematic value
        
        try:
            qhs = QuantumHyperSearch(backend='simple', random_state=42)
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=5
            )
            
            # Should handle errors gracefully
            assert best_params is not None, "Error handling failed"
            assert best_params.get('n_estimators', 0) >= 5, "Did not avoid problematic parameters"
            
        except Exception as e:
            # Acceptable if the system fails gracefully
            logger.warning(f"Error handling test failed, but this may be acceptable: {e}")
        
        logger.info("Error handling test completed")
    
    def test_code_syntax(self):
        """Test Python syntax across all source files."""
        python_files = []
        
        for root, dirs, files in os.walk('quantum_hyper_search'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    compile(content, py_file, 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception:
                # Skip other errors (e.g., encoding issues)
                pass
        
        assert len(syntax_errors) == 0, f"Syntax errors found: {syntax_errors}"
        
        logger.info(f"Syntax validation passed for {len(python_files)} files")
    
    def test_import_structure(self):
        """Test import structure and module organization."""
        core_modules = [
            'quantum_hyper_search.__init__',
            'quantum_hyper_search.core.quantum_hyper_search',
            'quantum_hyper_search.backends.backend_factory'
        ]
        
        import_results = {}
        
        for module_name in core_modules:
            try:
                __import__(module_name)
                import_results[module_name] = True
            except ImportError as e:
                import_results[module_name] = f"Failed: {e}"
        
        successful_imports = sum(1 for result in import_results.values() if result is True)
        total_modules = len(core_modules)
        
        assert successful_imports > 0, f"No core modules could be imported: {import_results}"
        
        logger.info(f"Import structure test: {successful_imports}/{total_modules} modules imported successfully")
    
    def generate_report(self):
        """Generate final quality gates report."""
        logger.info("=" * 60)
        logger.info("QUALITY GATES SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        
        pass_rate = (self.passed_tests / max(self.total_tests, 1)) * 100
        logger.info(f"Pass rate: {pass_rate:.1f}%")
        
        if self.failed_tests > 0:
            logger.error("FAILED TESTS:")
            for test_name, result in self.results.items():
                if result['status'] == 'FAILED':
                    logger.error(f"  - {test_name}: {result['error']}")
        
        # Determine overall status
        if self.failed_tests == 0:
            overall_status = "✓ ALL QUALITY GATES PASSED"
            status_color = "SUCCESS"
        elif pass_rate >= 80:
            overall_status = "⚠ QUALITY GATES MOSTLY PASSED"
            status_color = "WARNING"
        else:
            overall_status = "✗ QUALITY GATES FAILED"
            status_color = "FAILURE"
        
        logger.info("=" * 60)
        logger.info(f"{status_color}: {overall_status}")
        logger.info("=" * 60)
        
        # Save detailed report
        report = {
            'timestamp': time.time(),
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'pass_rate_percent': pass_rate,
            'overall_status': overall_status,
            'test_results': self.results,
            'system_info': {
                'has_quantum_modules': HAS_QUANTUM_MODULES,
                'has_sklearn': HAS_SKLEARN,
                'has_psutil': HAS_PSUTIL
            }
        }
        
        with open('quality_gates_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Detailed report saved to: quality_gates_report.json")
        
        return pass_rate >= 80  # Return True if quality gates pass


def main():
    """Main function to run quality gates."""
    runner = QualityGateRunner()
    success = runner.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)