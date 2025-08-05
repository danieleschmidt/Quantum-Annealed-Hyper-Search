"""
System health checks and diagnostics.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ..utils.logging_config import get_logger

logger = get_logger('health_check')


class HealthChecker:
    """
    Comprehensive system health checks for quantum hyperparameter search.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize health checker.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.checks = []
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Dictionary with check results
        """
        if self.verbose:
            logger.info("Running comprehensive health checks...")
        
        results = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # System checks
        results['checks']['system'] = self._check_system_requirements()
        results['checks']['dependencies'] = self._check_dependencies()
        results['checks']['memory'] = self._check_memory_availability()
        
        # Backend checks
        results['checks']['backends'] = self._check_backends()
        
        # Functionality checks
        results['checks']['basic_functionality'] = self._check_basic_functionality()
        results['checks']['data_processing'] = self._check_data_processing()
        
        # Performance checks
        results['checks']['performance'] = self._check_performance_baseline()
        
        # Determine overall status
        failed_checks = [
            name for name, result in results['checks'].items()
            if result.get('status') == 'failed'
        ]
        
        if failed_checks:
            results['overall_status'] = 'unhealthy'
            results['failed_checks'] = failed_checks
        elif any(result.get('status') == 'warning' for result in results['checks'].values()):
            results['overall_status'] = 'warning'
        
        if self.verbose:
            self._print_health_summary(results)
        
        return results
    
    def _check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements."""
        result = {'status': 'healthy', 'details': {}}
        
        try:
            # Python version
            python_version = sys.version_info
            result['details']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            if python_version < (3, 8):
                result['status'] = 'failed'
                result['message'] = f"Python 3.8+ required, found {result['details']['python_version']}"
                return result
            
            # Platform info
            import platform
            result['details']['platform'] = platform.platform()
            result['details']['architecture'] = platform.architecture()
            
            # CPU info
            result['details']['cpu_count'] = os.cpu_count()
            
            logger.info(f"System check passed: Python {result['details']['python_version']} on {result['details']['platform']}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['message'] = f"System check failed: {e}"
        
        return result
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        result = {'status': 'healthy', 'details': {}, 'missing': [], 'outdated': []}
        
        required_packages = {
            'numpy': '1.21.0',
            'scipy': '1.7.0',
            'scikit-learn': '1.0.0',
            'dimod': '0.12.0',
        }
        
        optional_packages = {
            'dwave-ocean-sdk': '6.0.0',
            'matplotlib': '3.5.0',
            'pandas': '1.3.0',
        }
        
        try:
            import pkg_resources
            
            # Check required packages
            for package, min_version in required_packages.items():
                try:
                    installed_version = pkg_resources.get_distribution(package).version
                    result['details'][package] = installed_version
                    
                    # Simple version comparison (not perfect but adequate)
                    if self._version_compare(installed_version, min_version) < 0:
                        result['outdated'].append(f"{package}: {installed_version} < {min_version}")
                        
                except pkg_resources.DistributionNotFound:
                    result['missing'].append(package)
            
            # Check optional packages
            for package, min_version in optional_packages.items():
                try:
                    installed_version = pkg_resources.get_distribution(package).version
                    result['details'][package] = installed_version
                except pkg_resources.DistributionNotFound:
                    result['details'][package] = 'not installed'
            
            # Determine status
            if result['missing']:
                result['status'] = 'failed'
                result['message'] = f"Missing required packages: {', '.join(result['missing'])}"
            elif result['outdated']:
                result['status'] = 'warning'
                result['message'] = f"Outdated packages: {', '.join(result['outdated'])}"
            
            logger.info(f"Dependencies check: {len(result['details'])} packages checked")
            
        except Exception as e:
            result['status'] = 'failed'
            result['message'] = f"Dependency check failed: {e}"
        
        return result
    
    def _check_memory_availability(self) -> Dict[str, Any]:
        """Check memory availability."""
        result = {'status': 'healthy', 'details': {}}
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            result['details']['total_memory_gb'] = round(memory.total / (1024**3), 2)
            result['details']['available_memory_gb'] = round(memory.available / (1024**3), 2)
            result['details']['memory_percent_used'] = memory.percent
            
            # Check if we have enough memory for typical workloads
            min_memory_gb = 2.0
            if result['details']['available_memory_gb'] < min_memory_gb:
                result['status'] = 'warning'
                result['message'] = f"Low memory: {result['details']['available_memory_gb']:.1f}GB available"
            
            logger.info(f"Memory check: {result['details']['available_memory_gb']:.1f}GB available")
            
        except ImportError:
            result['status'] = 'warning'
            result['message'] = "psutil not available for memory monitoring"
        except Exception as e:
            result['status'] = 'warning'
            result['message'] = f"Memory check failed: {e}"
        
        return result
    
    def _check_backends(self) -> Dict[str, Any]:
        """Check quantum backends availability."""
        result = {'status': 'healthy', 'details': {}}
        
        from ..backends.backend_factory import get_backend
        
        backends_to_check = ['simulator', 'dwave']
        
        for backend_name in backends_to_check:
            backend_result = {'available': False, 'error': None}
            
            try:
                backend = get_backend(backend_name)
                properties = backend.get_properties()
                backend_result['available'] = properties.get('available', False)
                backend_result['properties'] = properties
                
                if not backend_result['available']:
                    backend_result['error'] = properties.get('error', 'Unknown availability issue')
                
            except Exception as e:
                backend_result['error'] = str(e)
            
            result['details'][backend_name] = backend_result
        
        # Check if at least one backend is available
        available_backends = [
            name for name, details in result['details'].items()
            if details['available']
        ]
        
        if not available_backends:
            result['status'] = 'failed'
            result['message'] = "No quantum backends available"
        elif 'simulator' not in available_backends:
            result['status'] = 'warning'
            result['message'] = "Simulator backend not available"
        
        logger.info(f"Backend check: {len(available_backends)} backends available")
        
        return result
    
    def _check_basic_functionality(self) -> Dict[str, Any]:
        """Check basic functionality with a simple test case."""
        result = {'status': 'healthy', 'details': {}}
        
        try:
            from .. import QuantumHyperSearch
            
            # Create simple test case
            X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
            search_space = {'n_estimators': [10, 20], 'max_depth': [3, 5]}
            
            # Run quick optimization
            qhs = QuantumHyperSearch(backend='simulator', verbose=False)
            
            start_time = time.time()
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=3,
                quantum_reads=10,
                cv_folds=3
            )
            execution_time = time.time() - start_time
            
            result['details']['execution_time'] = round(execution_time, 3)
            result['details']['trials_completed'] = len(history.trials)
            result['details']['best_score'] = round(history.best_score, 4)
            result['details']['best_params'] = best_params
            
            if history.best_score <= 0:
                result['status'] = 'warning'
                result['message'] = f"Low performance: score = {history.best_score}"
            
            logger.info(f"Functionality check passed: {result['details']['best_score']:.4f} score in {execution_time:.2f}s")
            
        except Exception as e:
            result['status'] = 'failed'
            result['message'] = f"Basic functionality test failed: {e}"
        
        return result
    
    def _check_data_processing(self) -> Dict[str, Any]:
        """Check data processing capabilities."""
        result = {'status': 'healthy', 'details': {}}
        
        try:
            from ..utils.validation import validate_data, validate_search_space
            
            # Test data validation
            X = np.random.randn(100, 10)
            y = np.random.randint(0, 2, 100)
            
            X_val, y_val = validate_data(X, y)
            result['details']['data_validation'] = 'passed'
            
            # Test search space validation
            search_space = {
                'param1': [1, 2, 3, 4, 5],
                'param2': ['a', 'b', 'c'],
                'param3': [True, False]
            }
            
            validated_space = validate_search_space(search_space)
            result['details']['search_space_validation'] = 'passed'
            result['details']['search_space_size'] = len(validated_space)
            
            logger.info("Data processing check passed")
            
        except Exception as e:
            result['status'] = 'failed'
            result['message'] = f"Data processing check failed: {e}"
        
        return result
    
    def _check_performance_baseline(self) -> Dict[str, Any]:
        """Check performance against baseline."""
        result = {'status': 'healthy', 'details': {}}
        
        try:
            # Simple performance benchmark
            n_samples = 200
            n_features = 20
            
            X, y = make_classification(
                n_samples=n_samples, 
                n_features=n_features, 
                n_classes=2, 
                random_state=42
            )
            
            search_space = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15]
            }
            
            from .. import QuantumHyperSearch
            qhs = QuantumHyperSearch(backend='simulator', verbose=False)
            
            start_time = time.time()
            best_params, history = qhs.optimize(
                model_class=RandomForestClassifier,
                param_space=search_space,
                X=X, y=y,
                n_iterations=5,
                quantum_reads=20,
                cv_folds=3
            )
            execution_time = time.time() - start_time
            
            result['details']['benchmark_time'] = round(execution_time, 3)
            result['details']['benchmark_score'] = round(history.best_score, 4)
            result['details']['samples_per_second'] = round(n_samples / execution_time, 1)
            
            # Performance thresholds
            max_time_per_iteration = 10.0  # seconds
            min_accuracy = 0.7
            
            if execution_time / 5 > max_time_per_iteration:
                result['status'] = 'warning'
                result['message'] = f"Slow performance: {execution_time/5:.1f}s per iteration"
            elif history.best_score < min_accuracy:
                result['status'] = 'warning'
                result['message'] = f"Low accuracy: {history.best_score:.3f}"
            
            logger.info(f"Performance benchmark: {history.best_score:.4f} score in {execution_time:.2f}s")
            
        except Exception as e:
            result['status'] = 'failed'
            result['message'] = f"Performance check failed: {e}"
        
        return result
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Simple version comparison. Returns -1, 0, or 1."""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            if v1_parts < v2_parts:
                return -1
            elif v1_parts > v2_parts:
                return 1
            else:
                return 0
        except:
            return 0  # Assume equal if parsing fails
    
    def _print_health_summary(self, results: Dict[str, Any]) -> None:
        """Print health check summary."""
        status = results['overall_status']
        
        if status == 'healthy':
            print("✅ System Health: HEALTHY")
        elif status == 'warning':
            print("⚠️  System Health: WARNING")
        else:
            print("❌ System Health: UNHEALTHY")
        
        print("-" * 50)
        
        for check_name, check_result in results['checks'].items():
            status_icon = {
                'healthy': '✅',
                'warning': '⚠️',
                'failed': '❌'
            }.get(check_result['status'], '❓')
            
            print(f"{status_icon} {check_name.title()}: {check_result['status'].upper()}")
            
            if check_result.get('message'):
                print(f"   {check_result['message']}")
        
        print("-" * 50)
        
        if 'failed_checks' in results:
            print(f"Failed checks: {', '.join(results['failed_checks'])}")
    
    def check_quantum_connectivity(self, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Check D-Wave quantum computer connectivity.
        
        Args:
            token: D-Wave API token
            
        Returns:
            Connectivity check results
        """
        result = {'status': 'unknown', 'details': {}}
        
        if not token:
            result['status'] = 'skipped'
            result['message'] = 'No D-Wave token provided'
            return result
        
        try:
            from ..backends.backend_factory import get_backend
            
            backend = get_backend('dwave', token=token)
            properties = backend.get_properties()
            
            if properties.get('available'):
                result['status'] = 'healthy'
                result['details'] = properties
                logger.info(f"D-Wave connectivity: Connected to {properties.get('solver', 'unknown')}")
            else:
                result['status'] = 'failed'
                result['message'] = properties.get('error', 'Connection failed')
                
        except Exception as e:
            result['status'] = 'failed'
            result['message'] = f"D-Wave connection error: {e}"
        
        return result


def run_health_check(verbose: bool = True, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run health checks.
    
    Args:
        verbose: Enable verbose output
        token: Optional D-Wave API token for quantum connectivity check
        
    Returns:
        Health check results
    """
    checker = HealthChecker(verbose=verbose)
    results = checker.run_all_checks()
    
    # Add quantum connectivity check if token provided
    if token:
        results['checks']['quantum_connectivity'] = checker.check_quantum_connectivity(token)
    
    return results


if __name__ == '__main__':
    # Command line health check
    import argparse
    
    parser = argparse.ArgumentParser(description='Run quantum hyperparameter search health checks')
    parser.add_argument('--token', help='D-Wave API token for quantum connectivity check')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    results = run_health_check(verbose=not args.quiet, token=args.token)
    
    exit_code = 0 if results['overall_status'] == 'healthy' else 1
    sys.exit(exit_code)