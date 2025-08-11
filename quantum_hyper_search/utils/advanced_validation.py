"""
Advanced Validation System for Quantum Optimization

Comprehensive validation utilities for quantum hyperparameter search
with enterprise-grade error handling and security validation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
from enum import Enum
import json
import re
import time
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"

class ValidationSeverity(Enum):
    """Validation error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field_name: Optional[str] = None
    suggested_fix: Optional[str] = None
    validation_time: float = 0.0

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    overall_valid: bool
    validation_level: ValidationLevel
    results: List[ValidationResult]
    total_checks: int
    passed_checks: int
    warnings: int
    errors: int
    critical_issues: int
    validation_time: float

class QuantumOptimizationValidator:
    """
    Advanced validator for quantum optimization parameters and configurations
    with security, performance, and correctness validation.
    """
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_security_checks: bool = True,
        enable_performance_checks: bool = True,
        custom_validators: Optional[Dict[str, Callable]] = None
    ):
        self.validation_level = validation_level
        self.enable_security_checks = enable_security_checks
        self.enable_performance_checks = enable_performance_checks
        self.custom_validators = custom_validators or {}
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'validation_times': [],
            'common_errors': {}
        }
    
    def validate_optimization_config(
        self,
        config: Dict[str, Any]
    ) -> ValidationReport:
        """
        Validate complete optimization configuration
        
        Args:
            config: Optimization configuration dictionary
            
        Returns:
            ValidationReport with detailed validation results
        """
        
        start_time = time.time()
        results = []
        
        try:
            logger.info(f"Starting validation with level: {self.validation_level.value}")
            
            # Core parameter validation
            results.extend(self._validate_core_parameters(config))
            
            # QUBO-specific validation
            if 'qubo_matrix' in config:
                results.extend(self._validate_qubo_matrix(config['qubo_matrix']))
            
            # Backend validation
            if 'backend_config' in config:
                results.extend(self._validate_backend_config(config['backend_config']))
            
            # Objective function validation
            if 'objective_function' in config:
                results.extend(self._validate_objective_function(config['objective_function']))
            
            # Parameter space validation
            if 'parameter_space' in config:
                results.extend(self._validate_parameter_space(config['parameter_space']))
            
            # Security validation
            if self.enable_security_checks:
                results.extend(self._validate_security_aspects(config))
            
            # Performance validation
            if self.enable_performance_checks:
                results.extend(self._validate_performance_aspects(config))
            
            # Enterprise-level validation
            if self.validation_level == ValidationLevel.ENTERPRISE:
                results.extend(self._validate_enterprise_requirements(config))
            
            # Custom validation
            for validator_name, validator_func in self.custom_validators.items():
                try:
                    custom_result = validator_func(config)
                    if isinstance(custom_result, ValidationResult):
                        results.append(custom_result)
                    elif isinstance(custom_result, list):
                        results.extend(custom_result)
                except Exception as e:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Custom validator '{validator_name}' failed: {str(e)}",
                        field_name="custom_validation"
                    ))
            
            # Generate report
            report = self._generate_validation_report(results, time.time() - start_time)
            
            # Update statistics
            self._update_validation_statistics(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            return ValidationReport(
                overall_valid=False,
                validation_level=self.validation_level,
                results=[ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validation system error: {str(e)}",
                    validation_time=time.time() - start_time
                )],
                total_checks=1,
                passed_checks=0,
                warnings=0,
                errors=0,
                critical_issues=1,
                validation_time=time.time() - start_time
            )
    
    def _validate_core_parameters(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate core optimization parameters"""
        
        results = []
        
        # Required parameters check
        required_params = ['max_iterations', 'convergence_threshold']
        for param in required_params:
            if param not in config:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required parameter '{param}' is missing",
                    field_name=param,
                    suggested_fix=f"Add '{param}' to configuration"
                ))
        
        # Type validation
        if 'max_iterations' in config:
            if not isinstance(config['max_iterations'], int) or config['max_iterations'] <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="max_iterations must be a positive integer",
                    field_name='max_iterations',
                    suggested_fix="Set max_iterations to a positive integer (e.g., 1000)"
                ))
        
        if 'convergence_threshold' in config:
            if not isinstance(config['convergence_threshold'], (int, float)) or config['convergence_threshold'] <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="convergence_threshold must be a positive number",
                    field_name='convergence_threshold',
                    suggested_fix="Set convergence_threshold to a small positive number (e.g., 1e-6)"
                ))
        
        # Range validation
        if 'temperature' in config:
            temp = config['temperature']
            if not isinstance(temp, (int, float)) or temp < 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Temperature must be non-negative",
                    field_name='temperature',
                    suggested_fix="Set temperature to a non-negative value"
                ))
            elif temp > 100:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message="Temperature is very high and may cause poor optimization",
                    field_name='temperature',
                    suggested_fix="Consider reducing temperature to 0.1-10.0 range"
                ))
        
        return results
    
    def _validate_qubo_matrix(self, qubo_matrix: Any) -> List[ValidationResult]:
        """Validate QUBO matrix properties"""
        
        results = []
        
        try:
            # Type check
            if not isinstance(qubo_matrix, np.ndarray):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="QUBO matrix must be a numpy array",
                    field_name='qubo_matrix',
                    suggested_fix="Convert to numpy array using np.array()"
                ))
                return results
            
            # Dimension check
            if len(qubo_matrix.shape) != 2:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="QUBO matrix must be 2-dimensional",
                    field_name='qubo_matrix'
                ))
                return results
            
            # Square matrix check
            if qubo_matrix.shape[0] != qubo_matrix.shape[1]:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="QUBO matrix must be square",
                    field_name='qubo_matrix'
                ))
            
            # Size validation
            if qubo_matrix.shape[0] > 1000:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message="Large QUBO matrix may cause performance issues",
                    field_name='qubo_matrix',
                    suggested_fix="Consider problem decomposition for matrices >1000x1000"
                ))
            
            # Numerical validation
            if np.any(np.isnan(qubo_matrix)):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="QUBO matrix contains NaN values",
                    field_name='qubo_matrix'
                ))
            
            if np.any(np.isinf(qubo_matrix)):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="QUBO matrix contains infinite values",
                    field_name='qubo_matrix'
                ))
            
            # Symmetry check (for strict validation)
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]:
                if not np.allclose(qubo_matrix, qubo_matrix.T, rtol=1e-10):
                    results.append(ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        message="QUBO matrix is not symmetric (may be intentional for upper triangular)",
                        field_name='qubo_matrix'
                    ))
            
            # Condition number check
            try:
                condition_number = np.linalg.cond(qubo_matrix)
                if condition_number > 1e12:
                    results.append(ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        message="QUBO matrix is poorly conditioned",
                        field_name='qubo_matrix',
                        suggested_fix="Consider regularization or problem reformulation"
                    ))
            except np.linalg.LinAlgError:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message="Could not compute QUBO matrix condition number",
                    field_name='qubo_matrix'
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"QUBO matrix validation failed: {str(e)}",
                field_name='qubo_matrix'
            ))
        
        return results
    
    def _validate_backend_config(self, backend_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate quantum backend configuration"""
        
        results = []
        
        # Backend type validation
        if 'backend_type' not in backend_config:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Backend type is required",
                field_name='backend_type',
                suggested_fix="Specify backend_type (e.g., 'simulator', 'dwave', 'qiskit')"
            ))
        else:
            valid_backends = ['simulator', 'dwave', 'qiskit', 'simple_simulator']
            if backend_config['backend_type'] not in valid_backends:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid backend type: {backend_config['backend_type']}",
                    field_name='backend_type',
                    suggested_fix=f"Use one of: {', '.join(valid_backends)}"
                ))
        
        # D-Wave specific validation
        if backend_config.get('backend_type') == 'dwave':
            if 'token' in backend_config:
                token = backend_config['token']
                if not isinstance(token, str) or len(token) < 20:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="D-Wave token appears invalid",
                        field_name='token',
                        suggested_fix="Provide valid D-Wave API token"
                    ))
            
            # Solver validation
            if 'solver_name' in backend_config:
                solver = backend_config['solver_name']
                if not isinstance(solver, str):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Solver name must be a string",
                        field_name='solver_name'
                    ))
        
        # Simulator specific validation
        if backend_config.get('backend_type') == 'simulator':
            if 'num_reads' in backend_config:
                num_reads = backend_config['num_reads']
                if not isinstance(num_reads, int) or num_reads <= 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="num_reads must be a positive integer",
                        field_name='num_reads'
                    ))
                elif num_reads > 10000:
                    results.append(ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        message="Very high num_reads may cause slow performance",
                        field_name='num_reads'
                    ))
        
        return results
    
    def _validate_objective_function(self, objective_function: Any) -> List[ValidationResult]:
        """Validate objective function"""
        
        results = []
        
        # Callable check
        if not callable(objective_function):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Objective function must be callable",
                field_name='objective_function'
            ))
            return results
        
        # Basic functionality test
        try:
            # Test with dummy parameters
            test_params = {'x': 1.0, 'y': 2.0}
            result = objective_function(test_params)
            
            if not isinstance(result, (int, float)):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Objective function must return a numeric value",
                    field_name='objective_function'
                ))
            
            if np.isnan(result) or np.isinf(result):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="Objective function returned NaN or infinite value in test",
                    field_name='objective_function'
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Could not test objective function: {str(e)}",
                field_name='objective_function',
                suggested_fix="Ensure objective function accepts dict parameters"
            ))
        
        return results
    
    def _validate_parameter_space(self, parameter_space: Dict[str, Any]) -> List[ValidationResult]:
        """Validate parameter search space"""
        
        results = []
        
        if not isinstance(parameter_space, dict):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Parameter space must be a dictionary",
                field_name='parameter_space'
            ))
            return results
        
        if not parameter_space:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Parameter space cannot be empty",
                field_name='parameter_space'
            ))
            return results
        
        # Validate each parameter
        for param_name, param_config in parameter_space.items():
            if not isinstance(param_name, str):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Parameter name must be string, got {type(param_name)}",
                    field_name='parameter_space'
                ))
                continue
            
            # Validate parameter bounds
            if isinstance(param_config, (tuple, list)) and len(param_config) == 2:
                lower, upper = param_config
                if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Parameter bounds for '{param_name}' must be numeric",
                        field_name='parameter_space'
                    ))
                elif lower >= upper:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid bounds for '{param_name}': lower >= upper",
                        field_name='parameter_space'
                    ))
            elif isinstance(param_config, dict):
                # Extended parameter configuration
                if 'bounds' in param_config:
                    bounds = param_config['bounds']
                    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"Bounds for '{param_name}' must be (lower, upper) tuple",
                            field_name='parameter_space'
                        ))
            else:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid parameter configuration for '{param_name}'",
                    field_name='parameter_space'
                ))
        
        return results
    
    def _validate_security_aspects(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate security-related aspects"""
        
        results = []
        
        # Check for sensitive data exposure
        sensitive_patterns = [
            r'password',
            r'secret',
            r'token',
            r'key',
            r'credential'
        ]
        
        config_str = json.dumps(config, default=str).lower()
        
        for pattern in sensitive_patterns:
            if re.search(pattern, config_str):
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"Configuration may contain sensitive data: {pattern}",
                    field_name='security',
                    suggested_fix="Use environment variables or secure storage for sensitive data"
                ))
        
        # Path traversal check
        for key, value in config.items():
            if isinstance(value, str) and ('..' in value or value.startswith('/')):
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"Potential path traversal in {key}: {value}",
                    field_name=key,
                    suggested_fix="Use relative paths and validate input"
                ))
        
        # Memory usage validation
        if 'max_memory_gb' in config:
            max_mem = config['max_memory_gb']
            if isinstance(max_mem, (int, float)) and max_mem > 32:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message="High memory limit may cause system issues",
                    field_name='max_memory_gb',
                    suggested_fix="Consider reducing memory limit for safety"
                ))
        
        return results
    
    def _validate_performance_aspects(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate performance-related aspects"""
        
        results = []
        
        # Check for performance bottlenecks
        if 'max_iterations' in config and config['max_iterations'] > 10000:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message="Very high max_iterations may cause long runtime",
                field_name='max_iterations',
                suggested_fix="Consider lower iteration count or early stopping"
            ))
        
        if 'num_parallel_workers' in config:
            num_workers = config['num_parallel_workers']
            import os
            cpu_count = os.cpu_count() or 1
            
            if num_workers > cpu_count * 2:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"num_parallel_workers ({num_workers}) exceeds 2x CPU count ({cpu_count})",
                    field_name='num_parallel_workers',
                    suggested_fix=f"Consider reducing to {cpu_count} workers"
                ))
        
        # Memory estimation
        if 'qubo_matrix' in config:
            matrix = config['qubo_matrix']
            if hasattr(matrix, 'shape'):
                n = matrix.shape[0]
                estimated_memory_mb = (n * n * 8) / (1024 * 1024)  # 8 bytes per float64
                
                if estimated_memory_mb > 1000:  # 1GB
                    results.append(ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        message=f"Large QUBO matrix may use ~{estimated_memory_mb:.1f}MB memory",
                        field_name='qubo_matrix',
                        suggested_fix="Monitor memory usage during optimization"
                    ))
        
        return results
    
    def _validate_enterprise_requirements(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate enterprise-level requirements"""
        
        results = []
        
        # Logging configuration
        if 'logging_config' not in config:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="No logging configuration found",
                field_name='logging_config',
                suggested_fix="Add structured logging for enterprise deployment"
            ))
        
        # Monitoring configuration
        if 'monitoring_enabled' not in config or not config['monitoring_enabled']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Monitoring not enabled",
                field_name='monitoring_enabled',
                suggested_fix="Enable monitoring for production deployment"
            ))
        
        # Backup configuration
        if 'backup_results' not in config or not config['backup_results']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Result backup not configured",
                field_name='backup_results',
                suggested_fix="Configure automatic result backup"
            ))
        
        # Error handling
        if 'error_handling_strategy' not in config:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="No error handling strategy specified",
                field_name='error_handling_strategy',
                suggested_fix="Define error handling and recovery strategy"
            ))
        
        return results
    
    def _generate_validation_report(
        self,
        results: List[ValidationResult],
        validation_time: float
    ) -> ValidationReport:
        """Generate comprehensive validation report"""
        
        # Count results by severity
        warnings = sum(1 for r in results if r.severity == ValidationSeverity.WARNING)
        errors = sum(1 for r in results if r.severity == ValidationSeverity.ERROR)
        critical = sum(1 for r in results if r.severity == ValidationSeverity.CRITICAL)
        
        # Overall validity
        overall_valid = critical == 0 and errors == 0
        
        # Passed checks
        passed = sum(1 for r in results if r.is_valid)
        
        return ValidationReport(
            overall_valid=overall_valid,
            validation_level=self.validation_level,
            results=results,
            total_checks=len(results),
            passed_checks=passed,
            warnings=warnings,
            errors=errors,
            critical_issues=critical,
            validation_time=validation_time
        )
    
    def _update_validation_statistics(self, report: ValidationReport):
        """Update internal validation statistics"""
        
        self.validation_stats['total_validations'] += 1
        
        if report.overall_valid:
            self.validation_stats['successful_validations'] += 1
        
        self.validation_stats['validation_times'].append(report.validation_time)
        
        # Track common errors
        for result in report.results:
            if not result.is_valid:
                error_key = result.field_name or 'unknown'
                self.validation_stats['common_errors'][error_key] = \
                    self.validation_stats['common_errors'].get(error_key, 0) + 1
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        
        stats = self.validation_stats.copy()
        
        if stats['validation_times']:
            stats['average_validation_time'] = np.mean(stats['validation_times'])
            stats['max_validation_time'] = max(stats['validation_times'])
        else:
            stats['average_validation_time'] = 0.0
            stats['max_validation_time'] = 0.0
        
        if stats['total_validations'] > 0:
            stats['success_rate'] = stats['successful_validations'] / stats['total_validations']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def print_validation_report(self, report: ValidationReport):
        """Print formatted validation report"""
        
        print(f"\n{'='*60}")
        print(f"QUANTUM OPTIMIZATION VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Validation Level: {report.validation_level.value}")
        print(f"Overall Valid: {'✓' if report.overall_valid else '✗'}")
        print(f"Validation Time: {report.validation_time:.3f}s")
        print(f"\nSummary:")
        print(f"  Total Checks: {report.total_checks}")
        print(f"  Passed: {report.passed_checks}")
        print(f"  Warnings: {report.warnings}")
        print(f"  Errors: {report.errors}")
        print(f"  Critical: {report.critical_issues}")
        
        if report.results:
            print(f"\nDetailed Results:")
            for i, result in enumerate(report.results, 1):
                severity_symbol = {
                    ValidationSeverity.INFO: "ℹ",
                    ValidationSeverity.WARNING: "⚠",
                    ValidationSeverity.ERROR: "✗",
                    ValidationSeverity.CRITICAL: "🚨"
                }.get(result.severity, "?")
                
                print(f"  {i}. {severity_symbol} {result.message}")
                if result.field_name:
                    print(f"     Field: {result.field_name}")
                if result.suggested_fix:
                    print(f"     Fix: {result.suggested_fix}")
        
        print(f"{'='*60}\n")