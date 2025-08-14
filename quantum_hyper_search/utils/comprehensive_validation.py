#!/usr/bin/env python3
"""
Comprehensive Validation Framework
Enterprise-grade input validation, data sanitization, and integrity checking.
"""

import re
import json
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationResult(Enum):
    """Validation result types."""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Validation issue details."""
    field: str
    message: str
    severity: ValidationResult
    suggested_fix: Optional[str] = None
    value: Optional[Any] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int
    errors_count: int
    critical_count: int
    validation_time: float
    
    def has_errors(self) -> bool:
        """Check if there are any errors or critical issues."""
        return self.errors_count > 0 or self.critical_count > 0
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid:
            status = "✅ VALID"
        else:
            status = "❌ INVALID"
        
        return f"{status} - Warnings: {self.warnings_count}, Errors: {self.errors_count}, Critical: {self.critical_count}"


class DataValidator:
    """
    Advanced Data Validation System
    
    Provides comprehensive validation for quantum optimization parameters,
    user inputs, and system configurations.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.custom_validators = {}
        self.validation_cache = {}
        self.validation_stats = defaultdict(int)
        
        # Common validation patterns
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'quantum_backend': re.compile(r'^(dwave|qiskit|cirq|simulated|neal)$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        }
        
        # Security validation rules
        self.security_rules = {
            'sql_injection': re.compile(r'(\bSELECT\b|\bINSERT\b|\bDELETE\b|\bDROP\b|\bUNION\b|\bEXEC\b)', re.IGNORECASE),
            'xss_script': re.compile(r'<script.*?>.*?</script>', re.IGNORECASE | re.DOTALL),
            'path_traversal': re.compile(r'\.\.\/|\.\.\\'),
            'command_injection': re.compile(r'[;&|`]')
        }
    
    def validate_quantum_parameters(self, params: Dict[str, Any]) -> ValidationReport:
        """Validate quantum optimization parameters."""
        
        start_time = time.time()
        issues = []
        
        # Validate QUBO matrix
        if 'Q' in params:
            Q = params['Q']
            issues.extend(self._validate_qubo_matrix(Q))
        
        # Validate quantum backend
        if 'backend' in params:
            backend = params['backend']
            issues.extend(self._validate_quantum_backend(backend))
        
        # Validate num_reads
        if 'num_reads' in params:
            num_reads = params['num_reads']
            issues.extend(self._validate_num_reads(num_reads))
        
        # Validate annealing parameters
        if 'annealing_time' in params:
            annealing_time = params['annealing_time']
            issues.extend(self._validate_annealing_time(annealing_time))
        
        # Validate parameter space
        if 'parameter_space' in params:
            param_space = params['parameter_space']
            issues.extend(self._validate_parameter_space(param_space))
        
        # Validate optimization constraints
        if 'constraints' in params:
            constraints = params['constraints']
            issues.extend(self._validate_constraints(constraints))
        
        validation_time = time.time() - start_time
        
        # Count issue types
        warnings = sum(1 for issue in issues if issue.severity == ValidationResult.WARNING)
        errors = sum(1 for issue in issues if issue.severity == ValidationResult.ERROR)
        critical = sum(1 for issue in issues if issue.severity == ValidationResult.CRITICAL)
        
        is_valid = errors == 0 and critical == 0
        
        # Update statistics
        self.validation_stats['quantum_parameters'] += 1
        if not is_valid:
            self.validation_stats['quantum_parameters_failed'] += 1
        
        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            warnings_count=warnings,
            errors_count=errors,
            critical_count=critical,
            validation_time=validation_time
        )
    
    def _validate_qubo_matrix(self, Q: Any) -> List[ValidationIssue]:
        """Validate QUBO matrix structure and values."""
        
        issues = []
        
        if not isinstance(Q, dict):
            issues.append(ValidationIssue(
                field="Q",
                message="QUBO matrix must be a dictionary",
                severity=ValidationResult.CRITICAL,
                suggested_fix="Provide Q as dict with (i,j) tuples as keys",
                value=type(Q).__name__
            ))
            return issues
        
        if len(Q) == 0:
            issues.append(ValidationIssue(
                field="Q",
                message="QUBO matrix is empty",
                severity=ValidationResult.ERROR,
                suggested_fix="Provide at least one QUBO coefficient"
            ))
            return issues
        
        # Validate keys
        for key in Q.keys():
            if not isinstance(key, tuple) or len(key) != 2:
                issues.append(ValidationIssue(
                    field="Q",
                    message=f"Invalid QUBO key: {key}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Keys must be (i,j) tuples",
                    value=key
                ))
                continue
            
            i, j = key
            if not isinstance(i, int) or not isinstance(j, int):
                issues.append(ValidationIssue(
                    field="Q",
                    message=f"QUBO indices must be integers: {key}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Use integer indices",
                    value=key
                ))
            
            if i < 0 or j < 0:
                issues.append(ValidationIssue(
                    field="Q",
                    message=f"QUBO indices must be non-negative: {key}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Use non-negative integer indices",
                    value=key
                ))
        
        # Validate values
        for key, value in Q.items():
            if not isinstance(value, (int, float, np.number)):
                issues.append(ValidationIssue(
                    field="Q",
                    message=f"QUBO coefficient must be numeric: {key}={value}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Use numeric values for coefficients",
                    value=value
                ))
                continue
            
            if np.isnan(value) or np.isinf(value):
                issues.append(ValidationIssue(
                    field="Q",
                    message=f"QUBO coefficient is NaN or infinite: {key}={value}",
                    severity=ValidationResult.CRITICAL,
                    suggested_fix="Use finite numeric values",
                    value=value
                ))
            
            # Check for extremely large values
            if abs(value) > 1e6:
                issues.append(ValidationIssue(
                    field="Q",
                    message=f"Very large QUBO coefficient: {key}={value}",
                    severity=ValidationResult.WARNING,
                    suggested_fix="Consider normalizing QUBO coefficients",
                    value=value
                ))
        
        # Validate matrix properties
        variables = set()
        for i, j in Q.keys():
            variables.add(i)
            variables.add(j)
        
        if len(variables) > 1000:
            issues.append(ValidationIssue(
                field="Q",
                message=f"Very large QUBO problem: {len(variables)} variables",
                severity=ValidationResult.WARNING,
                suggested_fix="Consider problem decomposition",
                value=len(variables)
            ))
        
        # Check matrix density
        max_edges = len(variables) * (len(variables) + 1) // 2
        density = len(Q) / max(max_edges, 1)
        
        if density > 0.5:
            issues.append(ValidationIssue(
                field="Q",
                message=f"Dense QUBO matrix (density: {density:.2f})",
                severity=ValidationResult.WARNING,
                suggested_fix="Consider sparsification techniques"
            ))
        
        return issues
    
    def _validate_quantum_backend(self, backend: str) -> List[ValidationIssue]:
        """Validate quantum backend specification."""
        
        issues = []
        
        if not isinstance(backend, str):
            issues.append(ValidationIssue(
                field="backend",
                message="Backend must be a string",
                severity=ValidationResult.ERROR,
                suggested_fix="Use string backend name",
                value=type(backend).__name__
            ))
            return issues
        
        valid_backends = ['dwave', 'qiskit', 'cirq', 'simulated', 'neal']
        
        if backend not in valid_backends:
            issues.append(ValidationIssue(
                field="backend",
                message=f"Unknown backend: {backend}",
                severity=ValidationResult.ERROR,
                suggested_fix=f"Use one of: {', '.join(valid_backends)}",
                value=backend
            ))
        
        # Backend-specific validation
        if backend == 'dwave':
            # Check if D-Wave libraries are available
            try:
                import dwave.system
            except ImportError:
                issues.append(ValidationIssue(
                    field="backend",
                    message="D-Wave libraries not available",
                    severity=ValidationResult.WARNING,
                    suggested_fix="Install dwave-ocean-sdk"
                ))
        
        return issues
    
    def _validate_num_reads(self, num_reads: Any) -> List[ValidationIssue]:
        """Validate number of reads parameter."""
        
        issues = []
        
        if not isinstance(num_reads, int):
            issues.append(ValidationIssue(
                field="num_reads",
                message="num_reads must be an integer",
                severity=ValidationResult.ERROR,
                suggested_fix="Use integer value",
                value=type(num_reads).__name__
            ))
            return issues
        
        if num_reads <= 0:
            issues.append(ValidationIssue(
                field="num_reads",
                message="num_reads must be positive",
                severity=ValidationResult.ERROR,
                suggested_fix="Use positive integer",
                value=num_reads
            ))
        
        if num_reads > 10000:
            issues.append(ValidationIssue(
                field="num_reads",
                message=f"Very large num_reads: {num_reads}",
                severity=ValidationResult.WARNING,
                suggested_fix="Consider reducing for faster execution",
                value=num_reads
            ))
        
        return issues
    
    def _validate_annealing_time(self, annealing_time: Any) -> List[ValidationIssue]:
        """Validate annealing time parameter."""
        
        issues = []
        
        if not isinstance(annealing_time, (int, float)):
            issues.append(ValidationIssue(
                field="annealing_time",
                message="annealing_time must be numeric",
                severity=ValidationResult.ERROR,
                suggested_fix="Use numeric value",
                value=type(annealing_time).__name__
            ))
            return issues
        
        if annealing_time <= 0:
            issues.append(ValidationIssue(
                field="annealing_time",
                message="annealing_time must be positive",
                severity=ValidationResult.ERROR,
                suggested_fix="Use positive value",
                value=annealing_time
            ))
        
        if annealing_time > 1000:
            issues.append(ValidationIssue(
                field="annealing_time",
                message=f"Very long annealing time: {annealing_time}",
                severity=ValidationResult.WARNING,
                suggested_fix="Consider shorter annealing time",
                value=annealing_time
            ))
        
        return issues
    
    def _validate_parameter_space(self, param_space: Any) -> List[ValidationIssue]:
        """Validate parameter space specification."""
        
        issues = []
        
        if not isinstance(param_space, dict):
            issues.append(ValidationIssue(
                field="parameter_space",
                message="Parameter space must be a dictionary",
                severity=ValidationResult.ERROR,
                suggested_fix="Use dict with parameter names as keys",
                value=type(param_space).__name__
            ))
            return issues
        
        if len(param_space) == 0:
            issues.append(ValidationIssue(
                field="parameter_space",
                message="Parameter space is empty",
                severity=ValidationResult.ERROR,
                suggested_fix="Define at least one parameter"
            ))
            return issues
        
        for param_name, param_values in param_space.items():
            if not isinstance(param_name, str):
                issues.append(ValidationIssue(
                    field="parameter_space",
                    message=f"Parameter name must be string: {param_name}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Use string parameter names",
                    value=param_name
                ))
                continue
            
            if not isinstance(param_values, list):
                issues.append(ValidationIssue(
                    field="parameter_space",
                    message=f"Parameter values must be list: {param_name}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Use list of possible values",
                    value=type(param_values).__name__
                ))
                continue
            
            if len(param_values) == 0:
                issues.append(ValidationIssue(
                    field="parameter_space",
                    message=f"Empty parameter values: {param_name}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Provide at least one value",
                    value=param_name
                ))
            
            if len(param_values) > 100:
                issues.append(ValidationIssue(
                    field="parameter_space",
                    message=f"Large parameter space: {param_name} has {len(param_values)} values",
                    severity=ValidationResult.WARNING,
                    suggested_fix="Consider reducing parameter space size"
                ))
        
        return issues
    
    def _validate_constraints(self, constraints: Any) -> List[ValidationIssue]:
        """Validate optimization constraints."""
        
        issues = []
        
        if not isinstance(constraints, dict):
            issues.append(ValidationIssue(
                field="constraints",
                message="Constraints must be a dictionary",
                severity=ValidationResult.ERROR,
                suggested_fix="Use dict with constraint names as keys",
                value=type(constraints).__name__
            ))
            return issues
        
        for constraint_name, constraint_value in constraints.items():
            if not isinstance(constraint_name, str):
                issues.append(ValidationIssue(
                    field="constraints",
                    message=f"Constraint name must be string: {constraint_name}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Use string constraint names",
                    value=constraint_name
                ))
            
            # Validate common constraint types
            if constraint_name in ['max_time', 'max_iterations', 'tolerance']:
                if not isinstance(constraint_value, (int, float)):
                    issues.append(ValidationIssue(
                        field="constraints",
                        message=f"{constraint_name} must be numeric",
                        severity=ValidationResult.ERROR,
                        suggested_fix="Use numeric value",
                        value=constraint_value
                    ))
                elif constraint_value <= 0:
                    issues.append(ValidationIssue(
                        field="constraints",
                        message=f"{constraint_name} must be positive",
                        severity=ValidationResult.ERROR,
                        suggested_fix="Use positive value",
                        value=constraint_value
                    ))
        
        return issues
    
    def validate_user_input(self, data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationReport:
        """Validate user input against a schema."""
        
        start_time = time.time()
        issues = []
        
        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in data:
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Required field missing: {field}",
                    severity=ValidationResult.ERROR,
                    suggested_fix=f"Provide value for {field}"
                ))
        
        # Validate each field
        field_schemas = schema.get('fields', {})
        for field_name, field_schema in field_schemas.items():
            if field_name in data:
                field_issues = self._validate_field(field_name, data[field_name], field_schema)
                issues.extend(field_issues)
        
        # Check for unexpected fields
        if schema.get('strict', False):
            expected_fields = set(field_schemas.keys())
            actual_fields = set(data.keys())
            unexpected_fields = actual_fields - expected_fields
            
            for field in unexpected_fields:
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Unexpected field: {field}",
                    severity=ValidationResult.WARNING,
                    suggested_fix="Remove unexpected field or update schema"
                ))
        
        validation_time = time.time() - start_time
        
        # Count issue types
        warnings = sum(1 for issue in issues if issue.severity == ValidationResult.WARNING)
        errors = sum(1 for issue in issues if issue.severity == ValidationResult.ERROR)
        critical = sum(1 for issue in issues if issue.severity == ValidationResult.CRITICAL)
        
        is_valid = errors == 0 and critical == 0
        
        # Update statistics
        self.validation_stats['user_input'] += 1
        if not is_valid:
            self.validation_stats['user_input_failed'] += 1
        
        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            warnings_count=warnings,
            errors_count=errors,
            critical_count=critical,
            validation_time=validation_time
        )
    
    def _validate_field(self, field_name: str, value: Any, schema: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate a single field against its schema."""
        
        issues = []
        
        # Type validation
        expected_type = schema.get('type')
        if expected_type:
            if expected_type == 'string' and not isinstance(value, str):
                issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Expected string, got {type(value).__name__}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Provide string value",
                    value=value
                ))
                return issues
            elif expected_type == 'integer' and not isinstance(value, int):
                issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Expected integer, got {type(value).__name__}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Provide integer value",
                    value=value
                ))
                return issues
            elif expected_type == 'number' and not isinstance(value, (int, float)):
                issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Expected number, got {type(value).__name__}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Provide numeric value",
                    value=value
                ))
                return issues
            elif expected_type == 'boolean' and not isinstance(value, bool):
                issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Expected boolean, got {type(value).__name__}",
                    severity=ValidationResult.ERROR,
                    suggested_fix="Provide boolean value",
                    value=value
                ))
                return issues
        
        # Pattern validation
        if isinstance(value, str):
            pattern = schema.get('pattern')
            if pattern and pattern in self.patterns:
                if not self.patterns[pattern].match(value):
                    issues.append(ValidationIssue(
                        field=field_name,
                        message=f"Value doesn't match pattern {pattern}",
                        severity=ValidationResult.ERROR,
                        suggested_fix=f"Provide value matching {pattern} pattern",
                        value=value
                    ))
            
            # Security validation
            issues.extend(self._validate_security(field_name, value))
            
            # Length validation
            min_length = schema.get('min_length')
            max_length = schema.get('max_length')
            
            if min_length and len(value) < min_length:
                issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Value too short: {len(value)} < {min_length}",
                    severity=ValidationResult.ERROR,
                    suggested_fix=f"Provide value with at least {min_length} characters",
                    value=len(value)
                ))
            
            if max_length and len(value) > max_length:
                issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Value too long: {len(value)} > {max_length}",
                    severity=ValidationResult.ERROR,
                    suggested_fix=f"Provide value with at most {max_length} characters",
                    value=len(value)
                ))
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            min_val = schema.get('min')
            max_val = schema.get('max')
            
            if min_val is not None and value < min_val:
                issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Value below minimum: {value} < {min_val}",
                    severity=ValidationResult.ERROR,
                    suggested_fix=f"Provide value >= {min_val}",
                    value=value
                ))
            
            if max_val is not None and value > max_val:
                issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Value above maximum: {value} > {max_val}",
                    severity=ValidationResult.ERROR,
                    suggested_fix=f"Provide value <= {max_val}",
                    value=value
                ))
        
        # Choice validation
        choices = schema.get('choices')
        if choices and value not in choices:
            issues.append(ValidationIssue(
                field=field_name,
                message=f"Invalid choice: {value}",
                severity=ValidationResult.ERROR,
                suggested_fix=f"Choose from: {', '.join(map(str, choices))}",
                value=value
            ))
        
        return issues
    
    def _validate_security(self, field_name: str, value: str) -> List[ValidationIssue]:
        """Validate for security vulnerabilities."""
        
        issues = []
        
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            # Check for SQL injection
            if self.security_rules['sql_injection'].search(value):
                issues.append(ValidationIssue(
                    field=field_name,
                    message="Potential SQL injection detected",
                    severity=ValidationResult.CRITICAL,
                    suggested_fix="Remove SQL keywords",
                    value=value
                ))
            
            # Check for XSS
            if self.security_rules['xss_script'].search(value):
                issues.append(ValidationIssue(
                    field=field_name,
                    message="Potential XSS script detected",
                    severity=ValidationResult.CRITICAL,
                    suggested_fix="Remove script tags",
                    value=value
                ))
            
            # Check for path traversal
            if self.security_rules['path_traversal'].search(value):
                issues.append(ValidationIssue(
                    field=field_name,
                    message="Potential path traversal detected",
                    severity=ValidationResult.CRITICAL,
                    suggested_fix="Remove path traversal sequences",
                    value=value
                ))
            
            # Check for command injection
            if self.security_rules['command_injection'].search(value):
                issues.append(ValidationIssue(
                    field=field_name,
                    message="Potential command injection detected",
                    severity=ValidationResult.CRITICAL,
                    suggested_fix="Remove shell metacharacters",
                    value=value
                ))
        
        return issues
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data to remove potentially harmful content."""
        
        if isinstance(data, str):
            # Remove null bytes
            data = data.replace('\x00', '')
            
            # HTML escape if needed
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                data = data.replace('<', '&lt;').replace('>', '&gt;')
                data = data.replace('"', '&quot;').replace("'", '&#x27;')
            
            # Remove excessive whitespace
            data = ' '.join(data.split())
            
            return data
        
        elif isinstance(data, dict):
            return {key: self.sanitize_input(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        
        else:
            return data
    
    def validate_data_integrity(self, data: Any, expected_hash: Optional[str] = None) -> ValidationReport:
        """Validate data integrity using checksums."""
        
        start_time = time.time()
        issues = []
        
        # Calculate data hash
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
        
        calculated_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Check against expected hash
        if expected_hash:
            if calculated_hash != expected_hash:
                issues.append(ValidationIssue(
                    field="data_integrity",
                    message="Data integrity check failed",
                    severity=ValidationResult.CRITICAL,
                    suggested_fix="Data may have been corrupted or tampered with",
                    value=f"Expected: {expected_hash}, Got: {calculated_hash}"
                ))
        
        validation_time = time.time() - start_time
        
        # Count issue types
        warnings = sum(1 for issue in issues if issue.severity == ValidationResult.WARNING)
        errors = sum(1 for issue in issues if issue.severity == ValidationResult.ERROR)
        critical = sum(1 for issue in issues if issue.severity == ValidationResult.CRITICAL)
        
        is_valid = errors == 0 and critical == 0
        
        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            warnings_count=warnings,
            errors_count=errors,
            critical_count=critical,
            validation_time=validation_time
        )
    
    def register_custom_validator(self, name: str, validator_func: Callable[[Any], List[ValidationIssue]]):
        """Register a custom validator function."""
        
        self.custom_validators[name] = validator_func
        logger.info(f"Registered custom validator: {name}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        
        total_validations = sum(self.validation_stats.values())
        
        stats = {
            'total_validations': total_validations,
            'validation_breakdown': dict(self.validation_stats),
            'cache_size': len(self.validation_cache),
            'custom_validators': len(self.custom_validators)
        }
        
        if total_validations > 0:
            failure_rate = (
                self.validation_stats.get('quantum_parameters_failed', 0) +
                self.validation_stats.get('user_input_failed', 0)
            ) / total_validations
            
            stats['failure_rate'] = failure_rate
        
        return stats


# Global validator instance
_global_validator = None


def get_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> DataValidator:
    """Get the global validator instance."""
    global _global_validator
    
    if _global_validator is None:
        _global_validator = DataValidator(validation_level)
    
    return _global_validator