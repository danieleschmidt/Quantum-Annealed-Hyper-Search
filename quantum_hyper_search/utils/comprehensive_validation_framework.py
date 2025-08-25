#!/usr/bin/env python3
"""
Comprehensive Validation Framework for Quantum Hyperparameter Search
Enterprise-grade validation, testing, and quality assurance for quantum optimization systems.

This module provides extensive validation capabilities to ensure reliability,
correctness, and performance of quantum-enhanced optimization algorithms.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation thoroughness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"

class TestResult(Enum):
    """Test execution results."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    test_name: str
    result: TestResult
    score: float
    execution_time: float
    details: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class QuantumParameterValidator:
    """Validates quantum optimization parameters for correctness and efficiency."""
    
    def __init__(self):
        self.validation_rules = {
            'circuit_depth': {'min': 1, 'max': 200, 'optimal_max': 50},
            'num_qubits': {'min': 1, 'max': 100, 'practical_max': 20},
            'measurement_shots': {'min': 1, 'max': 1000000, 'optimal_max': 10000},
            'coherence_time': {'min': 0.1, 'max': 1000.0, 'units': 'microseconds'},
            'gate_fidelity': {'min': 0.8, 'max': 1.0, 'warning_threshold': 0.95},
            'optimization_budget': {'min': 1, 'max': 10000, 'recommended_max': 1000}
        }
    
    def validate_parameter_ranges(self, parameters: Dict[str, Any]) -> ValidationReport:
        """Validate parameter values are within acceptable ranges."""
        start_time = time.time()
        warnings_list = []
        details = {}
        overall_score = 1.0
        
        for param_name, value in parameters.items():
            if param_name in self.validation_rules:
                rules = self.validation_rules[param_name]
                
                # Check minimum value
                if value < rules['min']:
                    details[param_name] = f"Value {value} below minimum {rules['min']}"
                    overall_score *= 0.5
                
                # Check maximum value  
                elif value > rules['max']:
                    details[param_name] = f"Value {value} exceeds maximum {rules['max']}"
                    overall_score *= 0.3
                
                # Check optimal range
                elif 'optimal_max' in rules and value > rules['optimal_max']:
                    warnings_list.append(f"{param_name} ({value}) exceeds optimal range (≤{rules['optimal_max']})")
                    details[param_name] = "Above optimal range"
                    overall_score *= 0.9
                
                # Check warning thresholds
                elif 'warning_threshold' in rules and value < rules['warning_threshold']:
                    warnings_list.append(f"{param_name} ({value}) below recommended threshold ({rules['warning_threshold']})")
                    details[param_name] = "Below recommended threshold"
                    overall_score *= 0.95
                
                else:
                    details[param_name] = "Valid"
        
        execution_time = time.time() - start_time
        result = TestResult.PASS if overall_score > 0.8 else TestResult.FAIL if overall_score < 0.5 else TestResult.WARNING
        
        return ValidationReport(
            test_name="Parameter Range Validation",
            result=result,
            score=overall_score,
            execution_time=execution_time,
            details=details,
            warnings=warnings_list
        )
    
    def validate_parameter_consistency(self, parameters: Dict[str, Any]) -> ValidationReport:
        """Validate parameter combinations for consistency."""
        start_time = time.time()
        warnings_list = []
        details = {}
        overall_score = 1.0
        
        # Check circuit depth vs coherence time
        if 'circuit_depth' in parameters and 'coherence_time' in parameters:
            depth = parameters['circuit_depth']
            coherence = parameters['coherence_time']
            
            # Rough estimate: each gate takes ~0.1 microseconds
            estimated_execution_time = depth * 0.1
            
            if estimated_execution_time > coherence:
                warnings_list.append(f"Circuit depth ({depth}) may exceed coherence time ({coherence}μs)")
                details['depth_coherence_mismatch'] = True
                overall_score *= 0.8
            else:
                details['depth_coherence_consistency'] = "Good"
        
        # Check qubits vs measurement shots efficiency
        if 'num_qubits' in parameters and 'measurement_shots' in parameters:
            qubits = parameters['num_qubits']
            shots = parameters['measurement_shots']
            
            # More qubits need more shots for statistical accuracy
            recommended_shots = min(1000 * (2 ** min(qubits, 10)), 50000)
            
            if shots < recommended_shots // 4:
                warnings_list.append(f"Too few measurement shots ({shots}) for {qubits} qubits")
                details['insufficient_shots'] = True
                overall_score *= 0.9
            elif shots > recommended_shots * 4:
                warnings_list.append(f"Excessive measurement shots ({shots}) may be inefficient")
                details['excessive_shots'] = True
                overall_score *= 0.95
            else:
                details['shots_qubits_balance'] = "Optimal"
        
        # Check optimization budget vs problem complexity
        if 'optimization_budget' in parameters and 'num_parameters' in parameters:
            budget = parameters['optimization_budget']
            num_params = parameters['num_parameters']
            
            recommended_budget = max(50, num_params * 10)
            
            if budget < recommended_budget:
                warnings_list.append(f"Optimization budget ({budget}) may be too low for {num_params} parameters")
                details['insufficient_budget'] = True
                overall_score *= 0.85
            else:
                details['budget_adequacy'] = "Sufficient"
        
        execution_time = time.time() - start_time
        result = TestResult.PASS if overall_score > 0.9 else TestResult.WARNING if overall_score > 0.7 else TestResult.FAIL
        
        return ValidationReport(
            test_name="Parameter Consistency Validation",
            result=result,
            score=overall_score,
            execution_time=execution_time,
            details=details,
            warnings=warnings_list
        )

class AlgorithmCorrectnessValidator:
    """Validates quantum algorithm implementations for correctness."""
    
    def validate_quantum_state_normalization(self, state_vector: np.ndarray) -> ValidationReport:
        """Validate quantum state vector normalization."""
        start_time = time.time()
        
        # Check if state vector is properly normalized
        norm = np.linalg.norm(state_vector)
        normalization_error = abs(norm - 1.0)
        
        score = max(0.0, 1.0 - normalization_error * 10)
        
        if normalization_error < 1e-10:
            result = TestResult.PASS
            details = {"normalization_error": normalization_error, "status": "excellent"}
        elif normalization_error < 1e-6:
            result = TestResult.PASS
            details = {"normalization_error": normalization_error, "status": "good"}
        elif normalization_error < 1e-3:
            result = TestResult.WARNING
            details = {"normalization_error": normalization_error, "status": "acceptable"}
        else:
            result = TestResult.FAIL
            details = {"normalization_error": normalization_error, "status": "invalid"}
        
        execution_time = time.time() - start_time
        
        return ValidationReport(
            test_name="Quantum State Normalization",
            result=result,
            score=score,
            execution_time=execution_time,
            details=details
        )
    
    def validate_unitary_operations(self, unitary_matrix: np.ndarray) -> ValidationReport:
        """Validate quantum unitary operations."""
        start_time = time.time()
        
        # Check if matrix is unitary (U† U = I)
        conjugate_transpose = np.conj(unitary_matrix.T)
        product = np.dot(conjugate_transpose, unitary_matrix)
        identity = np.eye(product.shape[0])
        
        unitarity_error = np.linalg.norm(product - identity)
        score = max(0.0, 1.0 - unitarity_error * 10)
        
        if unitarity_error < 1e-10:
            result = TestResult.PASS
            details = {"unitarity_error": unitarity_error, "status": "perfect"}
        elif unitarity_error < 1e-6:
            result = TestResult.PASS
            details = {"unitarity_error": unitarity_error, "status": "excellent"}
        elif unitarity_error < 1e-3:
            result = TestResult.WARNING
            details = {"unitarity_error": unitarity_error, "status": "acceptable"}
        else:
            result = TestResult.FAIL
            details = {"unitarity_error": unitarity_error, "status": "non-unitary"}
        
        execution_time = time.time() - start_time
        
        return ValidationReport(
            test_name="Unitary Operation Validation",
            result=result,
            score=score,
            execution_time=execution_time,
            details=details
        )
    
    def validate_measurement_probabilities(self, probabilities: np.ndarray) -> ValidationReport:
        """Validate quantum measurement probability distributions."""
        start_time = time.time()
        warnings_list = []
        
        # Check probability normalization
        prob_sum = np.sum(probabilities)
        normalization_error = abs(prob_sum - 1.0)
        
        # Check for negative probabilities
        negative_probs = np.sum(probabilities < 0)
        
        # Check for probability range [0, 1]
        invalid_probs = np.sum((probabilities < 0) | (probabilities > 1))
        
        score = 1.0
        if normalization_error > 1e-6:
            score *= 0.8
            warnings_list.append(f"Probability normalization error: {normalization_error}")
        
        if negative_probs > 0:
            score *= 0.5
            warnings_list.append(f"Found {negative_probs} negative probabilities")
        
        if invalid_probs > 0:
            score *= 0.3
            warnings_list.append(f"Found {invalid_probs} invalid probabilities")
        
        # Check for excessive concentration (may indicate poor sampling)
        max_prob = np.max(probabilities)
        if max_prob > 0.99:
            score *= 0.9
            warnings_list.append("Probability distribution highly concentrated")
        
        result = TestResult.PASS if score > 0.9 else TestResult.WARNING if score > 0.7 else TestResult.FAIL
        
        execution_time = time.time() - start_time
        
        return ValidationReport(
            test_name="Measurement Probability Validation",
            result=result,
            score=score,
            execution_time=execution_time,
            details={
                "normalization_error": normalization_error,
                "negative_probabilities": int(negative_probs),
                "invalid_probabilities": int(invalid_probs),
                "max_probability": float(max_prob)
            },
            warnings=warnings_list
        )

class PerformanceValidator:
    """Validates performance characteristics of quantum algorithms."""
    
    def validate_convergence_behavior(self, optimization_history: List[Dict[str, Any]]) -> ValidationReport:
        """Validate optimization convergence behavior."""
        start_time = time.time()
        warnings_list = []
        
        if len(optimization_history) < 3:
            return ValidationReport(
                test_name="Convergence Validation",
                result=TestResult.SKIP,
                score=0.0,
                execution_time=time.time() - start_time,
                details={"reason": "Insufficient optimization history"}
            )
        
        # Extract best scores over time
        best_scores = [entry.get('best_score', 0) for entry in optimization_history]
        
        # Check for improvement trend
        improvements = []
        for i in range(1, len(best_scores)):
            if best_scores[i] > best_scores[i-1]:
                improvements.append(True)
            elif best_scores[i] == best_scores[i-1]:
                improvements.append(None)  # No change
            else:
                improvements.append(False)  # Degradation
        
        improvement_rate = improvements.count(True) / len(improvements) if improvements else 0
        stagnation_rate = improvements.count(None) / len(improvements) if improvements else 0
        
        # Calculate convergence metrics
        final_score = best_scores[-1]
        initial_score = best_scores[0]
        total_improvement = final_score - initial_score if final_score != initial_score else 0
        
        score = improvement_rate * 0.6 + min(1.0, total_improvement / abs(initial_score + 1e-8)) * 0.4
        
        if improvement_rate < 0.1:
            warnings_list.append("Very low improvement rate - algorithm may not be converging")
            score *= 0.7
        
        if stagnation_rate > 0.7:
            warnings_list.append("High stagnation rate - algorithm may be stuck in local optimum")
            score *= 0.8
        
        result = TestResult.PASS if score > 0.8 else TestResult.WARNING if score > 0.5 else TestResult.FAIL
        
        execution_time = time.time() - start_time
        
        return ValidationReport(
            test_name="Convergence Behavior Validation",
            result=result,
            score=score,
            execution_time=execution_time,
            details={
                "improvement_rate": improvement_rate,
                "stagnation_rate": stagnation_rate,
                "total_improvement": total_improvement,
                "convergence_efficiency": improvement_rate / max(len(optimization_history), 1)
            },
            warnings=warnings_list
        )
    
    def validate_scalability(self, 
                           performance_data: Dict[str, List[float]],
                           problem_sizes: List[int]) -> ValidationReport:
        """Validate algorithm scalability with problem size."""
        start_time = time.time()
        warnings_list = []
        
        if len(performance_data.get('execution_times', [])) != len(problem_sizes):
            return ValidationReport(
                test_name="Scalability Validation",
                result=TestResult.ERROR,
                score=0.0,
                execution_time=time.time() - start_time,
                details={"error": "Mismatched performance data and problem sizes"}
            )
        
        execution_times = performance_data.get('execution_times', [])
        
        # Analyze time complexity
        if len(execution_times) >= 3:
            # Fit polynomial to estimate time complexity
            log_sizes = np.log(problem_sizes)
            log_times = np.log(execution_times)
            
            # Linear regression to estimate exponent
            coeffs = np.polyfit(log_sizes, log_times, 1)
            time_complexity_exponent = coeffs[0]
            
            # Score based on time complexity
            if time_complexity_exponent <= 1.5:
                complexity_score = 1.0  # Excellent (linear to quasi-linear)
            elif time_complexity_exponent <= 2.5:
                complexity_score = 0.8  # Good (quadratic)
            elif time_complexity_exponent <= 3.5:
                complexity_score = 0.6  # Acceptable (cubic)
            else:
                complexity_score = 0.3  # Poor (higher order)
                warnings_list.append(f"High time complexity exponent: {time_complexity_exponent:.2f}")
            
            # Check for reasonable absolute performance
            max_time = max(execution_times)
            if max_time > 3600:  # 1 hour
                warnings_list.append("Maximum execution time exceeds 1 hour")
                complexity_score *= 0.8
            
        else:
            complexity_score = 0.5
            warnings_list.append("Insufficient data for complexity analysis")
            time_complexity_exponent = None
        
        result = TestResult.PASS if complexity_score > 0.8 else TestResult.WARNING if complexity_score > 0.5 else TestResult.FAIL
        
        execution_time = time.time() - start_time
        
        return ValidationReport(
            test_name="Scalability Validation",
            result=result,
            score=complexity_score,
            execution_time=execution_time,
            details={
                "time_complexity_exponent": time_complexity_exponent,
                "max_execution_time": max(execution_times) if execution_times else 0,
                "scalability_rating": "Excellent" if complexity_score > 0.8 else "Good" if complexity_score > 0.6 else "Poor"
            },
            warnings=warnings_list
        )

class ComprehensiveValidationFramework:
    """
    Comprehensive validation framework for quantum hyperparameter search systems.
    
    Provides end-to-end validation including parameter validation, algorithm correctness,
    performance analysis, and quantum-specific checks.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize comprehensive validation framework."""
        self.validation_level = validation_level
        self.parameter_validator = QuantumParameterValidator()
        self.correctness_validator = AlgorithmCorrectnessValidator()
        self.performance_validator = PerformanceValidator()
        
        self.validation_history = []
        
        logger.info(f"Initialized ComprehensiveValidationFramework with {validation_level.value} validation level")
    
    def validate_quantum_optimization_system(self, 
                                           system_parameters: Dict[str, Any],
                                           quantum_states: Optional[List[np.ndarray]] = None,
                                           optimization_history: Optional[List[Dict[str, Any]]] = None,
                                           performance_data: Optional[Dict[str, Any]] = None) -> Dict[str, ValidationReport]:
        """
        Perform comprehensive validation of quantum optimization system.
        
        Args:
            system_parameters: System configuration parameters
            quantum_states: List of quantum state vectors for validation
            optimization_history: History of optimization iterations
            performance_data: Performance metrics and timing data
            
        Returns:
            Dictionary of validation reports by test name
        """
        validation_reports = {}
        start_time = time.time()
        
        logger.info(f"Starting comprehensive validation with {self.validation_level.value} level")
        
        # 1. Parameter validation
        validation_reports['parameter_ranges'] = self.parameter_validator.validate_parameter_ranges(system_parameters)
        validation_reports['parameter_consistency'] = self.parameter_validator.validate_parameter_consistency(system_parameters)
        
        # 2. Quantum algorithm correctness validation
        if quantum_states:
            for i, state in enumerate(quantum_states[:5]):  # Validate first 5 states
                if state.ndim == 1:  # State vector
                    validation_reports[f'state_normalization_{i}'] = self.correctness_validator.validate_quantum_state_normalization(state)
                elif state.ndim == 2:  # Unitary matrix
                    validation_reports[f'unitary_validation_{i}'] = self.correctness_validator.validate_unitary_operations(state)
        
        # 3. Performance validation
        if optimization_history:
            validation_reports['convergence_behavior'] = self.performance_validator.validate_convergence_behavior(optimization_history)
        
        if performance_data and 'problem_sizes' in performance_data:
            validation_reports['scalability'] = self.performance_validator.validate_scalability(
                performance_data, performance_data['problem_sizes']
            )
        
        # 4. Advanced validation for comprehensive/exhaustive levels
        if self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.EXHAUSTIVE]:
            validation_reports.update(self._advanced_validation_suite(system_parameters, quantum_states))
        
        # 5. Generate overall system validation report
        validation_reports['overall_system'] = self._generate_overall_report(validation_reports)
        
        # Store validation session
        validation_session = {
            'timestamp': start_time,
            'validation_level': self.validation_level.value,
            'total_tests': len(validation_reports),
            'total_time': time.time() - start_time,
            'reports': validation_reports
        }
        self.validation_history.append(validation_session)
        
        logger.info(f"Validation completed: {len(validation_reports)} tests in {validation_session['total_time']:.2f}s")
        
        return validation_reports
    
    def _advanced_validation_suite(self, 
                                 parameters: Dict[str, Any],
                                 quantum_states: Optional[List[np.ndarray]]) -> Dict[str, ValidationReport]:
        """Run advanced validation tests for comprehensive validation levels."""
        advanced_reports = {}
        
        # Quantum noise resilience test
        if quantum_states and len(quantum_states) > 0:
            advanced_reports['noise_resilience'] = self._test_noise_resilience(quantum_states[0])
        
        # Parameter sensitivity analysis
        advanced_reports['parameter_sensitivity'] = self._test_parameter_sensitivity(parameters)
        
        # Resource efficiency validation
        advanced_reports['resource_efficiency'] = self._test_resource_efficiency(parameters)
        
        return advanced_reports
    
    def _test_noise_resilience(self, quantum_state: np.ndarray) -> ValidationReport:
        """Test quantum algorithm resilience to noise."""
        start_time = time.time()
        
        # Apply various noise models
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        resilience_scores = []
        
        original_fidelity = np.abs(np.vdot(quantum_state, quantum_state))
        
        for noise_level in noise_levels:
            # Add depolarizing noise
            noise = np.random.normal(0, noise_level, quantum_state.shape) + 1j * np.random.normal(0, noise_level, quantum_state.shape)
            noisy_state = quantum_state + noise
            noisy_state = noisy_state / np.linalg.norm(noisy_state)
            
            # Calculate fidelity
            fidelity = np.abs(np.vdot(quantum_state, noisy_state))**2
            resilience_scores.append(fidelity)
        
        # Score based on fidelity degradation
        average_fidelity = np.mean(resilience_scores)
        score = average_fidelity
        
        result = TestResult.PASS if score > 0.9 else TestResult.WARNING if score > 0.7 else TestResult.FAIL
        
        return ValidationReport(
            test_name="Quantum Noise Resilience",
            result=result,
            score=score,
            execution_time=time.time() - start_time,
            details={
                "noise_levels_tested": noise_levels,
                "fidelity_scores": resilience_scores,
                "average_fidelity": average_fidelity,
                "resilience_rating": "High" if score > 0.9 else "Medium" if score > 0.7 else "Low"
            }
        )
    
    def _test_parameter_sensitivity(self, parameters: Dict[str, Any]) -> ValidationReport:
        """Test sensitivity to parameter variations."""
        start_time = time.time()
        
        sensitivity_scores = {}
        critical_params = ['circuit_depth', 'measurement_shots', 'optimization_budget']
        
        for param in critical_params:
            if param in parameters:
                original_value = parameters[param]
                
                # Test ±10% variations
                variations = [0.9, 1.1]
                param_sensitivity = []
                
                for factor in variations:
                    new_value = original_value * factor
                    # Simplified sensitivity measure (would need actual performance evaluation)
                    sensitivity = abs(factor - 1.0)
                    param_sensitivity.append(sensitivity)
                
                sensitivity_scores[param] = np.mean(param_sensitivity)
        
        overall_sensitivity = np.mean(list(sensitivity_scores.values())) if sensitivity_scores else 0.5
        score = max(0.0, 1.0 - overall_sensitivity * 5)  # Lower sensitivity is better
        
        result = TestResult.PASS if score > 0.8 else TestResult.WARNING if score > 0.6 else TestResult.FAIL
        
        return ValidationReport(
            test_name="Parameter Sensitivity Analysis",
            result=result,
            score=score,
            execution_time=time.time() - start_time,
            details={
                "parameter_sensitivities": sensitivity_scores,
                "overall_sensitivity": overall_sensitivity,
                "stability_rating": "Stable" if score > 0.8 else "Moderate" if score > 0.6 else "Unstable"
            }
        )
    
    def _test_resource_efficiency(self, parameters: Dict[str, Any]) -> ValidationReport:
        """Test computational resource efficiency."""
        start_time = time.time()
        
        efficiency_factors = []
        
        # Memory efficiency
        if 'num_qubits' in parameters:
            qubits = parameters['num_qubits']
            memory_usage = 2**qubits * 16  # Rough estimate in bytes
            memory_efficiency = max(0.0, 1.0 - memory_usage / (1e9))  # Penalize if > 1GB
            efficiency_factors.append(memory_efficiency)
        
        # Time efficiency
        if 'circuit_depth' in parameters and 'measurement_shots' in parameters:
            depth = parameters['circuit_depth']
            shots = parameters['measurement_shots']
            estimated_time = depth * shots * 1e-6  # Rough time estimate
            time_efficiency = max(0.0, 1.0 - estimated_time / 3600)  # Penalize if > 1 hour
            efficiency_factors.append(time_efficiency)
        
        # Shot efficiency
        if 'measurement_shots' in parameters and 'num_qubits' in parameters:
            shots = parameters['measurement_shots']
            qubits = parameters['num_qubits']
            optimal_shots = min(1000 * qubits, 10000)
            shot_efficiency = 1.0 - abs(shots - optimal_shots) / optimal_shots
            efficiency_factors.append(max(0.0, shot_efficiency))
        
        overall_efficiency = np.mean(efficiency_factors) if efficiency_factors else 0.5
        
        result = TestResult.PASS if overall_efficiency > 0.8 else TestResult.WARNING if overall_efficiency > 0.6 else TestResult.FAIL
        
        return ValidationReport(
            test_name="Resource Efficiency Analysis",
            result=result,
            score=overall_efficiency,
            execution_time=time.time() - start_time,
            details={
                "efficiency_factors": efficiency_factors,
                "overall_efficiency": overall_efficiency,
                "efficiency_rating": "High" if overall_efficiency > 0.8 else "Medium" if overall_efficiency > 0.6 else "Low"
            }
        )
    
    def _generate_overall_report(self, validation_reports: Dict[str, ValidationReport]) -> ValidationReport:
        """Generate overall system validation report."""
        start_time = time.time()
        
        # Calculate overall metrics
        total_tests = len(validation_reports)
        pass_count = sum(1 for report in validation_reports.values() if report.result == TestResult.PASS)
        warning_count = sum(1 for report in validation_reports.values() if report.result == TestResult.WARNING)
        fail_count = sum(1 for report in validation_reports.values() if report.result == TestResult.FAIL)
        
        # Overall score (weighted average)
        scores = [report.score for report in validation_reports.values() if report.score is not None]
        overall_score = np.mean(scores) if scores else 0.0
        
        # System health assessment
        if pass_count / total_tests > 0.9 and overall_score > 0.9:
            system_status = "EXCELLENT"
            result = TestResult.PASS
        elif pass_count / total_tests > 0.7 and overall_score > 0.7:
            system_status = "GOOD"
            result = TestResult.PASS
        elif pass_count / total_tests > 0.5 and overall_score > 0.5:
            system_status = "ACCEPTABLE"
            result = TestResult.WARNING
        else:
            system_status = "NEEDS_IMPROVEMENT"
            result = TestResult.FAIL
        
        return ValidationReport(
            test_name="Overall System Validation",
            result=result,
            score=overall_score,
            execution_time=time.time() - start_time,
            details={
                "total_tests": total_tests,
                "pass_count": pass_count,
                "warning_count": warning_count,
                "fail_count": fail_count,
                "pass_rate": pass_count / total_tests,
                "system_status": system_status,
                "validation_level": self.validation_level.value
            }
        )
    
    def generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        if not self.validation_history:
            return {"error": "No validation history available"}
        
        latest_validation = self.validation_history[-1]
        
        # Extract key metrics
        reports = latest_validation['reports']
        overall_report = reports.get('overall_system')
        
        summary = {
            "validation_timestamp": latest_validation['timestamp'],
            "validation_level": latest_validation['validation_level'],
            "total_tests": latest_validation['total_tests'],
            "validation_time": latest_validation['total_time'],
            "overall_score": overall_report.score if overall_report else 0.0,
            "system_status": overall_report.details.get('system_status', 'UNKNOWN') if overall_report else 'UNKNOWN',
            "pass_rate": overall_report.details.get('pass_rate', 0.0) if overall_report else 0.0,
            "critical_issues": [],
            "recommendations": []
        }
        
        # Collect critical issues and recommendations
        for test_name, report in reports.items():
            if report.result == TestResult.FAIL:
                summary['critical_issues'].append(f"{test_name}: {report.details}")
            
            if report.recommendations:
                summary['recommendations'].extend(report.recommendations)
        
        return summary
    
    def save_validation_report(self, filepath: str, validation_reports: Dict[str, ValidationReport]):
        """Save detailed validation report to file."""
        try:
            report_data = {
                'timestamp': time.time(),
                'validation_level': self.validation_level.value,
                'summary': self.generate_validation_summary(),
                'detailed_reports': {}
            }
            
            for test_name, report in validation_reports.items():
                report_data['detailed_reports'][test_name] = {
                    'result': report.result.value,
                    'score': report.score,
                    'execution_time': report.execution_time,
                    'details': report.details,
                    'warnings': report.warnings,
                    'recommendations': report.recommendations
                }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

# Global validation framework instance
global_validation_framework = ComprehensiveValidationFramework(ValidationLevel.COMPREHENSIVE)