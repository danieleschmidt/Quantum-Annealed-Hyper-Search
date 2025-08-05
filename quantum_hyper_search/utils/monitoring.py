"""
Performance monitoring and health checking for quantum hyperparameter search.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization run."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Timing metrics
    total_duration: float = 0.0
    quantum_sampling_time: float = 0.0
    evaluation_time: float = 0.0
    encoding_time: float = 0.0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    
    # Optimization metrics
    total_evaluations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    convergence_iteration: int = -1
    
    # Quality metrics
    best_score_history: List[float] = field(default_factory=list)
    score_improvements: int = 0
    exploration_diversity: float = 0.0


class PerformanceMonitor:
    """Monitor performance and resource usage during optimization."""
    
    def __init__(self, sample_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            sample_interval: Interval in seconds between resource samples
        """
        self.sample_interval = sample_interval
        self.metrics = PerformanceMetrics()
        self.monitoring_active = False
        self._monitor_thread = None
        
        # Resource usage tracking
        self._cpu_samples = []
        self._memory_samples = []
        self._process = psutil.Process()
        
        # Timing contexts
        self._timing_stack = []
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.metrics.start_time = datetime.now()
        
        # Start resource monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        self.metrics.end_time = datetime.now()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        # Calculate final metrics
        self._calculate_final_metrics()
    
    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = self._process.cpu_percent()
                self._cpu_samples.append(cpu_percent)
                
                # Memory usage
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self._memory_samples.append(memory_mb)
                
                # Update peak values
                self.metrics.peak_cpu_percent = max(self.metrics.peak_cpu_percent, cpu_percent)
                self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)
                
                time.sleep(self.sample_interval)
                
            except Exception:
                # Continue monitoring even if individual samples fail
                time.sleep(self.sample_interval)
    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics."""
        if self.metrics.end_time and self.metrics.start_time:
            self.metrics.total_duration = (
                self.metrics.end_time - self.metrics.start_time
            ).total_seconds()
        
        # Average CPU usage
        if self._cpu_samples:
            self.metrics.avg_cpu_percent = np.mean(self._cpu_samples)
        
        # Calculate convergence iteration
        if len(self.metrics.best_score_history) > 1:
            best_score = max(self.metrics.best_score_history)
            for i, score in enumerate(self.metrics.best_score_history):
                if abs(score - best_score) < 1e-6:  # Found convergence
                    self.metrics.convergence_iteration = i
                    break
        
        # Calculate exploration diversity (std of scores)
        if len(self.metrics.best_score_history) > 1:
            self.metrics.exploration_diversity = np.std(self.metrics.best_score_history)
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return TimingContext(self, operation_name)
    
    def record_evaluation(self, score: float, success: bool = True):
        """Record evaluation result."""
        self.metrics.total_evaluations += 1
        
        if success:
            self.metrics.successful_evaluations += 1
            self.metrics.best_score_history.append(score)
            
            # Check if this is an improvement
            if len(self.metrics.best_score_history) > 1:
                if score > max(self.metrics.best_score_history[:-1]):
                    self.metrics.score_improvements += 1
        else:
            self.metrics.failed_evaluations += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary as dictionary."""
        return {
            'total_duration_seconds': self.metrics.total_duration,
            'quantum_sampling_time_seconds': self.metrics.quantum_sampling_time,
            'evaluation_time_seconds': self.metrics.evaluation_time,
            'encoding_time_seconds': self.metrics.encoding_time,
            'peak_memory_mb': self.metrics.peak_memory_mb,
            'avg_cpu_percent': self.metrics.avg_cpu_percent,
            'peak_cpu_percent': self.metrics.peak_cpu_percent,
            'total_evaluations': self.metrics.total_evaluations,
            'successful_evaluations': self.metrics.successful_evaluations,
            'failed_evaluations': self.metrics.failed_evaluations,
            'success_rate': (
                self.metrics.successful_evaluations / max(self.metrics.total_evaluations, 1)
            ),
            'convergence_iteration': self.metrics.convergence_iteration,
            'score_improvements': self.metrics.score_improvements,
            'exploration_diversity': self.metrics.exploration_diversity,
            'evaluations_per_second': (
                self.metrics.total_evaluations / max(self.metrics.total_duration, 1e-6)
            )
        }


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            # Record timing in appropriate metric
            if 'quantum' in self.operation_name.lower():
                self.monitor.metrics.quantum_sampling_time += duration
            elif 'evaluation' in self.operation_name.lower():
                self.monitor.metrics.evaluation_time += duration
            elif 'encoding' in self.operation_name.lower():
                self.monitor.metrics.encoding_time += duration


class HealthChecker:
    """Health checker for quantum optimization process."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks = []
        self.alerts = []
        
    def add_check(self, name: str, func: Callable[[], bool], critical: bool = False):
        """
        Add health check.
        
        Args:
            name: Check name
            func: Function that returns True if healthy
            critical: Whether failure should stop optimization
        """
        self.checks.append({
            'name': name,
            'func': func,
            'critical': critical,
            'last_result': None,
            'last_check': None
        })
    
    def run_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Dictionary with check results
        """
        results = {
            'overall_health': True,
            'critical_failures': [],
            'warnings': [],
            'checks': {}
        }
        
        for check in self.checks:
            try:
                check_result = check['func']()
                check['last_result'] = check_result
                check['last_check'] = datetime.now()
                
                results['checks'][check['name']] = {
                    'status': 'healthy' if check_result else 'unhealthy',
                    'critical': check['critical'],
                    'timestamp': check['last_check']
                }
                
                if not check_result:
                    if check['critical']:
                        results['critical_failures'].append(check['name'])
                        results['overall_health'] = False
                    else:
                        results['warnings'].append(check['name'])
                        
            except Exception as e:
                check['last_result'] = False
                check['last_check'] = datetime.now()
                
                results['checks'][check['name']] = {
                    'status': 'error',
                    'error': str(e),
                    'critical': check['critical'],
                    'timestamp': check['last_check']
                }
                
                if check['critical']:
                    results['critical_failures'].append(check['name'])
                    results['overall_health'] = False
        
        return results
    
    def get_default_checks(self) -> List[Dict[str, Any]]:
        """Get default health checks for quantum optimization."""
        def check_memory():
            """Check available memory."""
            memory = psutil.virtual_memory()
            return memory.percent < 90
        
        def check_cpu():
            """Check CPU usage."""
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return cpu_percent < 95
        
        def check_disk_space():
            """Check disk space."""
            disk = psutil.disk_usage('/')
            return disk.percent < 95
        
        return [
            {'name': 'memory_usage', 'func': check_memory, 'critical': True},
            {'name': 'cpu_usage', 'func': check_cpu, 'critical': False},
            {'name': 'disk_space', 'func': check_disk_space, 'critical': True},
        ]
    
    def setup_default_checks(self):
        """Setup default health checks."""
        for check_config in self.get_default_checks():
            self.add_check(**check_config)


class OptimizationMonitor:
    """Combined monitoring for quantum optimization."""
    
    def __init__(self, enable_performance: bool = True, enable_health: bool = True):
        """
        Initialize optimization monitor.
        
        Args:
            enable_performance: Enable performance monitoring
            enable_health: Enable health checking
        """
        self.performance_monitor = PerformanceMonitor() if enable_performance else None
        self.health_checker = HealthChecker() if enable_health else None
        
        if self.health_checker:
            self.health_checker.setup_default_checks()
    
    def start_monitoring(self):
        """Start all monitoring."""
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop all monitoring."""
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
    
    def check_health(self) -> bool:
        """
        Check system health.
        
        Returns:
            True if system is healthy
        """
        if not self.health_checker:
            return True
            
        results = self.health_checker.run_checks()
        return results['overall_health']
    
    def record_evaluation(self, score: float, success: bool = True):
        """Record evaluation for performance tracking."""
        if self.performance_monitor:
            self.performance_monitor.record_evaluation(score, success)
    
    def time_operation(self, operation_name: str):
        """Time operation for performance tracking."""
        if self.performance_monitor:
            return self.performance_monitor.time_operation(operation_name)
        else:
            return NullTimingContext()
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        report = {}
        
        if self.performance_monitor:
            report['performance'] = self.performance_monitor.get_summary()
        
        if self.health_checker:
            report['health'] = self.health_checker.run_checks()
        
        return report


class NullTimingContext:
    """Null timing context when monitoring is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass