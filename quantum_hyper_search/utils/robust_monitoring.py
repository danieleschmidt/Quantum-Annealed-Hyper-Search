#!/usr/bin/env python3
"""
Robust Monitoring System
Enterprise-grade monitoring with quantum-specific metrics and alerting.
"""

import time
import threading
import queue
import logging
import json
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import warnings

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QuantumMetric:
    """Quantum-specific metric data."""
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any]
    alert_level: AlertLevel = AlertLevel.INFO


@dataclass
class SystemHealth:
    """System health status."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    quantum_coherence: float
    error_rate: float
    response_time: float
    uptime: float


class MetricsCollector:
    """
    Advanced Metrics Collection System
    
    Collects system, quantum, and application metrics with
    real-time monitoring and alerting capabilities.
    """
    
    def __init__(self, collection_interval: float = 10.0,
                 max_history_size: int = 10000):
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size
        
        # Metric storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history_size))
        self.alert_queue = queue.Queue()
        self.alert_callbacks = []
        
        # System monitoring
        self.start_time = time.time()
        self.error_count = 0
        self.total_requests = 0
        self.response_times = deque(maxlen=1000)
        
        # Quantum-specific metrics
        self.quantum_operations = 0
        self.quantum_errors = 0
        self.coherence_measurements = deque(maxlen=100)
        
        # Threading
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Prometheus integration
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 5.0,
            'response_time': 2000.0,
            'quantum_error_rate': 10.0
        }
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.prom_cpu = Gauge('system_cpu_usage_percent', 
                             'CPU usage percentage', registry=self.registry)
        self.prom_memory = Gauge('system_memory_usage_percent', 
                                'Memory usage percentage', registry=self.registry)
        self.prom_quantum_ops = Counter('quantum_operations_total', 
                                       'Total quantum operations', registry=self.registry)
        self.prom_quantum_errors = Counter('quantum_errors_total', 
                                          'Total quantum errors', registry=self.registry)
        self.prom_response_time = Histogram('response_time_seconds', 
                                           'Response time in seconds', registry=self.registry)
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect quantum metrics
                self._collect_quantum_metrics()
                
                # Check alert conditions
                self._check_alerts()
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            self._record_metric('cpu_usage', cpu_usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            self._record_metric('memory_usage', memory_usage)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self._record_metric('disk_usage', disk_usage)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self._record_metric('network_bytes_sent', net_io.bytes_sent)
            self._record_metric('network_bytes_recv', net_io.bytes_recv)
            
            # Calculate error rate
            error_rate = (self.error_count / max(self.total_requests, 1)) * 100
            self._record_metric('error_rate', error_rate)
            
            # Calculate average response time
            if self.response_times:
                avg_response_time = np.mean(list(self.response_times))
                self._record_metric('response_time', avg_response_time)
            
            # System uptime
            uptime = time.time() - self.start_time
            self._record_metric('uptime', uptime)
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def _collect_quantum_metrics(self):
        """Collect quantum-specific metrics."""
        try:
            # Quantum operation rate
            current_time = time.time()
            self._record_metric('quantum_operations_total', self.quantum_operations)
            
            # Quantum error rate
            quantum_error_rate = (self.quantum_errors / max(self.quantum_operations, 1)) * 100
            self._record_metric('quantum_error_rate', quantum_error_rate)
            
            # Quantum coherence (if measurements available)
            if self.coherence_measurements:
                avg_coherence = np.mean(list(self.coherence_measurements))
                self._record_metric('quantum_coherence', avg_coherence)
            
            # Quantum advantage estimation
            if self.quantum_operations > 10:
                quantum_advantage = self._estimate_quantum_advantage()
                self._record_metric('quantum_advantage', quantum_advantage)
            
        except Exception as e:
            logger.error(f"Quantum metrics collection failed: {e}")
    
    def _record_metric(self, name: str, value: float, metadata: Optional[Dict] = None):
        """Record a metric value."""
        with self.lock:
            metric = QuantumMetric(
                name=name,
                value=value,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            self.metrics_history[name].append(metric)
    
    def _check_alerts(self):
        """Check for alert conditions."""
        current_metrics = self.get_current_metrics()
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in current_metrics:
                value = current_metrics[metric_name].value
                
                if value > threshold:
                    alert_level = AlertLevel.WARNING
                    if value > threshold * 1.2:
                        alert_level = AlertLevel.ERROR
                    if value > threshold * 1.5:
                        alert_level = AlertLevel.CRITICAL
                    
                    self._trigger_alert(metric_name, value, threshold, alert_level)
    
    def _trigger_alert(self, metric_name: str, value: float, 
                      threshold: float, level: AlertLevel):
        """Trigger an alert."""
        alert_message = f"Alert: {metric_name} = {value:.2f} exceeds threshold {threshold:.2f}"
        
        alert_data = {
            'metric': metric_name,
            'value': value,
            'threshold': threshold,
            'level': level.value,
            'timestamp': time.time(),
            'message': alert_message
        }
        
        # Add to alert queue
        self.alert_queue.put(alert_data)
        
        # Log alert
        if level == AlertLevel.CRITICAL:
            logger.critical(alert_message)
        elif level == AlertLevel.ERROR:
            logger.error(alert_message)
        elif level == AlertLevel.WARNING:
            logger.warning(alert_message)
        else:
            logger.info(alert_message)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        current_metrics = self.get_current_metrics()
        
        if 'cpu_usage' in current_metrics:
            self.prom_cpu.set(current_metrics['cpu_usage'].value)
        
        if 'memory_usage' in current_metrics:
            self.prom_memory.set(current_metrics['memory_usage'].value)
        
        self.prom_quantum_ops.inc(0)  # Increment by 0 to update timestamp
        self.prom_quantum_errors.inc(0)
    
    def _estimate_quantum_advantage(self) -> float:
        """Estimate quantum advantage based on collected metrics."""
        
        # Simple heuristic based on performance metrics
        if not self.response_times or not self.coherence_measurements:
            return 1.0
        
        avg_response_time = np.mean(list(self.response_times))
        avg_coherence = np.mean(list(self.coherence_measurements))
        error_rate = (self.quantum_errors / max(self.quantum_operations, 1)) * 100
        
        # Quantum advantage increases with:
        # - Lower response times
        # - Higher coherence
        # - Lower error rates
        
        time_factor = max(0.1, 2.0 / max(avg_response_time, 0.1))
        coherence_factor = avg_coherence
        error_factor = max(0.1, 1.0 - error_rate / 100)
        
        quantum_advantage = time_factor * coherence_factor * error_factor
        return min(10.0, quantum_advantage)  # Cap at 10x advantage
    
    def record_quantum_operation(self, operation_type: str, success: bool,
                                execution_time: float, coherence: Optional[float] = None):
        """Record a quantum operation."""
        with self.lock:
            self.quantum_operations += 1
            self.total_requests += 1
            
            if not success:
                self.quantum_errors += 1
                self.error_count += 1
            
            self.response_times.append(execution_time)
            
            if coherence is not None:
                self.coherence_measurements.append(coherence)
            
            # Update Prometheus counters
            if PROMETHEUS_AVAILABLE:
                self.prom_quantum_ops.inc()
                if not success:
                    self.prom_quantum_errors.inc()
                self.prom_response_time.observe(execution_time)
    
    def get_current_metrics(self) -> Dict[str, QuantumMetric]:
        """Get the most recent metrics."""
        current_metrics = {}
        
        with self.lock:
            for name, history in self.metrics_history.items():
                if history:
                    current_metrics[name] = history[-1]
        
        return current_metrics
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health status."""
        current_metrics = self.get_current_metrics()
        
        health = SystemHealth(
            cpu_usage=current_metrics.get('cpu_usage', QuantumMetric('', 0, 0, {})).value,
            memory_usage=current_metrics.get('memory_usage', QuantumMetric('', 0, 0, {})).value,
            disk_usage=current_metrics.get('disk_usage', QuantumMetric('', 0, 0, {})).value,
            quantum_coherence=current_metrics.get('quantum_coherence', QuantumMetric('', 1.0, 0, {})).value,
            error_rate=current_metrics.get('error_rate', QuantumMetric('', 0, 0, {})).value,
            response_time=current_metrics.get('response_time', QuantumMetric('', 0, 0, {})).value,
            uptime=current_metrics.get('uptime', QuantumMetric('', 0, 0, {})).value
        )
        
        return health
    
    def get_metrics_export(self) -> str:
        """Export metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            # Fallback JSON export
            current_metrics = self.get_current_metrics()
            export_data = {name: asdict(metric) for name, metric in current_metrics.items()}
            return json.dumps(export_data, indent=2)
    
    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a metric."""
        self.alert_thresholds[metric_name] = threshold
    
    def get_metric_history(self, metric_name: str, duration_seconds: Optional[float] = None) -> List[QuantumMetric]:
        """Get historical data for a metric."""
        with self.lock:
            if metric_name not in self.metrics_history:
                return []
            
            history = list(self.metrics_history[metric_name])
            
            if duration_seconds is not None:
                cutoff_time = time.time() - duration_seconds
                history = [m for m in history if m.timestamp >= cutoff_time]
            
            return history


class HealthCheckManager:
    """
    Comprehensive Health Check System
    
    Provides detailed health checks for all system components
    including quantum hardware and classical infrastructure.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = {}
        self.last_health_check = None
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health check functions."""
        
        self.register_health_check('system_resources', self._check_system_resources)
        self.register_health_check('quantum_backend', self._check_quantum_backend)
        self.register_health_check('error_rates', self._check_error_rates)
        self.register_health_check('response_times', self._check_response_times)
        self.register_health_check('disk_space', self._check_disk_space)
    
    def register_health_check(self, name: str, check_function: Callable[[], Dict[str, Any]]):
        """Register a health check function."""
        self.health_checks[name] = check_function
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        
        health_report = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        failed_checks = 0
        warning_checks = 0
        
        for check_name, check_function in self.health_checks.items():
            try:
                check_result = check_function()
                health_report['checks'][check_name] = check_result
                
                if check_result.get('status') == 'failed':
                    failed_checks += 1
                elif check_result.get('status') == 'warning':
                    warning_checks += 1
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                health_report['checks'][check_name] = {
                    'status': 'failed',
                    'message': f"Health check error: {str(e)}",
                    'timestamp': time.time()
                }
                failed_checks += 1
        
        # Determine overall status
        if failed_checks > 0:
            health_report['overall_status'] = 'unhealthy'
        elif warning_checks > 0:
            health_report['overall_status'] = 'degraded'
        
        self.last_health_check = health_report
        return health_report
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        
        health = self.metrics_collector.get_system_health()
        
        status = 'healthy'
        issues = []
        
        if health.cpu_usage > 90:
            status = 'failed'
            issues.append(f"High CPU usage: {health.cpu_usage:.1f}%")
        elif health.cpu_usage > 80:
            status = 'warning'
            issues.append(f"Elevated CPU usage: {health.cpu_usage:.1f}%")
        
        if health.memory_usage > 95:
            status = 'failed'
            issues.append(f"Critical memory usage: {health.memory_usage:.1f}%")
        elif health.memory_usage > 85:
            status = 'warning'
            issues.append(f"High memory usage: {health.memory_usage:.1f}%")
        
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'System resources normal',
            'details': {
                'cpu_usage': health.cpu_usage,
                'memory_usage': health.memory_usage,
                'disk_usage': health.disk_usage
            },
            'timestamp': time.time()
        }
    
    def _check_quantum_backend(self) -> Dict[str, Any]:
        """Check quantum backend availability and performance."""
        
        try:
            # Test quantum backend connectivity
            from ..backends.backend_factory import BackendFactory
            
            factory = BackendFactory()
            backend = factory.create_backend('simulated')
            
            # Simple connectivity test
            test_qubo = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
            start_time = time.time()
            
            response = backend.sample_qubo(test_qubo, num_reads=10)
            test_time = time.time() - start_time
            
            if test_time > 5.0:
                return {
                    'status': 'warning',
                    'message': f'Slow quantum backend response: {test_time:.2f}s',
                    'details': {'response_time': test_time},
                    'timestamp': time.time()
                }
            
            return {
                'status': 'healthy',
                'message': 'Quantum backend operational',
                'details': {'response_time': test_time},
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Quantum backend unavailable: {str(e)}',
                'details': {'error': str(e)},
                'timestamp': time.time()
            }
    
    def _check_error_rates(self) -> Dict[str, Any]:
        """Check system and quantum error rates."""
        
        health = self.metrics_collector.get_system_health()
        
        status = 'healthy'
        issues = []
        
        if health.error_rate > 10:
            status = 'failed'
            issues.append(f"High error rate: {health.error_rate:.1f}%")
        elif health.error_rate > 5:
            status = 'warning'
            issues.append(f"Elevated error rate: {health.error_rate:.1f}%")
        
        # Check quantum-specific error rates
        current_metrics = self.metrics_collector.get_current_metrics()
        if 'quantum_error_rate' in current_metrics:
            quantum_error_rate = current_metrics['quantum_error_rate'].value
            
            if quantum_error_rate > 20:
                status = 'failed'
                issues.append(f"High quantum error rate: {quantum_error_rate:.1f}%")
            elif quantum_error_rate > 10:
                status = 'warning'
                issues.append(f"Elevated quantum error rate: {quantum_error_rate:.1f}%")
        
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'Error rates normal',
            'details': {
                'overall_error_rate': health.error_rate,
                'quantum_error_rate': current_metrics.get('quantum_error_rate', QuantumMetric('', 0, 0, {})).value
            },
            'timestamp': time.time()
        }
    
    def _check_response_times(self) -> Dict[str, Any]:
        """Check system response times."""
        
        health = self.metrics_collector.get_system_health()
        
        status = 'healthy'
        message = 'Response times normal'
        
        if health.response_time > 2000:  # 2 seconds
            status = 'failed'
            message = f"Slow response times: {health.response_time:.0f}ms"
        elif health.response_time > 1000:  # 1 second
            status = 'warning'
            message = f"Elevated response times: {health.response_time:.0f}ms"
        
        return {
            'status': status,
            'message': message,
            'details': {'avg_response_time_ms': health.response_time},
            'timestamp': time.time()
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        
        health = self.metrics_collector.get_system_health()
        
        status = 'healthy'
        message = 'Disk space sufficient'
        
        if health.disk_usage > 95:
            status = 'failed'
            message = f"Critical disk usage: {health.disk_usage:.1f}%"
        elif health.disk_usage > 85:
            status = 'warning'
            message = f"High disk usage: {health.disk_usage:.1f}%"
        
        return {
            'status': status,
            'message': message,
            'details': {'disk_usage_percent': health.disk_usage},
            'timestamp': time.time()
        }
    
    def get_health_summary(self) -> str:
        """Get a human-readable health summary."""
        
        if not self.last_health_check:
            self.run_health_checks()
        
        report = self.last_health_check
        
        summary = f"""
# System Health Report

**Overall Status**: {report['overall_status'].upper()}
**Last Check**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['timestamp']))}

## Component Status
"""
        
        for check_name, check_result in report['checks'].items():
            status_emoji = {
                'healthy': '🟢',
                'warning': '🟡',
                'failed': '🔴'
            }.get(check_result['status'], '⚪')
            
            summary += f"- **{check_name.replace('_', ' ').title()}**: {status_emoji} {check_result['message']}\n"
        
        return summary


# Global monitoring instance
_global_metrics_collector = None
_global_health_manager = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
        _global_metrics_collector.start_monitoring()
    
    return _global_metrics_collector


def get_health_manager() -> HealthCheckManager:
    """Get the global health check manager instance."""
    global _global_health_manager
    
    if _global_health_manager is None:
        collector = get_metrics_collector()
        _global_health_manager = HealthCheckManager(collector)
    
    return _global_health_manager