"""
Comprehensive Monitoring - Enterprise-grade monitoring and alerting system.

Provides real-time monitoring, anomaly detection, performance tracking,
and automated alerting for quantum optimization systems.
"""

import time
import threading
import queue
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    name: str
    value: float
    timestamp: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'unit': self.unit,
            'tags': self.tags
        }


@dataclass
class Alert:
    """Represents a monitoring alert."""
    id: str
    severity: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    timestamp: float
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'source': self.source,
            'tags': self.tags,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


class MetricCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics = deque(maxlen=max_metrics)
        self.metric_sums = defaultdict(float)
        self.metric_counts = defaultdict(int)
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        if tags is None:
            tags = {}
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            unit=unit,
            tags=tags
        )
        
        with self._lock:
            self.metrics.append(metric)
            self.metric_sums[name] += value
            self.metric_counts[name] += 1
            self.metric_history[name].append(value)
    
    def get_metric_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        with self._lock:
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                values = [m.value for m in self.metrics 
                         if m.name == name and m.timestamp >= cutoff_time]
            else:
                values = list(self.metric_history[name])
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    def get_all_metrics(self, window_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get all metrics within a time window."""
        with self._lock:
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                return [m.to_dict() for m in self.metrics if m.timestamp >= cutoff_time]
            else:
                return [m.to_dict() for m in self.metrics]


class AnomalyDetector:
    """Detects anomalies in performance metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baselines = defaultdict(lambda: {'mean': 0.0, 'std': 0.0, 'count': 0})
        self._lock = threading.Lock()
    
    def update_baseline(self, metric_name: str, value: float):
        """Update baseline statistics for a metric."""
        with self._lock:
            baseline = self.baselines[metric_name]
            n = baseline['count']
            
            if n == 0:
                baseline['mean'] = value
                baseline['std'] = 0.0
            else:
                # Online calculation of mean and variance
                delta = value - baseline['mean']
                baseline['mean'] += delta / (n + 1)
                delta2 = value - baseline['mean']
                baseline['std'] = np.sqrt((n * baseline['std']**2 + delta * delta2) / (n + 1))
            
            baseline['count'] += 1
    
    def detect_anomaly(self, metric_name: str, value: float) -> bool:
        """Detect if a metric value is anomalous."""
        with self._lock:
            baseline = self.baselines[metric_name]
            
            if baseline['count'] < 10:  # Need minimum samples
                return False
            
            if baseline['std'] == 0:  # No variance
                return value != baseline['mean']
            
            z_score = abs(value - baseline['mean']) / baseline['std']
            return z_score > self.sensitivity
    
    def get_anomaly_score(self, metric_name: str, value: float) -> float:
        """Get anomaly score for a metric value."""
        with self._lock:
            baseline = self.baselines[metric_name]
            
            if baseline['count'] < 10 or baseline['std'] == 0:
                return 0.0
            
            return abs(value - baseline['mean']) / baseline['std']


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts = deque(maxlen=max_alerts)
        self.alert_handlers = []
        self.alert_rules = []
        self._lock = threading.Lock()
        self._alert_counter = 0
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def add_alert_rule(self, rule: Callable[[PerformanceMetric], Optional[Alert]]):
        """Add an alert rule function."""
        self.alert_rules.append(rule)
    
    def create_alert(self, severity: str, title: str, message: str, 
                    source: str, tags: Optional[Dict[str, str]] = None) -> Alert:
        """Create and process a new alert."""
        if tags is None:
            tags = {}
        
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=time.time(),
            source=source,
            tags=tags
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        return alert
    
    def process_metric(self, metric: PerformanceMetric):
        """Process a metric against alert rules."""
        for rule in self.alert_rules:
            try:
                alert = rule(metric)
                if alert:
                    with self._lock:
                        self.alerts.append(alert)
                    
                    # Trigger alert handlers
                    for handler in self.alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logger.error(f"Alert handler failed: {e}")
            except Exception as e:
                logger.error(f"Alert rule failed: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    return True
        return False


class HealthChecker:
    """Monitors system health and availability."""
    
    def __init__(self):
        self.health_checks = {}
        self.health_status = {}
        self._lock = threading.Lock()
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        with self._lock:
            self.health_checks[name] = check_func
            self.health_status[name] = {'status': 'unknown', 'last_check': 0, 'message': ''}
    
    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return {'status': 'unknown', 'message': f'Health check {name} not found'}
        
        try:
            start_time = time.time()
            result = self.health_checks[name]()
            check_duration = time.time() - start_time
            
            status = {
                'status': 'healthy' if result else 'unhealthy',
                'last_check': time.time(),
                'duration': check_duration,
                'message': 'Health check passed' if result else 'Health check failed'
            }
            
        except Exception as e:
            status = {
                'status': 'error',
                'last_check': time.time(),
                'message': f'Health check error: {str(e)}'
            }
        
        with self._lock:
            self.health_status[name] = status
        
        return status
    
    def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        for name in list(self.health_checks.keys()):
            results[name] = self.run_health_check(name)
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = self.run_all_health_checks()
        
        healthy_count = sum(1 for status in results.values() if status['status'] == 'healthy')
        total_count = len(results)
        
        overall_status = 'healthy' if healthy_count == total_count else 'degraded'
        if healthy_count == 0:
            overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'healthy_checks': healthy_count,
            'total_checks': total_count,
            'timestamp': time.time(),
            'checks': results
        }


class ComprehensiveMonitor:
    """Main monitoring system that coordinates all monitoring components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = MetricCollector(max_metrics=self.config.get('max_metrics', 10000))
        self.anomaly_detector = AnomalyDetector(sensitivity=self.config.get('anomaly_sensitivity', 2.0))
        self.alerts = AlertManager(max_alerts=self.config.get('max_alerts', 1000))
        self.health = HealthChecker()
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._setup_default_alert_rules()
        self._setup_default_health_checks()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        
        def high_error_rate_rule(metric: PerformanceMetric) -> Optional[Alert]:
            if metric.name == 'error_rate' and metric.value > 0.1:  # 10% error rate
                return Alert(
                    id=f"high_error_rate_{int(time.time())}",
                    severity='critical',
                    title='High Error Rate Detected',
                    message=f'Error rate is {metric.value:.2%}, exceeding 10% threshold',
                    timestamp=metric.timestamp,
                    source='monitoring_system'
                )
            return None
        
        def slow_response_rule(metric: PerformanceMetric) -> Optional[Alert]:
            if metric.name == 'response_time' and metric.value > 5.0:  # 5 seconds
                return Alert(
                    id=f"slow_response_{int(time.time())}",
                    severity='warning',
                    title='Slow Response Time',
                    message=f'Response time is {metric.value:.2f}s, exceeding 5s threshold',
                    timestamp=metric.timestamp,
                    source='monitoring_system'
                )
            return None
        
        def memory_usage_rule(metric: PerformanceMetric) -> Optional[Alert]:
            if metric.name == 'memory_usage_percent' and metric.value > 90:  # 90% memory
                return Alert(
                    id=f"high_memory_{int(time.time())}",
                    severity='warning',
                    title='High Memory Usage',
                    message=f'Memory usage is {metric.value:.1f}%, exceeding 90% threshold',
                    timestamp=metric.timestamp,
                    source='monitoring_system'
                )
            return None
        
        self.alerts.add_alert_rule(high_error_rate_rule)
        self.alerts.add_alert_rule(slow_response_rule)
        self.alerts.add_alert_rule(memory_usage_rule)
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        def memory_health_check() -> bool:
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < 95  # Less than 95% memory usage
            except ImportError:
                return True  # Assume healthy if psutil not available
        
        def disk_health_check() -> bool:
            try:
                import psutil
                disk = psutil.disk_usage('/')
                return disk.percent < 90  # Less than 90% disk usage
            except ImportError:
                return True  # Assume healthy if psutil not available
        
        self.health.register_health_check('memory', memory_health_check)
        self.health.register_health_check('disk', disk_health_check)
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        self.metrics.record_metric(name, value, unit, tags)
        
        # Update anomaly detection baseline
        self.anomaly_detector.update_baseline(name, value)
        
        # Check for anomalies
        if self.anomaly_detector.detect_anomaly(name, value):
            anomaly_score = self.anomaly_detector.get_anomaly_score(name, value)
            self.alerts.create_alert(
                severity='warning',
                title=f'Anomaly Detected: {name}',
                message=f'Metric {name} value {value} is anomalous (score: {anomaly_score:.2f})',
                source='anomaly_detector',
                tags=tags or {}
            )
        
        # Process alert rules
        metric = PerformanceMetric(name, value, time.time(), unit, tags or {})
        self.alerts.process_metric(metric)
    
    def start_monitoring(self, interval: float = 30.0):
        """Start background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        def monitoring_loop():
            while not self._stop_monitoring.wait(interval):
                try:
                    # Run health checks
                    health_status = self.health.get_overall_health()
                    
                    # Record health metrics
                    self.record_metric('health_score', 
                                     health_status['healthy_checks'] / max(1, health_status['total_checks']))
                    
                    # Check for system health issues
                    if health_status['overall_status'] != 'healthy':
                        self.alerts.create_alert(
                            severity='critical' if health_status['overall_status'] == 'unhealthy' else 'warning',
                            title=f'System Health: {health_status["overall_status"]}',
                            message=f'System health is {health_status["overall_status"]} '
                                   f'({health_status["healthy_checks"]}/{health_status["total_checks"]} checks passing)',
                            source='health_monitor'
                        )
                
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Started monitoring thread")
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped monitoring thread")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            'timestamp': time.time(),
            'metrics_summary': {
                'total_metrics': len(self.metrics.metrics),
                'metric_types': len(self.metrics.metric_history)
            },
            'alerts_summary': {
                'total_alerts': len(self.alerts.alerts),
                'active_alerts': len(self.alerts.get_active_alerts())
            },
            'health_summary': self.health.get_overall_health(),
            'anomaly_summary': {
                'baselines_count': len(self.anomaly_detector.baselines)
            }
        }


# Global monitoring instance
global_monitor = ComprehensiveMonitor()


def monitor_performance(metric_name: str, unit: str = "", tags: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                global_monitor.record_metric(
                    f"{metric_name}_execution_time",
                    execution_time,
                    "seconds",
                    tags
                )
                global_monitor.record_metric(
                    f"{metric_name}_success_count",
                    1,
                    "count",
                    tags
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                global_monitor.record_metric(
                    f"{metric_name}_execution_time",
                    execution_time,
                    "seconds",
                    tags
                )
                global_monitor.record_metric(
                    f"{metric_name}_error_count",
                    1,
                    "count",
                    tags
                )
                
                raise e
        
        return wrapper
    return decorator