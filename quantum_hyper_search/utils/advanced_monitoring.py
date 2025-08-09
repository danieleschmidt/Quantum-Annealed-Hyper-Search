"""
Advanced Monitoring - Real-time monitoring and alerting system for quantum optimization.

Provides comprehensive monitoring, anomaly detection, and automated alerting
for quantum hyperparameter optimization processes.
"""

import time
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MonitoringEvent:
    """Represents a monitoring event."""
    timestamp: float
    event_type: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type,
            'severity': self.severity,
            'message': self.message,
            'data': self.data,
            'source': self.source
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    quantum_queue_length: int = 0
    evaluation_rate: float = 0.0
    error_rate: float = 0.0
    average_response_time: float = 0.0
    best_score_trend: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'quantum_queue_length': self.quantum_queue_length,
            'evaluation_rate': self.evaluation_rate,
            'error_rate': self.error_rate,
            'average_response_time': self.average_response_time,
            'best_score_trend': self.best_score_trend
        }


class AnomalyDetector:
    """Real-time anomaly detection for quantum optimization processes."""
    
    def __init__(self, window_size: int = 50, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines = {}
        self.anomaly_count = defaultdict(int)
        
    def add_metric(self, metric_name: str, value: float) -> Optional[MonitoringEvent]:
        """Add metric value and detect anomalies."""
        history = self.metric_history[metric_name]
        history.append(value)
        
        # Need sufficient history for anomaly detection
        if len(history) < 10:
            return None
        
        # Calculate baseline statistics
        mean = np.mean(history)
        std = np.std(history)
        
        # Update baseline
        self.baselines[metric_name] = {'mean': mean, 'std': std}
        
        # Detect anomaly using z-score
        if std > 0:
            z_score = abs(value - mean) / std
            
            if z_score > self.sensitivity:
                self.anomaly_count[metric_name] += 1
                
                severity = 'warning'
                if z_score > self.sensitivity * 2:
                    severity = 'error'
                if z_score > self.sensitivity * 3:
                    severity = 'critical'
                
                return MonitoringEvent(
                    timestamp=time.time(),
                    event_type='anomaly_detected',
                    severity=severity,
                    message=f'Anomaly detected in {metric_name}: {value:.3f} (z-score: {z_score:.2f})',
                    data={
                        'metric_name': metric_name,
                        'value': value,
                        'z_score': z_score,
                        'baseline_mean': mean,
                        'baseline_std': std
                    },
                    source='anomaly_detector'
                )
        
        return None
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        return {
            'total_anomalies': sum(self.anomaly_count.values()),
            'anomalies_by_metric': dict(self.anomaly_count),
            'monitored_metrics': list(self.baselines.keys()),
            'baselines': self.baselines
        }


class AlertManager:
    """Manages alerting for monitoring events."""
    
    def __init__(self, 
                 email_config: Optional[Dict] = None,
                 webhook_urls: Optional[List[str]] = None,
                 alert_cooldown: float = 300.0):  # 5 minutes
        self.email_config = email_config or {}
        self.webhook_urls = webhook_urls or []
        self.alert_cooldown = alert_cooldown
        self.last_alert_times = defaultdict(float)
        self.alert_queue = queue.Queue()
        self.running = False
        self.alert_thread = None
        
    def start(self):
        """Start alert processing thread."""
        if not self.running:
            self.running = True
            self.alert_thread = threading.Thread(target=self._process_alerts)
            self.alert_thread.daemon = True
            self.alert_thread.start()
            logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert processing."""
        self.running = False
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=5)
        logger.info("Alert manager stopped")
    
    def send_alert(self, event: MonitoringEvent) -> bool:
        """Queue alert for processing."""
        # Check cooldown
        alert_key = f"{event.event_type}_{event.severity}"
        now = time.time()
        
        if now - self.last_alert_times[alert_key] < self.alert_cooldown:
            logger.debug(f"Alert {alert_key} suppressed due to cooldown")
            return False
        
        self.last_alert_times[alert_key] = now
        
        try:
            self.alert_queue.put_nowait(event)
            return True
        except queue.Full:
            logger.error("Alert queue full, dropping alert")
            return False
    
    def _process_alerts(self):
        """Process alerts in background thread."""
        while self.running:
            try:
                event = self.alert_queue.get(timeout=1.0)
                
                # Send email alert
                if self.email_config:
                    self._send_email_alert(event)
                
                # Send webhook alerts
                if self.webhook_urls:
                    self._send_webhook_alert(event)
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
    
    def _send_email_alert(self, event: MonitoringEvent):
        """Send email alert."""
        try:
            if not all(k in self.email_config for k in ['smtp_server', 'smtp_port', 'username', 'password', 'to_addresses']):
                logger.warning("Incomplete email configuration, skipping email alert")
                return
            
            # Create email content
            subject = f"[{event.severity.upper()}] Quantum Optimization Alert: {event.event_type}"
            body = f"""
Quantum Hyperparameter Optimization Alert

Event: {event.event_type}
Severity: {event.severity}
Timestamp: {datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
Source: {event.source}

Message:
{event.message}

Additional Data:
{json.dumps(event.data, indent=2)}
"""
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.email_config['username']
            msg['To'] = ', '.join(self.email_config['to_addresses'])
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {event.event_type}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, event: MonitoringEvent):
        """Send webhook alert."""
        try:
            import requests
            
            payload = {
                'event': event.to_dict(),
                'alert_type': 'quantum_optimization',
                'timestamp': time.time()
            }
            
            for webhook_url in self.webhook_urls:
                try:
                    response = requests.post(
                        webhook_url,
                        json=payload,
                        timeout=10,
                        headers={'Content-Type': 'application/json'}
                    )
                    response.raise_for_status()
                    logger.info(f"Webhook alert sent to {webhook_url}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to send webhook to {webhook_url}: {e}")
                    
        except ImportError:
            logger.warning("requests library not available for webhook alerts")
        except Exception as e:
            logger.error(f"Webhook alert error: {e}")


class AdvancedQuantumMonitor:
    """
    Advanced monitoring system for quantum hyperparameter optimization.
    
    Provides real-time monitoring, anomaly detection, alerting, and
    comprehensive performance analysis.
    """
    
    def __init__(self,
                 alert_config: Optional[Dict] = None,
                 anomaly_sensitivity: float = 2.0,
                 metric_retention_size: int = 1000):
        """
        Initialize advanced monitoring.
        
        Args:
            alert_config: Alerting configuration
            anomaly_sensitivity: Sensitivity for anomaly detection
            metric_retention_size: Number of metrics to retain in memory
        """
        self.session_id = f"monitor_{int(time.time())}"
        self.start_time = time.time()
        
        # Components
        self.anomaly_detector = AnomalyDetector(sensitivity=anomaly_sensitivity)
        self.alert_manager = AlertManager(
            email_config=alert_config.get('email') if alert_config else None,
            webhook_urls=alert_config.get('webhooks') if alert_config else None
        )
        
        # Data storage
        self.events = deque(maxlen=metric_retention_size)
        self.performance_history = deque(maxlen=metric_retention_size)
        self.optimization_metrics = {}
        
        # State tracking
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Performance tracking
        self.last_metrics_update = time.time()
        self.evaluation_count = 0
        self.error_count = 0
        
        # Start alert manager
        self.alert_manager.start()
        
        logger.info(f"Advanced monitor initialized (session: {self.session_id})")
    
    def start_monitoring(self):
        """Start monitoring process."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.log_event(MonitoringEvent(
                timestamp=time.time(),
                event_type='monitoring_started',
                severity='info',
                message='Advanced monitoring started',
                source='monitor'
            ))
            
            print("📊 Advanced monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring process."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.alert_manager.stop()
        
        self.log_event(MonitoringEvent(
            timestamp=time.time(),
            event_type='monitoring_stopped',
            severity='info',
            message='Advanced monitoring stopped',
            source='monitor'
        ))
        
        print("⚠️ Advanced monitoring stopped")
    
    def log_event(self, event: MonitoringEvent):
        """Log monitoring event."""
        self.events.append(event)
        
        # Send alert for warnings and errors
        if event.severity in ['warning', 'error', 'critical']:
            self.alert_manager.send_alert(event)
        
        logger.log(
            getattr(logging, event.severity.upper(), logging.INFO),
            f"[{event.event_type}] {event.message}"
        )
    
    def update_optimization_metrics(self, metrics: Dict[str, Any]):
        """Update optimization metrics and detect anomalies."""
        current_time = time.time()
        
        # Update core metrics
        self.optimization_metrics.update(metrics)
        self.last_metrics_update = current_time
        
        # Extract key performance indicators
        performance = PerformanceMetrics(
            evaluation_rate=metrics.get('evaluations_per_second', 0.0),
            error_rate=metrics.get('error_rate', 0.0),
            average_response_time=metrics.get('avg_evaluation_time', 0.0),
            best_score_trend=metrics.get('best_score_trend', 0.0)
        )
        
        self.performance_history.append((current_time, performance))
        
        # Anomaly detection
        for metric_name, value in performance.to_dict().items():
            if isinstance(value, (int, float)):
                anomaly_event = self.anomaly_detector.add_metric(metric_name, value)
                if anomaly_event:
                    self.log_event(anomaly_event)
        
        # Check for specific conditions
        self._check_performance_conditions(performance)
    
    def _check_performance_conditions(self, performance: PerformanceMetrics):
        """Check for specific performance conditions that warrant alerts."""
        current_time = time.time()
        
        # High error rate
        if performance.error_rate > 0.3:  # 30% error rate
            self.log_event(MonitoringEvent(
                timestamp=current_time,
                event_type='high_error_rate',
                severity='warning',
                message=f'High error rate detected: {performance.error_rate:.1%}',
                data={'error_rate': performance.error_rate},
                source='monitor'
            ))
        
        # Very high error rate
        if performance.error_rate > 0.7:  # 70% error rate
            self.log_event(MonitoringEvent(
                timestamp=current_time,
                event_type='critical_error_rate',
                severity='critical',
                message=f'Critical error rate: {performance.error_rate:.1%}',
                data={'error_rate': performance.error_rate},
                source='monitor'
            ))
        
        # Low evaluation rate
        if performance.evaluation_rate < 0.1 and current_time - self.start_time > 60:  # After 1 minute
            self.log_event(MonitoringEvent(
                timestamp=current_time,
                event_type='low_performance',
                severity='warning',
                message=f'Low evaluation rate: {performance.evaluation_rate:.3f} evals/sec',
                data={'evaluation_rate': performance.evaluation_rate},
                source='monitor'
            ))
        
        # Declining trend
        if performance.best_score_trend < -0.01:  # Significant decline
            self.log_event(MonitoringEvent(
                timestamp=current_time,
                event_type='declining_performance',
                severity='warning',
                message=f'Best score trend declining: {performance.best_score_trend:.4f}',
                data={'trend': performance.best_score_trend},
                source='monitor'
            ))
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Check if metrics are being updated
                if current_time - self.last_metrics_update > 300:  # 5 minutes without update
                    self.log_event(MonitoringEvent(
                        timestamp=current_time,
                        event_type='metrics_stale',
                        severity='warning',
                        message='No metric updates received for 5 minutes',
                        source='monitor'
                    ))
                
                # System resource monitoring (if available)
                try:
                    import psutil
                    
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # Check for resource issues
                    if cpu_percent > 90:
                        self.log_event(MonitoringEvent(
                            timestamp=current_time,
                            event_type='high_cpu_usage',
                            severity='warning',
                            message=f'High CPU usage: {cpu_percent:.1f}%',
                            data={'cpu_percent': cpu_percent},
                            source='system_monitor'
                        ))
                    
                    if memory.percent > 90:
                        self.log_event(MonitoringEvent(
                            timestamp=current_time,
                            event_type='high_memory_usage',
                            severity='warning',
                            message=f'High memory usage: {memory.percent:.1f}%',
                            data={'memory_percent': memory.percent},
                            source='system_monitor'
                        ))
                        
                except ImportError:
                    # psutil not available, skip system monitoring
                    pass
                except Exception as e:
                    logger.debug(f"System monitoring error: {e}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(60)  # Wait longer on errors
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Event statistics
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in self.events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
        
        # Recent performance
        recent_performance = None
        if self.performance_history:
            _, recent_performance = self.performance_history[-1]
        
        return {
            'session_id': self.session_id,
            'uptime_seconds': uptime,
            'uptime_formatted': str(timedelta(seconds=int(uptime))),
            'is_monitoring': self.is_monitoring,
            'total_events': len(self.events),
            'events_by_type': dict(event_counts),
            'events_by_severity': dict(severity_counts),
            'recent_performance': recent_performance.to_dict() if recent_performance else None,
            'anomaly_summary': self.anomaly_detector.get_anomaly_summary(),
            'last_metrics_update': self.last_metrics_update,
            'metrics_age_seconds': current_time - self.last_metrics_update
        }
    
    def export_monitoring_data(self, filename: str):
        """Export monitoring data to file."""
        try:
            export_data = {
                'summary': self.get_monitoring_summary(),
                'events': [event.to_dict() for event in self.events],
                'performance_history': [
                    {
                        'timestamp': timestamp,
                        'metrics': performance.to_dict()
                    }
                    for timestamp, performance in self.performance_history
                ],
                'optimization_metrics': self.optimization_metrics
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"💾 Monitoring data exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export monitoring data: {e}")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate system health report."""
        summary = self.get_monitoring_summary()
        
        # Calculate health score
        health_score = 100.0
        issues = []
        
        # Deduct points for various issues
        critical_events = summary['events_by_severity'].get('critical', 0)
        error_events = summary['events_by_severity'].get('error', 0)
        warning_events = summary['events_by_severity'].get('warning', 0)
        
        health_score -= critical_events * 25  # Major deduction for critical events
        health_score -= error_events * 10     # Moderate deduction for errors
        health_score -= warning_events * 2    # Minor deduction for warnings
        
        if critical_events > 0:
            issues.append(f"{critical_events} critical events")
        if error_events > 0:
            issues.append(f"{error_events} error events")
        if warning_events > 5:
            issues.append(f"{warning_events} warning events")
        
        # Check for stale metrics
        metrics_age = summary['metrics_age_seconds']
        if metrics_age > 300:  # 5 minutes
            health_score -= 20
            issues.append(f"Stale metrics ({int(metrics_age/60)} minutes old)")
        
        health_score = max(0, health_score)  # Ensure non-negative
        
        # Determine health status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 50:
            status = 'fair'
        elif health_score >= 25:
            status = 'poor'
        else:
            status = 'critical'
        
        return {
            'health_score': health_score,
            'status': status,
            'issues': issues,
            'recommendations': self._generate_health_recommendations(summary, issues),
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_health_recommendations(self, summary: Dict, issues: List[str]) -> List[str]:
        """Generate health recommendations based on issues."""
        recommendations = []
        
        if 'critical events' in ' '.join(issues):
            recommendations.append("Review critical events and address underlying causes")
        
        if 'Stale metrics' in ' '.join(issues):
            recommendations.append("Check optimization process - metrics not updating")
        
        if summary['events_by_severity'].get('error', 0) > 10:
            recommendations.append("High error rate - review error logs and improve error handling")
        
        anomaly_count = summary['anomaly_summary']['total_anomalies']
        if anomaly_count > 20:
            recommendations.append("Many anomalies detected - review system stability")
        
        if not recommendations:
            recommendations.append("System operating normally - continue monitoring")
        
        return recommendations
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
