"""
Auto-scaling and resource management for quantum hyperparameter search.
"""

import time
import threading
import psutil
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ScalingDecision(Enum):
    """Scaling decision types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes_per_sec: float = 0.0
    active_workers: int = 0
    queue_size: int = 0
    throughput_tasks_per_sec: float = 0.0


@dataclass
class ScalingPolicy:
    """Scaling policy configuration."""
    min_workers: int = 1
    max_workers: int = 8
    scale_up_cpu_threshold: float = 0.7
    scale_down_cpu_threshold: float = 0.3
    scale_up_memory_threshold: float = 0.8
    scale_down_memory_threshold: float = 0.4
    scale_up_queue_threshold: int = 10
    scale_down_queue_threshold: int = 2
    cooldown_seconds: float = 30.0
    evaluation_window_seconds: float = 60.0


class AutoScaler:
    """
    Automatic scaling system for quantum optimization workloads.
    """
    
    def __init__(
        self,
        policy: Optional[ScalingPolicy] = None,
        monitoring_interval: float = 5.0
    ):
        """
        Initialize auto-scaler.
        
        Args:
            policy: Scaling policy configuration
            monitoring_interval: How often to check metrics (seconds)
        """
        self.policy = policy or ScalingPolicy()
        self.monitoring_interval = monitoring_interval
        
        # Current state
        self.current_workers = self.policy.min_workers
        self.last_scaling_action = 0.0
        self.is_monitoring = False
        
        # Metrics history
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 1000
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        # Scaling callbacks
        self.scale_up_callback: Optional[Callable[[int], None]] = None
        self.scale_down_callback: Optional[Callable[[int], None]] = None
    
    def start_monitoring(self) -> None:
        """Start resource monitoring and auto-scaling."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.stop_monitoring.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring_system(self) -> None:
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        self.stop_monitoring.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.monitoring_interval * 2)
    
    def set_scaling_callbacks(
        self,
        scale_up_callback: Optional[Callable[[int], None]] = None,
        scale_down_callback: Optional[Callable[[int], None]] = None
    ) -> None:
        """
        Set callbacks for scaling actions.
        
        Args:
            scale_up_callback: Called when scaling up with new worker count
            scale_down_callback: Called when scaling down with new worker count
        """
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    def update_queue_metrics(self, queue_size: int, throughput: float) -> None:
        """
        Update queue-related metrics.
        
        Args:
            queue_size: Current queue size
            throughput: Current throughput (tasks/second)
        """
        if self.metrics_history:
            self.metrics_history[-1].queue_size = queue_size
            self.metrics_history[-1].throughput_tasks_per_sec = throughput
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # Make scaling decision
                decision = self._make_scaling_decision(metrics)
                
                # Execute scaling action
                if decision != ScalingDecision.MAINTAIN:
                    self._execute_scaling_decision(decision)
                    
            except Exception:
                # Continue monitoring even if individual cycles fail
                continue
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / 1024 / 1024
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Network I/O (simplified)
        try:
            net_io = psutil.net_io_counters()
            network_io_bytes_per_sec = getattr(net_io, 'bytes_sent', 0) + getattr(net_io, 'bytes_recv', 0)
        except:
            network_io_bytes_per_sec = 0.0
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_usage_percent=disk_usage_percent,
            network_io_bytes_per_sec=network_io_bytes_per_sec,
            active_workers=self.current_workers
        )
    
    def _make_scaling_decision(self, current_metrics: ResourceMetrics) -> ScalingDecision:
        """
        Make scaling decision based on current and historical metrics.
        
        Args:
            current_metrics: Current resource metrics
            
        Returns:
            Scaling decision
        """
        # Check cooldown period
        if time.time() - self.last_scaling_action < self.policy.cooldown_seconds:
            return ScalingDecision.MAINTAIN
        
        # Get recent metrics for trend analysis
        recent_metrics = self._get_recent_metrics()
        if len(recent_metrics) < 3:  # Need enough data
            return ScalingDecision.MAINTAIN
        
        # Calculate average metrics over evaluation window
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics]) / 100.0
        avg_memory = np.mean([m.memory_percent for m in recent_metrics]) / 100.0
        avg_queue_size = np.mean([m.queue_size for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_tasks_per_sec for m in recent_metrics])
        
        # Scaling up conditions
        scale_up_reasons = []
        
        if avg_cpu > self.policy.scale_up_cpu_threshold:
            scale_up_reasons.append("high_cpu")
        
        if avg_memory > self.policy.scale_up_memory_threshold:
            scale_up_reasons.append("high_memory")
        
        if avg_queue_size > self.policy.scale_up_queue_threshold:
            scale_up_reasons.append("high_queue")
        
        # Scaling down conditions
        scale_down_reasons = []
        
        if avg_cpu < self.policy.scale_down_cpu_threshold:
            scale_down_reasons.append("low_cpu")
        
        if avg_memory < self.policy.scale_down_memory_threshold:
            scale_down_reasons.append("low_memory")
        
        if avg_queue_size < self.policy.scale_down_queue_threshold:
            scale_down_reasons.append("low_queue")
        
        # Decision logic
        if scale_up_reasons and self.current_workers < self.policy.max_workers:
            return ScalingDecision.SCALE_UP
        elif (len(scale_down_reasons) >= 2 and  # Multiple indicators
              self.current_workers > self.policy.min_workers and
              not scale_up_reasons):  # No conflicting signals
            return ScalingDecision.SCALE_DOWN
        else:
            return ScalingDecision.MAINTAIN
    
    def _get_recent_metrics(self) -> List[ResourceMetrics]:
        """Get metrics from recent evaluation window."""
        if not self.metrics_history:
            return []
        
        cutoff_time = time.time() - self.policy.evaluation_window_seconds
        return [
            m for m in self.metrics_history[-50:]  # Look at last 50 entries max
            if m.timestamp >= cutoff_time
        ]
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """
        Execute scaling decision.
        
        Args:
            decision: Scaling decision to execute
        """
        if decision == ScalingDecision.SCALE_UP:
            new_workers = min(self.current_workers + 1, self.policy.max_workers)
            if new_workers > self.current_workers:
                self.current_workers = new_workers
                self.last_scaling_action = time.time()
                
                if self.scale_up_callback:
                    self.scale_up_callback(new_workers)
        
        elif decision == ScalingDecision.SCALE_DOWN:
            new_workers = max(self.current_workers - 1, self.policy.min_workers)
            if new_workers < self.current_workers:
                self.current_workers = new_workers
                self.last_scaling_action = time.time()
                
                if self.scale_down_callback:
                    self.scale_down_callback(new_workers)
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling metrics and status."""
        recent_metrics = self._get_recent_metrics()
        
        if not recent_metrics:
            return {
                'current_workers': self.current_workers,
                'policy': {
                    'min_workers': self.policy.min_workers,
                    'max_workers': self.policy.max_workers
                }
            }
        
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_queue_size = np.mean([m.queue_size for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_tasks_per_sec for m in recent_metrics])
        
        return {
            'current_workers': self.current_workers,
            'policy': {
                'min_workers': self.policy.min_workers,
                'max_workers': self.policy.max_workers,
                'scale_up_cpu_threshold': self.policy.scale_up_cpu_threshold,
                'scale_down_cpu_threshold': self.policy.scale_down_cpu_threshold
            },
            'current_metrics': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'avg_queue_size': avg_queue_size,
                'avg_throughput': avg_throughput
            },
            'last_scaling_action': self.last_scaling_action,
            'time_since_last_scaling': time.time() - self.last_scaling_action,
            'metrics_history_size': len(self.metrics_history)
        }


class ResourceManager:
    """
    Resource manager for optimizing quantum hyperparameter search performance.
    """
    
    def __init__(
        self,
        memory_limit_mb: Optional[float] = None,
        cpu_limit_percent: Optional[float] = None
    ):
        """
        Initialize resource manager.
        
        Args:
            memory_limit_mb: Memory limit in MB (None for no limit)
            cpu_limit_percent: CPU usage limit as percentage (None for no limit)
        """
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        
        # Resource usage tracking
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.resource_warnings = []
        
        # Auto-scaling integration
        self.auto_scaler: Optional[AutoScaler] = None
    
    def set_auto_scaler(self, auto_scaler: AutoScaler) -> None:
        """Set auto-scaler for integration."""
        self.auto_scaler = auto_scaler
    
    def check_resource_constraints(self) -> Dict[str, Any]:
        """
        Check current resource usage against constraints.
        
        Returns:
            Resource status and warnings
        """
        # Get current usage
        memory = psutil.virtual_memory()
        current_memory_mb = memory.used / 1024 / 1024
        current_cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Track peaks
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)
        self.peak_cpu_percent = max(self.peak_cpu_percent, current_cpu_percent)
        
        warnings = []
        status = "healthy"
        
        # Check memory constraints
        if self.memory_limit_mb and current_memory_mb > self.memory_limit_mb:
            warnings.append(f"Memory usage ({current_memory_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)")
            status = "constrained"
        
        # Check CPU constraints
        if self.cpu_limit_percent and current_cpu_percent > self.cpu_limit_percent:
            warnings.append(f"CPU usage ({current_cpu_percent:.1f}%) exceeds limit ({self.cpu_limit_percent}%)")
            status = "constrained"
        
        # Check for resource pressure
        if memory.percent > 90:
            warnings.append(f"System memory usage is high ({memory.percent:.1f}%)")
            status = "pressured"
        
        if current_cpu_percent > 95:
            warnings.append(f"System CPU usage is high ({current_cpu_percent:.1f}%)")
            status = "pressured"
        
        self.resource_warnings.extend(warnings)
        
        return {
            'status': status,
            'current_memory_mb': current_memory_mb,
            'current_cpu_percent': current_cpu_percent,
            'peak_memory_mb': self.peak_memory_mb,
            'peak_cpu_percent': self.peak_cpu_percent,
            'warnings': warnings,
            'memory_limit_mb': self.memory_limit_mb,
            'cpu_limit_percent': self.cpu_limit_percent
        }
    
    def optimize_for_workload(self, workload_type: str = "mixed") -> Dict[str, Any]:
        """
        Optimize resource allocation for specific workload type.
        
        Args:
            workload_type: Type of workload ('cpu_intensive', 'memory_intensive', 'mixed')
            
        Returns:
            Optimization recommendations
        """
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'current_load': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        }
        
        recommendations = {}
        
        if workload_type == "cpu_intensive":
            # Optimize for CPU-bound tasks
            recommended_workers = min(system_info['cpu_count'], 8)
            recommended_chunk_size = 1
            
        elif workload_type == "memory_intensive":
            # Optimize for memory-bound tasks
            # Reduce workers to avoid memory pressure
            recommended_workers = max(1, system_info['cpu_count'] // 2)
            recommended_chunk_size = 2
            
        else:  # mixed
            # Balanced optimization
            recommended_workers = max(1, int(system_info['cpu_count'] * 0.75))
            recommended_chunk_size = 1
        
        recommendations.update({
            'recommended_workers': recommended_workers,
            'recommended_chunk_size': recommended_chunk_size,
            'workload_type': workload_type,
            'system_info': system_info
        })
        
        return recommendations
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary."""
        current_status = self.check_resource_constraints()
        
        return {
            'current_status': current_status,
            'total_warnings': len(self.resource_warnings),
            'recent_warnings': self.resource_warnings[-5:],  # Last 5 warnings
            'auto_scaler_active': self.auto_scaler is not None and self.auto_scaler.is_monitoring,
            'auto_scaler_metrics': (
                self.auto_scaler.get_scaling_metrics() 
                if self.auto_scaler else None
            )
        }