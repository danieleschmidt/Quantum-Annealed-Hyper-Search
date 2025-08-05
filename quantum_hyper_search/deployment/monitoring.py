"""
Production monitoring for quantum hyperparameter search.
"""

from typing import Dict, List, Any, Optional
from ..utils.monitoring import OptimizationMonitor
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ProductionMonitoring:
    """Production-level monitoring for quantum systems."""
    
    def __init__(self):
        """Initialize production monitoring."""
        self.monitor = OptimizationMonitor()
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get production metrics."""
        return self.monitor.get_report()
    
    def start(self):
        """Start monitoring."""
        self.monitor.start_monitoring()
    
    def stop(self):
        """Stop monitoring."""
        self.monitor.stop_monitoring()