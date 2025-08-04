"""
Performance monitoring and profiling utilities.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_logger

logger = get_logger('performance_monitor')


class PerformanceMonitor:
    """
    Monitor and profile quantum hyperparameter search performance.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.timings = {}
        self.counters = {}
        self.current_timers = {}
    
    @contextmanager
    def time_block(self, name: str):
        """
        Context manager for timing code blocks.
        
        Args:
            name: Name of the timed block
        """
        start_time = time.time()
        self.current_timers[name] = start_time
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            
            del self.current_timers[name]
    
    def increment_counter(self, name: str, value: int = 1):
        """
        Increment a performance counter.
        
        Args:
            name: Counter name
            value: Value to add
        """
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Performance statistics
        """
        summary = {
            'timings': {},
            'counters': self.counters.copy()
        }
        
        for name, times in self.timings.items():
            if times:
                import numpy as np
                summary['timings'][name] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': min(times),
                    'max': max(times)
                }
        
        return summary
    
    def reset(self):
        """Reset all performance data."""
        self.timings.clear()
        self.counters.clear()
        self.current_timers.clear()