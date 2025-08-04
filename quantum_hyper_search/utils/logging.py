"""
Comprehensive logging system for quantum hyperparameter search.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json


class QuantumSearchFormatter(logging.Formatter):
    """Custom formatter for quantum search logs with structured output."""
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.now()
        
    def format(self, record):
        """Format log record with quantum search context."""
        # Add timing information
        record.elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'elapsed_seconds': record.elapsed
        }
        
        # Add extra fields if present
        if hasattr(record, 'quantum_backend'):
            log_entry['quantum_backend'] = record.quantum_backend
        if hasattr(record, 'iteration'):
            log_entry['iteration'] = record.iteration
        if hasattr(record, 'energy'):
            log_entry['energy'] = record.energy
        if hasattr(record, 'score'):
            log_entry['score'] = record.score
        if hasattr(record, 'parameters'):
            log_entry['parameters'] = record.parameters
            
        return json.dumps(log_entry, default=str)


def setup_logger(
    name: str = "quantum_hyper_search",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    structured: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for quantum hyperparameter search.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
        structured: Whether to use structured JSON logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if structured:
        console_formatter = QuantumSearchFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(QuantumSearchFormatter())
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "quantum_hyper_search") -> logging.Logger:
    """
    Get or create logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class OptimizationLogger:
    """Specialized logger for optimization events."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize optimization logger."""
        self.logger = logger or get_logger()
        
    def log_optimization_start(
        self,
        backend: str,
        search_space: Dict[str, Any],
        n_iterations: int,
        quantum_reads: int
    ):
        """Log optimization start."""
        self.logger.info(
            "Starting quantum hyperparameter optimization",
            extra={
                'quantum_backend': backend,
                'search_space_size': len(search_space),
                'total_combinations': self._calculate_combinations(search_space),
                'n_iterations': n_iterations,
                'quantum_reads': quantum_reads
            }
        )
    
    def log_iteration_start(self, iteration: int, total: int):
        """Log iteration start."""
        self.logger.info(
            f"Starting iteration {iteration}/{total}",
            extra={'iteration': iteration}
        )
    
    def log_quantum_sampling(self, backend: str, num_reads: int, energy: float):
        """Log quantum sampling results."""
        self.logger.info(
            f"Quantum sampling complete with {backend}",
            extra={
                'quantum_backend': backend,
                'num_reads': num_reads,
                'best_energy': energy
            }
        )
    
    def log_evaluation(
        self,
        parameters: Dict[str, Any],
        score: float,
        iteration: int,
        is_best: bool = False
    ):
        """Log parameter evaluation."""
        level = logging.INFO if is_best else logging.DEBUG
        message = "New best configuration found!" if is_best else "Configuration evaluated"
        
        self.logger.log(
            level,
            message,
            extra={
                'parameters': parameters,
                'score': score,
                'iteration': iteration,
                'is_best': is_best
            }
        )
    
    def log_optimization_complete(
        self,
        best_score: float,
        best_params: Dict[str, Any],
        total_evaluations: int,
        elapsed_time: float
    ):
        """Log optimization completion."""
        self.logger.info(
            "Quantum hyperparameter optimization complete",
            extra={
                'best_score': best_score,
                'best_parameters': best_params,
                'total_evaluations': total_evaluations,
                'elapsed_time_seconds': elapsed_time
            }
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log optimization error with context."""
        self.logger.error(
            f"Optimization error: {str(error)}",
            extra={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            },
            exc_info=True
        )
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log optimization warning."""
        self.logger.warning(
            message,
            extra={'context': context or {}}
        )
    
    def _calculate_combinations(self, search_space: Dict[str, Any]) -> int:
        """Calculate total parameter combinations."""
        try:
            import numpy as np
            return int(np.prod([len(v) for v in search_space.values()]))
        except Exception:
            return -1