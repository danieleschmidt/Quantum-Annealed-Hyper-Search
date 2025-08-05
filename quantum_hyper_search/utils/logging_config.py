"""
Centralized logging configuration for quantum hyperparameter search.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path for logging output
        console: Whether to log to console
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logger
    logger = logging.getLogger('quantum_hyper_search')
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Rotating file handler to prevent large log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name under the quantum_hyper_search hierarchy.
    
    Args:
        name: Logger name (will be prefixed with 'quantum_hyper_search.')
        
    Returns:
        Logger instance
    """
    full_name = f'quantum_hyper_search.{name}'
    return logging.getLogger(full_name)


class QuantumLoggingContext:
    """Context manager for temporary logging configuration changes."""
    
    def __init__(self, level: str = 'INFO', quiet: bool = False):
        """
        Initialize logging context.
        
        Args:
            level: Temporary logging level
            quiet: If True, suppress all console output
        """
        self.level = level
        self.quiet = quiet
        self.original_level = None
        self.original_handlers = []
    
    def __enter__(self):
        """Enter the logging context."""
        logger = logging.getLogger('quantum_hyper_search')
        
        # Store original configuration
        self.original_level = logger.level
        self.original_handlers = logger.handlers[:]
        
        # Apply new configuration
        if self.quiet:
            # Remove all handlers to suppress output
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        else:
            # Just change the level
            numeric_level = getattr(logging, self.level.upper(), None)
            if isinstance(numeric_level, int):
                logger.setLevel(numeric_level)
                for handler in logger.handlers:
                    handler.setLevel(numeric_level)
        
        return logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the logging context and restore original configuration."""
        logger = logging.getLogger('quantum_hyper_search')
        
        # Restore original configuration
        logger.setLevel(self.original_level)
        
        # Remove current handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Restore original handlers
        for handler in self.original_handlers:
            logger.addHandler(handler)


def configure_third_party_loggers(level: str = 'WARNING') -> None:
    """
    Configure third-party library loggers to reduce noise.
    
    Args:
        level: Logging level for third-party libraries
    """
    third_party_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'matplotlib',
        'dwave',
        'dimod',
        'neal'
    ]
    
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(numeric_level)


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform
    import sys
    
    logger = get_logger('system')
    
    logger.info("System Information:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Architecture: {platform.architecture()}")
    logger.info(f"  Processor: {platform.processor()}")
    
    # Log installed packages
    try:
        import pkg_resources
        installed_packages = [d.project_name + '==' + d.version 
                            for d in pkg_resources.working_set]
        relevant_packages = [p for p in installed_packages 
                           if any(lib in p.lower() for lib in 
                                ['quantum', 'dwave', 'dimod', 'neal', 'numpy', 'sklearn'])]
        
        if relevant_packages:
            logger.info("Relevant packages:")
            for package in sorted(relevant_packages):
                logger.info(f"  {package}")
    except ImportError:
        logger.info("Package information not available")


def log_optimization_start(
    search_space: dict,
    n_iterations: int,
    backend: str,
    dataset_info: dict
) -> None:
    """
    Log optimization start information.
    
    Args:
        search_space: Hyperparameter search space
        n_iterations: Number of iterations
        backend: Backend name
        dataset_info: Dataset information
    """
    logger = get_logger('optimization')
    
    logger.info("="*60)
    logger.info("QUANTUM HYPERPARAMETER OPTIMIZATION STARTED")
    logger.info("="*60)
    
    logger.info(f"Backend: {backend}")
    logger.info(f"Iterations: {n_iterations}")
    
    logger.info("Search Space:")
    total_combinations = 1
    for param, values in search_space.items():
        logger.info(f"  {param}: {len(values)} options")
        total_combinations *= len(values)
    logger.info(f"  Total combinations: {total_combinations:,}")
    
    logger.info("Dataset:")
    for key, value in dataset_info.items():
        logger.info(f"  {key}: {value}")


def log_optimization_result(
    best_params: dict,
    best_score: float,
    history: any,
    execution_time: float
) -> None:
    """
    Log optimization results.
    
    Args:
        best_params: Best parameters found
        best_score: Best score achieved
        history: Optimization history
        execution_time: Total execution time in seconds
    """
    logger = get_logger('optimization')
    
    logger.info("="*60)
    logger.info("QUANTUM HYPERPARAMETER OPTIMIZATION COMPLETED")
    logger.info("="*60)
    
    logger.info(f"Execution time: {execution_time:.2f} seconds")
    logger.info(f"Best score: {best_score:.6f}")
    logger.info(f"Total trials: {len(history.trials)}")
    
    # Calculate improvement
    if len(history.scores) > 1:
        improvement = (best_score - min(history.scores)) / abs(min(history.scores)) * 100
        logger.info(f"Improvement over worst trial: {improvement:.2f}%")
    
    logger.info("Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    
    # Convergence analysis
    best_iteration = history.scores.index(best_score) + 1
    logger.info(f"Best result found at iteration: {best_iteration}")
    
    if best_iteration < len(history.scores):
        logger.info(f"No improvement in last {len(history.scores) - best_iteration} iterations")
    
    logger.info("="*60)