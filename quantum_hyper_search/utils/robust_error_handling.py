"""
Robust Error Handling - Comprehensive error handling and recovery for quantum optimization.

Provides resilient error handling, automatic recovery, and graceful degradation
for quantum hyperparameter optimization systems.
"""

import logging
import time
import traceback
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    max_retry_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'component': self.component,
            'operation': self.operation,
            'parameters': self.parameters,
            'stack_trace': self.stack_trace,
            'recovery_attempts': self.recovery_attempts,
            'max_retry_attempts': self.max_retry_attempts
        }


@dataclass
class RecoveryAction:
    """Defines a recovery action for specific errors."""
    strategy: RecoveryStrategy
    handler: Callable
    conditions: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    backoff_factor: float = 1.5
    timeout: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker pattern for handling failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED")
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                
                raise e


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, max_error_history: int = 1000):
        self.error_history = deque(maxlen=max_error_history)
        self.recovery_handlers = defaultdict(list)
        self.circuit_breakers = {}
        self.error_patterns = defaultdict(int)
        self._lock = threading.Lock()
        
        # Default recovery handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default error recovery handlers."""
        
        # Quantum backend failures
        self.register_recovery_handler(
            "QuantumBackendError",
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                handler=self._fallback_to_classical,
                max_attempts=1
            )
        )
        
        # Memory errors
        self.register_recovery_handler(
            "MemoryError",
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                handler=self._reduce_batch_size,
                max_attempts=3
            )
        )
        
        # Network timeouts
        self.register_recovery_handler(
            "TimeoutError",
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                handler=self._exponential_backoff,
                max_attempts=5,
                backoff_factor=2.0
            )
        )
        
        # Validation errors
        self.register_recovery_handler(
            "ValidationError",
            RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                handler=self._log_and_skip,
                max_attempts=1
            )
        )
    
    def register_recovery_handler(self, error_type: str, action: RecoveryAction):
        """Register a recovery handler for specific error types."""
        self.recovery_handlers[error_type].append(action)
        logger.info(f"Registered recovery handler for {error_type}: {action.strategy.value}")
    
    def handle_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle an error with appropriate recovery strategy."""
        with self._lock:
            self.error_history.append(context)
            self._update_error_patterns(context)
        
        error_type = type(error).__name__
        logger.error(f"Handling {error_type}: {str(error)}")
        
        # Check for circuit breaker
        if error_type in self.circuit_breakers:
            breaker = self.circuit_breakers[error_type]
            if breaker.state == 'OPEN':
                raise Exception(f"Circuit breaker open for {error_type}")
        
        # Try recovery handlers
        for action in self.recovery_handlers.get(error_type, []):
            if context.recovery_attempts < action.max_attempts:
                try:
                    logger.info(f"Attempting recovery with {action.strategy.value}")
                    context.recovery_attempts += 1
                    
                    if action.strategy == RecoveryStrategy.RETRY:
                        time.sleep(action.backoff_factor ** context.recovery_attempts)
                    
                    result = action.handler(error, context)
                    logger.info(f"Recovery successful with {action.strategy.value}")
                    return result
                    
                except Exception as recovery_error:
                    logger.warning(f"Recovery attempt failed: {recovery_error}")
                    continue
        
        # If all recovery attempts failed
        logger.error(f"All recovery attempts failed for {error_type}")
        self._escalate_error(error, context)
        raise error
    
    def _fallback_to_classical(self, error: Exception, context: ErrorContext) -> Any:
        """Fallback to classical optimization when quantum backend fails."""
        logger.info("Falling back to classical optimization")
        
        # Switch to classical backend
        if 'optimizer' in context.parameters:
            optimizer = context.parameters['optimizer']
            if hasattr(optimizer, 'backend'):
                optimizer.backend = optimizer._get_fallback_backend()
                return "fallback_applied"
        
        return None
    
    def _reduce_batch_size(self, error: Exception, context: ErrorContext) -> Any:
        """Reduce batch size when memory errors occur."""
        logger.info("Reducing batch size due to memory constraints")
        
        if 'batch_size' in context.parameters:
            original_size = context.parameters['batch_size']
            new_size = max(1, original_size // 2)
            context.parameters['batch_size'] = new_size
            logger.info(f"Reduced batch size from {original_size} to {new_size}")
            return "batch_size_reduced"
        
        return None
    
    def _exponential_backoff(self, error: Exception, context: ErrorContext) -> Any:
        """Apply exponential backoff for retry operations."""
        backoff_time = (2 ** context.recovery_attempts) * 0.1
        logger.info(f"Applying exponential backoff: {backoff_time:.2f}s")
        time.sleep(backoff_time)
        return "backoff_applied"
    
    def _log_and_skip(self, error: Exception, context: ErrorContext) -> Any:
        """Log error and skip the current operation."""
        logger.warning(f"Skipping operation due to {type(error).__name__}: {str(error)}")
        return "operation_skipped"
    
    def _update_error_patterns(self, context: ErrorContext):
        """Update error pattern tracking."""
        pattern_key = f"{context.component}:{context.error_type}"
        self.error_patterns[pattern_key] += 1
        
        # Check for concerning patterns
        if self.error_patterns[pattern_key] > 10:
            logger.warning(f"High error frequency detected: {pattern_key}")
    
    def _escalate_error(self, error: Exception, context: ErrorContext):
        """Escalate critical errors."""
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {context.error_message}")
            # Could trigger alerts, notifications, etc.
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_history:
            return {"total_errors": 0, "patterns": {}}
        
        recent_errors = list(self.error_history)
        error_types = defaultdict(int)
        severities = defaultdict(int)
        components = defaultdict(int)
        
        for error_ctx in recent_errors:
            error_types[error_ctx.error_type] += 1
            severities[error_ctx.severity.value] += 1
            components[error_ctx.component] += 1
        
        return {
            "total_errors": len(recent_errors),
            "error_types": dict(error_types),
            "severities": dict(severities),
            "components": dict(components),
            "patterns": dict(self.error_patterns)
        }
    
    def create_circuit_breaker(self, error_type: str, failure_threshold: int = 5, 
                              recovery_timeout: float = 60.0):
        """Create a circuit breaker for specific error types."""
        self.circuit_breakers[error_type] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        logger.info(f"Created circuit breaker for {error_type}")


def robust_operation(component: str = "unknown", operation: str = "unknown", 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    max_retry_attempts: int = 3):
    """Decorator for robust error handling of operations."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = getattr(wrapper, '_error_handler', None)
            if error_handler is None:
                error_handler = RobustErrorHandler()
                wrapper._error_handler = error_handler
            
            for attempt in range(max_retry_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    context = ErrorContext(
                        timestamp=time.time(),
                        error_type=type(e).__name__,
                        error_message=str(e),
                        severity=severity,
                        component=component,
                        operation=operation,
                        parameters={'args': args, 'kwargs': kwargs},
                        stack_trace=traceback.format_exc(),
                        recovery_attempts=attempt,
                        max_retry_attempts=max_retry_attempts
                    )
                    
                    if attempt < max_retry_attempts:
                        try:
                            recovery_result = error_handler.handle_error(e, context)
                            if recovery_result:
                                logger.info(f"Recovery successful, retrying operation")
                                continue
                        except Exception:
                            pass
                    
                    # Final attempt failed
                    logger.error(f"Operation failed after {max_retry_attempts} attempts")
                    raise e
        
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = RobustErrorHandler()


def handle_quantum_errors(func: Callable) -> Callable:
    """Decorator specifically for quantum operations."""
    return robust_operation(
        component="quantum_backend",
        operation=func.__name__,
        severity=ErrorSeverity.HIGH,
        max_retry_attempts=3
    )(func)


def handle_optimization_errors(func: Callable) -> Callable:
    """Decorator specifically for optimization operations."""
    return robust_operation(
        component="optimization",
        operation=func.__name__,
        severity=ErrorSeverity.MEDIUM,
        max_retry_attempts=5
    )(func)


def handle_validation_errors(func: Callable) -> Callable:
    """Decorator specifically for validation operations."""
    return robust_operation(
        component="validation",
        operation=func.__name__,
        severity=ErrorSeverity.LOW,
        max_retry_attempts=1
    )(func)