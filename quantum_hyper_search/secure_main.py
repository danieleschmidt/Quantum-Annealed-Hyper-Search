"""
Secure Quantum Hyperparameter Search - Enterprise Security Integration

Production-ready quantum hyperparameter optimization with comprehensive
enterprise security, compliance, and monitoring capabilities.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

# Core quantum optimization
from .optimized_main import QuantumHyperSearchOptimized
from .core.base import QuantumBackend

# Security framework
from .security.quantum_security_framework import (
    QuantumSecurityFramework,
    SecurityPolicy,
    SecurityEvent
)
from .security.authentication import AuthenticationManager
from .security.authorization import AuthorizationManager, Permission
from .security.encryption import EncryptionManager, DataProtection
from .security.compliance import ComplianceManager, ComplianceFramework

# Monitoring and utilities
from .utils.comprehensive_monitoring import ComprehensiveMonitoring
from .utils.robust_error_handling import RobustErrorHandler


logger = logging.getLogger(__name__)


@dataclass
class SecureOptimizationConfig:
    """Configuration for secure quantum optimization."""
    # Security settings
    enable_security_framework: bool = True
    compliance_frameworks: List[str] = field(default_factory=lambda: ['standard'])
    require_authentication: bool = True
    require_authorization: bool = True
    enable_encryption: bool = True
    audit_all_operations: bool = True
    
    # Performance settings
    enable_monitoring: bool = True
    enable_caching: bool = True
    max_optimization_time: float = 3600.0  # 1 hour
    
    # Quantum settings
    quantum_backend: str = 'simulator'
    enable_quantum_advantage: bool = False
    
    # Data protection
    data_classification: str = 'internal'
    encrypt_results: bool = True
    secure_parameter_handling: bool = True


class SecureQuantumHyperSearch:
    """
    Enterprise-grade secure quantum hyperparameter optimization.
    
    Combines quantum optimization capabilities with comprehensive security,
    compliance, monitoring, and enterprise features.
    """
    
    def __init__(self, config: Optional[SecureOptimizationConfig] = None):
        """Initialize secure quantum hyperparameter search."""
        self.config = config or SecureOptimizationConfig()
        
        # Initialize security framework
        if self.config.enable_security_framework:
            self._initialize_security()
        
        # Initialize core optimization engine
        self._initialize_optimization_engine()
        
        # Initialize monitoring
        if self.config.enable_monitoring:
            self._initialize_monitoring()
        
        # Initialize error handling
        self.error_handler = RobustErrorHandler({
            'max_retries': 3,
            'backoff_factor': 2.0,
            'enable_circuit_breaker': True
        })
        
        # Session management
        self.current_session = None
        self.current_user = None
        
        logger.info("Secure Quantum Hyperparameter Search initialized")
    
    def _initialize_security(self):
        """Initialize comprehensive security framework."""
        # Security policy
        security_policy = SecurityPolicy(
            compliance_mode=self.config.compliance_frameworks[0] if self.config.compliance_frameworks else 'standard',
            require_mfa=True,
            require_encryption_at_rest=self.config.enable_encryption,
            audit_all_actions=self.config.audit_all_operations,
            max_session_duration=3600
        )
        
        # Main security framework
        self.security_framework = QuantumSecurityFramework(
            policy=security_policy,
            audit_log_path="quantum_security_audit.log"
        )
        
        # Authentication manager
        auth_config = {
            'min_password_length': 12,
            'require_mfa': True,
            'max_failed_attempts': 5,
            'lockout_duration': 900
        }
        self.auth_manager = AuthenticationManager(auth_config)
        
        # Authorization manager
        authz_config = {
            'rbac_enabled': True,
            'default_permissions': ['quantum_view_results']
        }
        self.authz_manager = AuthorizationManager(authz_config)
        
        # Initialize default roles
        self._setup_default_roles()
        
        # Encryption and data protection
        encryption_config = {
            'default_algorithm': 'AES-256-GCM',
            'key_rotation_interval': 86400,
            'quantum_safe_mode': False
        }
        self.encryption_manager = EncryptionManager(encryption_config)
        self.data_protection = DataProtection(encryption_config)
        
        # Compliance management
        compliance_config = {
            'enabled_frameworks': self.config.compliance_frameworks,
            'audit_logging_enabled': True,
            'encryption_in_transit': True,
            'access_controls_implemented': True
        }
        self.compliance_manager = ComplianceManager(compliance_config)
        
        logger.info("Security framework initialized")
    
    def _setup_default_roles(self):
        """Setup default authorization roles."""
        # Assign researcher role to admin user
        self.authz_manager.assign_role('user_admin', 'quantum_researcher')
        
        # Create custom optimization role
        from .security.authorization import Permission
        optimization_permissions = [
            Permission.QUANTUM_OPTIMIZE,
            Permission.QUANTUM_VIEW_RESULTS,
            Permission.QUANTUM_EXPORT_DATA,
            Permission.DATA_READ,
            Permission.DATA_WRITE
        ]
        
        self.authz_manager.create_role(
            'quantum_optimizer',
            optimization_permissions,
            'Custom role for quantum optimization tasks'
        )
    
    def _initialize_optimization_engine(self):
        """Initialize quantum optimization engine."""
        # Use the optimized version as base
        optimization_config = {
            'backend': self.config.quantum_backend,
            'enable_caching': self.config.enable_caching,
            'enable_monitoring': self.config.enable_monitoring,
            'enable_quantum_advantage': self.config.enable_quantum_advantage
        }
        
        self.quantum_optimizer = QuantumHyperSearchOptimized(
            optimization_config
        )
        
        logger.info("Quantum optimization engine initialized")
    
    def _initialize_monitoring(self):
        """Initialize comprehensive monitoring."""
        monitoring_config = {
            'enable_performance_tracking': True,
            'enable_security_monitoring': True,
            'enable_compliance_monitoring': True,
            'alert_thresholds': {
                'optimization_time': self.config.max_optimization_time,
                'error_rate': 0.05,
                'security_events': 10
            }
        }
        
        self.monitoring = ComprehensiveMonitoring(monitoring_config)
        
        logger.info("Monitoring system initialized")
    
    def authenticate(self, username: str, password: str, 
                    mfa_token: Optional[str] = None) -> Dict[str, Any]:
        """Authenticate user with security framework."""
        if not self.config.enable_security_framework:
            # Skip authentication if security is disabled
            return {
                'success': True,
                'session_token': 'mock_session',
                'user_id': f'user_{username}'
            }
        
        try:
            # Use security framework for authentication
            session_token = self.security_framework.authenticate_user(
                username=username,
                password=password,
                mfa_token=mfa_token,
                metadata={'login_time': time.time()}
            )
            
            if session_token:
                self.current_session = session_token
                self.current_user = f'user_{username}'
                
                return {
                    'success': True,
                    'session_token': session_token,
                    'user_id': self.current_user,
                    'requires_mfa': False
                }
            else:
                return {
                    'success': False,
                    'reason': 'authentication_failed',
                    'requires_mfa': self.config.require_authentication
                }
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                'success': False,
                'reason': 'authentication_error',
                'error': str(e)
            }
    
    def authorize_optimization(self, session_token: str, 
                             optimization_params: Dict[str, Any]) -> bool:
        """Authorize optimization operation."""
        if not self.config.require_authorization:
            return True
        
        return self.security_framework.authorize_action(
            session_token=session_token,
            action='optimize',
            resource='hyperparameters',
            context=optimization_params
        )
    
    def secure_optimize(self, 
                       session_token: str,
                       objective_function: Callable,
                       parameter_space: Dict[str, Any],
                       max_iterations: int = 50,
                       optimization_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform secure quantum hyperparameter optimization.
        
        Args:
            session_token: User session token for authorization
            objective_function: Function to optimize
            parameter_space: Parameter search space
            max_iterations: Maximum optimization iterations
            optimization_config: Additional optimization configuration
        
        Returns:
            Dictionary containing optimization results and security metadata
        """
        start_time = time.time()
        operation_id = f"opt_{int(start_time)}_{hash(str(parameter_space)) % 10000}"
        
        try:
            # Validate session
            if self.config.require_authentication:
                session_info = self.security_framework.session_manager.validate_session(session_token)
                if not session_info:
                    raise PermissionError("Invalid or expired session")
                
                user_id = session_info['user_id']
            else:
                user_id = 'anonymous'
            
            # Authorize operation
            if not self.authorize_optimization(session_token, parameter_space):
                raise PermissionError("Insufficient permissions for optimization")
            
            # Sanitize and validate parameters
            sanitized_params = self.security_framework.validate_and_sanitize_input(
                parameter_space, 
                context='optimization'
            )
            
            # Encrypt sensitive parameters if required
            if self.config.secure_parameter_handling:
                encrypted_params = self.security_framework.encrypt_sensitive_data(
                    sanitized_params,
                    classification=self.config.data_classification
                )
            else:
                encrypted_params = {'encrypted_data': sanitized_params, 'metadata': {'encrypted': False}}
            
            # Start monitoring
            if self.config.enable_monitoring:
                self.monitoring.start_operation_tracking(operation_id, 'optimization')
            
            # Perform optimization with security context
            with self.security_framework.secure_operation('quantum_optimization', user_id):
                optimization_result = self._run_secure_optimization(
                    objective_function=objective_function,
                    parameter_space=sanitized_params,
                    max_iterations=max_iterations,
                    config=optimization_config or {},
                    operation_id=operation_id
                )
            
            # Encrypt results if required
            if self.config.encrypt_results:
                encrypted_results = self.security_framework.encrypt_sensitive_data(
                    optimization_result,
                    classification=self.config.data_classification
                )
            else:
                encrypted_results = optimization_result
            
            # Create secure response
            secure_response = {
                'operation_id': operation_id,
                'success': True,
                'results': encrypted_results,
                'security_metadata': {
                    'user_id': user_id,
                    'operation_time': time.time() - start_time,
                    'parameters_encrypted': self.config.secure_parameter_handling,
                    'results_encrypted': self.config.encrypt_results,
                    'compliance_frameworks': self.config.compliance_frameworks
                },
                'monitoring_data': self.monitoring.get_operation_summary(operation_id) if self.config.enable_monitoring else None
            }
            
            # Log successful operation
            self.security_framework._log_event(
                'optimization_completed',
                user_id,
                'success',
                details={'operation_id': operation_id, 'duration': time.time() - start_time}
            )
            
            return secure_response
            
        except PermissionError as e:
            # Log security violation
            self.security_framework._log_event(
                'optimization_denied',
                user_id if 'user_id' in locals() else None,
                'denied',
                details={'operation_id': operation_id, 'reason': str(e)},
                risk_level='high'
            )
            
            return {
                'operation_id': operation_id,
                'success': False,
                'error': 'Permission denied',
                'error_type': 'security',
                'details': str(e)
            }
            
        except Exception as e:
            # Log general error
            logger.error(f"Optimization error: {e}")
            
            if hasattr(self, 'security_framework'):
                self.security_framework._log_event(
                    'optimization_failed',
                    user_id if 'user_id' in locals() else None,
                    'failure',
                    details={'operation_id': operation_id, 'error': str(e)},
                    risk_level='medium'
                )
            
            return {
                'operation_id': operation_id,
                'success': False,
                'error': str(e),
                'error_type': 'optimization',
                'duration': time.time() - start_time
            }
    
    def _run_secure_optimization(self,
                               objective_function: Callable,
                               parameter_space: Dict[str, Any],
                               max_iterations: int,
                               config: Dict[str, Any],
                               operation_id: str) -> Dict[str, Any]:
        """Run optimization with error handling and monitoring."""
        try:
            # Use the underlying quantum optimizer
            result = self.quantum_optimizer.optimize(
                objective_function=objective_function,
                parameter_space=parameter_space,
                max_iterations=max_iterations,
                **config
            )
            
            # Add security metadata
            result['security_validated'] = True
            result['operation_id'] = operation_id
            
            return result
            
        except Exception as e:
            # Use error handler for retries
            return self.error_handler.handle_optimization_error(
                error=e,
                context={
                    'operation_id': operation_id,
                    'parameter_space': parameter_space,
                    'max_iterations': max_iterations
                }
            )
    
    def get_security_report(self, session_token: str) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        if not self.config.enable_security_framework:
            return {'security_enabled': False}
        
        # Authorize report access
        if not self.security_framework.authorize_action(
            session_token, 'view', 'security_report'
        ):
            raise PermissionError("Insufficient permissions for security report")
        
        # Generate comprehensive report
        security_report = self.security_framework.generate_security_report()
        
        # Add compliance assessment
        if hasattr(self, 'compliance_manager'):
            compliance_report = self.compliance_manager.run_comprehensive_assessment()
            security_report['compliance_assessment'] = compliance_report
        
        return security_report
    
    def export_secure_data(self, session_token: str, data_type: str, 
                          filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Export data with security controls."""
        # Authorize export
        if not self.security_framework.authorize_action(
            session_token, 'export', data_type
        ):
            raise PermissionError("Insufficient permissions for data export")
        
        # Apply data protection
        exported_data = self._get_export_data(data_type, filters)
        
        # Encrypt for export
        if self.config.enable_encryption:
            protected_data = self.data_protection.protect_data(
                exported_data,
                context={'export': True, 'data_type': data_type}
            )
        else:
            protected_data = exported_data
        
        return {
            'data': protected_data,
            'export_metadata': {
                'timestamp': time.time(),
                'data_type': data_type,
                'filters_applied': filters,
                'protection_applied': self.config.enable_encryption
            }
        }
    
    def _get_export_data(self, data_type: str, filters: Optional[Dict]) -> Any:
        """Get data for export based on type and filters."""
        if data_type == 'optimization_history':
            return self.quantum_optimizer.get_optimization_history()
        elif data_type == 'security_audit':
            return self.security_framework.audit_logger.export_logs()
        elif data_type == 'monitoring_metrics':
            return self.monitoring.get_comprehensive_metrics() if hasattr(self, 'monitoring') else {}
        else:
            return {'message': f'Unknown data type: {data_type}'}
    
    @contextmanager
    def secure_session(self, username: str, password: str, 
                      mfa_token: Optional[str] = None):
        """Context manager for secure operations."""
        # Authenticate
        auth_result = self.authenticate(username, password, mfa_token)
        
        if not auth_result['success']:
            raise PermissionError(f"Authentication failed: {auth_result.get('reason', 'unknown')}")
        
        session_token = auth_result['session_token']
        
        try:
            yield session_token
        finally:
            # Cleanup session
            if self.config.enable_security_framework:
                self.security_framework.session_manager.revoke_session(session_token)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': time.time(),
            'security_enabled': self.config.enable_security_framework,
            'quantum_backend': self.config.quantum_backend,
            'compliance_frameworks': self.config.compliance_frameworks
        }
        
        if self.config.enable_security_framework:
            status['security_status'] = {
                'active_sessions': len(self.security_framework.session_manager.active_sessions),
                'audit_events': len(self.security_framework.audit_logger.events),
                'encryption_enabled': self.config.enable_encryption
            }
        
        if self.config.enable_monitoring:
            status['monitoring_status'] = self.monitoring.get_system_health()
        
        return status
    
    def shutdown(self):
        """Graceful shutdown with security cleanup."""
        logger.info("Initiating secure shutdown...")
        
        try:
            # Security framework cleanup
            if self.config.enable_security_framework:
                self.security_framework.shutdown()
            
            # Monitoring cleanup
            if self.config.enable_monitoring:
                self.monitoring.shutdown()
            
            # Quantum optimizer cleanup
            if hasattr(self.quantum_optimizer, 'shutdown'):
                self.quantum_optimizer.shutdown()
            
            logger.info("Secure shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Convenience functions for common use cases
def create_secure_optimizer(compliance_mode: str = 'standard',
                          enable_quantum: bool = False) -> SecureQuantumHyperSearch:
    """Create secure optimizer with common configuration."""
    config = SecureOptimizationConfig(
        compliance_frameworks=[compliance_mode],
        enable_quantum_advantage=enable_quantum,
        quantum_backend='simulator' if not enable_quantum else 'quantum',
        enable_security_framework=True,
        require_authentication=True,
        enable_encryption=True
    )
    
    return SecureQuantumHyperSearch(config)


def quick_secure_optimization(objective_function: Callable,
                            parameter_space: Dict[str, Any],
                            username: str,
                            password: str,
                            max_iterations: int = 50) -> Dict[str, Any]:
    """Quick secure optimization with automatic authentication."""
    with create_secure_optimizer() as optimizer:
        with optimizer.secure_session(username, password) as session_token:
            return optimizer.secure_optimize(
                session_token=session_token,
                objective_function=objective_function,
                parameter_space=parameter_space,
                max_iterations=max_iterations
            )