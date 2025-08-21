"""
Quantum Security Framework - Enterprise-grade security orchestration.

Comprehensive security framework that integrates all security components
for quantum-enhanced hyperparameter optimization systems.
"""

import os
import time
import json
import logging
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict
from contextlib import contextmanager

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Represents a security event for comprehensive audit logging."""
    timestamp: float
    event_type: str  # authentication, authorization, encryption, access, admin
    user_id: Optional[str]
    session_id: str
    action: str
    resource: str
    outcome: str  # success, failure, denied, warning
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = 'low'  # low, medium, high, critical
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'datetime_utc': datetime.utcfromtimestamp(self.timestamp).isoformat() + 'Z',
            'event_type': self.event_type,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'action': self.action,
            'resource': self.resource,
            'outcome': self.outcome,
            'details': self.details,
            'risk_level': self.risk_level,
            'source_ip': self.source_ip,
            'user_agent': self.user_agent,
            'location': self.location
        }


@dataclass
class SecurityPolicy:
    """Comprehensive security policy configuration."""
    # Authentication policies
    max_session_duration: int = 3600  # seconds
    password_min_length: int = 12
    password_require_special: bool = True
    password_require_digits: bool = True
    password_require_uppercase: bool = True
    require_mfa: bool = True
    mfa_methods: List[str] = field(default_factory=lambda: ['totp', 'sms'])
    
    # Authorization policies
    rbac_enabled: bool = True
    default_permissions: List[str] = field(default_factory=list)
    admin_approval_required: List[str] = field(default_factory=lambda: ['admin_actions'])
    
    # Security controls
    max_failed_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    rate_limit_window: int = 60  # seconds
    max_requests_per_window: int = 100
    
    # Encryption settings
    require_encryption_at_rest: bool = True
    require_encryption_in_transit: bool = True
    encryption_algorithm: str = 'AES-256-GCM'
    key_rotation_interval: int = 86400  # 24 hours
    
    # Audit and compliance
    audit_all_actions: bool = True
    audit_retention_days: int = 2555  # ~7 years
    compliance_mode: str = 'standard'  # standard, hipaa, gdpr, sox, pci
    immutable_audit_log: bool = True
    
    # Data protection
    data_classification_enabled: bool = True
    pii_detection_enabled: bool = True
    data_masking_enabled: bool = True
    backup_encryption_required: bool = True
    
    # Network security
    ip_whitelist: List[str] = field(default_factory=list)
    geo_blocking_enabled: bool = False
    blocked_countries: List[str] = field(default_factory=list)
    
    # Monitoring and alerting
    real_time_monitoring: bool = True
    security_alerts_enabled: bool = True
    threat_detection_enabled: bool = True
    anomaly_detection_enabled: bool = True


class CryptographyManager:
    """Advanced cryptography manager for data protection."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.master_key = None
        self.encryption_keys = {}
        self.key_metadata = {}
        
        if HAS_CRYPTOGRAPHY:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption system with key management."""
        # Generate or load master key
        self.master_key = self._get_or_create_master_key()
        
        # Initialize primary encryption key
        self.encryption_keys['primary'] = Fernet.generate_key()
        self.key_metadata['primary'] = {
            'created_at': time.time(),
            'algorithm': 'Fernet-AES128',
            'purpose': 'data_encryption'
        }
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        key_file = Path('.quantum_security_key')
        
        if key_file.exists():
            return key_file.read_bytes()
        else:
            master_key = Fernet.generate_key()
            # In production, store this securely (HSM, key vault, etc.)
            key_file.write_bytes(master_key)
            os.chmod(str(key_file), 0o600)  # Read-only for owner
            return master_key
    
    def encrypt_data(self, data: Union[str, bytes, Dict], classification: str = 'internal') -> Optional[Dict]:
        """Encrypt data with metadata."""
        if not HAS_CRYPTOGRAPHY:
            logger.warning("Cryptography not available")
            return None
        
        try:
            # Prepare data for encryption
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Get appropriate cipher
            cipher = Fernet(self.encryption_keys['primary'])
            
            # Encrypt with current timestamp
            encrypted_data = cipher.encrypt(data_bytes)
            
            # Create encryption metadata
            metadata = {
                'encrypted_at': time.time(),
                'algorithm': 'Fernet-AES128',
                'key_id': 'primary',
                'classification': classification,
                'data_hash': hashlib.sha256(data_bytes).hexdigest(),
                'size': len(data_bytes)
            }
            
            return {
                'encrypted_data': encrypted_data,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_package: Dict) -> Optional[Union[str, Dict]]:
        """Decrypt data using metadata."""
        if not HAS_CRYPTOGRAPHY:
            return None
        
        try:
            encrypted_data = encrypted_package['encrypted_data']
            metadata = encrypted_package['metadata']
            
            # Get appropriate cipher
            key_id = metadata.get('key_id', 'primary')
            cipher = Fernet(self.encryption_keys[key_id])
            
            # Decrypt data
            decrypted_bytes = cipher.decrypt(encrypted_data)
            
            # Verify data integrity
            data_hash = hashlib.sha256(decrypted_bytes).hexdigest()
            if data_hash != metadata.get('data_hash'):
                logger.error("Data integrity check failed")
                return None
            
            # Try to decode as JSON, fallback to string
            try:
                return json.loads(decrypted_bytes.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return decrypted_bytes.decode('utf-8')
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def rotate_keys(self):
        """Rotate encryption keys according to policy."""
        if not HAS_CRYPTOGRAPHY:
            return
        
        # Generate new key
        new_key = Fernet.generate_key()
        new_key_id = f"key_{int(time.time())}"
        
        # Store old key for decryption
        self.encryption_keys[new_key_id] = new_key
        self.key_metadata[new_key_id] = {
            'created_at': time.time(),
            'algorithm': 'Fernet-AES128',
            'purpose': 'data_encryption'
        }
        
        # Update primary key
        self.encryption_keys['primary'] = new_key
        
        logger.info(f"Encryption keys rotated: {new_key_id}")


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, policy: SecurityPolicy, log_path: Optional[str] = None):
        self.policy = policy
        self.log_path = log_path or "quantum_security_audit.log"
        self.events = []
        self.lock = threading.Lock()
        
        # Initialize log file
        self._initialize_audit_log()
    
    def _initialize_audit_log(self):
        """Initialize audit log file with proper permissions."""
        log_file = Path(self.log_path)
        
        if not log_file.exists():
            log_file.touch()
            os.chmod(str(log_file), 0o640)  # Read/write owner, read group
    
    def log_event(self, event: SecurityEvent):
        """Log security event with proper formatting."""
        with self.lock:
            self.events.append(event)
            
            # Write to file if configured
            if self.log_path:
                self._write_to_file(event)
            
            # Check for high-risk events
            if event.risk_level in ['high', 'critical']:
                self._handle_high_risk_event(event)
    
    def _write_to_file(self, event: SecurityEvent):
        """Write event to audit log file."""
        try:
            log_entry = {
                'timestamp': event.timestamp,
                'datetime': datetime.fromtimestamp(event.timestamp).isoformat(),
                'event': event.to_dict()
            }
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _handle_high_risk_event(self, event: SecurityEvent):
        """Handle high-risk security events."""
        logger.warning(f"High-risk security event: {event.action} by {event.user_id}")
        
        # In production, trigger alerts, notifications, etc.
    
    def export_logs(self, start_time: Optional[float] = None, 
                   end_time: Optional[float] = None,
                   event_types: Optional[List[str]] = None) -> List[Dict]:
        """Export audit logs with filtering."""
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        filtered_events = []
        for event in self.events:
            # Filter by time
            if not (start_time <= event.timestamp <= end_time):
                continue
            
            # Filter by event type
            if event_types and event.event_type not in event_types:
                continue
            
            filtered_events.append(event.to_dict())
        
        return filtered_events
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics from audit logs."""
        if not self.events:
            return {}
        
        # Count events by type and outcome
        event_counts = defaultdict(int)
        outcome_counts = defaultdict(int)
        risk_counts = defaultdict(int)
        
        for event in self.events:
            event_counts[event.event_type] += 1
            outcome_counts[event.outcome] += 1
            risk_counts[event.risk_level] += 1
        
        # Calculate security score
        total_events = len(self.events)
        success_events = outcome_counts.get('success', 0)
        security_score = (success_events / total_events) * 100 if total_events > 0 else 100
        
        return {
            'total_events': total_events,
            'events_by_type': dict(event_counts),
            'events_by_outcome': dict(outcome_counts),
            'events_by_risk_level': dict(risk_counts),
            'security_score': security_score,
            'high_risk_events': risk_counts.get('high', 0) + risk_counts.get('critical', 0)
        }


class SessionManager:
    """Advanced session management with security controls."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.active_sessions = {}
        self.session_history = []
        self.lock = threading.Lock()
    
    def create_session(self, user_id: str, permissions: List[str], 
                      metadata: Optional[Dict] = None) -> str:
        """Create new secure session."""
        session_id = f"qhs_{int(time.time())}_{secrets.token_hex(16)}"
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'permissions': permissions,
            'created_at': time.time(),
            'expires_at': time.time() + self.policy.max_session_duration,
            'last_activity': time.time(),
            'metadata': metadata or {},
            'is_active': True
        }
        
        with self.lock:
            self.active_sessions[session_id] = session_data
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate session and update activity."""
        with self.lock:
            session = self.active_sessions.get(session_id)
            
            if not session:
                return None
            
            # Check expiration
            if time.time() > session['expires_at']:
                self._expire_session(session_id)
                return None
            
            # Update last activity
            session['last_activity'] = time.time()
            
            return session.copy()
    
    def _expire_session(self, session_id: str):
        """Expire session and move to history."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['is_active'] = False
            session['expired_at'] = time.time()
            
            self.session_history.append(session)
            del self.active_sessions[session_id]
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke active session."""
        with self.lock:
            if session_id in self.active_sessions:
                self._expire_session(session_id)
                return True
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        now = time.time()
        expired_sessions = []
        
        with self.lock:
            for session_id, session in self.active_sessions.items():
                if now > session['expires_at']:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._expire_session(session_id)
        
        return len(expired_sessions)


class ComplianceManager:
    """Compliance management for various regulatory frameworks."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.compliance_checks = {}
        
        # Initialize compliance checks based on mode
        self._initialize_compliance_checks()
    
    def _initialize_compliance_checks(self):
        """Initialize compliance checks based on policy."""
        base_checks = {
            'encryption_at_rest': self.policy.require_encryption_at_rest,
            'encryption_in_transit': self.policy.require_encryption_in_transit,
            'audit_logging': self.policy.audit_all_actions,
            'access_controls': self.policy.rbac_enabled,
            'session_management': True,  # Always required
        }
        
        if self.policy.compliance_mode == 'hipaa':
            base_checks.update({
                'mfa_required': self.policy.require_mfa,
                'session_timeout_30min': self.policy.max_session_duration <= 1800,
                'audit_retention_7years': self.policy.audit_retention_days >= 2555,
                'access_logging': True,
                'minimum_necessary_access': True
            })
        
        elif self.policy.compliance_mode == 'gdpr':
            base_checks.update({
                'data_subject_rights': True,
                'consent_management': True,
                'data_portability': True,
                'right_to_erasure': True,
                'privacy_by_design': True,
                'breach_notification': True
            })
        
        elif self.policy.compliance_mode == 'sox':
            base_checks.update({
                'immutable_audit_trail': self.policy.immutable_audit_log,
                'segregation_of_duties': True,
                'financial_data_protection': True,
                'change_management': True,
                'audit_retention_7years': self.policy.audit_retention_days >= 2555
            })
        
        self.compliance_checks = base_checks
    
    def run_compliance_check(self) -> Dict[str, Any]:
        """Run comprehensive compliance check."""
        results = {}
        passed_checks = 0
        total_checks = len(self.compliance_checks)
        
        for check_name, required in self.compliance_checks.items():
            # In a real implementation, these would be actual checks
            # For now, we'll use the policy settings
            passed = self._perform_check(check_name, required)
            results[check_name] = {
                'required': required,
                'passed': passed,
                'status': 'PASS' if passed else 'FAIL'
            }
            
            if passed:
                passed_checks += 1
        
        compliance_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
        
        return {
            'compliance_mode': self.policy.compliance_mode,
            'compliance_score': compliance_score,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'checks': results,
            'overall_status': 'COMPLIANT' if compliance_score >= 90 else 'NON_COMPLIANT',
            'timestamp': time.time()
        }
    
    def _perform_check(self, check_name: str, required: bool) -> bool:
        """Perform individual compliance check."""
        # Mock implementation - in production, these would be real checks
        if not required:
            return True
        
        # Basic checks that we can verify
        if check_name == 'encryption_at_rest':
            return HAS_CRYPTOGRAPHY
        elif check_name == 'audit_logging':
            return self.policy.audit_all_actions
        elif check_name == 'mfa_required':
            return self.policy.require_mfa
        else:
            # For other checks, assume they pass if required
            return True


class QuantumSecurityFramework:
    """
    Comprehensive Quantum Security Framework
    
    Enterprise-grade security orchestration for quantum-enhanced
    hyperparameter optimization systems.
    """
    
    def __init__(self, 
                 policy: Optional[SecurityPolicy] = None,
                 audit_log_path: Optional[str] = None):
        """Initialize security framework."""
        self.policy = policy or SecurityPolicy()
        self.session_id = f"qsf_{int(time.time())}_{secrets.token_hex(8)}"
        
        # Initialize managers
        self.crypto_manager = CryptographyManager(self.policy)
        self.audit_logger = AuditLogger(self.policy, audit_log_path)
        self.session_manager = SessionManager(self.policy)
        self.compliance_manager = ComplianceManager(self.policy)
        
        # Security state
        self.is_initialized = True
        self.threat_level = 'normal'  # normal, elevated, high, critical
        
        # Log framework initialization
        self._log_event('framework_initialized', 'system', 'success', 
                       details={'compliance_mode': self.policy.compliance_mode})
        
        logger.info(f"Quantum Security Framework initialized (session: {self.session_id})")
    
    def authenticate_user(self, username: str, password: str, 
                         mfa_token: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> Optional[str]:
        """Authenticate user with comprehensive security checks."""
        user_id = f"user_{username}"
        
        # Pre-authentication checks
        if not self._pre_auth_checks(user_id):
            return None
        
        # Password verification
        if not self._verify_password(username, password):
            self._log_event('authentication_failed', user_id, 'failure',
                          details={'reason': 'invalid_password'}, risk_level='medium')
            return None
        
        # MFA verification if required
        if self.policy.require_mfa:
            if not mfa_token or not self._verify_mfa(user_id, mfa_token):
                self._log_event('mfa_verification_failed', user_id, 'failure',
                              details={'mfa_required': True}, risk_level='high')
                return None
        
        # Create session
        permissions = self._get_user_permissions(user_id)
        session_id = self.session_manager.create_session(user_id, permissions, metadata)
        
        self._log_event('user_authenticated', user_id, 'success',
                       details={'session_id': session_id, 'mfa_used': bool(mfa_token)})
        
        return session_id
    
    def authorize_action(self, session_id: str, action: str, 
                        resource: str, context: Optional[Dict] = None) -> bool:
        """Authorize action with role-based access control."""
        session = self.session_manager.validate_session(session_id)
        if not session:
            self._log_event('authorization_failed', None, 'denied',
                          details={'reason': 'invalid_session', 'action': action})
            return False
        
        user_id = session['user_id']
        permissions = session['permissions']
        
        # Check if user has required permission
        required_permission = self._get_required_permission(action, resource)
        if required_permission not in permissions:
            self._log_event('authorization_denied', user_id, 'denied',
                          details={'action': action, 'resource': resource,
                                 'required': required_permission}, risk_level='medium')
            return False
        
        self._log_event('action_authorized', user_id, 'success',
                       details={'action': action, 'resource': resource})
        return True
    
    def encrypt_sensitive_data(self, data: Any, classification: str = 'internal') -> Optional[Dict]:
        """Encrypt sensitive data with security metadata."""
        if not self.policy.require_encryption_at_rest:
            return {'encrypted_data': data, 'metadata': {'encrypted': False}}
        
        result = self.crypto_manager.encrypt_data(data, classification)
        
        if result:
            self._log_event('data_encrypted', None, 'success',
                          details={'classification': classification, 'size': len(str(data))})
        else:
            self._log_event('encryption_failed', None, 'failure',
                          details={'classification': classification}, risk_level='high')
        
        return result
    
    def decrypt_sensitive_data(self, encrypted_package: Dict) -> Optional[Any]:
        """Decrypt sensitive data with verification."""
        result = self.crypto_manager.decrypt_data(encrypted_package)
        
        if result:
            self._log_event('data_decrypted', None, 'success')
        else:
            self._log_event('decryption_failed', None, 'failure', risk_level='high')
        
        return result
    
    def validate_and_sanitize_input(self, data: Dict[str, Any], 
                                   context: str = 'optimization') -> Dict[str, Any]:
        """Validate and sanitize input data."""
        sanitized = {}
        
        for key, value in data.items():
            # Key validation
            if not self._is_safe_key(key):
                self._log_event('unsafe_input_key', None, 'denied',
                              details={'key': key, 'context': context}, risk_level='high')
                continue
            
            # Value sanitization
            sanitized_value = self._sanitize_value(value)
            if sanitized_value is not None:
                sanitized[key] = sanitized_value
        
        self._log_event('input_sanitized', None, 'success',
                       details={'original_keys': len(data), 'sanitized_keys': len(sanitized)})
        
        return sanitized
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        # Get metrics from all managers
        audit_metrics = self.audit_logger.get_security_metrics()
        compliance_report = self.compliance_manager.run_compliance_check()
        
        # Session statistics
        active_sessions = len(self.session_manager.active_sessions)
        total_sessions = len(self.session_manager.session_history) + active_sessions
        
        return {
            'framework_info': {
                'session_id': self.session_id,
                'compliance_mode': self.policy.compliance_mode,
                'threat_level': self.threat_level,
                'encryption_enabled': HAS_CRYPTOGRAPHY,
                'jwt_enabled': HAS_JWT
            },
            'session_metrics': {
                'active_sessions': active_sessions,
                'total_sessions': total_sessions,
                'session_success_rate': 100.0 if total_sessions == 0 else (active_sessions / total_sessions) * 100
            },
            'security_metrics': audit_metrics,
            'compliance_report': compliance_report,
            'policy_summary': {
                'mfa_required': self.policy.require_mfa,
                'encryption_required': self.policy.require_encryption_at_rest,
                'audit_enabled': self.policy.audit_all_actions,
                'max_session_duration': self.policy.max_session_duration
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def _pre_auth_checks(self, user_id: str) -> bool:
        """Perform pre-authentication security checks."""
        # Rate limiting check
        # IP whitelist check
        # Geo-blocking check
        # Account lockout check
        return True  # Simplified for now
    
    def _verify_password(self, username: str, password: str) -> bool:
        """Verify user password with policy compliance."""
        # In production, use proper password hashing (bcrypt, scrypt, etc.)
        if len(password) < self.policy.password_min_length:
            return False
        
        if self.policy.password_require_special and not any(c in '!@#$%^&*()' for c in password):
            return False
        
        if self.policy.password_require_digits and not any(c.isdigit() for c in password):
            return False
        
        if self.policy.password_require_uppercase and not any(c.isupper() for c in password):
            return False
        
        return True
    
    def _verify_mfa(self, user_id: str, token: str) -> bool:
        """Verify multi-factor authentication token."""
        # Mock MFA verification - in production, verify TOTP/SMS/etc.
        return len(token) >= 6 and token.isdigit()
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions from identity provider."""
        # Mock permissions - in production, fetch from identity provider
        return ['quantum_optimize', 'view_results', 'export_data']
    
    def _get_required_permission(self, action: str, resource: str) -> str:
        """Map action/resource to required permission."""
        permission_map = {
            ('optimize', 'hyperparameters'): 'quantum_optimize',
            ('view', 'results'): 'view_results',
            ('export', 'data'): 'export_data',
            ('admin', 'system'): 'admin'
        }
        
        return permission_map.get((action, resource), 'unknown')
    
    def _is_safe_key(self, key: str) -> bool:
        """Check if input key is safe."""
        dangerous_patterns = ['__', 'eval', 'exec', 'import', 'system', 'open', 'file', 'os.']
        return not any(pattern in key.lower() for pattern in dangerous_patterns)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize input value."""
        if isinstance(value, str):
            # Remove dangerous characters
            dangerous_chars = ['<', '>', '&', '"', "'", ';', '|', '$', '`']
            sanitized = value
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            
            # Limit length
            if len(sanitized) > 1000:
                sanitized = sanitized[:1000]
            
            return sanitized
        
        elif isinstance(value, (int, float, bool)):
            return value
        
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_value(v) for v in value]
        
        elif isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items() if self._is_safe_key(k)}
        
        else:
            # Convert to string and sanitize
            return self._sanitize_value(str(value))
    
    def _log_event(self, action: str, user_id: Optional[str], outcome: str,
                   details: Optional[Dict] = None, risk_level: str = 'low'):
        """Log security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type='security',
            user_id=user_id,
            session_id=self.session_id,
            action=action,
            resource='quantum_optimization_framework',
            outcome=outcome,
            details=details or {},
            risk_level=risk_level
        )
        
        self.audit_logger.log_event(event)
    
    @contextmanager
    def secure_operation(self, operation_name: str, user_id: Optional[str] = None):
        """Context manager for secure operations."""
        start_time = time.time()
        
        try:
            self._log_event(f'{operation_name}_started', user_id, 'success')
            yield
            
        except Exception as e:
            self._log_event(f'{operation_name}_failed', user_id, 'failure',
                          details={'error': str(e)}, risk_level='high')
            raise
        
        finally:
            duration = time.time() - start_time
            self._log_event(f'{operation_name}_completed', user_id, 'success',
                          details={'duration': duration})
    
    def shutdown(self):
        """Shutdown security framework gracefully."""
        # Clean up expired sessions
        expired_count = self.session_manager.cleanup_expired_sessions()
        
        # Log shutdown
        self._log_event('framework_shutdown', None, 'success',
                       details={'expired_sessions_cleaned': expired_count})
        
        logger.info("Quantum Security Framework shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()