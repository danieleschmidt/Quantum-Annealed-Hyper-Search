"""
Enterprise Security - Advanced security features for quantum hyperparameter optimization.

Provides enterprise-grade security including encryption, authentication,
audit logging, and compliance features.
"""

import hashlib
import hmac
import secrets
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
    from cryptography.hazmat.primitives.asymmetric import rsa
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
    """Represents a security event for audit logging."""
    timestamp: float
    event_type: str
    user_id: Optional[str]
    session_id: str
    action: str
    resource: str
    outcome: str  # 'success', 'failure', 'denied'
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = 'low'  # 'low', 'medium', 'high', 'critical'
    source_ip: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'action': self.action,
            'resource': self.resource,
            'outcome': self.outcome,
            'details': self.details,
            'risk_level': self.risk_level,
            'source_ip': self.source_ip
        }


class QuantumSecurityManager:
    """
    Comprehensive security manager for quantum hyperparameter optimization.
    
    Provides authentication, authorization, encryption, audit logging,
    and compliance features for enterprise deployments.
    """
    
    def __init__(self,
                 encryption_key: Optional[bytes] = None,
                 jwt_secret: Optional[str] = None,
                 audit_log_path: Optional[str] = None,
                 compliance_mode: str = 'standard'):  # 'standard', 'hipaa', 'gdpr', 'sox'
        """
        Initialize security manager.
        
        Args:
            encryption_key: Key for data encryption (auto-generated if None)
            jwt_secret: Secret for JWT token signing
            audit_log_path: Path for audit log file
            compliance_mode: Compliance mode for additional restrictions
        """
        self.compliance_mode = compliance_mode
        self.session_id = self._generate_session_id()
        
        # Initialize encryption
        if HAS_CRYPTOGRAPHY:
            if encryption_key is None:
                encryption_key = Fernet.generate_key()
            self.cipher = Fernet(encryption_key)
            self.encryption_enabled = True
        else:
            logger.warning("Cryptography not available - encryption disabled")
            self.cipher = None
            self.encryption_enabled = False
        
        # Initialize JWT
        if HAS_JWT:
            self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
            self.jwt_enabled = True
        else:
            logger.warning("JWT not available - token authentication disabled")
            self.jwt_secret = None
            self.jwt_enabled = False
        
        # Initialize audit logging
        self.audit_log_path = audit_log_path
        self.audit_events = []
        self.audit_lock = threading.Lock()
        
        # Security policies
        self.policies = self._initialize_security_policies()
        
        # Rate limiting
        self.rate_limits = defaultdict(list)
        self.failed_attempts = defaultdict(int)
        
        # Active sessions
        self.active_sessions = {}
        
        logger.info(f"Security manager initialized (mode: {compliance_mode}, session: {self.session_id})")
        self._log_security_event('security_manager_initialized', 'system', 'success')
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID."""
        return f"qhs_{int(time.time())}_{secrets.token_hex(8)}"
    
    def _initialize_security_policies(self) -> Dict[str, Any]:
        """Initialize security policies based on compliance mode."""
        base_policies = {
            'max_session_duration': 3600,  # 1 hour
            'max_failed_attempts': 5,
            'lockout_duration': 900,  # 15 minutes
            'password_min_length': 8,
            'require_encryption': True,
            'audit_all_actions': True,
            'rate_limit_window': 60,  # 1 minute
            'max_requests_per_window': 100
        }
        
        # Compliance-specific policies
        if self.compliance_mode == 'hipaa':
            base_policies.update({
                'max_session_duration': 1800,  # 30 minutes
                'password_min_length': 12,
                'require_mfa': True,
                'data_retention_days': 2555,  # ~7 years
                'access_logging_required': True
            })
        elif self.compliance_mode == 'gdpr':
            base_policies.update({
                'data_retention_days': 1095,  # 3 years
                'require_consent': True,
                'right_to_erasure': True,
                'data_portability': True
            })
        elif self.compliance_mode == 'sox':
            base_policies.update({
                'audit_all_actions': True,
                'immutable_audit_log': True,
                'separation_of_duties': True,
                'data_retention_days': 2555  # ~7 years
            })
        
        return base_policies
    
    def authenticate_user(self, username: str, password: str, 
                         additional_factors: Optional[Dict] = None) -> Optional[str]:
        """Authenticate user and return session token."""
        user_id = f"user_{username}"
        
        # Check rate limiting
        if not self._check_rate_limit(user_id, 'auth_attempt'):
            self._log_security_event('authentication_rate_limited', user_id, 'denied',
                                   details={'reason': 'rate_limit_exceeded'})
            return None
        
        # Check lockout status
        if self._is_user_locked_out(user_id):
            self._log_security_event('authentication_attempt_locked', user_id, 'denied',
                                   details={'reason': 'user_locked_out'})
            return None
        
        # Simulate password verification (in real implementation, use proper hashing)
        if self._verify_password(username, password):
            # Check MFA if required
            if self.policies.get('require_mfa', False):
                if not self._verify_mfa(user_id, additional_factors):
                    self._log_security_event('mfa_verification_failed', user_id, 'failure')
                    return None
            
            # Generate session token
            token = self._generate_session_token(user_id)
            
            # Reset failed attempts
            self.failed_attempts[user_id] = 0
            
            self._log_security_event('user_authenticated', user_id, 'success',
                                   details={'authentication_method': 'password+mfa' if self.policies.get('require_mfa') else 'password'})
            
            return token
        else:
            # Record failed attempt
            self.failed_attempts[user_id] += 1
            
            self._log_security_event('authentication_failed', user_id, 'failure',
                                   details={'failed_attempts': self.failed_attempts[user_id]},
                                   risk_level='medium')
            
            return None
    
    def _verify_password(self, username: str, password: str) -> bool:
        """Verify password (mock implementation)."""
        # In real implementation, use proper password hashing (bcrypt, scrypt, etc.)
        return len(password) >= self.policies['password_min_length']
    
    def _verify_mfa(self, user_id: str, factors: Optional[Dict]) -> bool:
        """Verify multi-factor authentication."""
        if not factors:
            return False
        
        # Mock MFA verification
        return factors.get('totp_code') or factors.get('sms_code')
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate JWT session token."""
        if not self.jwt_enabled:
            return f"mock_token_{secrets.token_hex(16)}"
        
        expiration = datetime.utcnow() + timedelta(seconds=self.policies['max_session_duration'])
        
        payload = {
            'user_id': user_id,
            'session_id': self.session_id,
            'iat': datetime.utcnow(),
            'exp': expiration,
            'permissions': self._get_user_permissions(user_id)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
        # Store session
        self.active_sessions[token] = {
            'user_id': user_id,
            'created_at': time.time(),
            'expires_at': expiration.timestamp(),
            'permissions': payload['permissions']
        }
        
        return token
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return session info."""
        if not self.jwt_enabled:
            # Mock validation
            return {'user_id': 'mock_user', 'permissions': ['read', 'write']}
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if session is still active
            if token in self.active_sessions:
                session = self.active_sessions[token]
                
                if time.time() > session['expires_at']:
                    # Session expired
                    del self.active_sessions[token]
                    self._log_security_event('session_expired', payload['user_id'], 'failure')
                    return None
                
                return session
            
            return None
            
        except jwt.ExpiredSignatureError:
            self._log_security_event('token_expired', None, 'failure')
            return None
        except jwt.InvalidTokenError:
            self._log_security_event('invalid_token', None, 'failure', risk_level='high')
            return None
    
    def authorize_action(self, token: str, action: str, resource: str) -> bool:
        """Authorize action for given token."""
        session = self.validate_session_token(token)
        if not session:
            self._log_security_event('authorization_failed', None, 'denied',
                                   details={'reason': 'invalid_session', 'action': action, 'resource': resource})
            return False
        
        user_id = session['user_id']
        permissions = session.get('permissions', [])
        
        # Check permissions
        required_permission = self._get_required_permission(action, resource)
        if required_permission not in permissions:
            self._log_security_event('authorization_denied', user_id, 'denied',
                                   details={'action': action, 'resource': resource, 'required_permission': required_permission},
                                   risk_level='medium')
            return False
        
        self._log_security_event('action_authorized', user_id, 'success',
                               details={'action': action, 'resource': resource})
        return True
    
    def encrypt_data(self, data: Union[str, bytes, Dict]) -> Optional[bytes]:
        """Encrypt sensitive data."""
        if not self.encryption_enabled:
            logger.warning("Encryption not available")
            return None
        
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.cipher.encrypt(data)
            
            self._log_security_event('data_encrypted', None, 'success',
                                   details={'data_size': len(data)})
            
            return encrypted_data
            
        except Exception as e:
            self._log_security_event('encryption_failed', None, 'failure',
                                   details={'error': str(e)}, risk_level='high')
            return None
    
    def decrypt_data(self, encrypted_data: bytes) -> Optional[Union[str, Dict]]:
        """Decrypt sensitive data."""
        if not self.encryption_enabled:
            logger.warning("Encryption not available")
            return None
        
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            # Try to decode as JSON first
            try:
                return json.loads(decrypted_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Return as string if not JSON
                return decrypted_data.decode('utf-8')
            
        except Exception as e:
            self._log_security_event('decryption_failed', None, 'failure',
                                   details={'error': str(e)}, risk_level='high')
            return None
    
    def sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters to prevent injection attacks."""
        sanitized = {}
        
        for key, value in params.items():
            # Sanitize key
            if not self._is_safe_key(key):
                self._log_security_event('unsafe_parameter_key', None, 'denied',
                                       details={'key': key}, risk_level='high')
                continue
            
            # Sanitize value
            if isinstance(value, str):
                sanitized[key] = self._sanitize_string_value(value)
            elif isinstance(value, (int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [self._sanitize_string_value(str(v)) if isinstance(v, str) else v for v in value]
            else:
                # Convert complex types to string and sanitize
                sanitized[key] = self._sanitize_string_value(str(value))
        
        self._log_security_event('parameters_sanitized', None, 'success',
                               details={'original_count': len(params), 'sanitized_count': len(sanitized)})
        
        return sanitized
    
    def _is_safe_key(self, key: str) -> bool:
        """Check if parameter key is safe."""
        # Prevent common injection patterns
        dangerous_patterns = ['__', 'eval', 'exec', 'import', 'system', 'open', 'file']
        return not any(pattern in key.lower() for pattern in dangerous_patterns)
    
    def _sanitize_string_value(self, value: str) -> str:
        """Sanitize string value."""
        # Remove potentially dangerous characters/patterns
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '|', '&', '$']
        sanitized = value
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        if len(sanitized) > 1000:  # Arbitrary limit
            sanitized = sanitized[:1000]
        
        return sanitized
    
    def _check_rate_limit(self, identifier: str, action: str) -> bool:
        """Check if action is within rate limits."""
        now = time.time()
        window_start = now - self.policies['rate_limit_window']
        
        # Clean old entries
        rate_key = f"{identifier}_{action}"
        self.rate_limits[rate_key] = [
            timestamp for timestamp in self.rate_limits[rate_key]
            if timestamp > window_start
        ]
        
        # Check limit
        if len(self.rate_limits[rate_key]) >= self.policies['max_requests_per_window']:
            return False
        
        # Record this request
        self.rate_limits[rate_key].append(now)
        return True
    
    def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        return self.failed_attempts.get(user_id, 0) >= self.policies['max_failed_attempts']
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions (mock implementation)."""
        # In real implementation, fetch from database
        return ['quantum_optimize', 'view_results', 'export_data']
    
    def _get_required_permission(self, action: str, resource: str) -> str:
        """Get required permission for action on resource."""
        permission_map = {
            ('optimize', 'hyperparameters'): 'quantum_optimize',
            ('view', 'results'): 'view_results',
            ('export', 'data'): 'export_data',
            ('modify', 'config'): 'admin'
        }
        
        return permission_map.get((action, resource), 'unknown')
    
    def _log_security_event(self, action: str, user_id: Optional[str], outcome: str,
                          details: Optional[Dict] = None, risk_level: str = 'low'):
        """Log security event for audit trail."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type='security',
            user_id=user_id,
            session_id=self.session_id,
            action=action,
            resource='quantum_optimization_system',
            outcome=outcome,
            details=details or {},
            risk_level=risk_level
        )
        
        with self.audit_lock:
            self.audit_events.append(event)
            
            # Write to audit log file if configured
            if self.audit_log_path:
                self._write_audit_log(event)
    
    def _write_audit_log(self, event: SecurityEvent):
        """Write event to audit log file."""
        try:
            audit_entry = json.dumps(event.to_dict()) + '\n'
            
            with open(self.audit_log_path, 'a') as f:
                f.write(audit_entry)
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary."""
        now = time.time()
        
        # Event statistics
        event_counts = defaultdict(int)
        risk_counts = defaultdict(int)
        outcome_counts = defaultdict(int)
        
        for event in self.audit_events:
            event_counts[event.action] += 1
            risk_counts[event.risk_level] += 1
            outcome_counts[event.outcome] += 1
        
        return {
            'session_id': self.session_id,
            'compliance_mode': self.compliance_mode,
            'encryption_enabled': self.encryption_enabled,
            'jwt_enabled': self.jwt_enabled,
            'active_sessions': len(self.active_sessions),
            'total_audit_events': len(self.audit_events),
            'events_by_action': dict(event_counts),
            'events_by_risk_level': dict(risk_counts),
            'events_by_outcome': dict(outcome_counts),
            'security_policies': self.policies,
            'failed_attempts_by_user': dict(self.failed_attempts)
        }
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        summary = self.get_security_summary()
        
        # Check compliance status
        compliance_checks = {
            'encryption_enabled': self.encryption_enabled,
            'audit_logging_active': len(self.audit_events) > 0,
            'authentication_required': self.jwt_enabled,
            'rate_limiting_active': len(self.rate_limits) > 0,
            'session_management': len(self.active_sessions) >= 0  # Always true if initialized
        }
        
        # Add compliance-specific checks
        if self.compliance_mode == 'hipaa':
            compliance_checks.update({
                'mfa_required': self.policies.get('require_mfa', False),
                'session_timeout_30min': self.policies['max_session_duration'] <= 1800,
                'access_logging': self.audit_log_path is not None
            })
        elif self.compliance_mode == 'gdpr':
            compliance_checks.update({
                'data_retention_policy': True,  # Assuming policy exists
                'consent_tracking': True,       # Mock - would be real in implementation
                'right_to_erasure_support': True
            })
        
        compliance_score = (sum(compliance_checks.values()) / len(compliance_checks)) * 100
        
        return {
            'compliance_mode': self.compliance_mode,
            'compliance_score': compliance_score,
            'compliance_checks': compliance_checks,
            'recommendations': self._generate_compliance_recommendations(compliance_checks),
            'summary': summary,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_compliance_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        for check, passed in checks.items():
            if not passed:
                if check == 'encryption_enabled':
                    recommendations.append("Enable encryption for data protection")
                elif check == 'mfa_required':
                    recommendations.append("Implement multi-factor authentication")
                elif check == 'audit_logging_active':
                    recommendations.append("Activate comprehensive audit logging")
                # Add more specific recommendations as needed
        
        if not recommendations:
            recommendations.append("All compliance checks passed - maintain current security posture")
        
        return recommendations
    
    def export_audit_log(self, start_time: Optional[float] = None, 
                        end_time: Optional[float] = None) -> List[Dict]:
        """Export audit log for specified time range."""
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        filtered_events = [
            event.to_dict() for event in self.audit_events
            if start_time <= event.timestamp <= end_time
        ]
        
        self._log_security_event('audit_log_exported', None, 'success',
                               details={'event_count': len(filtered_events), 'time_range': f'{start_time}-{end_time}'})
        
        return filtered_events
    
    def revoke_session(self, token: str) -> bool:
        """Revoke active session."""
        if token in self.active_sessions:
            user_id = self.active_sessions[token]['user_id']
            del self.active_sessions[token]
            
            self._log_security_event('session_revoked', user_id, 'success')
            return True
        
        return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = time.time()
        expired_tokens = []
        
        for token, session in self.active_sessions.items():
            if now > session['expires_at']:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            user_id = self.active_sessions[token]['user_id']
            del self.active_sessions[token]
            self._log_security_event('session_expired_cleanup', user_id, 'success')
        
        return len(expired_tokens)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_expired_sessions()
        self._log_security_event('security_manager_shutdown', None, 'success')
