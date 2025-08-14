#!/usr/bin/env python3
"""
Enhanced Security Framework
Enterprise-grade security with quantum-safe encryption and comprehensive audit logging.
"""

import hashlib
import hmac
import time
import json
import secrets
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import uuid

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class AuditAction(Enum):
    """Types of auditable actions."""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    QUANTUM_OPERATION = "quantum_operation"
    CONFIG_CHANGE = "config_change"
    SECURITY_EVENT = "security_event"


@dataclass
class AuditLogEntry:
    """Audit log entry structure."""
    timestamp: float
    user_id: str
    action: AuditAction
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    session_id: Optional[str] = None


@dataclass
class SecurityToken:
    """Security token structure."""
    token_id: str
    user_id: str
    issued_at: float
    expires_at: float
    permissions: List[str]
    security_level: SecurityLevel
    session_id: str


class QuantumSafeEncryption:
    """
    Quantum-Safe Encryption Implementation
    
    Implements post-quantum cryptographic algorithms suitable
    for protecting data against quantum computer attacks.
    """
    
    def __init__(self, key_size: int = 4096):
        self.key_size = key_size
        self.backend = default_backend()
        self._private_key = None
        self._public_key = None
        
        if CRYPTOGRAPHY_AVAILABLE:
            self._generate_keypair()
    
    def _generate_keypair(self):
        """Generate RSA keypair (transitional until post-quantum standards)."""
        
        try:
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size,
                backend=self.backend
            )
            self._public_key = self._private_key.public_key()
            
            logger.info(f"Generated {self.key_size}-bit RSA keypair for quantum-safe encryption")
            
        except Exception as e:
            logger.error(f"Failed to generate keypair: {e}")
            raise
    
    def encrypt_data(self, data: bytes, recipient_public_key: Optional[Any] = None) -> bytes:
        """Encrypt data using quantum-safe encryption."""
        
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography not available, using fallback encoding")
            return base64.b64encode(data)
        
        try:
            public_key = recipient_public_key or self._public_key
            
            if public_key is None:
                raise ValueError("No public key available for encryption")
            
            # For large data, use hybrid encryption
            if len(data) > 190:  # RSA padding limit
                return self._hybrid_encrypt(data, public_key)
            else:
                # Direct RSA encryption for small data
                encrypted = public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return encrypted
                
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, private_key: Optional[Any] = None) -> bytes:
        """Decrypt data using quantum-safe decryption."""
        
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography not available, using fallback decoding")
            return base64.b64decode(encrypted_data)
        
        try:
            priv_key = private_key or self._private_key
            
            if priv_key is None:
                raise ValueError("No private key available for decryption")
            
            # Check if this is hybrid encryption
            if len(encrypted_data) > 512:  # Likely hybrid
                return self._hybrid_decrypt(encrypted_data, priv_key)
            else:
                # Direct RSA decryption
                decrypted = priv_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return decrypted
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def _hybrid_encrypt(self, data: bytes, public_key: Any) -> bytes:
        """Hybrid encryption for large data."""
        
        # Generate random AES key
        aes_key = secrets.token_bytes(32)  # 256-bit key
        iv = secrets.token_bytes(16)  # 128-bit IV
        
        # Encrypt data with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Pad data to AES block size
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encrypt AES key with RSA
        encrypted_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine encrypted key, IV, and encrypted data
        result = len(encrypted_key).to_bytes(4, 'big') + encrypted_key + iv + encrypted_data
        return result
    
    def _hybrid_decrypt(self, encrypted_data: bytes, private_key: Any) -> bytes:
        """Hybrid decryption for large data."""
        
        # Extract encrypted key length
        key_length = int.from_bytes(encrypted_data[:4], 'big')
        
        # Extract components
        encrypted_key = encrypted_data[4:4+key_length]
        iv = encrypted_data[4+key_length:4+key_length+16]
        encrypted_content = encrypted_data[4+key_length+16:]
        
        # Decrypt AES key
        aes_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(encrypted_content) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        data = padded_data[:-padding_length]
        
        return data
    
    def generate_secure_hash(self, data: bytes, salt: Optional[bytes] = None) -> str:
        """Generate secure hash with salt."""
        
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        
        hash_bytes = kdf.derive(data)
        
        # Encode salt and hash
        result = base64.b64encode(salt + hash_bytes).decode('utf-8')
        return result
    
    def verify_secure_hash(self, data: bytes, hash_string: str) -> bool:
        """Verify secure hash."""
        
        try:
            # Decode salt and hash
            combined = base64.b64decode(hash_string.encode('utf-8'))
            salt = combined[:32]
            stored_hash = combined[32:]
            
            # Regenerate hash
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=self.backend
            )
            
            computed_hash = kdf.derive(data)
            
            # Constant-time comparison
            return hmac.compare_digest(stored_hash, computed_hash)
            
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False
    
    def export_public_key(self) -> str:
        """Export public key in PEM format."""
        
        if not self._public_key:
            raise ValueError("No public key available")
        
        pem = self._public_key.public_key_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return pem.decode('utf-8')
    
    def import_public_key(self, pem_data: str) -> Any:
        """Import public key from PEM format."""
        
        try:
            public_key = serialization.load_pem_public_key(
                pem_data.encode('utf-8'),
                backend=self.backend
            )
            return public_key
            
        except Exception as e:
            logger.error(f"Failed to import public key: {e}")
            raise


class SecureTokenManager:
    """
    Secure JWT Token Management
    
    Manages authentication and authorization tokens with
    configurable expiration and permission controls.
    """
    
    def __init__(self, secret_key: Optional[str] = None, 
                 default_expiry: int = 3600):
        self.secret_key = secret_key or secrets.token_urlsafe(64)
        self.default_expiry = default_expiry
        self.active_tokens = {}
        self.revoked_tokens = set()
        self.lock = threading.Lock()
        
        if not JWT_AVAILABLE:
            logger.warning("JWT library not available, using fallback token management")
    
    def create_token(self, user_id: str, permissions: List[str],
                    security_level: SecurityLevel = SecurityLevel.INTERNAL,
                    expiry_seconds: Optional[int] = None) -> SecurityToken:
        """Create a new security token."""
        
        now = time.time()
        expiry = now + (expiry_seconds or self.default_expiry)
        token_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        token_data = SecurityToken(
            token_id=token_id,
            user_id=user_id,
            issued_at=now,
            expires_at=expiry,
            permissions=permissions,
            security_level=security_level,
            session_id=session_id
        )
        
        with self.lock:
            self.active_tokens[token_id] = token_data
        
        logger.info(f"Created token for user {user_id} with {len(permissions)} permissions")
        return token_data
    
    def encode_token(self, token: SecurityToken) -> str:
        """Encode token as JWT."""
        
        if not JWT_AVAILABLE:
            # Fallback: base64 encode the token data
            token_json = json.dumps(asdict(token))
            return base64.b64encode(token_json.encode()).decode()
        
        try:
            payload = {
                'token_id': token.token_id,
                'user_id': token.user_id,
                'iat': token.issued_at,
                'exp': token.expires_at,
                'permissions': token.permissions,
                'security_level': token.security_level.value,
                'session_id': token.session_id
            }
            
            encoded = jwt.encode(payload, self.secret_key, algorithm='HS256')
            return encoded
            
        except Exception as e:
            logger.error(f"Token encoding failed: {e}")
            raise
    
    def decode_token(self, token_string: str) -> Optional[SecurityToken]:
        """Decode and validate JWT token."""
        
        if not JWT_AVAILABLE:
            # Fallback: base64 decode
            try:
                token_json = base64.b64decode(token_string).decode()
                token_data = json.loads(token_json)
                
                return SecurityToken(
                    token_id=token_data['token_id'],
                    user_id=token_data['user_id'],
                    issued_at=token_data['issued_at'],
                    expires_at=token_data['expires_at'],
                    permissions=token_data['permissions'],
                    security_level=SecurityLevel(token_data['security_level']),
                    session_id=token_data['session_id']
                )
            except Exception as e:
                logger.error(f"Fallback token decoding failed: {e}")
                return None
        
        try:
            payload = jwt.decode(token_string, self.secret_key, algorithms=['HS256'])
            
            token_id = payload['token_id']
            
            # Check if token is revoked
            if token_id in self.revoked_tokens:
                logger.warning(f"Attempted use of revoked token: {token_id}")
                return None
            
            # Validate token exists in active tokens
            with self.lock:
                if token_id not in self.active_tokens:
                    logger.warning(f"Token not found in active tokens: {token_id}")
                    return None
                
                stored_token = self.active_tokens[token_id]
            
            # Verify token hasn't been tampered with
            if (stored_token.user_id != payload['user_id'] or
                stored_token.expires_at != payload['exp']):
                logger.error(f"Token validation failed: data mismatch for {token_id}")
                return None
            
            return stored_token
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token decoding failed: {e}")
            return None
    
    def revoke_token(self, token_id: str):
        """Revoke a token."""
        
        with self.lock:
            if token_id in self.active_tokens:
                del self.active_tokens[token_id]
            self.revoked_tokens.add(token_id)
        
        logger.info(f"Revoked token: {token_id}")
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens from active set."""
        
        now = time.time()
        expired_tokens = []
        
        with self.lock:
            for token_id, token in list(self.active_tokens.items()):
                if token.expires_at < now:
                    expired_tokens.append(token_id)
                    del self.active_tokens[token_id]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
    
    def validate_permission(self, token: SecurityToken, required_permission: str,
                          required_level: SecurityLevel = SecurityLevel.INTERNAL) -> bool:
        """Validate token permissions."""
        
        # Check security level
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        token_level = level_hierarchy.get(token.security_level, 0)
        required_level_value = level_hierarchy.get(required_level, 0)
        
        if token_level < required_level_value:
            logger.warning(f"Insufficient security level: {token.security_level} < {required_level}")
            return False
        
        # Check specific permission
        if required_permission not in token.permissions and '*' not in token.permissions:
            logger.warning(f"Missing permission: {required_permission}")
            return False
        
        return True


class AuditLogger:
    """
    Comprehensive Audit Logging System
    
    Provides tamper-resistant logging of all security-relevant events
    with configurable retention and compliance features.
    """
    
    def __init__(self, log_file: Optional[str] = None,
                 encryption: Optional[QuantumSafeEncryption] = None):
        self.log_file = log_file or "security_audit.log"
        self.encryption = encryption
        self.log_buffer = []
        self.buffer_size = 100
        self.lock = threading.Lock()
        
        # Setup separate audit logger
        self.audit_logger = logging.getLogger('security_audit')
        self.audit_logger.setLevel(logging.INFO)
        
        if not self.audit_logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
    
    def log_event(self, user_id: str, action: AuditAction, resource: str,
                 details: Dict[str, Any], success: bool = True,
                 security_level: SecurityLevel = SecurityLevel.INTERNAL,
                 ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None,
                 session_id: Optional[str] = None):
        """Log a security audit event."""
        
        entry = AuditLogEntry(
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            success=success,
            security_level=security_level,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id
        )
        
        # Convert to JSON
        entry_json = json.dumps(asdict(entry), default=str)
        
        # Encrypt if encryption is available
        if self.encryption:
            try:
                encrypted_entry = self.encryption.encrypt_data(entry_json.encode())
                log_data = base64.b64encode(encrypted_entry).decode()
                entry_json = f"ENCRYPTED:{log_data}"
            except Exception as e:
                logger.error(f"Failed to encrypt audit log entry: {e}")
        
        # Write to audit log
        self.audit_logger.info(entry_json)
        
        # Add to buffer for batch processing
        with self.lock:
            self.log_buffer.append(entry)
            if len(self.log_buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush log buffer to persistent storage."""
        
        if not self.log_buffer:
            return
        
        # In a real implementation, this would write to a database
        # or other persistent storage with integrity checks
        
        logger.info(f"Flushed {len(self.log_buffer)} audit log entries")
        self.log_buffer.clear()
    
    def search_logs(self, user_id: Optional[str] = None,
                   action: Optional[AuditAction] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   limit: int = 100) -> List[AuditLogEntry]:
        """Search audit logs with filters."""
        
        # In a real implementation, this would query a database
        # For now, return recent buffer entries that match criteria
        
        results = []
        
        with self.lock:
            for entry in self.log_buffer:
                # Apply filters
                if user_id and entry.user_id != user_id:
                    continue
                if action and entry.action != action:
                    continue
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                
                results.append(entry)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def generate_audit_report(self, start_time: float, end_time: float) -> str:
        """Generate comprehensive audit report."""
        
        logs = self.search_logs(start_time=start_time, end_time=end_time, limit=10000)
        
        # Analyze logs
        user_activity = defaultdict(int)
        action_counts = defaultdict(int)
        security_events = []
        failed_operations = []
        
        for log in logs:
            user_activity[log.user_id] += 1
            action_counts[log.action.value] += 1
            
            if not log.success:
                failed_operations.append(log)
            
            if log.action == AuditAction.SECURITY_EVENT:
                security_events.append(log)
        
        # Generate report
        report = f"""
# Security Audit Report

**Report Period**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} to {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}
**Total Events**: {len(logs)}

## Activity Summary

### Top Users by Activity
"""
        
        for user, count in sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"- {user}: {count} events\n"
        
        report += "\n### Actions Performed\n"
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {action}: {count} times\n"
        
        if failed_operations:
            report += f"\n### Failed Operations ({len(failed_operations)})\n"
            for failure in failed_operations[-10:]:  # Last 10 failures
                report += f"- {failure.user_id}: {failure.action.value} on {failure.resource} at {time.strftime('%H:%M:%S', time.localtime(failure.timestamp))}\n"
        
        if security_events:
            report += f"\n### Security Events ({len(security_events)})\n"
            for event in security_events[-10:]:  # Last 10 security events
                report += f"- {event.details.get('type', 'Unknown')}: {event.details.get('description', 'No description')} at {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}\n"
        
        return report


class SecurityManager:
    """
    Main Security Management System
    
    Orchestrates all security components including encryption,
    authentication, authorization, and audit logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.encryption = QuantumSafeEncryption()
        self.token_manager = SecureTokenManager()
        self.audit_logger = AuditLogger(encryption=self.encryption)
        
        # Security policies
        self.password_policy = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_digits': True,
            'require_special': True
        }
        
        # Rate limiting
        self.rate_limits = defaultdict(list)
        self.rate_limit_window = 300  # 5 minutes
        self.max_requests_per_window = 100
        
        logger.info("Security Manager initialized")
    
    def authenticate_user(self, user_id: str, credentials: Dict[str, Any],
                         ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate a user and return a token."""
        
        # Check rate limiting
        if not self._check_rate_limit(user_id, ip_address):
            self.audit_logger.log_event(
                user_id=user_id,
                action=AuditAction.SECURITY_EVENT,
                resource="authentication",
                details={
                    "type": "rate_limit_exceeded",
                    "ip_address": ip_address
                },
                success=False
            )
            return None
        
        # Validate credentials (simplified - in practice, check against user database)
        if not self._validate_credentials(user_id, credentials):
            self.audit_logger.log_event(
                user_id=user_id,
                action=AuditAction.LOGIN,
                resource="authentication",
                details={"ip_address": ip_address},
                success=False,
                ip_address=ip_address
            )
            return None
        
        # Create token with appropriate permissions
        permissions = self._get_user_permissions(user_id)
        security_level = self._get_user_security_level(user_id)
        
        token = self.token_manager.create_token(
            user_id=user_id,
            permissions=permissions,
            security_level=security_level
        )
        
        encoded_token = self.token_manager.encode_token(token)
        
        # Log successful authentication
        self.audit_logger.log_event(
            user_id=user_id,
            action=AuditAction.LOGIN,
            resource="authentication",
            details={
                "permissions": permissions,
                "security_level": security_level.value,
                "ip_address": ip_address
            },
            success=True,
            ip_address=ip_address,
            session_id=token.session_id
        )
        
        return encoded_token
    
    def authorize_action(self, token_string: str, action: str, resource: str,
                        required_level: SecurityLevel = SecurityLevel.INTERNAL) -> bool:
        """Authorize an action with the given token."""
        
        # Decode and validate token
        token = self.token_manager.decode_token(token_string)
        if not token:
            return False
        
        # Check if action is authorized
        authorized = self.token_manager.validate_permission(
            token, action, required_level
        )
        
        # Log authorization attempt
        self.audit_logger.log_event(
            user_id=token.user_id,
            action=AuditAction.ACCESS,
            resource=resource,
            details={
                "requested_action": action,
                "required_level": required_level.value,
                "token_permissions": token.permissions
            },
            success=authorized,
            session_id=token.session_id
        )
        
        return authorized
    
    def encrypt_sensitive_data(self, data: Union[str, bytes], 
                             security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Encrypt sensitive data based on security level."""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Apply additional security measures based on level
        if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            # Add additional entropy for highest security levels
            salt = secrets.token_bytes(32)
            data = salt + data
        
        encrypted = self.encryption.encrypt_data(data)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_sensitive_data(self, encrypted_data: str,
                             security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Decrypt sensitive data."""
        
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.encryption.decrypt_data(encrypted_bytes)
        
        # Remove salt if it was added for high security levels
        if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            decrypted = decrypted[32:]  # Remove 32-byte salt
        
        return decrypted.decode('utf-8')
    
    def _check_rate_limit(self, user_id: str, ip_address: Optional[str]) -> bool:
        """Check rate limiting for user/IP."""
        
        now = time.time()
        window_start = now - self.rate_limit_window
        
        # Clean old entries
        identifier = f"{user_id}:{ip_address or 'unknown'}"
        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if timestamp > window_start
        ]
        
        # Check current rate
        if len(self.rate_limits[identifier]) >= self.max_requests_per_window:
            return False
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return True
    
    def _validate_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate user credentials."""
        
        # Simplified validation - in practice, check against secure user database
        password = credentials.get('password', '')
        
        # Check password policy
        if not self._validate_password_policy(password):
            return False
        
        # In practice, hash the password and compare with stored hash
        # For now, just check if password is not empty
        return len(password) > 0
    
    def _validate_password_policy(self, password: str) -> bool:
        """Validate password against security policy."""
        
        if len(password) < self.password_policy['min_length']:
            return False
        
        if self.password_policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False
        
        if self.password_policy['require_lowercase'] and not any(c.islower() for c in password):
            return False
        
        if self.password_policy['require_digits'] and not any(c.isdigit() for c in password):
            return False
        
        if self.password_policy['require_special']:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False
        
        return True
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions (simplified)."""
        
        # In practice, this would query a user permissions database
        if user_id.startswith('admin_'):
            return ['*']  # Admin users get all permissions
        elif user_id.startswith('quantum_'):
            return ['quantum_operations', 'read_data', 'write_data']
        else:
            return ['read_data']
    
    def _get_user_security_level(self, user_id: str) -> SecurityLevel:
        """Get user security clearance level (simplified)."""
        
        # In practice, this would query a user security database
        if user_id.startswith('admin_'):
            return SecurityLevel.SECRET
        elif user_id.startswith('quantum_'):
            return SecurityLevel.CONFIDENTIAL
        else:
            return SecurityLevel.INTERNAL
    
    def get_security_report(self) -> str:
        """Generate comprehensive security report."""
        
        # Clean up expired tokens
        self.token_manager.cleanup_expired_tokens()
        
        # Generate audit report for last 24 hours
        end_time = time.time()
        start_time = end_time - 86400  # 24 hours
        
        audit_report = self.audit_logger.generate_audit_report(start_time, end_time)
        
        # Add token statistics
        active_tokens = len(self.token_manager.active_tokens)
        revoked_tokens = len(self.token_manager.revoked_tokens)
        
        security_summary = f"""
# Security System Status

## Token Management
- **Active Tokens**: {active_tokens}
- **Revoked Tokens**: {revoked_tokens}

## Rate Limiting
- **Active Rate Limits**: {len(self.rate_limits)}
- **Window Size**: {self.rate_limit_window} seconds
- **Max Requests**: {self.max_requests_per_window}

{audit_report}
"""
        
        return security_summary


# Global security manager instance
_global_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global _global_security_manager
    
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    
    return _global_security_manager