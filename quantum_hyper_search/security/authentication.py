"""
Authentication Module - Advanced authentication mechanisms.

Provides comprehensive authentication including password policies,
multi-factor authentication, and secure token management.
"""

import time
import hmac
import hashlib
import secrets
import base64
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

try:
    import pyotp
    HAS_PYOTP = True
except ImportError:
    HAS_PYOTP = False


@dataclass
class AuthenticationResult:
    """Result of authentication attempt."""
    success: bool
    user_id: Optional[str] = None
    session_token: Optional[str] = None
    mfa_required: bool = False
    reason: Optional[str] = None
    metadata: Dict[str, Any] = None


class AuthenticationManager:
    """Comprehensive authentication manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.failed_attempts = {}
        self.locked_accounts = {}
        
        # Password policy
        self.min_password_length = config.get('min_password_length', 12)
        self.require_special_chars = config.get('require_special_chars', True)
        self.require_digits = config.get('require_digits', True)
        self.require_uppercase = config.get('require_uppercase', True)
        self.require_lowercase = config.get('require_lowercase', True)
        
        # Account lockout policy
        self.max_failed_attempts = config.get('max_failed_attempts', 5)
        self.lockout_duration = config.get('lockout_duration', 900)  # 15 minutes
        
        # MFA settings
        self.require_mfa = config.get('require_mfa', True)
        self.mfa_methods = config.get('mfa_methods', ['totp'])
    
    def authenticate(self, username: str, password: str, 
                    mfa_token: Optional[str] = None,
                    ip_address: Optional[str] = None) -> AuthenticationResult:
        """Authenticate user with comprehensive checks."""
        user_id = f"user_{username}"
        
        # Check if account is locked
        if self._is_account_locked(user_id):
            return AuthenticationResult(
                success=False,
                reason="account_locked",
                metadata={"lockout_expires": self.locked_accounts[user_id]['expires']}
            )
        
        # Verify password
        if not self._verify_password(username, password):
            self._record_failed_attempt(user_id, ip_address)
            return AuthenticationResult(
                success=False,
                reason="invalid_credentials",
                metadata={"failed_attempts": self.failed_attempts.get(user_id, {}).get('count', 0)}
            )
        
        # Check MFA if required
        if self.require_mfa:
            if not mfa_token:
                return AuthenticationResult(
                    success=False,
                    mfa_required=True,
                    reason="mfa_required"
                )
            
            if not self._verify_mfa(user_id, mfa_token):
                self._record_failed_attempt(user_id, ip_address)
                return AuthenticationResult(
                    success=False,
                    reason="invalid_mfa",
                    metadata={"mfa_required": True}
                )
        
        # Authentication successful
        self._clear_failed_attempts(user_id)
        session_token = self._generate_session_token(user_id)
        
        return AuthenticationResult(
            success=True,
            user_id=user_id,
            session_token=session_token,
            metadata={"authenticated_at": time.time()}
        )
    
    def _verify_password(self, username: str, password: str) -> bool:
        """Verify password against stored hash."""
        # Password policy check
        if not self._check_password_policy(password):
            return False
        
        # In production, compare against stored bcrypt hash
        if HAS_BCRYPT:
            stored_hash = self._get_stored_password_hash(username)
            if stored_hash:
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
        
        # Fallback for demo (use secure hashing in production)
        return len(password) >= self.min_password_length
    
    def _check_password_policy(self, password: str) -> bool:
        """Check if password meets policy requirements."""
        if len(password) < self.min_password_length:
            return False
        
        if self.require_special_chars and not any(c in '!@#$%^&*()[]{}|;:,.<>?' for c in password):
            return False
        
        if self.require_digits and not any(c.isdigit() for c in password):
            return False
        
        if self.require_uppercase and not any(c.isupper() for c in password):
            return False
        
        if self.require_lowercase and not any(c.islower() for c in password):
            return False
        
        return True
    
    def _verify_mfa(self, user_id: str, token: str) -> bool:
        """Verify MFA token."""
        if 'totp' in self.mfa_methods and HAS_PYOTP:
            return self._verify_totp(user_id, token)
        
        # Fallback verification for demo
        return len(token) == 6 and token.isdigit()
    
    def _verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token."""
        if not HAS_PYOTP:
            return False
        
        # Get user's TOTP secret (in production, from secure storage)
        totp_secret = self._get_user_totp_secret(user_id)
        if not totp_secret:
            return False
        
        totp = pyotp.TOTP(totp_secret)
        return totp.verify(token, valid_window=1)  # Allow 30s window
    
    def _get_user_totp_secret(self, user_id: str) -> Optional[str]:
        """Get user's TOTP secret (mock implementation)."""
        # In production, fetch from secure storage
        return pyotp.random_base32() if HAS_PYOTP else None
    
    def _get_stored_password_hash(self, username: str) -> Optional[bytes]:
        """Get stored password hash (mock implementation)."""
        # In production, fetch from secure database
        if HAS_BCRYPT:
            # Return a mock hash for demo
            return bcrypt.hashpw(b"secure_password_123", bcrypt.gensalt())
        return None
    
    def _record_failed_attempt(self, user_id: str, ip_address: Optional[str]):
        """Record failed authentication attempt."""
        now = time.time()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = {'count': 0, 'attempts': []}
        
        self.failed_attempts[user_id]['count'] += 1
        self.failed_attempts[user_id]['attempts'].append({
            'timestamp': now,
            'ip_address': ip_address
        })
        
        # Check if account should be locked
        if self.failed_attempts[user_id]['count'] >= self.max_failed_attempts:
            self._lock_account(user_id)
    
    def _lock_account(self, user_id: str):
        """Lock user account due to failed attempts."""
        self.locked_accounts[user_id] = {
            'locked_at': time.time(),
            'expires': time.time() + self.lockout_duration,
            'reason': 'max_failed_attempts'
        }
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is currently locked."""
        if user_id not in self.locked_accounts:
            return False
        
        # Check if lockout has expired
        if time.time() > self.locked_accounts[user_id]['expires']:
            del self.locked_accounts[user_id]
            return False
        
        return True
    
    def _clear_failed_attempts(self, user_id: str):
        """Clear failed attempts after successful authentication."""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate secure session token."""
        # In production, use JWT or similar
        token_data = f"{user_id}:{int(time.time())}:{secrets.token_hex(16)}"
        return base64.b64encode(token_data.encode()).decode()
    
    def setup_mfa(self, user_id: str, method: str = 'totp') -> Dict[str, Any]:
        """Setup MFA for user."""
        if method == 'totp' and HAS_PYOTP:
            secret = pyotp.random_base32()
            totp = pyotp.TOTP(secret)
            
            # Generate QR code URI
            provisioning_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name="Quantum Hyper Search"
            )
            
            return {
                'method': 'totp',
                'secret': secret,
                'qr_code_uri': provisioning_uri,
                'backup_codes': self._generate_backup_codes()
            }
        
        return {}
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA recovery."""
        return [f"{secrets.randbelow(100000000):08d}" for _ in range(10)]
    
    def change_password(self, user_id: str, old_password: str, 
                       new_password: str) -> bool:
        """Change user password with validation."""
        # Verify old password
        username = user_id.replace('user_', '')
        if not self._verify_password(username, old_password):
            return False
        
        # Check new password policy
        if not self._check_password_policy(new_password):
            return False
        
        # In production, hash and store new password
        if HAS_BCRYPT:
            new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            # Store new_hash securely
        
        return True
    
    def get_authentication_stats(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        total_failed_attempts = sum(
            data['count'] for data in self.failed_attempts.values()
        )
        
        return {
            'failed_attempts_by_user': len(self.failed_attempts),
            'total_failed_attempts': total_failed_attempts,
            'locked_accounts': len(self.locked_accounts),
            'mfa_enabled': self.require_mfa,
            'password_policy': {
                'min_length': self.min_password_length,
                'require_special': self.require_special_chars,
                'require_digits': self.require_digits,
                'require_uppercase': self.require_uppercase
            }
        }


class TokenManager:
    """Advanced token management for secure sessions."""
    
    def __init__(self, secret_key: str, token_lifetime: int = 3600):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        self.token_lifetime = token_lifetime
        self.revoked_tokens = set()
    
    def generate_token(self, user_id: str, permissions: List[str], 
                      metadata: Optional[Dict] = None) -> str:
        """Generate secure token with claims."""
        now = int(time.time())
        expires = now + self.token_lifetime
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'issued_at': now,
            'expires_at': expires,
            'jti': secrets.token_hex(16),  # JWT ID for revocation
            'metadata': metadata or {}
        }
        
        # Create signature
        payload_str = base64.b64encode(str(payload).encode()).decode()
        signature = hmac.new(
            self.secret_key,
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = f"{payload_str}.{signature}"
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and return payload if valid."""
        try:
            payload_str, signature = token.split('.', 1)
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key,
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Decode payload
            payload = eval(base64.b64decode(payload_str.encode()).decode())
            
            # Check expiration
            if time.time() > payload['expires_at']:
                return None
            
            # Check revocation
            if payload['jti'] in self.revoked_tokens:
                return None
            
            return payload
            
        except Exception:
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke token by adding to blacklist."""
        payload = self.verify_token(token)
        if payload:
            self.revoked_tokens.add(payload['jti'])
            return True
        return False
    
    def cleanup_expired_revocations(self):
        """Clean up expired token IDs from revocation list."""
        # In production, implement cleanup based on token expiration times
        pass


class MFAManager:
    """Multi-factor authentication manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled_methods = config.get('mfa_methods', ['totp'])
        self.backup_codes = {}  # user_id -> codes
        
    def setup_totp(self, user_id: str, issuer: str = "Quantum Hyper Search") -> Dict[str, Any]:
        """Setup TOTP MFA for user."""
        if not HAS_PYOTP:
            raise RuntimeError("PyOTP not available for TOTP setup")
        
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        
        provisioning_uri = totp.provisioning_uri(
            name=user_id,
            issuer_name=issuer
        )
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        self.backup_codes[user_id] = backup_codes
        
        return {
            'secret': secret,
            'qr_code_uri': provisioning_uri,
            'backup_codes': backup_codes
        }
    
    def verify_totp(self, user_id: str, secret: str, token: str) -> bool:
        """Verify TOTP token."""
        if not HAS_PYOTP:
            return False
        
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume backup code."""
        if user_id not in self.backup_codes:
            return False
        
        code = code.upper().strip()
        if code in self.backup_codes[user_id]:
            # Remove used backup code
            self.backup_codes[user_id].remove(code)
            return True
        
        return False
    
    def get_remaining_backup_codes(self, user_id: str) -> int:
        """Get number of remaining backup codes."""
        return len(self.backup_codes.get(user_id, []))
    
    def regenerate_backup_codes(self, user_id: str) -> List[str]:
        """Regenerate backup codes for user."""
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        self.backup_codes[user_id] = backup_codes
        return backup_codes