"""
Encryption Module - Advanced encryption and key management.

Provides comprehensive encryption including symmetric/asymmetric encryption,
key management, and quantum-safe cryptography for future-proofing.
"""

import os
import time
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

try:
    # Quantum-safe cryptography (placeholder for future implementation)
    # from pqcrypto.kem import kyber512, kyber768, kyber1024
    # from pqcrypto.sign import dilithium2, dilithium3, dilithium5
    HAS_PQC = False  # Set to True when PQC libraries are available
except ImportError:
    HAS_PQC = False


@dataclass
class EncryptionKey:
    """Represents an encryption key with metadata."""
    key_id: str
    algorithm: str
    key_data: bytes
    created_at: float
    expires_at: Optional[float] = None
    purpose: str = "data_encryption"
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert key metadata to dictionary (excluding key data)."""
        return {
            'key_id': self.key_id,
            'algorithm': self.algorithm,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'purpose': self.purpose,
            'metadata': self.metadata,
            'is_active': self.is_active,
            'is_expired': self.is_expired()
        }


@dataclass
class EncryptedData:
    """Represents encrypted data with metadata."""
    encrypted_data: bytes
    algorithm: str
    key_id: str
    iv: Optional[bytes] = None
    salt: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)
    data_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'encrypted_data': self.encrypted_data.hex(),
            'algorithm': self.algorithm,
            'key_id': self.key_id,
            'iv': self.iv.hex() if self.iv else None,
            'salt': self.salt.hex() if self.salt else None,
            'timestamp': self.timestamp,
            'data_hash': self.data_hash,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedData':
        """Create from dictionary."""
        return cls(
            encrypted_data=bytes.fromhex(data['encrypted_data']),
            algorithm=data['algorithm'],
            key_id=data['key_id'],
            iv=bytes.fromhex(data['iv']) if data.get('iv') else None,
            salt=bytes.fromhex(data['salt']) if data.get('salt') else None,
            timestamp=data['timestamp'],
            data_hash=data.get('data_hash'),
            metadata=data.get('metadata', {})
        )


class KeyManager:
    """Advanced key management with rotation and security."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.keys: Dict[str, EncryptionKey] = {}
        self.key_history: List[EncryptionKey] = []
        self.master_key: Optional[bytes] = None
        
        # Key rotation settings
        self.rotation_interval = config.get('key_rotation_interval', 86400)  # 24 hours
        self.max_key_age = config.get('max_key_age', 7 * 86400)  # 7 days
        
        # Initialize master key
        self._initialize_master_key()
        
        # Create initial encryption keys
        self._create_initial_keys()
    
    def _initialize_master_key(self):
        """Initialize or load master key."""
        if not HAS_CRYPTOGRAPHY:
            return
        
        master_key_file = Path(self.config.get('master_key_file', '.master_key'))
        
        if master_key_file.exists():
            # Load existing master key
            with open(master_key_file, 'rb') as f:
                self.master_key = f.read()
        else:
            # Generate new master key
            self.master_key = os.urandom(32)  # 256-bit key
            
            # Save master key securely
            with open(master_key_file, 'wb') as f:
                f.write(self.master_key)
            
            # Set restrictive permissions
            os.chmod(str(master_key_file), 0o600)
    
    def _create_initial_keys(self):
        """Create initial set of encryption keys."""
        if not HAS_CRYPTOGRAPHY:
            return
        
        # Create primary symmetric key
        self.create_symmetric_key('primary', 'AES-256-GCM')
        
        # Create asymmetric key pair
        self.create_asymmetric_key_pair('primary_rsa', 'RSA-2048')
    
    def create_symmetric_key(self, key_id: str, algorithm: str = 'AES-256-GCM') -> Optional[EncryptionKey]:
        """Create symmetric encryption key."""
        if not HAS_CRYPTOGRAPHY:
            return None
        
        if algorithm == 'Fernet':
            key_data = Fernet.generate_key()
        elif algorithm == 'AES-256-GCM':
            key_data = os.urandom(32)  # 256-bit key
        else:
            raise ValueError(f"Unsupported symmetric algorithm: {algorithm}")
        
        key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            key_data=key_data,
            created_at=time.time(),
            expires_at=time.time() + self.max_key_age,
            purpose="symmetric_encryption"
        )
        
        self.keys[key_id] = key
        return key
    
    def create_asymmetric_key_pair(self, key_id: str, algorithm: str = 'RSA-2048') -> Optional[Tuple[EncryptionKey, EncryptionKey]]:
        """Create asymmetric key pair."""
        if not HAS_CRYPTOGRAPHY:
            return None
        
        if algorithm == 'RSA-2048':
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
        else:
            raise ValueError(f"Unsupported asymmetric algorithm: {algorithm}")
        
        # Create private key object
        private_key_obj = EncryptionKey(
            key_id=f"{key_id}_private",
            algorithm=algorithm,
            key_data=private_pem,
            created_at=time.time(),
            expires_at=time.time() + self.max_key_age,
            purpose="asymmetric_private"
        )
        
        # Create public key object
        public_key_obj = EncryptionKey(
            key_id=f"{key_id}_public",
            algorithm=algorithm,
            key_data=public_pem,
            created_at=time.time(),
            expires_at=time.time() + self.max_key_age,
            purpose="asymmetric_public"
        )
        
        self.keys[private_key_obj.key_id] = private_key_obj
        self.keys[public_key_obj.key_id] = public_key_obj
        
        return private_key_obj, public_key_obj
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get encryption key by ID."""
        key = self.keys.get(key_id)
        
        if key and key.is_expired():
            # Key is expired, move to history and remove from active keys
            self.key_history.append(key)
            key.is_active = False
            del self.keys[key_id]
            return None
        
        return key
    
    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Rotate encryption key."""
        old_key = self.keys.get(key_id)
        if not old_key:
            return None
        
        # Create new key with same algorithm and purpose
        if old_key.purpose == "symmetric_encryption":
            new_key = self.create_symmetric_key(key_id, old_key.algorithm)
        elif old_key.purpose == "asymmetric_private":
            # For asymmetric keys, rotate the pair
            base_id = key_id.replace('_private', '')
            pair = self.create_asymmetric_key_pair(base_id, old_key.algorithm)
            new_key = pair[0] if pair else None
        else:
            return None
        
        # Move old key to history
        if old_key:
            old_key.is_active = False
            self.key_history.append(old_key)
        
        return new_key
    
    def cleanup_expired_keys(self) -> int:
        """Clean up expired keys."""
        expired_keys = []
        
        for key_id, key in self.keys.items():
            if key.is_expired():
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            key = self.keys[key_id]
            key.is_active = False
            self.key_history.append(key)
            del self.keys[key_id]
        
        return len(expired_keys)
    
    def get_key_metrics(self) -> Dict[str, Any]:
        """Get key management metrics."""
        active_keys = len(self.keys)
        expired_keys = len([k for k in self.key_history if k.is_expired()])
        
        algorithms = {}
        for key in self.keys.values():
            algorithms[key.algorithm] = algorithms.get(key.algorithm, 0) + 1
        
        return {
            'active_keys': active_keys,
            'historical_keys': len(self.key_history),
            'expired_keys': expired_keys,
            'algorithms_in_use': algorithms,
            'rotation_interval': self.rotation_interval,
            'max_key_age': self.max_key_age
        }


class EncryptionManager:
    """Comprehensive encryption manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.key_manager = KeyManager(config)
        
        # Encryption settings
        self.default_algorithm = config.get('default_algorithm', 'AES-256-GCM')
        self.compression_enabled = config.get('compression_enabled', True)
        
        # Quantum-safe preparation
        self.quantum_safe_mode = config.get('quantum_safe_mode', False)
    
    def encrypt_data(self, data: Union[str, bytes, Dict], 
                    algorithm: Optional[str] = None,
                    key_id: Optional[str] = None,
                    classification: str = 'internal') -> Optional[EncryptedData]:
        """Encrypt data with specified or default algorithm."""
        if not HAS_CRYPTOGRAPHY:
            return None
        
        # Prepare data
        if isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Calculate hash for integrity
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Choose algorithm and key
        algorithm = algorithm or self.default_algorithm
        key_id = key_id or 'primary'
        
        # Get encryption key
        encryption_key = self.key_manager.get_key(key_id)
        if not encryption_key:
            return None
        
        try:
            if algorithm == 'Fernet':
                return self._encrypt_fernet(data_bytes, encryption_key, data_hash, classification)
            elif algorithm == 'AES-256-GCM':
                return self._encrypt_aes_gcm(data_bytes, encryption_key, data_hash, classification)
            elif algorithm == 'RSA-2048':
                return self._encrypt_rsa(data_bytes, encryption_key, data_hash, classification)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
                
        except Exception as e:
            print(f"Encryption failed: {e}")
            return None
    
    def _encrypt_fernet(self, data: bytes, key: EncryptionKey, 
                       data_hash: str, classification: str) -> EncryptedData:
        """Encrypt using Fernet (symmetric)."""
        cipher = Fernet(key.key_data)
        encrypted_data = cipher.encrypt(data)
        
        return EncryptedData(
            encrypted_data=encrypted_data,
            algorithm='Fernet',
            key_id=key.key_id,
            data_hash=data_hash,
            metadata={'classification': classification}
        )
    
    def _encrypt_aes_gcm(self, data: bytes, key: EncryptionKey, 
                        data_hash: str, classification: str) -> EncryptedData:
        """Encrypt using AES-256-GCM."""
        # Generate random IV
        iv = os.urandom(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data) + encryptor.finalize()
        
        # Combine encrypted data with authentication tag
        encrypted_with_tag = encrypted_data + encryptor.tag
        
        return EncryptedData(
            encrypted_data=encrypted_with_tag,
            algorithm='AES-256-GCM',
            key_id=key.key_id,
            iv=iv,
            data_hash=data_hash,
            metadata={'classification': classification}
        )
    
    def _encrypt_rsa(self, data: bytes, key: EncryptionKey, 
                    data_hash: str, classification: str) -> EncryptedData:
        """Encrypt using RSA (asymmetric)."""
        # Load public key for encryption
        public_key_id = key.key_id.replace('_private', '_public')
        public_key_obj = self.key_manager.get_key(public_key_id)
        
        if not public_key_obj:
            raise ValueError("Public key not found for RSA encryption")
        
        public_key = serialization.load_pem_public_key(
            public_key_obj.key_data,
            backend=default_backend()
        )
        
        # RSA has size limitations, so we might need to chunk large data
        max_chunk_size = (public_key.key_size // 8) - 42  # OAEP padding overhead
        
        if len(data) <= max_chunk_size:
            # Encrypt directly
            encrypted_data = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            # Use hybrid encryption (RSA + AES)
            # Generate random AES key
            aes_key = os.urandom(32)
            
            # Encrypt data with AES
            iv = os.urandom(12)
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            aes_encrypted = encryptor.update(data) + encryptor.finalize()
            aes_encrypted_with_tag = aes_encrypted + encryptor.tag
            
            # Encrypt AES key with RSA
            encrypted_aes_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key + IV + encrypted data
            encrypted_data = encrypted_aes_key + iv + aes_encrypted_with_tag
        
        return EncryptedData(
            encrypted_data=encrypted_data,
            algorithm='RSA-2048',
            key_id=key.key_id,
            data_hash=data_hash,
            metadata={'classification': classification, 'hybrid': len(data) > max_chunk_size}
        )
    
    def decrypt_data(self, encrypted_data: EncryptedData) -> Optional[Union[str, bytes, Dict]]:
        """Decrypt data using stored metadata."""
        if not HAS_CRYPTOGRAPHY:
            return None
        
        # Get decryption key
        key = self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            # Try to find key in history
            for historical_key in self.key_manager.key_history:
                if historical_key.key_id == encrypted_data.key_id:
                    key = historical_key
                    break
        
        if not key:
            return None
        
        try:
            if encrypted_data.algorithm == 'Fernet':
                return self._decrypt_fernet(encrypted_data, key)
            elif encrypted_data.algorithm == 'AES-256-GCM':
                return self._decrypt_aes_gcm(encrypted_data, key)
            elif encrypted_data.algorithm == 'RSA-2048':
                return self._decrypt_rsa(encrypted_data, key)
            else:
                raise ValueError(f"Unsupported decryption algorithm: {encrypted_data.algorithm}")
                
        except Exception as e:
            print(f"Decryption failed: {e}")
            return None
    
    def _decrypt_fernet(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using Fernet."""
        cipher = Fernet(key.key_data)
        decrypted = cipher.decrypt(encrypted_data.encrypted_data)
        
        # Verify integrity
        if encrypted_data.data_hash:
            computed_hash = hashlib.sha256(decrypted).hexdigest()
            if computed_hash != encrypted_data.data_hash:
                raise ValueError("Data integrity check failed")
        
        return decrypted
    
    def _decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using AES-256-GCM."""
        # Split encrypted data and tag
        encrypted_bytes = encrypted_data.encrypted_data[:-16]  # All but last 16 bytes
        tag = encrypted_data.encrypted_data[-16:]  # Last 16 bytes
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(encrypted_data.iv, tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(encrypted_bytes) + decryptor.finalize()
        
        # Verify integrity
        if encrypted_data.data_hash:
            computed_hash = hashlib.sha256(decrypted).hexdigest()
            if computed_hash != encrypted_data.data_hash:
                raise ValueError("Data integrity check failed")
        
        return decrypted
    
    def _decrypt_rsa(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using RSA."""
        # Load private key
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=default_backend()
        )
        
        is_hybrid = encrypted_data.metadata.get('hybrid', False)
        
        if not is_hybrid:
            # Direct RSA decryption
            decrypted = private_key.decrypt(
                encrypted_data.encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            # Hybrid decryption (RSA + AES)
            # Extract components
            key_size = private_key.key_size // 8
            encrypted_aes_key = encrypted_data.encrypted_data[:key_size]
            iv = encrypted_data.encrypted_data[key_size:key_size + 12]
            aes_encrypted_with_tag = encrypted_data.encrypted_data[key_size + 12:]
            
            # Decrypt AES key with RSA
            aes_key = private_key.decrypt(
                encrypted_aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt data with AES
            aes_encrypted = aes_encrypted_with_tag[:-16]
            tag = aes_encrypted_with_tag[-16:]
            
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(aes_encrypted) + decryptor.finalize()
        
        # Verify integrity
        if encrypted_data.data_hash:
            computed_hash = hashlib.sha256(decrypted).hexdigest()
            if computed_hash != encrypted_data.data_hash:
                raise ValueError("Data integrity check failed")
        
        return decrypted
    
    def get_encryption_metrics(self) -> Dict[str, Any]:
        """Get encryption system metrics."""
        key_metrics = self.key_manager.get_key_metrics()
        
        return {
            'encryption_enabled': HAS_CRYPTOGRAPHY,
            'quantum_safe_ready': HAS_PQC and self.quantum_safe_mode,
            'default_algorithm': self.default_algorithm,
            'key_management': key_metrics,
            'supported_algorithms': [
                'Fernet', 'AES-256-GCM', 'RSA-2048'
            ] if HAS_CRYPTOGRAPHY else []
        }


class DataProtection:
    """High-level data protection with automatic encryption."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_manager = EncryptionManager(config)
        
        # Data classification rules
        self.classification_rules = config.get('classification_rules', {})
        self.auto_encrypt_patterns = config.get('auto_encrypt_patterns', [])
        
        # Protection policies
        self.require_encryption = config.get('require_encryption', True)
        self.encryption_at_rest = config.get('encryption_at_rest', True)
        self.encryption_in_transit = config.get('encryption_in_transit', True)
    
    def protect_data(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Protect data based on classification and policies."""
        context = context or {}
        
        # Classify data
        classification = self._classify_data(data, context)
        
        # Determine protection level
        protection_level = self._get_protection_level(classification)
        
        if protection_level in ['sensitive', 'restricted', 'confidential']:
            # Encrypt sensitive data
            encrypted = self.encryption_manager.encrypt_data(
                data, 
                classification=classification
            )
            
            if encrypted:
                return {
                    'protected': True,
                    'encrypted': True,
                    'classification': classification,
                    'data': encrypted.to_dict()
                }
        
        # Return unprotected for public/internal data
        return {
            'protected': False,
            'encrypted': False,
            'classification': classification,
            'data': data
        }
    
    def unprotect_data(self, protected_data: Dict[str, Any]) -> Any:
        """Unprotect data if encrypted."""
        if not protected_data.get('encrypted', False):
            return protected_data['data']
        
        # Reconstruct encrypted data object
        encrypted_data = EncryptedData.from_dict(protected_data['data'])
        
        # Decrypt
        decrypted = self.encryption_manager.decrypt_data(encrypted_data)
        
        # Try to parse as JSON if possible
        if isinstance(decrypted, bytes):
            try:
                return json.loads(decrypted.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return decrypted.decode('utf-8')
        
        return decrypted
    
    def _classify_data(self, data: Any, context: Dict) -> str:
        """Classify data based on content and context."""
        # Convert data to string for analysis
        data_str = str(data).lower()
        
        # Check for sensitive patterns
        sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'credential',
            'ssn', 'social security', 'credit card', 'bank account'
        ]
        
        if any(pattern in data_str for pattern in sensitive_patterns):
            return 'confidential'
        
        # Check context-based classification
        if context.get('user_data', False):
            return 'sensitive'
        
        if context.get('research_data', False):
            return 'internal'
        
        # Default classification
        return 'internal'
    
    def _get_protection_level(self, classification: str) -> str:
        """Get protection level for data classification."""
        protection_map = {
            'public': 'none',
            'internal': 'basic',
            'sensitive': 'sensitive',
            'restricted': 'restricted',
            'confidential': 'confidential'
        }
        
        return protection_map.get(classification, 'basic')