"""
Quantum Hyper Search Security Framework

Comprehensive enterprise-grade security module providing:
- Authentication & Authorization
- Encryption & Data Protection
- Audit Logging & Compliance
- Rate Limiting & Session Management
- Input Validation & Sanitization
"""

from .quantum_security_framework import (
    QuantumSecurityFramework,
    SecurityPolicy,
    SecurityEvent,
    ComplianceManager,
    CryptographyManager,
    AuditLogger,
    SessionManager
)

from .authentication import (
    AuthenticationManager,
    TokenManager,
    MFAManager
)

from .authorization import (
    AuthorizationManager,
    RoleBasedAccessControl,
    PermissionManager
)

from .encryption import (
    EncryptionManager,
    KeyManager,
    DataProtection
)

from .compliance import (
    ComplianceFramework,
    HIPAACompliance,
    GDPRCompliance,
    SOXCompliance
)

__version__ = "1.0.0"

__all__ = [
    "QuantumSecurityFramework",
    "SecurityPolicy", 
    "SecurityEvent",
    "ComplianceManager",
    "CryptographyManager",
    "AuditLogger",
    "SessionManager",
    "AuthenticationManager",
    "TokenManager",
    "MFAManager",
    "AuthorizationManager",
    "RoleBasedAccessControl",
    "PermissionManager",
    "EncryptionManager",
    "KeyManager",
    "DataProtection",
    "ComplianceFramework",
    "HIPAACompliance",
    "GDPRCompliance",
    "SOXCompliance"
]