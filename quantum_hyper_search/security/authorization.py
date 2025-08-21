"""
Authorization Module - Role-based access control and permissions.

Provides comprehensive authorization including RBAC, permissions,
and fine-grained access control for quantum optimization resources.
"""

import time
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class Permission(Enum):
    """Enumeration of system permissions."""
    # Quantum optimization permissions
    QUANTUM_OPTIMIZE = "quantum:optimize"
    QUANTUM_VIEW_RESULTS = "quantum:view_results"
    QUANTUM_EXPORT_DATA = "quantum:export_data"
    QUANTUM_CONFIGURE = "quantum:configure"
    
    # Research permissions
    RESEARCH_ACCESS = "research:access"
    RESEARCH_EXPERIMENT = "research:experiment"
    RESEARCH_BENCHMARK = "research:benchmark"
    RESEARCH_PUBLISH = "research:publish"
    
    # Administrative permissions
    ADMIN_USER_MANAGE = "admin:user_manage"
    ADMIN_SYSTEM_CONFIG = "admin:system_config"
    ADMIN_SECURITY_MANAGE = "admin:security_manage"
    ADMIN_AUDIT_VIEW = "admin:audit_view"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    
    # Monitoring permissions
    MONITOR_VIEW = "monitor:view"
    MONITOR_ALERTS = "monitor:alerts"
    MONITOR_METRICS = "monitor:metrics"


@dataclass
class Role:
    """Represents a role with permissions and metadata."""
    name: str
    permissions: Set[Permission]
    description: str = ""
    is_system_role: bool = False
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission."""
        return permission in self.permissions
    
    def add_permission(self, permission: Permission):
        """Add permission to role."""
        if not self.is_system_role:
            self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role."""
        if not self.is_system_role:
            self.permissions.discard(permission)


@dataclass
class AccessContext:
    """Context for access control decisions."""
    user_id: str
    resource: str
    action: str
    ip_address: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthorizationManager:
    """Comprehensive authorization manager with RBAC."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = defaultdict(set)
        self.resource_policies: Dict[str, Dict[str, Any]] = {}
        self.access_history: List[Dict[str, Any]] = []
        
        # Initialize system roles
        self._initialize_system_roles()
        
        # Custom authorization functions
        self.custom_authorizers: Dict[str, Callable] = {}
    
    def _initialize_system_roles(self):
        """Initialize system-defined roles."""
        # Super Admin - all permissions
        super_admin = Role(
            name="super_admin",
            permissions=set(Permission),
            description="Full system access",
            is_system_role=True
        )
        
        # Quantum Researcher - research and optimization
        researcher = Role(
            name="quantum_researcher",
            permissions={
                Permission.QUANTUM_OPTIMIZE,
                Permission.QUANTUM_VIEW_RESULTS,
                Permission.QUANTUM_EXPORT_DATA,
                Permission.RESEARCH_ACCESS,
                Permission.RESEARCH_EXPERIMENT,
                Permission.RESEARCH_BENCHMARK,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.MONITOR_VIEW,
                Permission.MONITOR_METRICS
            },
            description="Quantum research and optimization access",
            is_system_role=True
        )
        
        # Data Scientist - analysis and viewing
        data_scientist = Role(
            name="data_scientist",
            permissions={
                Permission.QUANTUM_VIEW_RESULTS,
                Permission.QUANTUM_EXPORT_DATA,
                Permission.DATA_READ,
                Permission.DATA_EXPORT,
                Permission.MONITOR_VIEW
            },
            description="Data analysis and viewing access",
            is_system_role=True
        )
        
        # System Administrator - system management
        system_admin = Role(
            name="system_admin",
            permissions={
                Permission.ADMIN_USER_MANAGE,
                Permission.ADMIN_SYSTEM_CONFIG,
                Permission.ADMIN_SECURITY_MANAGE,
                Permission.ADMIN_AUDIT_VIEW,
                Permission.MONITOR_VIEW,
                Permission.MONITOR_ALERTS,
                Permission.MONITOR_METRICS
            },
            description="System administration access",
            is_system_role=True
        )
        
        # Viewer - read-only access
        viewer = Role(
            name="viewer",
            permissions={
                Permission.QUANTUM_VIEW_RESULTS,
                Permission.DATA_READ,
                Permission.MONITOR_VIEW
            },
            description="Read-only access",
            is_system_role=True
        )
        
        # Store system roles
        for role in [super_admin, researcher, data_scientist, system_admin, viewer]:
            self.roles[role.name] = role
    
    def create_role(self, name: str, permissions: List[Permission], 
                   description: str = "") -> bool:
        """Create custom role."""
        if name in self.roles:
            return False  # Role already exists
        
        role = Role(
            name=name,
            permissions=set(permissions),
            description=description,
            is_system_role=False
        )
        
        self.roles[name] = role
        return True
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign role to user."""
        if role_name not in self.roles:
            return False
        
        self.user_roles[user_id].add(role_name)
        return True
    
    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke role from user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            return True
        return False
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user across all roles."""
        permissions = set()
        
        for role_name in self.user_roles.get(user_id, set()):
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)
        
        return permissions
    
    def authorize(self, user_id: str, permission: Permission, 
                 context: Optional[AccessContext] = None) -> bool:
        """Authorize user action with comprehensive checks."""
        # Get user permissions
        user_permissions = self.get_user_permissions(user_id)
        
        # Check basic permission
        if permission not in user_permissions:
            self._record_access_attempt(user_id, permission, False, "insufficient_permissions")
            return False
        
        # Apply resource-specific policies
        if context and not self._check_resource_policies(user_id, context):
            self._record_access_attempt(user_id, permission, False, "resource_policy_denied")
            return False
        
        # Apply custom authorization logic
        if context and not self._check_custom_authorizers(user_id, permission, context):
            self._record_access_attempt(user_id, permission, False, "custom_policy_denied")
            return False
        
        # Authorization successful
        self._record_access_attempt(user_id, permission, True, "authorized")
        return True
    
    def _check_resource_policies(self, user_id: str, context: AccessContext) -> bool:
        """Check resource-specific access policies."""
        resource_policy = self.resource_policies.get(context.resource)
        if not resource_policy:
            return True  # No specific policy, allow
        
        # Check time-based restrictions
        if 'allowed_hours' in resource_policy:
            current_hour = time.localtime(context.timestamp).tm_hour
            if current_hour not in resource_policy['allowed_hours']:
                return False
        
        # Check IP-based restrictions
        if 'allowed_ips' in resource_policy and context.ip_address:
            if context.ip_address not in resource_policy['allowed_ips']:
                return False
        
        # Check action-specific restrictions
        if 'restricted_actions' in resource_policy:
            if context.action in resource_policy['restricted_actions']:
                return False
        
        return True
    
    def _check_custom_authorizers(self, user_id: str, permission: Permission, 
                                 context: AccessContext) -> bool:
        """Apply custom authorization logic."""
        resource_key = f"{context.resource}:{context.action}"
        
        if resource_key in self.custom_authorizers:
            authorizer = self.custom_authorizers[resource_key]
            try:
                return authorizer(user_id, permission, context)
            except Exception:
                # If custom authorizer fails, deny access
                return False
        
        return True
    
    def register_custom_authorizer(self, resource: str, action: str, 
                                  authorizer: Callable[[str, Permission, AccessContext], bool]):
        """Register custom authorization function."""
        key = f"{resource}:{action}"
        self.custom_authorizers[key] = authorizer
    
    def set_resource_policy(self, resource: str, policy: Dict[str, Any]):
        """Set access policy for specific resource."""
        self.resource_policies[resource] = policy
    
    def _record_access_attempt(self, user_id: str, permission: Permission, 
                              granted: bool, reason: str):
        """Record access attempt for audit."""
        attempt = {
            'timestamp': time.time(),
            'user_id': user_id,
            'permission': permission.value,
            'granted': granted,
            'reason': reason
        }
        
        self.access_history.append(attempt)
        
        # Keep only recent history (e.g., last 1000 attempts)
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]
    
    def get_authorization_metrics(self) -> Dict[str, Any]:
        """Get authorization metrics and statistics."""
        if not self.access_history:
            return {}
        
        total_attempts = len(self.access_history)
        granted_attempts = sum(1 for attempt in self.access_history if attempt['granted'])
        
        # Permission usage statistics
        permission_counts = defaultdict(int)
        for attempt in self.access_history:
            permission_counts[attempt['permission']] += 1
        
        # User activity statistics
        user_activity = defaultdict(int)
        for attempt in self.access_history:
            user_activity[attempt['user_id']] += 1
        
        return {
            'total_attempts': total_attempts,
            'granted_attempts': granted_attempts,
            'denied_attempts': total_attempts - granted_attempts,
            'success_rate': (granted_attempts / total_attempts) * 100 if total_attempts > 0 else 0,
            'permission_usage': dict(permission_counts),
            'user_activity': dict(user_activity),
            'active_roles': len(self.roles),
            'users_with_roles': len(self.user_roles)
        }


class RoleBasedAccessControl:
    """Enhanced RBAC with hierarchical roles and dynamic permissions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.authorization_manager = AuthorizationManager(config)
        self.role_hierarchy: Dict[str, Set[str]] = {}  # role -> parent roles
        self.permission_groups: Dict[str, Set[Permission]] = {}
        
        self._initialize_permission_groups()
    
    def _initialize_permission_groups(self):
        """Initialize logical permission groups."""
        self.permission_groups = {
            'quantum_basic': {
                Permission.QUANTUM_VIEW_RESULTS,
                Permission.DATA_READ,
                Permission.MONITOR_VIEW
            },
            'quantum_advanced': {
                Permission.QUANTUM_OPTIMIZE,
                Permission.QUANTUM_CONFIGURE,
                Permission.QUANTUM_EXPORT_DATA,
                Permission.DATA_WRITE
            },
            'research_full': {
                Permission.RESEARCH_ACCESS,
                Permission.RESEARCH_EXPERIMENT,
                Permission.RESEARCH_BENCHMARK,
                Permission.RESEARCH_PUBLISH
            },
            'admin_basic': {
                Permission.ADMIN_AUDIT_VIEW,
                Permission.MONITOR_ALERTS,
                Permission.MONITOR_METRICS
            },
            'admin_full': {
                Permission.ADMIN_USER_MANAGE,
                Permission.ADMIN_SYSTEM_CONFIG,
                Permission.ADMIN_SECURITY_MANAGE
            }
        }
    
    def create_role_hierarchy(self, child_role: str, parent_roles: List[str]) -> bool:
        """Create role hierarchy (child inherits from parents)."""
        if child_role not in self.authorization_manager.roles:
            return False
        
        # Validate parent roles exist
        for parent in parent_roles:
            if parent not in self.authorization_manager.roles:
                return False
        
        self.role_hierarchy[child_role] = set(parent_roles)
        
        # Update child role permissions
        self._update_inherited_permissions(child_role)
        return True
    
    def _update_inherited_permissions(self, role_name: str):
        """Update role permissions based on hierarchy."""
        if role_name not in self.role_hierarchy:
            return
        
        role = self.authorization_manager.roles[role_name]
        if role.is_system_role:
            return  # Don't modify system roles
        
        # Collect inherited permissions
        inherited_permissions = set()
        for parent_role in self.role_hierarchy[role_name]:
            if parent_role in self.authorization_manager.roles:
                parent = self.authorization_manager.roles[parent_role]
                inherited_permissions.update(parent.permissions)
        
        # Add inherited permissions
        role.permissions.update(inherited_permissions)
    
    def assign_permission_group(self, role_name: str, group_name: str) -> bool:
        """Assign all permissions from a permission group to role."""
        if role_name not in self.authorization_manager.roles:
            return False
        
        if group_name not in self.permission_groups:
            return False
        
        role = self.authorization_manager.roles[role_name]
        if role.is_system_role:
            return False  # Can't modify system roles
        
        role.permissions.update(self.permission_groups[group_name])
        return True
    
    def check_access(self, user_id: str, resource: str, action: str, 
                    metadata: Optional[Dict] = None) -> bool:
        """High-level access check with resource and action."""
        # Map resource/action to permission
        permission = self._map_to_permission(resource, action)
        if not permission:
            return False
        
        # Create access context
        context = AccessContext(
            user_id=user_id,
            resource=resource,
            action=action,
            metadata=metadata or {}
        )
        
        return self.authorization_manager.authorize(user_id, permission, context)
    
    def _map_to_permission(self, resource: str, action: str) -> Optional[Permission]:
        """Map resource/action combination to permission."""
        mapping = {
            ('quantum', 'optimize'): Permission.QUANTUM_OPTIMIZE,
            ('quantum', 'view'): Permission.QUANTUM_VIEW_RESULTS,
            ('quantum', 'export'): Permission.QUANTUM_EXPORT_DATA,
            ('quantum', 'configure'): Permission.QUANTUM_CONFIGURE,
            ('research', 'access'): Permission.RESEARCH_ACCESS,
            ('research', 'experiment'): Permission.RESEARCH_EXPERIMENT,
            ('research', 'benchmark'): Permission.RESEARCH_BENCHMARK,
            ('data', 'read'): Permission.DATA_READ,
            ('data', 'write'): Permission.DATA_WRITE,
            ('data', 'delete'): Permission.DATA_DELETE,
            ('admin', 'users'): Permission.ADMIN_USER_MANAGE,
            ('admin', 'system'): Permission.ADMIN_SYSTEM_CONFIG,
            ('monitor', 'view'): Permission.MONITOR_VIEW
        }
        
        return mapping.get((resource, action))
    
    def get_user_accessible_resources(self, user_id: str) -> Dict[str, List[str]]:
        """Get all resources and actions user can access."""
        user_permissions = self.authorization_manager.get_user_permissions(user_id)
        accessible = defaultdict(list)
        
        permission_to_resource = {
            Permission.QUANTUM_OPTIMIZE: ('quantum', 'optimize'),
            Permission.QUANTUM_VIEW_RESULTS: ('quantum', 'view'),
            Permission.QUANTUM_EXPORT_DATA: ('quantum', 'export'),
            Permission.RESEARCH_ACCESS: ('research', 'access'),
            Permission.DATA_READ: ('data', 'read'),
            Permission.DATA_WRITE: ('data', 'write'),
            Permission.ADMIN_USER_MANAGE: ('admin', 'users'),
            Permission.MONITOR_VIEW: ('monitor', 'view')
        }
        
        for permission in user_permissions:
            if permission in permission_to_resource:
                resource, action = permission_to_resource[permission]
                accessible[resource].append(action)
        
        return dict(accessible)


class PermissionManager:
    """Fine-grained permission management."""
    
    def __init__(self):
        self.dynamic_permissions: Dict[str, Permission] = {}
        self.permission_dependencies: Dict[Permission, Set[Permission]] = {}
        self.conditional_permissions: Dict[Permission, Callable] = {}
    
    def create_dynamic_permission(self, name: str, description: str = "") -> Permission:
        """Create dynamic permission at runtime."""
        permission = Permission(name)
        self.dynamic_permissions[name] = permission
        return permission
    
    def set_permission_dependency(self, permission: Permission, 
                                 dependencies: List[Permission]):
        """Set dependencies for permission (requires other permissions)."""
        self.permission_dependencies[permission] = set(dependencies)
    
    def set_conditional_permission(self, permission: Permission, 
                                  condition_func: Callable[[str, Dict], bool]):
        """Set conditional logic for permission evaluation."""
        self.conditional_permissions[permission] = condition_func
    
    def evaluate_permission(self, user_id: str, permission: Permission, 
                          context: Dict[str, Any]) -> bool:
        """Evaluate permission with dependencies and conditions."""
        # Check dependencies
        if permission in self.permission_dependencies:
            # User must have all dependency permissions
            user_permissions = set()  # Would get from authorization manager
            dependencies = self.permission_dependencies[permission]
            if not dependencies.issubset(user_permissions):
                return False
        
        # Check conditional logic
        if permission in self.conditional_permissions:
            condition_func = self.conditional_permissions[permission]
            try:
                return condition_func(user_id, context)
            except Exception:
                return False
        
        return True