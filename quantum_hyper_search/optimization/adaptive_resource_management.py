"""
Adaptive Resource Management for Quantum Optimization

Intelligent resource allocation and management system that dynamically
optimizes computational resources based on workload and performance.
"""

import numpy as np
import psutil
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class ResourceType(Enum):
    """Types of computational resources"""
    CPU = "cpu"
    MEMORY = "memory"
    QUANTUM_QPU = "quantum_qpu"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"

class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    GREEDY = "greedy"
    FAIR_SHARE = "fair_share"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE_LEARNING = "adaptive_learning"
    QUANTUM_AWARE = "quantum_aware"

@dataclass
class ResourceMetrics:
    """System resource metrics"""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mbps: float = 0.0
    gpu_usage_percent: float = 0.0
    quantum_qpu_usage: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ResourceRequest:
    """Resource allocation request"""
    request_id: str
    task_id: str
    priority: int
    estimated_duration: float
    cpu_cores: int = 1
    memory_gb: float = 1.0
    quantum_qpu_time: float = 0.0
    gpu_memory_gb: float = 0.0
    network_bandwidth_mbps: float = 10.0
    storage_gb: float = 1.0
    quantum_advantage_expected: bool = False

@dataclass
class ResourceAllocation:
    """Allocated resources"""
    allocation_id: str
    request_id: str
    allocated_resources: Dict[ResourceType, float]
    start_time: float
    expected_end_time: float
    actual_usage: Dict[ResourceType, float] = field(default_factory=dict)
    performance_score: float = 0.0

class AdaptiveResourceManager:
    """
    Intelligent resource management system that adaptively allocates
    computational resources to optimize quantum optimization performance.
    """
    
    def __init__(
        self,
        allocation_strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE_LEARNING,
        enable_quantum_awareness: bool = True,
        monitoring_interval: float = 1.0,
        learning_rate: float = 0.1
    ):
        self.allocation_strategy = allocation_strategy
        self.enable_quantum_awareness = enable_quantum_awareness
        self.monitoring_interval = monitoring_interval
        self.learning_rate = learning_rate
        
        # System resource limits
        self.system_resources = self._detect_system_resources()
        
        # Active allocations
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_queue: List[ResourceRequest] = []
        
        # Performance history for adaptive learning
        self.performance_history = deque(maxlen=1000)
        self.resource_efficiency_map = {}
        
        # Monitoring
        self.current_metrics = ResourceMetrics()
        self.metrics_history = deque(maxlen=300)  # 5 minutes at 1s intervals
        
        # Control
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = threading.RLock()
    
    def _detect_system_resources(self) -> Dict[ResourceType, float]:
        """Detect available system resources"""
        
        try:
            resources = {}
            
            # CPU cores
            resources[ResourceType.CPU] = psutil.cpu_count(logical=True)
            
            # Memory in GB
            memory = psutil.virtual_memory()
            resources[ResourceType.MEMORY] = memory.total / (1024**3)
            
            # Storage in GB (available on root partition)
            disk = psutil.disk_usage('/')
            resources[ResourceType.STORAGE] = disk.free / (1024**3)
            
            # Network (assume 1Gbps default)
            resources[ResourceType.NETWORK] = 1000.0  # Mbps
            
            # GPU detection (simplified)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                resources[ResourceType.GPU] = sum(gpu.memoryTotal for gpu in gpus) / 1024  # GB
            except ImportError:
                resources[ResourceType.GPU] = 0.0
            
            # Quantum QPU (mock - would integrate with actual quantum backends)
            resources[ResourceType.QUANTUM_QPU] = 100.0  # Arbitrary QPU time units
            
            logger.info(f"Detected system resources: {resources}")
            return resources
            
        except Exception as e:
            logger.error(f"Failed to detect system resources: {e}")
            # Fallback defaults
            return {
                ResourceType.CPU: 4.0,
                ResourceType.MEMORY: 8.0,
                ResourceType.STORAGE: 100.0,
                ResourceType.NETWORK: 100.0,
                ResourceType.GPU: 0.0,
                ResourceType.QUANTUM_QPU: 10.0
            }
    
    def start_monitoring(self):
        """Start resource monitoring"""
        
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self):
        """Main resource monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect current metrics
                self.current_metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(self.current_metrics)
                    
                    # Update active allocation usage
                    self._update_allocation_usage()
                    
                    # Check for resource violations
                    self._check_resource_violations()
                    
                    # Adaptive learning update
                    if self.allocation_strategy == AllocationStrategy.ADAPTIVE_LEARNING:
                        self._update_learning_model()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics"""
        
        try:
            metrics = ResourceMetrics()
            
            # CPU usage
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.memory_usage_percent = memory.percent
            metrics.memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O (simplified)
            network = psutil.net_io_counters()
            if hasattr(self, '_last_network_bytes'):
                bytes_diff = (network.bytes_sent + network.bytes_recv) - self._last_network_bytes
                metrics.network_io_mbps = (bytes_diff * 8) / (1024 * 1024 * self.monitoring_interval)
            self._last_network_bytes = network.bytes_sent + network.bytes_recv
            
            # GPU usage (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    metrics.gpu_usage_percent = np.mean([gpu.load * 100 for gpu in gpus])
            except ImportError:
                pass
            
            # Quantum QPU usage (mock)
            active_quantum_allocations = sum(
                alloc.allocated_resources.get(ResourceType.QUANTUM_QPU, 0)
                for alloc in self.active_allocations.values()
            )
            max_quantum = self.system_resources.get(ResourceType.QUANTUM_QPU, 100)
            metrics.quantum_qpu_usage = (active_quantum_allocations / max_quantum) * 100
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return ResourceMetrics()
    
    def request_resources(self, request: ResourceRequest) -> Optional[str]:
        """
        Request resource allocation
        
        Args:
            request: Resource request specification
            
        Returns:
            Allocation ID if successful, None if cannot allocate
        """
        
        with self._lock:
            # Check if resources are available
            if self._can_allocate_resources(request):
                allocation = self._allocate_resources(request)
                self.active_allocations[allocation.allocation_id] = allocation
                
                logger.info(f"Allocated resources for request {request.request_id}: "
                          f"allocation {allocation.allocation_id}")
                
                return allocation.allocation_id
            else:
                # Queue the request for later allocation
                self._queue_request(request)
                logger.info(f"Queued resource request {request.request_id}")
                return None
    
    def _can_allocate_resources(self, request: ResourceRequest) -> bool:
        """Check if requested resources can be allocated"""
        
        # Calculate current resource usage
        current_usage = self._calculate_current_usage()
        
        # Check each resource type
        required_resources = {
            ResourceType.CPU: request.cpu_cores,
            ResourceType.MEMORY: request.memory_gb,
            ResourceType.QUANTUM_QPU: request.quantum_qpu_time,
            ResourceType.GPU: request.gpu_memory_gb,
            ResourceType.NETWORK: request.network_bandwidth_mbps,
            ResourceType.STORAGE: request.storage_gb
        }
        
        for resource_type, required in required_resources.items():
            if required > 0:
                available = (
                    self.system_resources.get(resource_type, 0) - 
                    current_usage.get(resource_type, 0)
                )
                
                if required > available:
                    logger.debug(f"Insufficient {resource_type.value}: "
                               f"required {required}, available {available}")
                    return False
        
        return True
    
    def _calculate_current_usage(self) -> Dict[ResourceType, float]:
        """Calculate current resource usage from active allocations"""
        
        usage = {resource_type: 0.0 for resource_type in ResourceType}
        
        for allocation in self.active_allocations.values():
            for resource_type, amount in allocation.allocated_resources.items():
                usage[resource_type] += amount
        
        return usage
    
    def _allocate_resources(self, request: ResourceRequest) -> ResourceAllocation:
        """Actually allocate resources for a request"""
        
        allocation_id = f"alloc_{int(time.time() * 1000)}_{request.request_id[:8]}"
        
        # Apply allocation strategy
        if self.allocation_strategy == AllocationStrategy.ADAPTIVE_LEARNING:
            allocated_resources = self._adaptive_allocation(request)
        elif self.allocation_strategy == AllocationStrategy.QUANTUM_AWARE:
            allocated_resources = self._quantum_aware_allocation(request)
        else:
            allocated_resources = self._basic_allocation(request)
        
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            request_id=request.request_id,
            allocated_resources=allocated_resources,
            start_time=time.time(),
            expected_end_time=time.time() + request.estimated_duration
        )
        
        return allocation
    
    def _basic_allocation(self, request: ResourceRequest) -> Dict[ResourceType, float]:
        """Basic resource allocation - give exactly what's requested"""
        
        return {
            ResourceType.CPU: request.cpu_cores,
            ResourceType.MEMORY: request.memory_gb,
            ResourceType.QUANTUM_QPU: request.quantum_qpu_time,
            ResourceType.GPU: request.gpu_memory_gb,
            ResourceType.NETWORK: request.network_bandwidth_mbps,
            ResourceType.STORAGE: request.storage_gb
        }
    
    def _adaptive_allocation(self, request: ResourceRequest) -> Dict[ResourceType, float]:
        """Adaptive allocation based on historical performance"""
        
        base_allocation = self._basic_allocation(request)
        
        # Apply performance-based adjustments
        for resource_type in ResourceType:
            if resource_type in self.resource_efficiency_map:
                efficiency = self.resource_efficiency_map[resource_type]
                
                # Increase allocation for resources showing good efficiency
                if efficiency > 1.2:  # 20% better than average
                    base_allocation[resource_type] *= 1.1
                elif efficiency < 0.8:  # 20% worse than average
                    base_allocation[resource_type] *= 0.9
        
        # Quantum-specific adaptations
        if request.quantum_advantage_expected and self.enable_quantum_awareness:
            # Prioritize quantum resources and complementary classical resources
            base_allocation[ResourceType.QUANTUM_QPU] *= 1.2
            base_allocation[ResourceType.CPU] *= 0.8  # Reduce CPU when using quantum
        
        return base_allocation
    
    def _quantum_aware_allocation(self, request: ResourceRequest) -> Dict[ResourceType, float]:
        """Quantum-aware resource allocation strategy"""
        
        base_allocation = self._basic_allocation(request)
        
        if request.quantum_advantage_expected:
            # Quantum jobs get priority and optimized resource mix
            base_allocation[ResourceType.QUANTUM_QPU] *= 1.5
            base_allocation[ResourceType.MEMORY] *= 1.2  # More memory for quantum state preparation
            base_allocation[ResourceType.CPU] *= 0.7     # Less CPU needed with quantum acceleration
            
            # Ensure we don't exceed system limits
            for resource_type, amount in base_allocation.items():
                max_available = self.system_resources.get(resource_type, 0) * 0.8  # 80% max
                base_allocation[resource_type] = min(amount, max_available)
        
        return base_allocation
    
    def _queue_request(self, request: ResourceRequest):
        """Queue a resource request for later allocation"""
        
        # Insert in priority order
        inserted = False
        for i, queued_request in enumerate(self.allocation_queue):
            if request.priority > queued_request.priority:
                self.allocation_queue.insert(i, request)
                inserted = True
                break
        
        if not inserted:
            self.allocation_queue.append(request)
        
        # Try to process queue
        self._process_allocation_queue()
    
    def _process_allocation_queue(self):
        """Process queued allocation requests"""
        
        processed = []
        
        for i, request in enumerate(self.allocation_queue):
            if self._can_allocate_resources(request):
                allocation = self._allocate_resources(request)
                self.active_allocations[allocation.allocation_id] = allocation
                processed.append(i)
                
                logger.info(f"Processed queued request {request.request_id}")
        
        # Remove processed requests
        for i in reversed(processed):
            del self.allocation_queue[i]
    
    def release_resources(self, allocation_id: str) -> bool:
        """
        Release allocated resources
        
        Args:
            allocation_id: ID of allocation to release
            
        Returns:
            True if successfully released
        """
        
        with self._lock:
            if allocation_id in self.active_allocations:
                allocation = self.active_allocations[allocation_id]
                
                # Record performance metrics
                self._record_allocation_performance(allocation)
                
                # Remove allocation
                del self.active_allocations[allocation_id]
                
                # Process any queued requests
                self._process_allocation_queue()
                
                logger.info(f"Released resources for allocation {allocation_id}")
                return True
            else:
                logger.warning(f"Attempted to release unknown allocation {allocation_id}")
                return False
    
    def _record_allocation_performance(self, allocation: ResourceAllocation):
        """Record performance metrics for adaptive learning"""
        
        actual_duration = time.time() - allocation.start_time
        expected_duration = allocation.expected_end_time - allocation.start_time
        
        # Calculate performance score
        duration_efficiency = expected_duration / max(actual_duration, 0.1)
        resource_efficiency = self._calculate_resource_efficiency(allocation)
        
        performance_score = (duration_efficiency + resource_efficiency) / 2.0
        allocation.performance_score = performance_score
        
        # Store in performance history
        performance_record = {
            'allocation_id': allocation.allocation_id,
            'duration_efficiency': duration_efficiency,
            'resource_efficiency': resource_efficiency,
            'performance_score': performance_score,
            'allocated_resources': allocation.allocated_resources.copy(),
            'timestamp': time.time()
        }
        
        self.performance_history.append(performance_record)
        
        logger.debug(f"Recorded performance for {allocation.allocation_id}: "
                    f"score={performance_score:.3f}")
    
    def _calculate_resource_efficiency(self, allocation: ResourceAllocation) -> float:
        """Calculate resource efficiency for an allocation"""
        
        if not allocation.actual_usage:
            return 1.0  # Assume perfect if no usage data
        
        efficiency_scores = []
        
        for resource_type, allocated in allocation.allocated_resources.items():
            if allocated > 0 and resource_type in allocation.actual_usage:
                actual = allocation.actual_usage[resource_type]
                efficiency = min(1.0, actual / allocated)  # Cap at 1.0
                efficiency_scores.append(efficiency)
        
        return np.mean(efficiency_scores) if efficiency_scores else 1.0
    
    def _update_allocation_usage(self):
        """Update actual resource usage for active allocations"""
        
        # Simplified usage estimation based on current metrics
        total_cpu_allocated = sum(
            alloc.allocated_resources.get(ResourceType.CPU, 0)
            for alloc in self.active_allocations.values()
        )
        
        total_memory_allocated = sum(
            alloc.allocated_resources.get(ResourceType.MEMORY, 0)
            for alloc in self.active_allocations.values()
        )
        
        if total_cpu_allocated > 0 and total_memory_allocated > 0:
            # Distribute actual usage proportionally
            cpu_usage_fraction = self.current_metrics.cpu_usage_percent / 100.0
            memory_usage_fraction = self.current_metrics.memory_usage_percent / 100.0
            
            for allocation in self.active_allocations.values():
                allocated_cpu = allocation.allocated_resources.get(ResourceType.CPU, 0)
                allocated_memory = allocation.allocated_resources.get(ResourceType.MEMORY, 0)
                
                if allocated_cpu > 0:
                    actual_cpu = (allocated_cpu / total_cpu_allocated) * cpu_usage_fraction * allocated_cpu
                    allocation.actual_usage[ResourceType.CPU] = actual_cpu
                
                if allocated_memory > 0:
                    actual_memory = (allocated_memory / total_memory_allocated) * memory_usage_fraction * allocated_memory
                    allocation.actual_usage[ResourceType.MEMORY] = actual_memory
    
    def _check_resource_violations(self):
        """Check for resource usage violations and take corrective action"""
        
        # Check for over-allocation
        if self.current_metrics.memory_usage_percent > 90:
            logger.warning("High memory usage detected, considering reallocation")
            self._handle_resource_pressure(ResourceType.MEMORY)
        
        if self.current_metrics.cpu_usage_percent > 95:
            logger.warning("High CPU usage detected, considering reallocation")
            self._handle_resource_pressure(ResourceType.CPU)
    
    def _handle_resource_pressure(self, resource_type: ResourceType):
        """Handle resource pressure by optimizing allocations"""
        
        # Find allocations that are using less than allocated
        underutilized_allocations = []
        
        for allocation in self.active_allocations.values():
            allocated = allocation.allocated_resources.get(resource_type, 0)
            actual = allocation.actual_usage.get(resource_type, allocated)
            
            if allocated > 0 and actual < allocated * 0.7:  # Using <70% of allocation
                underutilized_allocations.append((allocation, allocated - actual))
        
        # Sort by amount of unused resources
        underutilized_allocations.sort(key=lambda x: x[1], reverse=True)
        
        # Reduce allocation for top underutilized tasks
        for allocation, unused in underutilized_allocations[:3]:  # Top 3
            reduction = unused * 0.5  # Reduce by 50% of unused
            allocation.allocated_resources[resource_type] -= reduction
            
            logger.info(f"Reduced {resource_type.value} allocation for "
                       f"{allocation.allocation_id} by {reduction:.2f}")
    
    def _update_learning_model(self):
        """Update adaptive learning model with recent performance data"""
        
        if len(self.performance_history) < 10:
            return  # Need more data
        
        # Calculate efficiency for each resource type
        recent_records = list(self.performance_history)[-50:]  # Last 50 records
        
        resource_scores = {resource_type: [] for resource_type in ResourceType}
        
        for record in recent_records:
            for resource_type in ResourceType:
                if resource_type in record['allocated_resources']:
                    resource_scores[resource_type].append(record['performance_score'])
        
        # Update efficiency map
        for resource_type, scores in resource_scores.items():
            if scores:
                current_efficiency = np.mean(scores)
                
                if resource_type not in self.resource_efficiency_map:
                    self.resource_efficiency_map[resource_type] = current_efficiency
                else:
                    # Exponential moving average
                    old_efficiency = self.resource_efficiency_map[resource_type]
                    new_efficiency = (
                        (1 - self.learning_rate) * old_efficiency + 
                        self.learning_rate * current_efficiency
                    )
                    self.resource_efficiency_map[resource_type] = new_efficiency
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status"""
        
        with self._lock:
            current_usage = self._calculate_current_usage()
            
            status = {
                'system_resources': self.system_resources.copy(),
                'current_usage': current_usage,
                'current_metrics': {
                    'cpu_percent': self.current_metrics.cpu_usage_percent,
                    'memory_percent': self.current_metrics.memory_usage_percent,
                    'memory_available_gb': self.current_metrics.memory_available_gb,
                    'disk_percent': self.current_metrics.disk_usage_percent,
                    'quantum_qpu_percent': self.current_metrics.quantum_qpu_usage
                },
                'active_allocations': len(self.active_allocations),
                'queued_requests': len(self.allocation_queue),
                'resource_efficiency': self.resource_efficiency_map.copy(),
                'allocation_strategy': self.allocation_strategy.value
            }
            
            # Resource availability
            status['resource_availability'] = {}
            for resource_type, total in self.system_resources.items():
                used = current_usage.get(resource_type, 0)
                available = total - used
                utilization = (used / total * 100) if total > 0 else 0
                
                status['resource_availability'][resource_type.value] = {
                    'total': total,
                    'used': used,
                    'available': available,
                    'utilization_percent': utilization
                }
            
            return status
    
    def optimize_allocations(self) -> Dict[str, Any]:
        """Optimize current resource allocations"""
        
        with self._lock:
            optimization_actions = []
            
            # Find optimization opportunities
            for allocation_id, allocation in self.active_allocations.items():
                actions = self._analyze_allocation_optimization(allocation)
                if actions:
                    optimization_actions.extend(actions)
            
            # Apply optimizations
            applied_optimizations = []
            for action in optimization_actions:
                if self._apply_optimization_action(action):
                    applied_optimizations.append(action)
            
            return {
                'total_opportunities': len(optimization_actions),
                'applied_optimizations': len(applied_optimizations),
                'optimization_actions': applied_optimizations
            }
    
    def _analyze_allocation_optimization(self, allocation: ResourceAllocation) -> List[Dict[str, Any]]:
        """Analyze optimization opportunities for an allocation"""
        
        actions = []
        
        for resource_type, allocated in allocation.allocated_resources.items():
            actual = allocation.actual_usage.get(resource_type, allocated)
            
            # Significant over-allocation
            if allocated > 0 and actual < allocated * 0.5:
                actions.append({
                    'type': 'reduce_allocation',
                    'allocation_id': allocation.allocation_id,
                    'resource_type': resource_type,
                    'current_allocation': allocated,
                    'suggested_allocation': actual * 1.2,  # 20% buffer
                    'potential_savings': allocated - (actual * 1.2)
                })
            
            # Under-allocation causing performance issues
            elif actual > allocated * 1.1:
                actions.append({
                    'type': 'increase_allocation',
                    'allocation_id': allocation.allocation_id,
                    'resource_type': resource_type,
                    'current_allocation': allocated,
                    'suggested_allocation': actual * 1.1,  # 10% buffer
                    'additional_needed': (actual * 1.1) - allocated
                })
        
        return actions
    
    def _apply_optimization_action(self, action: Dict[str, Any]) -> bool:
        """Apply an optimization action"""
        
        allocation_id = action['allocation_id']
        if allocation_id not in self.active_allocations:
            return False
        
        allocation = self.active_allocations[allocation_id]
        resource_type = action['resource_type']
        
        try:
            if action['type'] == 'reduce_allocation':
                allocation.allocated_resources[resource_type] = action['suggested_allocation']
                logger.info(f"Reduced {resource_type.value} allocation for {allocation_id}")
                return True
                
            elif action['type'] == 'increase_allocation':
                # Check if additional resources are available
                current_usage = self._calculate_current_usage()
                available = (
                    self.system_resources.get(resource_type, 0) - 
                    current_usage.get(resource_type, 0)
                )
                
                additional_needed = action['additional_needed']
                if additional_needed <= available:
                    allocation.allocated_resources[resource_type] = action['suggested_allocation']
                    logger.info(f"Increased {resource_type.value} allocation for {allocation_id}")
                    return True
                else:
                    logger.debug(f"Cannot increase allocation: insufficient {resource_type.value}")
                    
        except Exception as e:
            logger.error(f"Failed to apply optimization action: {e}")
        
        return False