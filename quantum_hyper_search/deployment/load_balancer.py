"""
Intelligent load balancer for quantum hyperparameter search services.
"""

import time
import asyncio
import random
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    QUANTUM_AWARE = "quantum_aware"
    ADAPTIVE = "adaptive"


@dataclass
class ServiceEndpoint:
    """Represents a service endpoint."""
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    
    # Runtime metrics
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    last_health_check: float = 0.0
    is_healthy: bool = True
    
    # Quantum-specific metrics
    quantum_queue_length: int = 0
    quantum_success_rate: float = 1.0
    preferred_problem_sizes: List[int] = field(default_factory=list)
    
    @property
    def url(self) -> str:
        """Get endpoint URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times[-100:])  # Last 100 requests
    
    @property
    def load_score(self) -> float:
        """Calculate load score (lower is better)."""
        connection_load = self.current_connections / self.max_connections
        queue_load = min(self.quantum_queue_length / 10.0, 1.0)  # Normalize queue length
        response_load = min(self.avg_response_time / 5.0, 1.0)  # Normalize response time
        
        return (connection_load + queue_load + response_load) / 3.0


class QuantumLoadBalancer:
    """
    Intelligent load balancer optimized for quantum hyperparameter search workloads.
    """
    
    def __init__(self,
                 strategy: BalancingStrategy = BalancingStrategy.QUANTUM_AWARE,
                 health_check_interval: float = 30.0,
                 enable_circuit_breaker: bool = True,
                 circuit_breaker_threshold: int = 5,
                 circuit_breaker_timeout: float = 60.0):
        """
        Initialize quantum load balancer.
        
        Args:
            strategy: Load balancing strategy
            health_check_interval: Interval between health checks in seconds
            enable_circuit_breaker: Enable circuit breaker pattern
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout: Circuit breaker timeout in seconds
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        self.endpoints: List[ServiceEndpoint] = []
        self.round_robin_index = 0
        
        # Circuit breaker state
        self.circuit_breaker_state: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.total_requests = 0
        self.request_history: List[Dict[str, Any]] = []
        
        # Health checking
        self._health_check_task = None
        self._running = False
    
    def add_endpoint(self, 
                     host: str, 
                     port: int, 
                     weight: float = 1.0,
                     max_connections: int = 100) -> None:
        """
        Add service endpoint.
        
        Args:
            host: Endpoint host
            port: Endpoint port
            weight: Endpoint weight for weighted strategies
            max_connections: Maximum concurrent connections
        """
        endpoint = ServiceEndpoint(
            host=host,
            port=port,
            weight=weight,
            max_connections=max_connections
        )
        
        self.endpoints.append(endpoint)
        
        # Initialize circuit breaker state
        if self.enable_circuit_breaker:
            self.circuit_breaker_state[endpoint.url] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure': 0.0,
                'next_attempt': 0.0
            }
        
        logger.info(f"Added endpoint: {endpoint.url} (weight: {weight})")
    
    def remove_endpoint(self, host: str, port: int) -> None:
        """
        Remove service endpoint.
        
        Args:
            host: Endpoint host
            port: Endpoint port
        """
        url = f"http://{host}:{port}"
        self.endpoints = [ep for ep in self.endpoints if ep.url != url]
        
        if url in self.circuit_breaker_state:
            del self.circuit_breaker_state[url]
        
        logger.info(f"Removed endpoint: {url}")
    
    def get_endpoint(self, 
                     request_context: Dict[str, Any] = None) -> Optional[ServiceEndpoint]:
        """
        Get next endpoint based on load balancing strategy.
        
        Args:
            request_context: Context information about the request
            
        Returns:
            Selected endpoint or None if no healthy endpoints
        """
        healthy_endpoints = self._get_healthy_endpoints()
        
        if not healthy_endpoints:
            logger.warning("No healthy endpoints available")
            return None
        
        request_context = request_context or {}
        
        # Apply balancing strategy
        if self.strategy == BalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_endpoints)
        elif self.strategy == BalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_endpoints)
        elif self.strategy == BalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_endpoints)
        elif self.strategy == BalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(healthy_endpoints)
        elif self.strategy == BalancingStrategy.QUANTUM_AWARE:
            return self._quantum_aware_select(healthy_endpoints, request_context)
        elif self.strategy == BalancingStrategy.ADAPTIVE:
            return self._adaptive_select(healthy_endpoints, request_context)
        else:
            return self._round_robin_select(healthy_endpoints)
    
    def _get_healthy_endpoints(self) -> List[ServiceEndpoint]:
        """Get list of healthy endpoints."""
        healthy = []
        
        for endpoint in self.endpoints:
            if not endpoint.is_healthy:
                continue
                
            # Check circuit breaker state
            if self.enable_circuit_breaker:
                cb_state = self.circuit_breaker_state.get(endpoint.url, {})
                if cb_state.get('state') == 'open':
                    # Check if we should attempt to close circuit
                    if time.time() > cb_state.get('next_attempt', 0):
                        cb_state['state'] = 'half_open'
                        logger.info(f"Circuit breaker half-open for {endpoint.url}")
                    else:
                        continue  # Circuit still open
            
            healthy.append(endpoint)
        
        return healthy
    
    def _round_robin_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin selection."""
        endpoint = endpoints[self.round_robin_index % len(endpoints)]
        self.round_robin_index += 1
        return endpoint
    
    def _weighted_round_robin_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round robin selection."""
        # Calculate cumulative weights
        cumulative_weights = []
        total_weight = 0
        
        for endpoint in endpoints:
            total_weight += endpoint.weight
            cumulative_weights.append(total_weight)
        
        # Select based on weighted probability
        r = random.uniform(0, total_weight)
        for i, weight in enumerate(cumulative_weights):
            if r <= weight:
                return endpoints[i]
        
        return endpoints[-1]  # Fallback
    
    def _least_connections_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections selection."""
        return min(endpoints, key=lambda ep: ep.current_connections)
    
    def _least_response_time_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least response time selection."""
        return min(endpoints, key=lambda ep: ep.avg_response_time)
    
    def _quantum_aware_select(self, 
                             endpoints: List[ServiceEndpoint],
                             request_context: Dict[str, Any]) -> ServiceEndpoint:
        """
        Quantum-aware selection based on workload characteristics.
        
        Args:
            endpoints: Available endpoints
            request_context: Request context with quantum-specific info
            
        Returns:
            Best endpoint for the quantum workload
        """
        problem_size = request_context.get('problem_size', 0)
        backend_type = request_context.get('backend_type', 'simulator')
        requires_quantum = request_context.get('requires_quantum', False)
        
        # Score endpoints based on quantum-specific factors
        scored_endpoints = []
        
        for endpoint in endpoints:
            score = 0.0
            
            # Base load score (lower is better)
            load_penalty = endpoint.load_score * 10
            score -= load_penalty
            
            # Quantum success rate bonus
            quantum_bonus = endpoint.quantum_success_rate * 5
            score += quantum_bonus
            
            # Problem size preference
            if endpoint.preferred_problem_sizes:
                size_match = min(endpoint.preferred_problem_sizes, 
                               key=lambda x: abs(x - problem_size))
                size_bonus = max(0, 5 - abs(size_match - problem_size) / 10)
                score += size_bonus
            
            # Queue length penalty
            queue_penalty = endpoint.quantum_queue_length * 0.5
            score -= queue_penalty
            
            # Response time penalty
            response_penalty = endpoint.avg_response_time * 0.3
            score -= response_penalty
            
            scored_endpoints.append((endpoint, score))
        
        # Select endpoint with highest score
        best_endpoint = max(scored_endpoints, key=lambda x: x[1])[0]
        return best_endpoint
    
    def _adaptive_select(self,
                        endpoints: List[ServiceEndpoint],
                        request_context: Dict[str, Any]) -> ServiceEndpoint:
        """
        Adaptive selection that learns from request patterns.
        
        Args:
            endpoints: Available endpoints
            request_context: Request context
            
        Returns:
            Adaptively selected endpoint
        """
        # Analyze recent request patterns
        recent_requests = self.request_history[-100:]  # Last 100 requests
        
        if len(recent_requests) < 10:
            # Not enough data, fall back to quantum-aware
            return self._quantum_aware_select(endpoints, request_context)
        
        # Calculate success rates by endpoint for similar requests
        endpoint_performance = {}
        
        for req in recent_requests:
            endpoint_url = req.get('endpoint_url')
            if endpoint_url not in endpoint_performance:
                endpoint_performance[endpoint_url] = {
                    'success_count': 0,
                    'total_count': 0,
                    'avg_response_time': 0.0,
                    'response_times': []
                }
            
            perf = endpoint_performance[endpoint_url]
            perf['total_count'] += 1
            
            if req.get('success', False):
                perf['success_count'] += 1
            
            response_time = req.get('response_time', 0.0)
            perf['response_times'].append(response_time)
            perf['avg_response_time'] = statistics.mean(perf['response_times'])
        
        # Score endpoints based on historical performance
        best_endpoint = None
        best_score = float('-inf')
        
        for endpoint in endpoints:
            perf = endpoint_performance.get(endpoint.url, {})
            
            if perf.get('total_count', 0) == 0:
                # No historical data, give neutral score
                score = 0.0
            else:
                success_rate = perf['success_count'] / perf['total_count']
                response_time = perf.get('avg_response_time', 1.0)
                
                # Score based on success rate and response time
                score = success_rate * 10 - response_time * 0.1
            
            # Add current load penalty
            score -= endpoint.load_score * 5
            
            if score > best_score:
                best_score = score
                best_endpoint = endpoint
        
        return best_endpoint or endpoints[0]
    
    def record_request_start(self, 
                           endpoint: ServiceEndpoint,
                           request_context: Dict[str, Any] = None) -> str:
        """
        Record start of request to endpoint.
        
        Args:
            endpoint: Target endpoint
            request_context: Request context
            
        Returns:
            Request ID for tracking
        """
        request_id = f"req_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        endpoint.current_connections += 1
        endpoint.total_requests += 1
        self.total_requests += 1
        
        # Record request start
        request_record = {
            'request_id': request_id,
            'endpoint_url': endpoint.url,
            'start_time': time.time(),
            'context': request_context or {}
        }
        
        self.request_history.append(request_record)
        
        # Limit history size
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-500:]
        
        return request_id
    
    def record_request_end(self,
                          endpoint: ServiceEndpoint,
                          request_id: str,
                          success: bool,
                          response_time: float,
                          error_message: str = None) -> None:
        """
        Record end of request.
        
        Args:
            endpoint: Target endpoint
            request_id: Request ID from record_request_start
            success: Whether request succeeded
            response_time: Response time in seconds
            error_message: Error message if failed
        """
        endpoint.current_connections = max(0, endpoint.current_connections - 1)
        endpoint.response_times.append(response_time)
        
        # Limit response time history
        if len(endpoint.response_times) > 1000:
            endpoint.response_times = endpoint.response_times[-500:]
        
        if success:
            endpoint.successful_requests += 1
            
            # Reset circuit breaker on success
            if self.enable_circuit_breaker:
                cb_state = self.circuit_breaker_state.get(endpoint.url, {})
                if cb_state.get('state') == 'half_open':
                    cb_state['state'] = 'closed'
                    cb_state['failure_count'] = 0
                    logger.info(f"Circuit breaker closed for {endpoint.url}")
        else:
            endpoint.failed_requests += 1
            
            # Update circuit breaker on failure
            if self.enable_circuit_breaker:
                self._handle_circuit_breaker_failure(endpoint, error_message)
        
        # Update request history
        for req in reversed(self.request_history):
            if req.get('request_id') == request_id:
                req.update({
                    'end_time': time.time(),
                    'success': success,
                    'response_time': response_time,
                    'error_message': error_message
                })
                break
    
    def _handle_circuit_breaker_failure(self, 
                                       endpoint: ServiceEndpoint,
                                       error_message: str = None) -> None:
        """Handle circuit breaker failure logic."""
        cb_state = self.circuit_breaker_state.get(endpoint.url, {})
        cb_state['failure_count'] += 1
        cb_state['last_failure'] = time.time()
        
        if cb_state['failure_count'] >= self.circuit_breaker_threshold:
            if cb_state.get('state') != 'open':
                cb_state['state'] = 'open'
                cb_state['next_attempt'] = time.time() + self.circuit_breaker_timeout
                logger.warning(f"Circuit breaker opened for {endpoint.url} after {cb_state['failure_count']} failures")
    
    async def start_health_checking(self) -> None:
        """Start health checking background task."""
        if self._health_check_task:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started health checking")
    
    async def stop_health_checking(self) -> None:
        """Stop health checking background task."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        logger.info("Stopped health checking")
    
    async def _health_check_loop(self) -> None:
        """Health checking background loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all endpoints."""
        import aiohttp
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            tasks = []
            
            for endpoint in self.endpoints:
                task = self._check_endpoint_health(session, endpoint)
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_endpoint_health(self, 
                                   session: 'aiohttp.ClientSession',
                                   endpoint: ServiceEndpoint) -> None:
        """Check health of a single endpoint."""
        health_url = f"{endpoint.url}/health"
        
        try:
            start_time = time.time()
            async with session.get(health_url) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Update endpoint metrics from health check
                    endpoint.is_healthy = True
                    endpoint.last_health_check = time.time()
                    
                    # Extract quantum-specific metrics
                    if 'quantum_queue_length' in data:
                        endpoint.quantum_queue_length = data['quantum_queue_length']
                    
                    if 'quantum_success_rate' in data:
                        endpoint.quantum_success_rate = data['quantum_success_rate']
                    
                    if 'preferred_problem_sizes' in data:
                        endpoint.preferred_problem_sizes = data['preferred_problem_sizes']
                    
                else:
                    endpoint.is_healthy = False
                    logger.warning(f"Health check failed for {endpoint.url}: HTTP {response.status}")
        
        except Exception as e:
            endpoint.is_healthy = False
            logger.warning(f"Health check failed for {endpoint.url}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_count = sum(1 for ep in self.endpoints if ep.is_healthy)
        
        total_success = sum(ep.successful_requests for ep in self.endpoints)
        total_failed = sum(ep.failed_requests for ep in self.endpoints)
        total_endpoint_requests = sum(ep.total_requests for ep in self.endpoints)
        
        endpoint_stats = []
        for endpoint in self.endpoints:
            stats = {
                'url': endpoint.url,
                'healthy': endpoint.is_healthy,
                'weight': endpoint.weight,
                'current_connections': endpoint.current_connections,
                'total_requests': endpoint.total_requests,
                'success_rate': endpoint.success_rate,
                'avg_response_time': endpoint.avg_response_time,
                'load_score': endpoint.load_score,
                'quantum_queue_length': endpoint.quantum_queue_length,
                'quantum_success_rate': endpoint.quantum_success_rate
            }
            
            # Add circuit breaker state
            if self.enable_circuit_breaker:
                cb_state = self.circuit_breaker_state.get(endpoint.url, {})
                stats['circuit_breaker_state'] = cb_state.get('state', 'closed')
                stats['circuit_breaker_failures'] = cb_state.get('failure_count', 0)
            
            endpoint_stats.append(stats)
        
        return {
            'strategy': self.strategy.value,
            'total_endpoints': len(self.endpoints),
            'healthy_endpoints': healthy_count,
            'total_requests': self.total_requests,
            'total_success': total_success,
            'total_failed': total_failed,
            'overall_success_rate': total_success / max(total_endpoint_requests, 1),
            'endpoints': endpoint_stats
        }