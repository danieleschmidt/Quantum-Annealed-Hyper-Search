```python
"""
Intelligent caching system for quantum hyperparameter search results.
"""

import hashlib
import json
import logging
import os
import pickle
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..utils.logging_config import get_logger

logger = get_logger('caching')


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    hit_count: int = 0
    computation_time: float = 0.0
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.hit_count += 1


class ResultCache:
    """
    Intelligent caching system with LRU eviction and adaptive strategies.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 3600.0,
        enable_persistence: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize result cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
            enable_persistence: Enable persistent cache to disk
            cache_dir: Directory for persistent cache files
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        
        # In-memory cache using OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Persistent cache setup
        if enable_persistence:
            self.cache_dir = Path(cache_dir or "./cache")
            self.cache_dir.mkdir(exist_ok=True)
            self._load_persistent_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if time.time() - entry.timestamp > self.ttl_seconds:
                    del self._cache[key]
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                self.hits += 1
                
                return entry.value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            computation_time: Time taken to compute the value
        """
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                computation_time=computation_time
            )
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)  # Mark as most recently used
            
            # Evict if over capacity
            if len(self._cache) > self.max_size:
                self._evict_lru()
            
            # Save to persistent cache if enabled
            if self.enable_persistence:
                self._save_to_persistent(key, entry)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        while len(self._cache) > self.max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self.evictions += 1
            
            # Remove from persistent cache
            if self.enable_persistence:
                persistent_file = self.cache_dir / f"{oldest_key}.pkl"
                if persistent_file.exists():
                    persistent_file.unlink()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            
            if self.enable_persistence and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'ttl_seconds': self.ttl_seconds
        }
    
    def _save_to_persistent(self, key: str, entry: CacheEntry) -> None:
        """Save entry to persistent storage."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception:
            # Silently fail for persistent cache operations
            pass
    
    def _load_persistent_cache(self) -> None:
        """Load entries from persistent storage."""
        if not self.cache_dir.exists():
            return
            
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                key = cache_file.stem
                
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if expired
                if time.time() - entry.timestamp <= self.ttl_seconds:
                    self._cache[key] = entry
                else:
                    cache_file.unlink()  # Remove expired file
                    
        except Exception:
            # Silently fail for persistent cache operations
            pass
    
    def optimize_for_pattern(self, access_pattern: str = "lru") -> None:
        """
        Optimize cache for specific access patterns.
        
        Args:
            access_pattern: Access pattern ('lru', 'lfu', 'adaptive')
        """
        if access_pattern == "lfu":
            # Sort by access frequency
            with self._lock:
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].access_count,
                    reverse=True
                )
                self._cache.clear()
                self._cache.update(sorted_items)
        
        elif access_pattern == "adaptive":
            # Implement adaptive replacement based on hit rates
            self._adaptive_replacement()
    
    def _adaptive_replacement(self) -> None:
        """Implement adaptive replacement policy."""
        with self._lock:
            if len(self._cache) < self.max_size // 2:
                return
                
            # Calculate utility scores for entries
            current_time = time.time()
            scored_entries = []
            
            for key, entry in self._cache.items():
                age = current_time - entry.timestamp
                utility = (entry.hit_count / max(age / 3600, 1)) * \
                         (1 / max(entry.computation_time, 0.1))
                scored_entries.append((key, entry, utility))
            
            # Keep top entries by utility
            scored_entries.sort(key=lambda x: x[2], reverse=True)
            keep_count = int(self.max_size * 0.8)
            
            new_cache = OrderedDict()
            for key, entry, _ in scored_entries[:keep_count]:
                new_cache[key] = entry
            
            self._cache = new_cache


class OptimizationCache:
    """
    Intelligent caching system for hyperparameter optimization results.
    
    Caches evaluation results, QUBO matrices, and partial optimization states
    to accelerate repeated or similar optimization runs.
    """
    
    def __init__(
        self,
        cache_dir: str = '.quantum_cache',
        max_cache_size_mb: int = 500,
        ttl_hours: float = 24.0,
        enable_disk_cache: bool = True,
        enable_memory_cache: bool = True,
        max_memory_entries: int = 10000
    ):
        """
        Initialize optimization cache.
        
        Args:
            cache_dir: Directory for disk cache
            max_cache_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live for cache entries in hours
            enable_disk_cache: Enable persistent disk caching
            enable_memory_cache: Enable in-memory caching
            max_memory_entries: Maximum number of memory cache entries
        """
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.ttl_seconds = ttl_hours * 3600
        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        self.max_memory_entries = max_memory_entries
        
        # In-memory caches using OrderedDict for LRU
        self.eval_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.qubo_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.embedding_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'evictions': 0,
            'size_bytes': 0
        }
        
        if self.enable_disk_cache:
            self._setup_disk_cache()
    
    def _setup_disk_cache(self) -> None:
        """Setup disk cache directory."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _hash_parameters(self, params: Dict[str, Any]) -> str:
        """Create hash of parameters for cache key."""
        # Sort parameters to ensure consistent hashing
        sorted_items = sorted(params.items())
        param_str = str(sorted_items)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _hash_data(self, X: np.ndarray, y: np.ndarray) -> str:
        """Create hash of dataset for cache key."""
        # Use shape and sample of data for hashing
        data_signature = f"{X.shape}_{y.shape}_{np.sum(X[:5])}_{np.sum(y[:5])}"
        return hashlib.md5(data_signature.encode()).hexdigest()
    
    def _is_cache_entry_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        
        age = time.time() - cache_entry['timestamp']
        return age < self.ttl_seconds
    
    def get_evaluation_result(
        self,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        model_class: type,
        cv_folds: int,
        scoring: str
    ) -> Optional[float]:
        """
        Get cached evaluation result.
        
        Args:
            params: Model parameters
            X: Features
            y: Labels
            model_class: Model class
            cv_folds: CV folds
            scoring: Scoring metric
            
        Returns:
            Cached score or None if not found
        """
        if not self.enable_memory_cache:
            return None
        
        # Create cache key
        param_hash = self._hash_parameters(params)
        data_hash = self._hash_data(X, y)
        key = f"eval_{param_hash}_{data_hash}_{model_class.__name__}_{cv_folds}_{scoring}"
        
        with self._lock:
            if key in self.eval_cache:
                entry = self.eval_cache[key]
                if self._is_cache_entry_valid(entry):
                    # Move to end (most recently used)
                    self.eval_cache.move_to_end(key)
                    self.stats['hits'] += 1
                    logger.debug(f"Cache hit for evaluation: {param_hash[:8]}")
                    return entry['score']
                else:
                    # Remove expired entry
                    del self.eval_cache[key]
        
        self.stats['misses'] += 1
        return None
    
    def save_evaluation_result(
        self,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        model_class: type,
        cv_folds: int,
        scoring: str,
        score: float,
        computation_time: float = 0.0
    ) -> None:
        """
        Save evaluation result to cache.
        
        Args:
            params: Model parameters
            X: Features
            y: Labels
            model_class: Model class
            cv_folds: CV folds
            scoring: Scoring metric
            score: Evaluation score
            computation_time: Time taken to compute
        """
        if not self.enable_memory_cache:
            return
        
        # Create cache key
        param_hash = self._hash_parameters(params)
        data_hash = self._hash_data(X, y)
        key = f"eval_{param_hash}_{data_hash}_{model_class.__name__}_{cv_folds}_{scoring}"
        
        with self._lock:
            # Store in memory cache
            self.eval_cache[key] = {
                'score': score,
                'timestamp': time.time(),
                'params': params.copy(),
                'computation_time': computation_time
            }
            self.eval_cache.move_to_end(key)
            
            # Evict if over capacity
            self._evict_if_needed(self.eval_cache)
        
        self.stats['saves'] += 1
        logger.debug(f"Cached evaluation result: {param_hash[:8]} -> {score:.4f}")
    
    def get_qubo_matrix(
        self,
        param_space: Dict[str, List[Any]],
        history_hash: str
    ) -> Optional[Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]]]:
        """
        Get cached QUBO matrix.
        
        Args:
            param_space: Parameter search space
            history_hash: Hash of optimization history
            
        Returns:
            Cached QUBO matrix or None if not found
        """
        if not self.enable_memory_cache:
            return None
        
        space_hash = self._hash_parameters(param_space)
        key = f"qubo_{space_hash}_{history_hash}"
        
        with self._lock:
            if key in self.qubo_cache:
                entry = self.qubo_cache[key]
                if self._is_cache_entry_valid(entry):
                    self.qubo_cache.move_to_end(key)
                    self.stats['hits'] += 1
                    logger.debug(f"Cache hit for QUBO: {space_hash[:8]}")
                    return entry['qubo_data']
                else:
                    del self.qubo_cache[key]
        
        self.stats['misses'] += 1
        return None
    
    def save_qubo_matrix(
        self,
        param_space: Dict[str, List[Any]],
        history_hash: str,
        qubo_data: Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]],
        computation_time: float = 0.0
    ) -> None:
        """
        Save QUBO matrix to cache.
        
        Args:
            param_space: Parameter search space
            history_hash: Hash of optimization history
            qubo_data: QUBO matrix data
            computation_time: Time taken to compute
        """
        if not self.enable_memory_cache:
            return
        
        space_hash = self._hash_parameters(param_space)
        key = f"qubo_{space_hash}_{history_hash}"
        
        with self._lock:
            self.qubo_cache[key] = {
                'qubo_data': qubo_data,
                'timestamp': time.time(),
                'computation_time': computation_time
            }
            self.qubo_cache.move_to_end(key)
            
            # Evict if over capacity
            self._evict_if_needed(self.qubo_cache)
        
        self.stats['saves'] += 1
        logger.debug(f"Cached QUBO matrix: {space_hash[:8]}")
    
    def save_optimization_state(
        self,
        optimization_id: str,
        state: Dict[str, Any]
    ) -> None:
        """
        Save optimization state to disk for resuming.
        
        Args:
            optimization_id: Unique optimization identifier
            state: Optimization state to save
        """
        if not self.enable_disk_cache:
            return
        
        try:
            state_file = os.path.join(self.cache_dir, f"state_{optimization_id}.pkl")
            
            with open(state_file, 'wb') as f:
                pickle.dump({
                    'state': state,
                    'timestamp': time.time()
                }, f)
            
            logger.info(f"Saved optimization state: {optimization_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save optimization state: {e}")
    
    def load_optimization_state(
        self,
        optimization_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load optimization state from disk.
        
        Args:
            optimization_id: Unique optimization identifier
            
        Returns:
            Optimization state or None if not found
        """
        if not self.enable_disk_cache:
            return None
        
        try:
            state_file = os.path.join(self.cache_dir, f"state_{optimization_id}.pkl")
            
            if not os.path.exists(state_file):
                return None
            
            with open(state_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if state is still valid
            if not self._is_cache_entry_valid(data):
                os.remove(state_file)
                return None
            
            logger.info(f"Loaded optimization state: {optimization_id}")
            return data['state']
            
        except Exception as e:
            logger.warning(f"Failed to load optimization state: {e}")
            return None
    
    def _evict_if_needed(self, cache: OrderedDict) -> None:
        """Evict least recently used entries if cache is full."""
        max_size = self.max_memory_entries // 3  # Divide between 3 cache types
        
        while len(cache) > max_size:
            oldest_key, _ = cache.popitem(last=False)
            self.stats['evictions'] += 1
    
    def cleanup_disk_cache(self) -> None:
        """Clean up expired disk cache entries."""
        if not self.enable_disk_cache or not os.path.exists(self.cache_dir):
            return
        
        try:
            current_time = time.time()
            total_size = 0
            file_ages = []
            
            # Collect file information
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    age = current_time - stat.st_mtime
                    file_ages.append((file_path, age, stat.st_size))
                    total_size += stat.st_size
            
            # Remove expired files
            removed_count = 0
            for file_path, age, size in file_ages:
                if age > self.ttl_seconds:
                    os.remove(file_path)
                    total_size -= size
                    removed_count += 1
            
            # Remove oldest files if cache is too large
            if total_size > self.max_cache_size:
                # Sort by age (oldest first)
                file_ages.sort(key=lambda x: x[1], reverse=True)
                
                for file_path, age, size in file_ages:
                    if total_size <= self.max_cache_size * 0.8:  # Target 80% of max
                        break
                    
                    os.remove(file_path)
                    total_size -= size
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired cache files")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup disk cache: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        disk_size = 0
        disk_files = 0
        
        if self.enable_disk_cache and os.path.exists(self.cache_dir):
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path):
                        disk_size += os.path.getsize(file_path)
                        disk_files += 1
            except Exception:
                pass
        
        return {
            'memory_cache': {
                'eval_entries': len(self.eval_cache),
                'qubo_entries': len(self.qubo_cache),
                'embedding_entries': len(self.embedding_cache)
            },
            'disk_cache': {
                'size_bytes': disk_size,
                'size_mb': round(disk_size / (1024*1024), 2),
                'files': disk_files
            },
            'performance': {
                'hit_rate': round(hit_rate, 3),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'saves': self.stats['saves'],
                'evictions': self.stats['evictions']
            }
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            # Clear memory caches
            self.eval_cache.clear()
            self.qubo_cache.clear()
            self.embedding_cache.clear()
        
        # Clear disk cache
        if self.enable_disk_cache and os.path.exists(self.cache_dir):
            try:
                import shutil
                shutil.rmtree(self.cache_dir)
                self._setup_disk_cache()
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {e}")
        
        # Reset statistics
        self.stats = {'hits': 0, 'misses': 0, 'saves': 0, 'evictions': 0, 'size_bytes': 0}
        logger.info("Cleared all caches")
    
    def optimize_for_pattern(self, access_pattern: str = "adaptive") -> None:
        """
        Optimize cache based on access patterns.
        
        Args:
            access_pattern: Pattern type ('lru', 'lfu', 'adaptive')
        """
        with self._lock:
            if access_pattern == "adaptive":
                # Analyze hit rates and adjust TTL accordingly
                total_requests = self.stats['hits'] + self.stats['misses']
                if total_requests > 100:
                    hit_rate = self.stats['hits'] / total_requests
                    
                    if hit_rate < 0.3:
                        # Low hit rate - increase TTL
                        self.ttl_seconds = min(self.ttl_seconds * 1.2, 86400)  # Max 24 hours
                    elif hit_rate > 0.8:
                        # High hit rate - can reduce TTL
                        self.ttl_seconds = max(self.ttl_seconds * 0.9, 1800)  # Min 30 minutes


def generate_cache_key(
    params: Dict[str, Any],
    model_class: type,
    X_shape: Tuple[int, ...],
    y_shape: Tuple[int, ...],
    cv_folds: int,
    scoring: str
) -> str:
    """
    Generate deterministic cache key for parameter evaluation.
    
    Args:
        params: Parameter dictionary
        model_class: Model class
        X_shape: Shape of feature matrix
        y_shape: Shape of target vector
        cv_folds: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        Deterministic cache key string
    """
    # Create deterministic representation
    key_data = {
        'params': sorted(params.items()),
        'model': f"{model_class.__module__}.{model_class.__name__}",
        'X_shape': X_shape,
        'y_shape': y_shape,
        'cv_folds': cv_folds,
        'scoring': scoring
    }
    
    # Convert to JSON for consistent string representation
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    
    # Hash to fixed-length key
    return hashlib.sha256(key_str.encode()).hexdigest()


def adaptive_cache(cache: ResultCache):
    """
    Decorator for adaptive caching of function results.
    
    Args:
        cache: ResultCache instance to use
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            try:
                key_data = {
                    'func': f"{func.__module__}.{func.__name__}",
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                key_str = json.dumps(key_data, sort_keys=True, default=str)
                cache_key = hashlib.sha256(key_str.encode()).hexdigest()
                
                # Try to get from cache
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Compute result
                start_time = time.time()
                result = func(*args, **kwargs)
                computation_time = time.time() - start_time
                
                # Cache result
                cache.put(cache_key, result, computation_time)
                
                return result
                
            except Exception:
                # Fall back to direct computation if caching fails
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class SmartCache:
    """
    Smart cache that learns from usage patterns and optimizes itself.
    """
    
    def __init__(
        self,
        base_cache: ResultCache,
        learning_window: int = 1000,
        optimization_interval: int = 100
    ):
        """
        Initialize smart cache.
        
        Args:
            base_cache: Underlying cache implementation
            learning_window: Number of operations to learn from
            optimization_interval: How often to optimize cache
        """
        self.base_cache = base_cache
        self.learning_window = learning_window
        self.optimization_interval = optimization_interval
        
        # Learning data
        self.access_history = []
        self.operation_count = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get with learning."""
        result = self.base_cache.get(key)
        
        # Record access pattern
        self.access_history.append({
            'key': key,
            'timestamp': time.time(),
            'hit': result is not None
        })
        
        # Trim history
        if len(self.access_history) > self.learning_window:
            self.access_history = self.access_history[-self.learning_window:]
        
        self.operation_count += 1
        
        # Optimize periodically
        if self.operation_count % self.optimization_interval == 0:
            self._optimize_cache()
        
        return result
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """Put with learning."""
        self.base_cache.put(key, value, computation_time)
        self.operation_count += 1
    
    def _optimize_cache(self) -> None:
        """Optimize cache based on learned patterns."""
        if len(self.access_history) < 100:
            return
        
        # Analyze access patterns
        hit_rate = sum(1 for access in self.access_history if access['hit']) / len(self.access_history)
        
        # Adjust cache parameters based on hit rate
        if hit_rate < 0.3:
            # Low hit rate - increase cache size or adjust TTL
            new_ttl = min(self.base_cache.ttl_seconds * 1.2, 7200)  # Max 2 hours
            self.base_cache.ttl_seconds = new_ttl
        elif hit_rate > 0.8:
            # High hit rate - we can be more aggressive with eviction
            new_ttl = max(self.base_cache.ttl_seconds * 0.9, 300)  # Min 5 minutes
            self.base_cache.ttl_seconds = new_ttl
        
        # Optimize replacement policy
        self.base_cache.optimize_for_pattern("adaptive")


# Global cache instance
_global_cache = None


def get_global_cache() -> OptimizationCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = OptimizationCache()
    return _global_cache


def configure_global_cache(
    cache_dir: str = '.quantum_cache',
    max_cache_size_mb: int = 500,
    ttl_hours: float = 24.0,
    max_memory_entries: int = 10000
) -> OptimizationCache:
    """Configure global cache with custom settings."""
    global _global_cache
    _global_cache = OptimizationCache(
        cache_dir=cache_dir,
        max_cache_size_mb=max_cache_size_mb,
        ttl_hours=ttl_hours,
        max_memory_entries=max_memory_entries
    )
    return _global_cache
```
