"""
Intelligent caching system for quantum hyperparameter search results.
"""

import hashlib
import pickle
import json
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import numpy as np


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