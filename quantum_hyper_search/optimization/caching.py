"""
Intelligent caching system for optimization results.
"""

import hashlib
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging_config import get_logger

logger = get_logger('caching')


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
        enable_memory_cache: bool = True
    ):
        """
        Initialize optimization cache.
        
        Args:
            cache_dir: Directory for disk cache
            max_cache_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live for cache entries in hours
            enable_disk_cache: Enable persistent disk caching
            enable_memory_cache: Enable in-memory caching
        """
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.ttl_seconds = ttl_hours * 3600
        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        
        # In-memory caches
        self.eval_cache = {}  # Parameter evaluation results
        self.qubo_cache = {}  # QUBO matrices
        self.embedding_cache = {}  # Quantum embeddings
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
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
        
        if key in self.eval_cache:
            entry = self.eval_cache[key]
            if self._is_cache_entry_valid(entry):
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
        score: float
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
        """
        if not self.enable_memory_cache:
            return
        
        # Create cache key
        param_hash = self._hash_parameters(params)
        data_hash = self._hash_data(X, y)
        key = f"eval_{param_hash}_{data_hash}_{model_class.__name__}_{cv_folds}_{scoring}"
        
        # Store in memory cache
        self.eval_cache[key] = {
            'score': score,
            'timestamp': time.time(),
            'params': params.copy()
        }
        
        self.stats['saves'] += 1
        logger.debug(f"Cached evaluation result: {param_hash[:8]} -> {score:.4f}")
        
        # Clean up old entries if cache is getting too large
        self._cleanup_memory_cache()
    
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
        
        if key in self.qubo_cache:
            entry = self.qubo_cache[key]
            if self._is_cache_entry_valid(entry):
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
        qubo_data: Tuple[Dict[Tuple[int, int], float], float, Dict[int, Tuple[str, Any]]]
    ) -> None:
        """
        Save QUBO matrix to cache.
        
        Args:
            param_space: Parameter search space
            history_hash: Hash of optimization history
            qubo_data: QUBO matrix data
        """
        if not self.enable_memory_cache:
            return
        
        space_hash = self._hash_parameters(param_space)
        key = f"qubo_{space_hash}_{history_hash}"
        
        self.qubo_cache[key] = {
            'qubo_data': qubo_data,
            'timestamp': time.time()
        }
        
        self.stats['saves'] += 1
        logger.debug(f"Cached QUBO matrix: {space_hash[:8]}")
        
        self._cleanup_memory_cache()
    
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
    
    def _cleanup_memory_cache(self) -> None:
        """Clean up memory cache if it's getting too large."""
        # Simple cleanup: remove oldest entries
        current_size = len(self.eval_cache) + len(self.qubo_cache)
        max_entries = 1000  # Reasonable limit
        
        if current_size > max_entries:
            # Remove oldest evaluation cache entries
            if self.eval_cache:
                oldest_keys = sorted(
                    self.eval_cache.keys(),
                    key=lambda k: self.eval_cache[k].get('timestamp', 0)
                )[:50]  # Remove 50 oldest entries
                
                for key in oldest_keys:
                    del self.eval_cache[key]
            
            # Remove oldest QUBO cache entries
            if self.qubo_cache:
                oldest_keys = sorted(
                    self.qubo_cache.keys(),
                    key=lambda k: self.qubo_cache[k].get('timestamp', 0)
                )[:20]  # Remove 20 oldest entries
                
                for key in oldest_keys:
                    del self.qubo_cache[key]
            
            logger.debug("Cleaned up memory cache")
    
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
                'saves': self.stats['saves']
            }
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
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
        self.stats = {'hits': 0, 'misses': 0, 'saves': 0, 'size_bytes': 0}
        logger.info("Cleared all caches")


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
    ttl_hours: float = 24.0
) -> OptimizationCache:
    """Configure global cache with custom settings."""
    global _global_cache
    _global_cache = OptimizationCache(
        cache_dir=cache_dir,
        max_cache_size_mb=max_cache_size_mb,
        ttl_hours=ttl_hours
    )
    return _global_cache