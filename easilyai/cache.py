"""
Caching system for EasilyAI.

This module provides caching functionality to optimize performance and reduce API calls.
Supports both in-memory and file-based caching backends.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from functools import lru_cache
import pickle


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend using LRU cache."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                # Move to end for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return value
            else:
                # Expired, remove it
                self.delete(key)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
    
    def delete(self, key: str):
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key in self._cache:
            _, expiry = self._cache[key]
            if time.time() < expiry:
                return True
            else:
                self.delete(key)
        return False
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        valid_entries = sum(1 for k in self._cache if self.exists(k))
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "max_size": self.max_size,
            "usage_percent": (valid_entries / self.max_size) * 100 if self.max_size > 0 else 0
        }


class FileCache(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        self.metadata_file = self.cache_dir / "_metadata.json"
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self._metadata, f)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 chars of hash for directory structure
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir = self.cache_dir / key_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        if key in self._metadata:
            expiry = self._metadata[key].get("expiry", 0)
            if time.time() < expiry:
                file_path = self._get_file_path(key)
                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as f:
                            return pickle.load(f)
                    except:
                        self.delete(key)
            else:
                self.delete(key)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in file cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            self._metadata[key] = {
                "expiry": time.time() + ttl,
                "created": time.time(),
                "size": file_path.stat().st_size
            }
            self._save_metadata()
        except Exception as e:
            # Clean up on failure
            if file_path.exists():
                file_path.unlink()
            raise
    
    def delete(self, key: str):
        """Delete value from file cache."""
        if key in self._metadata:
            del self._metadata[key]
            self._save_metadata()
            
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
    
    def clear(self):
        """Clear all cache entries."""
        # Remove all cache files
        for key in list(self._metadata.keys()):
            self.delete(key)
        
        # Clean up empty subdirectories
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and subdir.name != "_metadata.json":
                try:
                    subdir.rmdir()
                except:
                    pass
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key in self._metadata:
            expiry = self._metadata[key].get("expiry", 0)
            if time.time() < expiry:
                return self._get_file_path(key).exists()
            else:
                self.delete(key)
        return False
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        valid_entries = sum(1 for k in self._metadata if self.exists(k))
        total_size = sum(meta.get("size", 0) for meta in self._metadata.values())
        
        return {
            "total_entries": len(self._metadata),
            "valid_entries": valid_entries,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }


class ResponseCache:
    """High-level response cache for AI requests."""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or MemoryCache()
        self.stats = {"hits": 0, "misses": 0}
    
    def _generate_key(self, service: str, model: str, prompt: str, **kwargs) -> str:
        """Generate cache key from request parameters."""
        # Create a deterministic string from all parameters
        key_data = {
            "service": service,
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        
        # Sort to ensure consistent ordering
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Create hash for compact storage
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, service: str, model: str, prompt: str, **kwargs) -> Optional[Any]:
        """Get cached response."""
        key = self._generate_key(service, model, prompt, **kwargs)
        value = self.backend.get(key)
        
        if value is not None:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
        
        return value
    
    def set(self, service: str, model: str, prompt: str, response: Any, 
            ttl: Optional[int] = None, **kwargs):
        """Cache a response."""
        key = self._generate_key(service, model, prompt, **kwargs)
        self.backend.set(key, response, ttl)
    
    def invalidate(self, service: str, model: str, prompt: str, **kwargs):
        """Invalidate a cached response."""
        key = self._generate_key(service, model, prompt, **kwargs)
        self.backend.delete(key)
    
    def clear(self):
        """Clear all cached responses."""
        self.backend.clear()
        self.stats = {"hits": 0, "misses": 0}
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage."""
        total = self.stats["hits"] + self.stats["misses"]
        if total == 0:
            return 0.0
        return (self.stats["hits"] / total) * 100
    
    def get_stats(self) -> dict:
        """Get comprehensive cache statistics."""
        backend_stats = {}
        if hasattr(self.backend, 'get_stats'):
            backend_stats = self.backend.get_stats()
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": self.get_hit_rate(),
            "backend": self.backend.__class__.__name__,
            **backend_stats
        }


# Global cache instance
_cache: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        from .config import get_config
        config = get_config()
        
        if config.cache.enabled:
            if config.cache.backend == "file":
                backend = FileCache(
                    cache_dir=config.cache.cache_dir,
                    default_ttl=config.cache.ttl
                )
            else:
                backend = MemoryCache(
                    max_size=config.cache.max_size,
                    default_ttl=config.cache.ttl
                )
            _cache = ResponseCache(backend)
        else:
            # Null cache that doesn't store anything
            _cache = ResponseCache(MemoryCache(max_size=0))
    
    return _cache


def set_cache(cache: ResponseCache):
    """Set the global cache instance."""
    global _cache
    _cache = cache


def reset_cache():
    """Reset the global cache instance."""
    global _cache
    if _cache:
        _cache.clear()
    _cache = None