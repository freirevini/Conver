"""
Performance Optimization Utilities.

Provides:
- Caching decorators
- Lazy loading patterns
- Memory optimization helpers
"""
import functools
import time
import logging
from typing import Callable, TypeVar, Any, Dict, Optional
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LRUCache:
    """
    Thread-safe LRU Cache implementation.
    
    Used for caching template lookups and parsed results.
    """
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3)
        }


def memoize(maxsize: int = 128):
    """
    Memoization decorator with LRU eviction.
    
    Usage:
        @memoize(maxsize=256)
        def expensive_function(arg):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = LRUCache(maxsize)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            result = cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        
        return wrapper
    return decorator


def timed_lru_cache(seconds: int = 300, maxsize: int = 128):
    """
    LRU cache with TTL (time-to-live).
    
    Cached values expire after specified seconds.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[str, tuple] = {}
        lock = Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            with lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < seconds:
                        return value
                    del cache[key]
                
                # Evict old entries if cache too large
                if len(cache) >= maxsize:
                    oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                    del cache[oldest_key]
            
            result = func(*args, **kwargs)
            
            with lock:
                cache[key] = (result, now)
            
            return result
        
        def clear():
            with lock:
                cache.clear()
        
        wrapper.cache_clear = clear
        return wrapper
    return decorator


class LazyLoader:
    """
    Lazy loading wrapper for expensive objects.
    
    Usage:
        expensive_data = LazyLoader(load_expensive_data)
        # Data not loaded yet
        value = expensive_data.get()  # Now it loads
    """
    
    def __init__(self, loader: Callable[[], T]):
        self._loader = loader
        self._value: Optional[T] = None
        self._loaded = False
        self._lock = Lock()
    
    def get(self) -> T:
        """Get the value, loading if necessary."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._value = self._loader()
                    self._loaded = True
        return self._value
    
    def is_loaded(self) -> bool:
        """Check if value has been loaded."""
        return self._loaded
    
    def reset(self) -> None:
        """Reset to unloaded state."""
        with self._lock:
            self._value = None
            self._loaded = False


def chunked_processing(iterable, chunk_size: int = 1000):
    """
    Process large iterables in chunks to reduce memory.
    
    Usage:
        for chunk in chunked_processing(large_list, 100):
            process_chunk(chunk)
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class ObjectPool:
    """
    Object pool for reusing expensive-to-create objects.
    
    Usage:
        pool = ObjectPool(lambda: ExpensiveObject(), max_size=10)
        
        with pool.acquire() as obj:
            obj.do_work()
    """
    
    def __init__(self, factory: Callable[[], T], max_size: int = 10):
        self._factory = factory
        self._max_size = max_size
        self._pool: list = []
        self._lock = Lock()
    
    def acquire(self):
        """Acquire an object from the pool."""
        return _PooledObject(self)
    
    def _get(self) -> T:
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return self._factory()
    
    def _return(self, obj: T) -> None:
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)


class _PooledObject:
    """Context manager for pooled objects."""
    
    def __init__(self, pool: ObjectPool):
        self._pool = pool
        self._obj = None
    
    def __enter__(self):
        self._obj = self._pool._get()
        return self._obj
    
    def __exit__(self, *args):
        if self._obj is not None:
            self._pool._return(self._obj)
