"""
Memory Optimization Utilities.

Provides:
- Memory-efficient data structures
- Object pooling
- Lazy loading
- Memory-bounded operations
"""
import gc
import sys
import weakref
import logging
from typing import TypeVar, Generic, Callable, Optional, Dict, Any, Iterator, List
from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache
from threading import Lock
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ==================== Memory-Efficient Structures ====================

class SlottedClass:
    """
    Base class using __slots__ for memory efficiency.
    
    Classes inheriting from this use __slots__ which:
    - Reduces memory per instance
    - Speeds up attribute access
    - Prevents dynamic attribute creation
    """
    __slots__ = ()


@dataclass(slots=True)
class SlottedNode:
    """Memory-efficient node representation."""
    id: str
    node_type: str
    name: str
    settings: Optional[Dict[str, Any]] = None


class CompactDict(dict):
    """
    Memory-optimized dict that shares keys across instances.
    
    When many dicts have the same keys, this can save significant memory.
    """
    _shared_keys: Dict[frozenset, tuple] = {}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compact()
    
    def _compact(self):
        """Intern string keys to save memory."""
        for key in list(self.keys()):
            if isinstance(key, str):
                self[sys.intern(key)] = self.pop(key)


# ==================== Object Pool ====================

class ObjectPool(Generic[T]):
    """
    Thread-safe object pool for reusing expensive objects.
    
    Reduces allocation overhead by reusing objects instead of
    creating new ones. Useful for parsers, generators, etc.
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        reset_func: Optional[Callable[[T], None]] = None
    ):
        self._factory = factory
        self._max_size = max_size
        self._reset_func = reset_func
        self._pool: List[T] = []
        self._lock = Lock()
        self._created = 0
        self._reused = 0
    
    def acquire(self) -> T:
        """Get an object from the pool or create new one."""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._reused += 1
                if self._reset_func:
                    self._reset_func(obj)
                return obj
            
            self._created += 1
            return self._factory()
    
    def release(self, obj: T):
        """Return object to pool."""
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)
    
    @contextmanager
    def borrow(self):
        """Context manager for borrowing objects."""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)
    
    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "pool_size": len(self._pool),
            "max_size": self._max_size,
            "created": self._created,
            "reused": self._reused,
            "reuse_rate": self._reused / max(1, self._created + self._reused)
        }
    
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self._pool.clear()


# ==================== Lazy Loading ====================

class LazyLoader(Generic[T]):
    """
    Lazy-loading wrapper that defers initialization.
    
    Useful for expensive objects that may not always be needed.
    """
    
    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._instance: Optional[T] = None
        self._loaded = False
    
    @property
    def value(self) -> T:
        """Get the loaded value."""
        if not self._loaded:
            self._instance = self._factory()
            self._loaded = True
        return self._instance
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def reset(self):
        """Reset to unloaded state."""
        self._instance = None
        self._loaded = False


class LazyDict(dict):
    """
    Dictionary with lazy-loaded values.
    
    Values are computed on first access.
    """
    
    def __init__(self, loaders: Dict[str, Callable[[], Any]]):
        super().__init__()
        self._loaders = loaders
        self._loaded = set()
    
    def __getitem__(self, key):
        if key not in self._loaded and key in self._loaders:
            super().__setitem__(key, self._loaders[key]())
            self._loaded.add(key)
        return super().__getitem__(key)
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# ==================== Memory-Bounded Cache ====================

class MemoryBoundedCache(Generic[T]):
    """
    Cache that evicts items when memory limit is reached.
    
    Uses LRU eviction with optional size tracking.
    """
    
    def __init__(
        self,
        max_items: int = 1000,
        max_memory_mb: float = 100.0
    ):
        self._max_items = max_items
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_size = 0
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    def put(self, key: str, value: T, size: Optional[int] = None):
        """Put item in cache."""
        if size is None:
            size = sys.getsizeof(value)
        
        with self._lock:
            # Evict if necessary
            while (
                (len(self._cache) >= self._max_items or 
                 self._total_size + size > self._max_memory_bytes) and
                self._cache
            ):
                old_key, _ = self._cache.popitem(last=False)
                self._total_size -= self._sizes.pop(old_key, 0)
            
            self._cache[key] = value
            self._sizes[key] = size
            self._total_size += size
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._total_size = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "items": len(self._cache),
            "max_items": self._max_items,
            "memory_used_mb": self._total_size / (1024 * 1024),
            "max_memory_mb": self._max_memory_bytes / (1024 * 1024),
        }


# ==================== Weak Reference Cache ====================

class WeakCache(Generic[T]):
    """
    Cache using weak references that allows GC to reclaim memory.
    
    Useful for caching objects that should be freed when memory is needed.
    """
    
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
    
    def get(self, key: str) -> Optional[T]:
        """Get item if still alive."""
        if key in self._cache:
            ref = self._cache[key]
            value = ref()
            if value is not None:
                return value
            else:
                del self._cache[key]
        return None
    
    def put(self, key: str, value: T):
        """Put item in cache."""
        self._cache[key] = weakref.ref(value)
    
    def cleanup(self):
        """Remove dead references."""
        dead = [k for k, v in self._cache.items() if v() is None]
        for k in dead:
            del self._cache[k]


# ==================== Memory Monitoring ====================

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    import tracemalloc
    
    # Process memory
    import os
    try:
        import psutil
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 * 1024)
        vms = process.memory_info().vms / (1024 * 1024)
    except ImportError:
        rss = vms = 0.0
    
    # Python objects
    gc.collect()
    
    return {
        "rss_mb": rss,
        "vms_mb": vms,
        "gc_objects": len(gc.get_objects()),
    }


@contextmanager
def memory_limit(max_mb: float):
    """
    Context manager that monitors memory and warns if limit exceeded.
    
    Note: Does not actually limit memory, just warns.
    """
    import tracemalloc
    
    gc.collect()
    tracemalloc.start()
    
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / (1024 * 1024)
        if peak_mb > max_mb:
            logger.warning(
                f"Memory usage {peak_mb:.2f}MB exceeded limit {max_mb:.2f}MB"
            )


def force_gc():
    """Force garbage collection and return freed count."""
    gc.collect()
    gc.collect()
    gc.collect()
    return gc.collect()
