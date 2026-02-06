"""
Performance Profiler for KNIME to Python Converter.

Provides:
- Function execution timing
- Memory usage tracking  
- Bottleneck detection
- Performance metrics collection
"""
import time
import functools
import logging
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    name: str
    duration_ms: float
    memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ProfilingResult:
    """Aggregated profiling results."""
    total_duration_ms: float
    peak_memory_mb: float
    metrics: List[PerformanceMetric]
    bottlenecks: List[str]
    
    def to_dict(self) -> dict:
        return {
            "total_duration_ms": round(self.total_duration_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "metrics": [
                {
                    "name": m.name,
                    "duration_ms": round(m.duration_ms, 2),
                    "memory_mb": round(m.memory_mb, 2)
                }
                for m in self.metrics
            ],
            "bottlenecks": self.bottlenecks
        }


class PerformanceProfiler:
    """
    Performance profiler for tracking execution metrics.
    
    Usage:
        profiler = PerformanceProfiler()
        
        with profiler.track("parsing"):
            parse_workflow()
            
        with profiler.track("generation"):
            generate_code()
            
        result = profiler.get_results()
    """
    
    # Thresholds for bottleneck detection (milliseconds)
    SLOW_OPERATION_MS = 1000
    VERY_SLOW_OPERATION_MS = 5000
    
    def __init__(self, enable_memory: bool = True):
        self._metrics: List[PerformanceMetric] = []
        self._enable_memory = enable_memory
        self._start_time: Optional[float] = None
        self._peak_memory: float = 0.0
        
        if enable_memory:
            tracemalloc.start()
    
    @contextmanager
    def track(self, name: str, **metadata):
        """
        Context manager for tracking operation performance.
        
        Args:
            name: Name of the operation being tracked
            **metadata: Additional metadata to store
        """
        start_time = time.perf_counter()
        start_memory = self._get_memory_mb() if self._enable_memory else 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_mb() if self._enable_memory else 0
            
            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            metric = PerformanceMetric(
                name=name,
                duration_ms=duration_ms,
                memory_mb=memory_delta,
                metadata=metadata
            )
            self._metrics.append(metric)
            
            # Track peak memory
            if end_memory > self._peak_memory:
                self._peak_memory = end_memory
            
            # Log slow operations
            if duration_ms > self.VERY_SLOW_OPERATION_MS:
                logger.warning(f"Very slow operation: {name} took {duration_ms:.0f}ms")
            elif duration_ms > self.SLOW_OPERATION_MS:
                logger.info(f"Slow operation: {name} took {duration_ms:.0f}ms")
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            current, peak = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)
        except Exception:
            return 0.0
    
    def get_results(self) -> ProfilingResult:
        """Get aggregated profiling results."""
        total_duration = sum(m.duration_ms for m in self._metrics)
        
        # Detect bottlenecks (operations taking >30% of total time)
        bottlenecks = []
        for metric in self._metrics:
            if total_duration > 0:
                percentage = (metric.duration_ms / total_duration) * 100
                if percentage > 30:
                    bottlenecks.append(
                        f"{metric.name}: {metric.duration_ms:.0f}ms ({percentage:.1f}%)"
                    )
        
        if self._enable_memory:
            tracemalloc.stop()
        
        return ProfilingResult(
            total_duration_ms=total_duration,
            peak_memory_mb=self._peak_memory,
            metrics=self._metrics,
            bottlenecks=bottlenecks
        )
    
    def reset(self):
        """Reset profiler state."""
        self._metrics.clear()
        self._peak_memory = 0.0
        if self._enable_memory:
            tracemalloc.reset_peak()


def profile_function(name: Optional[str] = None):
    """
    Decorator for profiling individual functions.
    
    Usage:
        @profile_function("parse_node")
        def parse_node(node_id):
            ...
    """
    def decorator(func: Callable) -> Callable:
        operation_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                if duration > 100:  # Only log significant operations
                    logger.debug(f"[PERF] {operation_name}: {duration:.2f}ms")
        
        return wrapper
    return decorator


# Global profiler instance for simple use cases
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler(enable_memory: bool = False) -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enable_memory=enable_memory)
    return _global_profiler


def reset_profiler():
    """Reset the global profiler."""
    global _global_profiler
    if _global_profiler:
        _global_profiler.reset()
    _global_profiler = None
