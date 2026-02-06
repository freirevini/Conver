"""
Performance Module.

Provides:
- benchmarks: Performance profiling and benchmarking
- memory_optimizer: Memory-efficient data structures
- streaming_parser: Memory-efficient large workflow parsing
"""
from .benchmarks import (
    PerformanceProfiler,
    BenchmarkResult,
    BenchmarkSuite,
    benchmark_decorator,
    check_performance_regression,
    DEFAULT_THRESHOLDS
)

from .memory_optimizer import (
    ObjectPool,
    LazyLoader,
    LazyDict,
    MemoryBoundedCache,
    WeakCache,
    get_memory_usage,
    memory_limit,
    force_gc
)

from .streaming_parser import (
    StreamingKNWFParser,
    StreamedNode,
    StreamedConnection,
    AsyncStreamingParser
)

__all__ = [
    # Benchmarks
    "PerformanceProfiler",
    "BenchmarkResult",
    "benchmark_decorator",
    "check_performance_regression",
    # Memory
    "ObjectPool",
    "LazyLoader",
    "MemoryBoundedCache",
    "get_memory_usage",
    "force_gc",
    # Streaming
    "StreamingKNWFParser",
    "StreamedNode",
    "AsyncStreamingParser",
]
