"""
Performance Benchmarking Suite.

Provides:
- Workflow parsing benchmarks
- Code generation benchmarks
- Memory profiling
- Performance regression detection
"""
import cProfile
import pstats
import io
import time
import tracemalloc
import gc
import logging
from typing import Callable, Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from contextlib import contextmanager
from statistics import mean, stdev, median

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    execution_time: float  # seconds
    memory_peak: int  # bytes
    memory_allocated: int  # bytes
    iterations: int = 1
    
    @property
    def execution_time_ms(self) -> float:
        return self.execution_time * 1000
    
    @property
    def memory_peak_mb(self) -> float:
        return self.memory_peak / (1024 * 1024)
    
    @property
    def memory_allocated_mb(self) -> float:
        return self.memory_allocated / (1024 * 1024)
    
    def __str__(self) -> str:
        return (
            f"{self.name}: {self.execution_time_ms:.2f}ms, "
            f"Peak: {self.memory_peak_mb:.2f}MB, "
            f"Allocated: {self.memory_allocated_mb:.2f}MB"
        )


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    
    @property
    def total_time(self) -> float:
        return sum(r.execution_time for r in self.results)
    
    @property
    def total_time_ms(self) -> float:
        return self.total_time * 1000
    
    def summary(self) -> str:
        lines = [
            f"Benchmark Suite: {self.name}",
            "=" * 50,
        ]
        for r in self.results:
            lines.append(f"  {r}")
        lines.append("-" * 50)
        lines.append(f"Total: {self.total_time_ms:.2f}ms")
        return "\n".join(lines)


class PerformanceProfiler:
    """
    Production performance profiler.
    
    Features:
    - Function timing with high precision
    - Memory tracking with tracemalloc
    - CPU profiling with cProfile
    - Statistical analysis of multiple runs
    """
    
    def __init__(self, enable_memory: bool = True):
        self.enable_memory = enable_memory
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def profile(self, name: str = "operation"):
        """Context manager for profiling code blocks."""
        gc.collect()
        
        if self.enable_memory:
            tracemalloc.start()
        
        start_time = time.perf_counter()
        
        yield
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        if self.enable_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            current, peak = 0, 0
        
        result = BenchmarkResult(
            name=name,
            execution_time=execution_time,
            memory_peak=peak,
            memory_allocated=current
        )
        
        self.results.append(result)
        logger.debug(str(result))
    
    def benchmark(
        self,
        func: Callable,
        *args,
        iterations: int = 10,
        warmup: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark a function with multiple iterations.
        
        Args:
            func: Function to benchmark
            iterations: Number of benchmark runs
            warmup: Warmup iterations (not counted)
            
        Returns:
            Statistics dictionary
        """
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        gc.collect()
        
        times = []
        memory_peaks = []
        
        for i in range(iterations):
            if self.enable_memory:
                tracemalloc.start()
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            
            times.append(end - start)
            
            if self.enable_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_peaks.append(peak)
            
            gc.collect()
        
        stats = {
            "function": func.__name__,
            "iterations": iterations,
            "mean_time": mean(times),
            "median_time": median(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": stdev(times) if len(times) > 1 else 0,
            "total_time": sum(times),
        }
        
        if memory_peaks:
            stats["mean_memory"] = mean(memory_peaks)
            stats["max_memory"] = max(memory_peaks)
        
        return stats
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, pstats.Stats]:
        """
        Profile function with cProfile.
        
        Returns:
            Tuple of (function result, profile stats)
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        
        return result, stats
    
    def get_profile_report(self, stats: pstats.Stats, lines: int = 20) -> str:
        """Generate profile report."""
        stream = io.StringIO()
        stats.stream = stream
        stats.print_stats(lines)
        return stream.getvalue()
    
    def reset(self):
        """Reset profiler state."""
        self.results = []


def benchmark_decorator(iterations: int = 5, warmup: int = 1):
    """Decorator for benchmarking functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler(enable_memory=True)
            stats = profiler.benchmark(func, *args, iterations=iterations, warmup=warmup, **kwargs)
            
            logger.info(
                f"Benchmark {func.__name__}: "
                f"mean={stats['mean_time']*1000:.2f}ms, "
                f"min={stats['min_time']*1000:.2f}ms, "
                f"max={stats['max_time']*1000:.2f}ms"
            )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==================== Workflow-Specific Benchmarks ====================

class WorkflowBenchmarks:
    """Benchmarks for KNIME workflow processing."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
    
    def benchmark_parsing(self, workflow_path: Path, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark workflow parsing."""
        from app.services.parser.knwf_parser import KNWFParser
        
        def parse_workflow():
            parser = KNWFParser()
            return parser.parse(workflow_path)
        
        return self.profiler.benchmark(parse_workflow, iterations=iterations)
    
    def benchmark_generation(self, workflow_data: Dict, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark code generation."""
        from app.services.generator.code_generator import CodeGenerator
        
        def generate_code():
            generator = CodeGenerator()
            return generator.generate(workflow_data)
        
        return self.profiler.benchmark(generate_code, iterations=iterations)
    
    def run_full_benchmark(self, workflow_path: Path) -> BenchmarkSuite:
        """Run all benchmarks for a workflow."""
        suite = BenchmarkSuite(name=f"Workflow: {workflow_path.name}")
        
        # Parsing
        with self.profiler.profile("parsing"):
            from app.services.parser.knwf_parser import KNWFParser
            parser = KNWFParser()
            workflow_data = parser.parse(workflow_path)
        
        suite.results.append(self.profiler.results[-1])
        
        # Template mapping
        with self.profiler.profile("template_mapping"):
            from app.services.generator.template_mapper import TemplateMapper
            mapper = TemplateMapper()
            # Process nodes...
        
        suite.results.append(self.profiler.results[-1])
        
        # Code generation
        with self.profiler.profile("code_generation"):
            from app.services.generator.code_generator import CodeGenerator
            generator = CodeGenerator()
            # Generate code...
        
        suite.results.append(self.profiler.results[-1])
        
        return suite


# ==================== Performance Thresholds ====================

@dataclass
class PerformanceThreshold:
    """Performance threshold for regression detection."""
    name: str
    max_time_ms: float
    max_memory_mb: float


DEFAULT_THRESHOLDS = {
    "small_workflow": PerformanceThreshold("small_workflow", 100, 50),    # <100 nodes
    "medium_workflow": PerformanceThreshold("medium_workflow", 1000, 100),  # 100-500 nodes
    "large_workflow": PerformanceThreshold("large_workflow", 10000, 500),  # >500 nodes
}


def check_performance_regression(
    result: BenchmarkResult,
    threshold: PerformanceThreshold
) -> Tuple[bool, List[str]]:
    """Check if result exceeds threshold."""
    passed = True
    issues = []
    
    if result.execution_time_ms > threshold.max_time_ms:
        passed = False
        issues.append(
            f"Time {result.execution_time_ms:.2f}ms exceeds {threshold.max_time_ms}ms"
        )
    
    if result.memory_peak_mb > threshold.max_memory_mb:
        passed = False
        issues.append(
            f"Memory {result.memory_peak_mb:.2f}MB exceeds {threshold.max_memory_mb}MB"
        )
    
    return passed, issues
