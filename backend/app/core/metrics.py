"""
Observability and Metrics for FastAPI Application.

Provides:
- Prometheus metrics export
- Request/response logging
- Health metrics
- Custom business metrics
"""
import time
import logging
from typing import Callable, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Lock

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ================== Metrics Storage ==================

@dataclass
class MetricValue:
    """Single metric with labels."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # counter, gauge, histogram


class MetricsRegistry:
    """
    Thread-safe metrics registry.
    
    Collects and exports metrics in Prometheus format.
    """
    
    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, list] = defaultdict(list)
        self._lock = Lock()
    
    def inc_counter(self, name: str, value: float = 1.0, **labels):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, **labels):
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, **labels):
        """Observe a value for histogram."""
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Keep only last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create unique key for metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Export counters
            for key, value in self._counters.items():
                name = key.split("{")[0] if "{" in key else key
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{key} {value}")
            
            # Export gauges
            for key, value in self._gauges.items():
                name = key.split("{")[0] if "{" in key else key
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{key} {value}")
            
            # Export histogram summaries
            for key, values in self._histograms.items():
                if values:
                    name = key.split("{")[0] if "{" in key else key
                    lines.append(f"# TYPE {name} summary")
                    lines.append(f"{key}_count {len(values)}")
                    lines.append(f"{key}_sum {sum(values)}")
                    
                    # Quantiles
                    sorted_vals = sorted(values)
                    n = len(sorted_vals)
                    if n > 0:
                        lines.append(f"{key}{{quantile=\"0.5\"}} {sorted_vals[n//2]}")
                        lines.append(f"{key}{{quantile=\"0.9\"}} {sorted_vals[int(n*0.9)]}")
                        lines.append(f"{key}{{quantile=\"0.99\"}} {sorted_vals[int(n*0.99)]}")
        
        return "\n".join(lines)
    
    def get_metrics_json(self) -> dict:
        """Export metrics as JSON for health endpoints."""
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            
            histogram_stats = {}
            for key, values in self._histograms.items():
                if values:
                    histogram_stats[key] = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
        
        return {
            "counters": counters,
            "gauges": gauges,
            "histograms": histogram_stats
        }


# Global registry
_registry: Optional[MetricsRegistry] = None


def get_registry() -> MetricsRegistry:
    """Get or create global metrics registry."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


# ================== Middleware ==================

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting HTTP request metrics.
    
    Collects:
    - Request count by endpoint and status
    - Request duration histogram
    - Active request gauge
    """
    
    def __init__(self, app, registry: Optional[MetricsRegistry] = None):
        super().__init__(app)
        self.registry = registry or get_registry()
        self._active_requests = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)
        
        method = request.method
        path = self._normalize_path(request.url.path)
        
        # Track active requests
        self._active_requests += 1
        self.registry.set_gauge("http_requests_active", self._active_requests)
        
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Record duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            self.registry.inc_counter(
                "http_requests_total",
                method=method,
                path=path,
                status=str(status_code)
            )
            
            self.registry.observe_histogram(
                "http_request_duration_ms",
                duration_ms,
                method=method,
                path=path
            )
            
            self._active_requests -= 1
            self.registry.set_gauge("http_requests_active", self._active_requests)
        
        return response
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path to reduce cardinality (replace IDs with placeholders)."""
        import re
        # Replace UUIDs
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path
        )
        # Replace numeric IDs
        path = re.sub(r"/\d+(/|$)", "/{id}\\1", path)
        return path


# ================== Business Metrics ==================

class WorkflowMetrics:
    """
    Business metrics for workflow processing.
    """
    
    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or get_registry()
    
    def record_upload(self, file_size_mb: float):
        """Record a workflow upload."""
        self.registry.inc_counter("workflows_uploaded_total")
        self.registry.observe_histogram("workflow_size_mb", file_size_mb)
    
    def record_processing_start(self):
        """Record processing started."""
        self.registry.inc_counter("workflows_processing_started_total")
    
    def record_processing_complete(self, duration_ms: float, node_count: int):
        """Record successful processing."""
        self.registry.inc_counter("workflows_processing_completed_total")
        self.registry.observe_histogram("workflow_processing_duration_ms", duration_ms)
        self.registry.observe_histogram("workflow_node_count", node_count)
    
    def record_processing_error(self, error_type: str):
        """Record processing error."""
        self.registry.inc_counter("workflows_processing_errors_total", error_type=error_type)
    
    def record_template_usage(self, template_name: str, hit: bool):
        """Record template cache hit/miss."""
        status = "hit" if hit else "miss"
        self.registry.inc_counter("template_usage_total", template=template_name, status=status)
    
    def record_llm_call(self, duration_ms: float, success: bool):
        """Record LLM API call."""
        status = "success" if success else "error"
        self.registry.inc_counter("llm_calls_total", status=status)
        self.registry.observe_histogram("llm_call_duration_ms", duration_ms)
    
    def set_active_jobs(self, count: int):
        """Set current active job count."""
        self.registry.set_gauge("active_jobs", count)


# ================== FastAPI Routes ==================

def create_metrics_routes(app):
    """
    Add metrics endpoints to FastAPI app.
    
    Endpoints:
    - GET /metrics - Prometheus format
    - GET /api/metrics - JSON format
    """
    from fastapi.responses import PlainTextResponse
    
    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Export metrics in Prometheus format."""
        return PlainTextResponse(
            content=get_registry().export_prometheus(),
            media_type="text/plain; charset=utf-8"
        )
    
    @app.get("/api/metrics", include_in_schema=False)
    async def json_metrics():
        """Export metrics as JSON."""
        return get_registry().get_metrics_json()


# Global workflow metrics instance
_workflow_metrics: Optional[WorkflowMetrics] = None


def get_workflow_metrics() -> WorkflowMetrics:
    """Get or create workflow metrics instance."""
    global _workflow_metrics
    if _workflow_metrics is None:
        _workflow_metrics = WorkflowMetrics()
    return _workflow_metrics
