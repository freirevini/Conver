"""
Security Middleware and Utilities for FastAPI.

Provides:
- Rate limiting per IP
- Request validation
- Security headers
- Input sanitization
"""
import time
import re
import logging
from collections import defaultdict
from typing import Callable, Dict, Optional
from functools import wraps
from uuid import UUID

from fastapi import Request, HTTPException, status
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ================== Rate Limiting ==================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Limits requests per IP address within a time window.
    """
    
    def __init__(
        self, 
        app, 
        requests_per_minute: int = 60,
        burst_limit: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self._request_counts: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/", "/health", "/api/health"]:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        self._request_counts[client_ip] = [
            t for t in self._request_counts[client_ip]
            if current_time - t < 60
        ]
        
        # Check rate limit
        if len(self._request_counts[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": "Muitas requisições. Tente novamente em alguns minutos.",
                    "retry_after": 60
                }
            )
        
        # Check burst limit (requests in last 1 second)
        recent_requests = [
            t for t in self._request_counts[client_ip]
            if current_time - t < 1
        ]
        if len(recent_requests) >= self.burst_limit:
            logger.warning(f"Burst limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "burst_limit_exceeded",
                    "message": "Requisições muito rápidas. Aguarde um momento."
                }
            )
        
        # Record this request
        self._request_counts[client_ip].append(current_time)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# ================== Security Headers ==================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove server header if present
        response.headers.pop("Server", None)
        
        return response


# ================== Input Validation ==================

def validate_uuid(value: str, field_name: str = "ID") -> str:
    """
    Validate that a string is a valid UUID.
    
    Args:
        value: String to validate
        field_name: Name for error messages
        
    Returns:
        Validated UUID string
        
    Raises:
        HTTPException: If invalid
    """
    try:
        UUID(value, version=4)
        return value
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_uuid",
                "message": f"{field_name} inválido",
                "suggestion": "Verifique o formato do ID"
            }
        )


def validate_filename(filename: str, max_length: int = 255) -> str:
    """
    Validate and sanitize a filename.
    
    Args:
        filename: Filename to validate
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
        
    Raises:
        HTTPException: If invalid
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "missing_filename",
                "message": "Nome do arquivo não fornecido"
            }
        )
    
    # Check length
    if len(filename) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "filename_too_long",
                "message": f"Nome do arquivo muito longo (máximo: {max_length} caracteres)"
            }
        )
    
    # Sanitize - remove path traversal attempts
    sanitized = filename.replace("..", "").replace("/", "_").replace("\\", "_")
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r"<script",
        r"javascript:",
        r"data:",
        r"\x00",  # Null byte
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning(f"Dangerous filename pattern detected: {filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_filename",
                    "message": "Nome do arquivo contém caracteres não permitidos"
                }
            )
    
    return sanitized


def validate_file_extension(filename: str, allowed: list[str]) -> str:
    """
    Validate file extension.
    
    Args:
        filename: Filename to check
        allowed: List of allowed extensions (e.g., [".zip", ".knwf"])
        
    Returns:
        The filename if valid
        
    Raises:
        HTTPException: If extension not allowed
    """
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_file_type",
                "message": f"Tipo de arquivo não permitido: {ext}",
                "allowed_types": allowed,
                "suggestion": "Use um dos formatos permitidos"
            }
        )
    
    return filename


# ================== Logging Sanitization ==================

def sanitize_for_logging(value: str, max_length: int = 100) -> str:
    """
    Sanitize a value for safe logging.
    
    Removes potentially sensitive or dangerous content.
    """
    if not value:
        return ""
    
    # Truncate
    if len(value) > max_length:
        value = value[:max_length] + "..."
    
    # Remove control characters
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)
    
    return value


def mask_sensitive(value: str, visible_chars: int = 4) -> str:
    """
    Mask a sensitive value for logging.
    
    Shows only first and last N characters.
    """
    if not value or len(value) <= visible_chars * 2:
        return "****"
    
    return value[:visible_chars] + "****" + value[-visible_chars:]


# ================== Request Size Limiting ==================

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Limit request body size.
    """
    
    def __init__(self, app, max_size_mb: float = 100.0):
        super().__init__(app)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={
                            "error": "request_too_large",
                            "message": f"Requisição muito grande (máximo: {self.max_size_bytes // (1024*1024)}MB)",
                            "max_size_mb": self.max_size_bytes // (1024 * 1024)
                        }
                    )
            except ValueError:
                pass
        
        return await call_next(request)


# ================== Safe Error Responses ==================

def create_safe_error_response(
    error_code: str,
    message: str,
    suggestion: Optional[str] = None,
    status_code: int = 400
) -> HTTPException:
    """
    Create a safe, consistent error response.
    
    Avoids leaking sensitive information in error messages.
    """
    detail = {
        "error": error_code,
        "message": message
    }
    
    if suggestion:
        detail["suggestion"] = suggestion
    
    return HTTPException(status_code=status_code, detail=detail)


# ================== Environment Check ==================

def is_production() -> bool:
    """Check if running in production environment."""
    import os
    env = os.getenv("ENVIRONMENT", "development").lower()
    return env in ("production", "prod", "prd")


def get_safe_health_response() -> dict:
    """
    Get health check response appropriate for environment.
    
    In production, hides sensitive details.
    """
    if is_production():
        return {
            "status": "healthy",
            "service": "KNIME to Python Converter API"
        }
    else:
        # Development mode - show more details
        from app.core.config import settings
        return {
            "status": "healthy",
            "service": "KNIME to Python Converter API",
            "environment": "development",
            "gcp_project": settings.google_cloud_project[:8] + "..." if settings.google_cloud_project else None
        }
