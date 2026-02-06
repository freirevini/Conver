"""
Vertex AI Client - Resilient client for gemini-2.5-pro LLM integration.

Implements:
- Mandatory gemini-2.5-pro model (no alternatives)
- Retry with exponential backoff
- Circuit breaker pattern
- Response caching
- Privacy-aware sanitization

Configuration via: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VertexConfig:
    """Vertex AI configuration - loads from environment."""
    
    # Credentials (from env vars)
    project_id: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    )
    location: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    
    # Model - MANDATORY: gemini-2.5-pro only
    model_id: str = "gemini-2.5-pro"
    
    # Generation parameters
    temperature: float = 0.0  # Deterministic
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    
    # Resilience
    timeout_seconds: int = 30
    max_retries: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 32.0
    
    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout: int = 60
    
    # Caching
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache/llm"))
    
    def __post_init__(self):
        if self.model_id != "gemini-2.5-pro":
            raise ValueError(
                f"Invalid model: {self.model_id}. "
                f"Only 'gemini-2.5-pro' is allowed."
            )
        if not self.project_id:
            logger.warning("GOOGLE_CLOUD_PROJECT not set - LLM will be disabled")


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker to prevent cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = Lock()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._state = CircuitState.HALF_OPEN
            return self._state
    
    def _should_attempt_recovery(self) -> bool:
        if self._last_failure_time is None:
            return True
        elapsed = datetime.now() - self._last_failure_time
        return elapsed > timedelta(seconds=self.recovery_timeout)
    
    def record_success(self):
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
    
    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit OPEN after {self._failure_count} failures. "
                    f"Recovery in {self.recovery_timeout}s"
                )
    
    def can_execute(self) -> bool:
        return self.state != CircuitState.OPEN


# =============================================================================
# Retry Decorator
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 32.0,
    jitter: bool = True
):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    delay = min(initial_backoff * (2 ** attempt), max_backoff)
                    if jitter:
                        delay *= (0.75 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# Cache Manager
# =============================================================================

class LLMCache:
    """File-based cache for LLM responses."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def _get_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
    
    def get(self, prompt: str) -> Optional[str]:
        key = self._get_key(prompt)
        path = self._get_path(key)
        
        if path.exists():
            try:
                data = json.loads(path.read_text())
                logger.debug(f"Cache hit for prompt hash: {key[:8]}")
                return data.get("response")
            except Exception:
                return None
        return None
    
    def set(self, prompt: str, response: str):
        key = self._get_key(prompt)
        path = self._get_path(key)
        
        data = {
            "prompt_hash": key,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Cached response for prompt hash: {key[:8]}")


# =============================================================================
# Privacy Sanitizer
# =============================================================================

SENSITIVE_PATTERNS = [
    "password", "api_key", "apikey", "secret", "token", "bearer",
    "credential", "private_key", "ssn", "credit_card", "cpf", "cnpj",
]


def sanitize_for_llm(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data before sending to LLM."""
    sanitized = {}
    
    for key, value in data.items():
        key_lower = key.lower()
        
        if any(pattern in key_lower for pattern in SENSITIVE_PATTERNS):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, str):
            value_lower = value.lower()
            if any(pattern in value_lower for pattern in SENSITIVE_PATTERNS):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_llm(value)
        else:
            sanitized[key] = value
    
    return sanitized


# =============================================================================
# Vertex AI Client
# =============================================================================

class VertexAIClient:
    """
    Resilient Vertex AI client for gemini-2.5-pro.
    
    Features:
    - Retry with exponential backoff
    - Circuit breaker pattern
    - Response caching
    - Privacy-aware sanitization
    """
    
    def __init__(self, config: Optional[VertexConfig] = None):
        self.config = config or VertexConfig()
        self.circuit = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout,
        )
        self.cache = LLMCache(self.config.cache_dir) if self.config.cache_enabled else None
        self._client = None
        self._model = None
    
    @property
    def is_available(self) -> bool:
        """Check if LLM is configured and available."""
        return bool(self.config.project_id) and self.circuit.can_execute()
    
    def _initialize_client(self):
        """Initialize Vertex AI client lazily."""
        if self._client is not None:
            return
        
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location,
            )
            
            self._model = GenerativeModel(
                self.config.model_id,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                }
            )
            self._client = True
            logger.info(
                f"Vertex AI initialized: {self.config.model_id} "
                f"@ {self.config.location}"
            )
        except ImportError:
            raise ImportError(
                "vertexai package not installed. "
                "Run: pip install google-cloud-aiplatform"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using gemini-2.5-pro.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If circuit is open or generation fails
        """
        if not self.config.project_id:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT not configured. "
                "LLM generation is disabled."
            )
        
        if not self.circuit.can_execute():
            raise RuntimeError(
                f"Circuit breaker OPEN. "
                f"Retry in {self.config.recovery_timeout}s"
            )
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(prompt)
            if cached:
                return cached
        
        # Generate with retry
        try:
            response = self._generate_with_retry(prompt)
            self.circuit.record_success()
            
            # Cache successful response
            if self.cache:
                self.cache.set(prompt, response)
            
            return response
            
        except Exception as e:
            self.circuit.record_failure()
            raise
    
    @retry_with_backoff(max_retries=3, initial_backoff=1.0, max_backoff=32.0)
    def _generate_with_retry(self, prompt: str) -> str:
        """Generate with retry logic."""
        self._initialize_client()
        
        response = self._model.generate_content(prompt)
        
        if response.text:
            return response.text.strip()
        else:
            raise RuntimeError("Empty response from LLM")
    
    def generate_node_code(
        self,
        node_name: str,
        factory_class: str,
        settings: Dict[str, Any],
        category: Optional[str] = None,
    ) -> str:
        """
        Generate Python code for a KNIME node.
        
        Args:
            node_name: Display name of the node
            factory_class: KNIME factory class
            settings: Node configuration settings
            category: Node category (optional)
            
        Returns:
            Generated Python function code
        """
        # Sanitize settings for privacy
        safe_settings = sanitize_for_llm(settings)
        settings_str = json.dumps(safe_settings, indent=2, default=str)
        
        prompt = f"""Generate a Python function that replicates the behavior of a KNIME node.

KNIME Node Information:
- Name: {node_name}
- Factory Class: {factory_class}
- Category: {category or 'unknown'}

Node Settings:
{settings_str}

Requirements:
1. Function must accept pd.DataFrame as input_0
2. Function must return pd.DataFrame
3. Use only pandas, numpy, and scikit-learn
4. Include proper logging with logger.info/warning
5. Handle edge cases (empty DataFrame, missing columns)
6. Add descriptive docstring with original KNIME settings

Generate ONLY the Python function code, no explanations or markdown."""

        return self.generate(prompt)


# =============================================================================
# Singleton Instance
# =============================================================================

_vertex_client: Optional[VertexAIClient] = None


def get_vertex_client() -> VertexAIClient:
    """Get singleton Vertex AI client instance."""
    global _vertex_client
    
    if _vertex_client is None:
        _vertex_client = VertexAIClient()
    
    return _vertex_client
