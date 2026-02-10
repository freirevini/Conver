"""
LLM Fallback Module for KNIME Transpiler

Uses Gemini 2.5 Pro (via Vertex AI) to generate Python code for
KNIME nodes that don't have predefined templates.

Following rules from 04-llm-config.md:
- Model: gemini-2.5-pro (MANDATORY)
- Retry with exponential backoff
- Circuit breaker pattern
- Configuration via environment variables
"""
import os
import time
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class LLMConfig:
    """Vertex AI Configuration - loads from environment."""
    
    # Credentials (from env vars, NEVER hardcoded)
    project_id: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    location: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    
    # Model (MANDATORY: gemini-2.5-pro)
    model_id: str = "gemini-2.5-pro"
    
    # Generation parameters
    temperature: float = 0.0  # Deterministic for code generation
    max_output_tokens: int = 2048
    
    # Resilience settings
    timeout_seconds: int = 30
    max_retries: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 32.0
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: int = 60
    
    def __post_init__(self):
        """Validate configuration."""
        if self.model_id != "gemini-2.5-pro":
            raise ValueError(
                f"Invalid model: {self.model_id}. "
                f"Only gemini-2.5-pro is allowed."
            )


# Global config instance
_config: Optional[LLMConfig] = None


def get_config() -> LLMConfig:
    """Get or create LLM configuration."""
    global _config
    if _config is None:
        _config = LLMConfig()
    return _config


# ============================================================
# CIRCUIT BREAKER
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
    
    @property
    def state(self) -> CircuitState:
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
        self._failure_count = 0
        self._state = CircuitState.CLOSED
    
    def record_failure(self):
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


# Global circuit breaker
_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create circuit breaker."""
    global _circuit_breaker
    if _circuit_breaker is None:
        config = get_config()
        _circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=config.circuit_breaker_recovery
        )
    return _circuit_breaker


# ============================================================
# RETRY WITH EXPONENTIAL BACKOFF
# ============================================================

def retry_with_backoff(max_retries: int = 3, initial_backoff: float = 1.0, 
                       max_backoff: float = 32.0, jitter: bool = True):
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
                    
                    # Calculate backoff
                    delay = min(initial_backoff * (2 ** attempt), max_backoff)
                    
                    # Add jitter (±25%)
                    if jitter:
                        delay *= (0.75 + random.random() * 0.5)
                    
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


# ============================================================
# LLM CLIENT
# ============================================================

class LLMClient:
    """Resilient Vertex AI client for code generation."""
    
    def __init__(self):
        self.config = get_config()
        self.circuit = get_circuit_breaker()
        self._client = None
        self._model = None
    
    def _init_client(self):
        """Lazy initialization of Vertex AI client."""
        if self._client is not None:
            return
        
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location
            )
            
            self._model = GenerativeModel(self.config.model_id)
            self._client = True
            logger.info(f"Vertex AI initialized: {self.config.model_id}")
            
        except ImportError:
            logger.warning("vertexai not installed. LLM fallback disabled.")
            self._client = False
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            self._client = False
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        self._init_client()
        return self._client is True and self.circuit.can_execute()
    
    @retry_with_backoff(max_retries=3)
    def generate(self, prompt: str) -> str:
        """Generate code using LLM with all resilience patterns."""
        if not self.circuit.can_execute():
            raise RuntimeError("Circuit breaker OPEN")
        
        self._init_client()
        
        if self._model is None:
            raise RuntimeError("LLM model not initialized")
        
        try:
            from vertexai.generative_models import GenerationConfig
            
            response = self._model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                )
            )
            
            self.circuit.record_success()
            return response.text
            
        except Exception as e:
            self.circuit.record_failure()
            raise


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# ============================================================
# KNIME NODE TRANSLATION PROMPT
# ============================================================

KNIME_TRANSLATION_PROMPT = """You are an expert Python developer specializing in data pipeline translation.

TASK: Translate a KNIME Analytics node to equivalent Python/pandas code.

NODE INFORMATION:
- Factory: {factory}
- Node Name: {node_name}
- Node Settings (if available): {settings}

REQUIREMENTS:
1. Generate ONLY the Python code, no explanations
2. Use pandas (as `df`) for data manipulation
3. Use numpy (as `np`) for numerical operations
4. Assume input DataFrame is `df` and output should also be `df`
5. Handle edge cases gracefully
6. Keep code concise (1-5 lines max)

EXAMPLES:
- Filter rows: `df = df[df["col"] > 0]`
- Transform: `df["new"] = df["old"].apply(func)`
- Aggregate: `df = df.groupby(["col"]).sum().reset_index()`

OUTPUT FORMAT:
Return ONLY the Python code line(s), nothing else.
If you cannot translate this node, return: `pass  # TODO: Manual implementation required`

Python code:"""


def generate_node_code(factory: str, node_name: str, settings: dict = None) -> str:
    """
    Generate Python code for a KNIME node using LLM.
    
    Args:
        factory: KNIME node factory class name
        node_name: Human-readable node name
        settings: Optional node settings extracted from settings.xml
    
    Returns:
        Python code string or fallback TODO comment
    """
    client = get_llm_client()
    
    if not client.is_available():
        logger.warning(f"LLM not available for {node_name}, using fallback")
        simple = factory.split('.')[-1].replace('NodeFactory', '')
        return f'pass  # TODO: Implement {simple}'
    
    # Format prompt
    prompt = KNIME_TRANSLATION_PROMPT.format(
        factory=factory,
        node_name=node_name,
        settings=str(settings) if settings else "Not available"
    )
    
    try:
        response = client.generate(prompt)
        
        # Clean response - extract only code
        code = response.strip()
        
        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        code = code.strip()
        
        # Validate - must be reasonable Python
        if not code or len(code) > 500:
            return f'pass  # TODO: LLM response invalid for {node_name}'
        
        return code
        
    except Exception as e:
        logger.error(f"LLM generation failed for {node_name}: {e}")
        return f'pass  # TODO: LLM error - {node_name}'


# ============================================================
# INTEGRATION HELPER
# ============================================================

def llm_fallback(factory: str, node_name: str, settings: dict = None) -> tuple[str, bool]:
    """
    Fallback function for transpiler integration.
    
    Returns:
        tuple: (generated_code, was_llm_used)
    """
    code = generate_node_code(factory, node_name, settings)
    was_llm = not code.startswith('pass  # TODO')
    return code, was_llm


if __name__ == "__main__":
    # Test the module
    print("Testing LLM Fallback Module...")
    print(f"Config: {get_config()}")
    print(f"Circuit breaker state: {get_circuit_breaker().state}")
    
    # Test generation (will fail if no Vertex AI configured)
    result = generate_node_code(
        factory="org.knime.base.node.preproc.filter.RangeFilterNodeFactory",
        node_name="Range Filter",
        settings={"min": 0, "max": 100}
    )
    print(f"Generated: {result}")
