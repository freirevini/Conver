"""
Structured Error Handling for KNIME Transpiler.

Provides:
- Custom exception hierarchy
- Error codes
- Recovery strategies
- Error context capture
"""
import logging
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Type
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standardized error codes."""
    # Parsing errors (1xxx)
    PARSE_XML_INVALID = "E1001"
    PARSE_NODE_MISSING = "E1002"
    PARSE_CONNECTION_INVALID = "E1003"
    PARSE_SETTINGS_CORRUPT = "E1004"
    
    # Template errors (2xxx)
    TEMPLATE_NOT_FOUND = "E2001"
    TEMPLATE_INVALID = "E2002"
    TEMPLATE_PLACEHOLDER_MISSING = "E2003"
    
    # Generation errors (3xxx)
    GENERATION_FAILED = "E3001"
    GENERATION_SYNTAX_ERROR = "E3002"
    GENERATION_CIRCULAR_DEP = "E3003"
    
    # LLM errors (4xxx)
    LLM_UNAVAILABLE = "E4001"
    LLM_RATE_LIMITED = "E4002"
    LLM_INVALID_RESPONSE = "E4003"
    LLM_TIMEOUT = "E4004"
    
    # File errors (5xxx)
    FILE_NOT_FOUND = "E5001"
    FILE_INVALID_FORMAT = "E5002"
    FILE_TOO_LARGE = "E5003"
    FILE_PERMISSION_DENIED = "E5004"
    
    # Validation errors (6xxx)
    VALIDATION_SYNTAX = "E6001"
    VALIDATION_IMPORT = "E6002"
    VALIDATION_TYPE = "E6003"
    
    # Unknown
    UNKNOWN = "E9999"


@dataclass
class ErrorContext:
    """Captured context when error occurred."""
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    node_factory: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    additional: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.node_id:
            result["node_id"] = self.node_id
        if self.node_name:
            result["node_name"] = self.node_name
        if self.node_factory:
            result["node_factory"] = self.node_factory
        if self.file_path:
            result["file_path"] = self.file_path
        if self.line_number:
            result["line_number"] = self.line_number
        if self.additional:
            result.update(self.additional)
        return result


class TranspilerError(Exception):
    """Base exception for all transpiler errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False,
        suggestion: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.suggestion = suggestion
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion,
            "context": self.context.to_dict(),
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self):
        return f"[{self.code.value}] {self.message}"


class ParseError(TranspilerError):
    """Error during workflow parsing."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            code=ErrorCode.PARSE_XML_INVALID,
            context=context,
            suggestion="Verify the workflow file is a valid KNIME export",
            **kwargs
        )


class TemplateError(TranspilerError):
    """Error with template lookup or application."""
    
    def __init__(self, message: str, factory: str, context: Optional[ErrorContext] = None, **kwargs):
        ctx = context or ErrorContext()
        ctx.node_factory = factory
        super().__init__(
            message=message,
            code=ErrorCode.TEMPLATE_NOT_FOUND,
            context=ctx,
            recoverable=True,
            suggestion="Node will use fallback generation (LLM or stub)",
            **kwargs
        )


class GenerationError(TranspilerError):
    """Error during code generation."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            code=ErrorCode.GENERATION_FAILED,
            context=context,
            suggestion="Check node configuration and try again",
            **kwargs
        )


class LLMError(TranspilerError):
    """Error with LLM service."""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.LLM_UNAVAILABLE, **kwargs):
        super().__init__(
            message=message,
            code=code,
            recoverable=True,
            suggestion="LLM temporarily unavailable, using stub generation",
            **kwargs
        )


class ValidationError(TranspilerError):
    """Error during code validation."""
    
    def __init__(self, message: str, line: Optional[int] = None, **kwargs):
        ctx = ErrorContext(line_number=line)
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_SYNTAX,
            context=ctx,
            recoverable=True,
            suggestion="Generated code has syntax issues, attempting correction",
            **kwargs
        )


class FileError(TranspilerError):
    """Error with file operations."""
    
    def __init__(self, message: str, file_path: str, code: ErrorCode = ErrorCode.FILE_NOT_FOUND, **kwargs):
        ctx = ErrorContext(file_path=file_path)
        super().__init__(
            message=message,
            code=code,
            context=ctx,
            **kwargs
        )


# ==================== Error Handler ====================

class ErrorHandler:
    """
    Centralized error handling with recovery strategies.
    """
    
    def __init__(self):
        self._errors: List[TranspilerError] = []
        self._warnings: List[str] = []
    
    def handle(self, error: Exception, context: Optional[ErrorContext] = None) -> Optional[Any]:
        """
        Handle an exception with appropriate recovery.
        
        Returns recovery result if recoverable, raises otherwise.
        """
        if isinstance(error, TranspilerError):
            self._errors.append(error)
            logger.error(str(error))
            
            if error.recoverable:
                return self._attempt_recovery(error)
            raise error
        
        # Wrap unknown exceptions
        wrapped = TranspilerError(
            message=str(error),
            code=ErrorCode.UNKNOWN,
            context=context,
            cause=error
        )
        self._errors.append(wrapped)
        logger.error(f"Unexpected error: {error}", exc_info=True)
        raise wrapped
    
    def _attempt_recovery(self, error: TranspilerError) -> Optional[Any]:
        """Attempt to recover from a recoverable error."""
        logger.info(f"Attempting recovery for {error.code.value}")
        
        if error.code == ErrorCode.TEMPLATE_NOT_FOUND:
            # Use stub generation as fallback
            return {"fallback": "stub", "reason": error.message}
        
        if error.code == ErrorCode.LLM_UNAVAILABLE:
            # Mark for stub generation
            return {"fallback": "stub", "reason": "LLM unavailable"}
        
        if error.code == ErrorCode.VALIDATION_SYNTAX:
            # Return code anyway with warning
            self._warnings.append(f"Syntax warning: {error.message}")
            return {"warn": True}
        
        return None
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self._warnings.append(message)
        logger.warning(message)
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all errors as dicts."""
        return [e.to_dict() for e in self._errors]
    
    def get_warnings(self) -> List[str]:
        """Get all warnings."""
        return self._warnings.copy()
    
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self._errors) > 0
    
    def clear(self):
        """Clear all errors and warnings."""
        self._errors.clear()
        self._warnings.clear()
    
    def summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return {
            "error_count": len(self._errors),
            "warning_count": len(self._warnings),
            "recoverable_count": sum(1 for e in self._errors if e.recoverable),
            "error_codes": list(set(e.code.value for e in self._errors))
        }


# Global error handler
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler
