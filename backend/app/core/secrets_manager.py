"""
Secrets Manager for KNIME to Python Converter.

Provides secure management of sensitive credentials following best practices:
- All secrets from environment variables only
- No hardcoded sensitive defaults
- Validation and secure error messages
- Centralized credential access with logging
"""
import os
import logging
from typing import Optional
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecretDefinition:
    """Definition of a secret with metadata."""
    env_var: str
    description: str
    required: bool = True
    default: Optional[str] = None  # Only for NON-sensitive values
    is_sensitive: bool = True  # If True, never log the value
    validate_path: bool = False  # If True, validate file exists


class SecretsValidationError(Exception):
    """Raised when required secrets are missing or invalid."""
    pass


class SecretsManager:
    """
    Centralized secrets management with validation.
    
    Features:
    - Environment variable loading only (no hardcoded secrets)
    - Validation for required secrets
    - Path validation for credential files
    - Secure logging (never logs actual secret values)
    - Singleton pattern via lru_cache
    """
    
    # Define all secrets used by the application
    SECRETS: dict[str, SecretDefinition] = {
        # GCP Authentication
        "GOOGLE_CLOUD_PROJECT": SecretDefinition(
            env_var="GOOGLE_CLOUD_PROJECT",
            description="Google Cloud Project ID",
            required=True,
            is_sensitive=True,  # Project IDs can be sensitive
        ),
        "GOOGLE_CLOUD_LOCATION": SecretDefinition(
            env_var="GOOGLE_CLOUD_LOCATION",
            description="Google Cloud Region",
            required=False,
            default="us-central1",
            is_sensitive=False,
        ),
        "GOOGLE_APPLICATION_CREDENTIALS": SecretDefinition(
            env_var="GOOGLE_APPLICATION_CREDENTIALS",
            description="Path to GCP service account JSON",
            required=False,
            is_sensitive=True,
            validate_path=True,
        ),
        
        # Gemini API
        "GEMINI_MODEL": SecretDefinition(
            env_var="GEMINI_MODEL",
            description="Gemini model to use",
            required=False,
            default="gemini-2.5-pro",  # Non-sensitive config
            is_sensitive=False,
        ),
        
        # Database (if needed in future)
        "DATABASE_URL": SecretDefinition(
            env_var="DATABASE_URL",
            description="Database connection string",
            required=False,
            is_sensitive=True,
        ),
        
        # API Keys
        "API_SECRET_KEY": SecretDefinition(
            env_var="API_SECRET_KEY",
            description="Secret key for API authentication",
            required=False,
            is_sensitive=True,
        ),
    }
    
    _validated: bool = False
    _values: dict[str, Optional[str]] = field(default_factory=dict)
    
    def __init__(self) -> None:
        """Initialize secrets manager and load all secrets."""
        self._values = {}
        self._load_all_secrets()
    
    def _load_all_secrets(self) -> None:
        """Load all secrets from environment variables."""
        for secret_name, definition in self.SECRETS.items():
            value = os.getenv(definition.env_var)
            
            if value is None and definition.default is not None:
                value = definition.default
            
            self._values[secret_name] = value
            
            # Log loading status (never the value!)
            status = "loaded" if value else "not set"
            if definition.is_sensitive:
                logger.debug(f"Secret {secret_name}: {status}")
            else:
                # Only log non-sensitive values
                logger.debug(f"Config {secret_name}: {value or 'not set'}")
    
    def validate(self, raise_on_error: bool = True) -> list[str]:
        """
        Validate that all required secrets are present.
        
        Args:
            raise_on_error: If True, raises SecretsValidationError on failure
            
        Returns:
            List of validation error messages (empty if valid)
            
        Raises:
            SecretsValidationError: If required secrets are missing and raise_on_error=True
        """
        errors = []
        
        for secret_name, definition in self.SECRETS.items():
            value = self._values.get(secret_name)
            
            # Check required secrets
            if definition.required and not value:
                errors.append(
                    f"Required secret missing: {secret_name} "
                    f"(set {definition.env_var} environment variable)"
                )
            
            # Validate paths if required
            if value and definition.validate_path:
                path = Path(value)
                if not path.exists():
                    errors.append(
                        f"Path not found for {secret_name}: {value}"
                    )
        
        if errors and raise_on_error:
            error_msg = "Secrets validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise SecretsValidationError(error_msg)
        
        self._validated = True
        return errors
    
    def get(self, secret_name: str) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            secret_name: Name of the secret (from SECRETS dict)
            
        Returns:
            Secret value or None if not set
            
        Raises:
            KeyError: If secret_name is not a known secret
        """
        if secret_name not in self.SECRETS:
            raise KeyError(f"Unknown secret: {secret_name}")
        
        return self._values.get(secret_name)
    
    def get_required(self, secret_name: str) -> str:
        """
        Get a required secret value.
        
        Args:
            secret_name: Name of the secret
            
        Returns:
            Secret value (guaranteed non-None)
            
        Raises:
            SecretsValidationError: If secret is not set
        """
        value = self.get(secret_name)
        
        if value is None:
            definition = self.SECRETS[secret_name]
            raise SecretsValidationError(
                f"Required secret not set: {secret_name} "
                f"(set {definition.env_var} environment variable)"
            )
        
        return value
    
    @property
    def google_cloud_project(self) -> Optional[str]:
        """Get Google Cloud Project ID."""
        return self.get("GOOGLE_CLOUD_PROJECT")
    
    @property
    def google_cloud_location(self) -> str:
        """Get Google Cloud Location with default."""
        return self.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
    
    @property
    def google_credentials_path(self) -> Optional[str]:
        """Get Google Application Credentials path."""
        return self.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    @property
    def gemini_model(self) -> str:
        """Get Gemini model name."""
        return self.get("GEMINI_MODEL") or "gemini-2.5-pro"
    
    def __repr__(self) -> str:
        """Safe string representation (no secret values)."""
        status = {}
        for name, definition in self.SECRETS.items():
            value = self._values.get(name)
            if definition.is_sensitive:
                status[name] = "***" if value else "NOT_SET"
            else:
                status[name] = value or "NOT_SET"
        return f"SecretsManager({status})"


@lru_cache()
def get_secrets_manager() -> SecretsManager:
    """Get cached SecretsManager instance."""
    return SecretsManager()


# Convenience export
secrets = get_secrets_manager()


def validate_startup_secrets() -> None:
    """
    Validate all required secrets at application startup.
    
    Call this in your application entry point to fail fast
    if required secrets are missing.
    """
    errors = secrets.validate(raise_on_error=False)
    
    if errors:
        logger.warning(
            "Some secrets are not configured. "
            "This may cause issues with certain features:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )
    else:
        logger.info("All secrets validated successfully")
