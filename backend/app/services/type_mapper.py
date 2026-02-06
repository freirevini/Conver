"""
KNIME to Python Type Mapper

Provides automatic type conversion from KNIME cell types to Python/Pandas types.
Ensures type safety in generated code by properly casting values.
"""
import logging
import re
from datetime import datetime, date, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


# ==================== KNIME Cell Type Mappings ====================

KNIME_TO_PYTHON: Dict[str, Type] = {
    # String types
    "StringCell": str,
    "org.knime.core.data.def.StringCell": str,
    "StringValue": str,
    
    # Numeric types - Integer
    "IntCell": int,
    "org.knime.core.data.def.IntCell": int,
    "IntValue": int,
    "LongCell": int,
    "org.knime.core.data.def.LongCell": int,
    "LongValue": int,
    
    # Numeric types - Float/Double
    "DoubleCell": float,
    "org.knime.core.data.def.DoubleCell": float,
    "DoubleValue": float,
    
    # Boolean
    "BooleanCell": bool,
    "org.knime.core.data.def.BooleanCell": bool,
    "BooleanValue": bool,
    
    # Date/Time types
    "DateAndTimeCell": datetime,
    "org.knime.core.data.date.DateAndTimeCell": datetime,
    "DateAndTimeValue": datetime,
    "LocalDateCell": date,
    "LocalTimeCell": time,
    "LocalDateTimeCell": datetime,
    "ZonedDateTimeCell": datetime,
    "DurationCell": timedelta,
    "PeriodCell": str,  # Keep as string for periods
    
    # Binary/Blob
    "BlobCell": bytes,
    "org.knime.core.data.blob.BlobCell": bytes,
    
    # Complex types (default to string for safety)
    "XMLCell": str,
    "JSONCell": str,
    "ListCell": list,
    "SetCell": set,
    
    # Missing value indicator
    "MissingCell": None,
}

# Pandas dtype mappings for DataFrame columns
KNIME_TO_PANDAS_DTYPE: Dict[str, str] = {
    # String
    "StringCell": "object",
    "StringValue": "object",
    
    # Integer (nullable)
    "IntCell": "Int64",
    "IntValue": "Int64",
    "LongCell": "Int64",
    "LongValue": "Int64",
    
    # Float
    "DoubleCell": "float64",
    "DoubleValue": "float64",
    
    # Boolean (nullable)
    "BooleanCell": "boolean",
    "BooleanValue": "boolean",
    
    # DateTime
    "DateAndTimeCell": "datetime64[ns]",
    "DateAndTimeValue": "datetime64[ns]",
    "LocalDateCell": "datetime64[ns]",
    "LocalDateTimeCell": "datetime64[ns]",
    "LocalTimeCell": "object",  # Time stored as object
    "ZonedDateTimeCell": "datetime64[ns, UTC]",
    "DurationCell": "timedelta64[ns]",
}

# NumPy dtype mappings
KNIME_TO_NUMPY_DTYPE: Dict[str, str] = {
    "IntCell": "int64",
    "LongCell": "int64",
    "DoubleCell": "float64",
    "BooleanCell": "bool",
    "StringCell": "object",
}


# ==================== Type Conversion Functions ====================

def convert_value(value: Any, knime_type: str) -> Any:
    """
    Convert a value from XML/string to the proper Python type.
    
    Args:
        value: Raw value (usually string from XML)
        knime_type: KNIME cell type name
        
    Returns:
        Converted value in proper Python type
    """
    if value is None or value == "" or knime_type == "MissingCell":
        return None
    
    # Normalize type name
    type_name = _normalize_type_name(knime_type)
    python_type = KNIME_TO_PYTHON.get(type_name, str)
    
    if python_type is None:
        return None
    
    try:
        # Handle special cases
        if python_type == bool:
            return _convert_bool(value)
        elif python_type == datetime:
            return _convert_datetime(value)
        elif python_type == date:
            return _convert_date(value)
        elif python_type == time:
            return _convert_time(value)
        elif python_type == bytes:
            return _convert_bytes(value)
        elif python_type in (list, set):
            return _convert_collection(value, python_type)
        else:
            return python_type(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{value}' to {python_type}: {e}")
        return str(value)  # Fallback to string


def _normalize_type_name(knime_type: str) -> str:
    """Normalize KNIME type name for lookup."""
    # Remove package prefix if present
    if "." in knime_type:
        parts = knime_type.split(".")
        knime_type = parts[-1]
    return knime_type


def _convert_bool(value: Any) -> bool:
    """Convert to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


def _convert_datetime(value: Any) -> datetime:
    """Convert to datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Try common formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return datetime.fromisoformat(str(value))


def _convert_date(value: Any) -> date:
    """Convert to date."""
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    return date.fromisoformat(str(value))


def _convert_time(value: Any) -> time:
    """Convert to time."""
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, str):
        for fmt in ["%H:%M:%S", "%H:%M:%S.%f", "%H:%M"]:
            try:
                return datetime.strptime(value, fmt).time()
            except ValueError:
                continue
    return time.fromisoformat(str(value))


def _convert_bytes(value: Any) -> bytes:
    """Convert to bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        # Check if base64 encoded
        import base64
        try:
            return base64.b64decode(value)
        except Exception:
            return value.encode("utf-8")
    return bytes(value)


def _convert_collection(value: Any, target_type: Type) -> Union[list, set]:
    """Convert to list or set."""
    if isinstance(value, (list, set, tuple)):
        return target_type(value)
    if isinstance(value, str):
        # Try JSON parse
        import json
        try:
            parsed = json.loads(value)
            return target_type(parsed)
        except json.JSONDecodeError:
            # Split by comma
            items = [x.strip() for x in value.split(",")]
            return target_type(items)
    return target_type([value])


# ==================== Code Generation Helpers ====================

def get_pandas_dtype(knime_type: str) -> str:
    """Get Pandas dtype string for a KNIME type."""
    type_name = _normalize_type_name(knime_type)
    return KNIME_TO_PANDAS_DTYPE.get(type_name, "object")


def get_python_type_name(knime_type: str) -> str:
    """Get Python type name as string for code generation."""
    type_name = _normalize_type_name(knime_type)
    python_type = KNIME_TO_PYTHON.get(type_name, str)
    
    if python_type is None:
        return "None"
    return python_type.__name__


def generate_cast_code(column: str, knime_type: str, df_var: str = "df") -> str:
    """
    Generate Python code to cast a column to the correct type.
    
    Args:
        column: Column name
        knime_type: KNIME cell type
        df_var: DataFrame variable name
        
    Returns:
        Python code for type conversion
    """
    type_name = _normalize_type_name(knime_type)
    
    if type_name in ("IntCell", "IntValue", "LongCell", "LongValue"):
        return f"{df_var}['{column}'] = pd.to_numeric({df_var}['{column}'], errors='coerce').astype('Int64')"
    
    elif type_name in ("DoubleCell", "DoubleValue"):
        return f"{df_var}['{column}'] = pd.to_numeric({df_var}['{column}'], errors='coerce')"
    
    elif type_name in ("BooleanCell", "BooleanValue"):
        return f"{df_var}['{column}'] = {df_var}['{column}'].map({{'true': True, 'false': False, '1': True, '0': False}}).astype('boolean')"
    
    elif type_name in ("DateAndTimeCell", "DateAndTimeValue", "LocalDateTimeCell", "LocalDateCell"):
        return f"{df_var}['{column}'] = pd.to_datetime({df_var}['{column}'], errors='coerce')"
    
    elif type_name == "DurationCell":
        return f"{df_var}['{column}'] = pd.to_timedelta({df_var}['{column}'], errors='coerce')"
    
    elif type_name in ("StringCell", "StringValue"):
        return f"{df_var}['{column}'] = {df_var}['{column}'].astype(str)"
    
    return f"# No type conversion needed for {type_name}"


def generate_column_type_conversions(
    column_types: Dict[str, str],
    df_var: str = "df"
) -> Tuple[str, List[str]]:
    """
    Generate code to convert multiple columns to their correct types.
    
    Args:
        column_types: Dict mapping column names to KNIME types
        df_var: DataFrame variable name
        
    Returns:
        Tuple of (code, imports)
    """
    if not column_types:
        return "", []
    
    lines = ["# Type conversions"]
    imports = ["import pandas as pd"]
    
    for column, knime_type in column_types.items():
        cast_code = generate_cast_code(column, knime_type, df_var)
        if not cast_code.startswith("#"):
            lines.append(cast_code)
    
    return "\n".join(lines), imports


def infer_type_from_value(value: Any) -> str:
    """
    Infer KNIME type from a Python value.
    Used for reverse mapping.
    """
    if value is None:
        return "MissingCell"
    elif isinstance(value, bool):
        return "BooleanCell"
    elif isinstance(value, int):
        return "IntCell"
    elif isinstance(value, float):
        return "DoubleCell"
    elif isinstance(value, datetime):
        return "DateAndTimeCell"
    elif isinstance(value, date):
        return "LocalDateCell"
    elif isinstance(value, time):
        return "LocalTimeCell"
    elif isinstance(value, bytes):
        return "BlobCell"
    elif isinstance(value, list):
        return "ListCell"
    elif isinstance(value, set):
        return "SetCell"
    else:
        return "StringCell"


# ==================== Validation Helpers ====================

def validate_value_type(value: Any, expected_knime_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a value matches the expected KNIME type.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    type_name = _normalize_type_name(expected_knime_type)
    expected_python_type = KNIME_TO_PYTHON.get(type_name, str)
    
    if value is None and type_name != "MissingCell":
        return True, None  # Allow None for any type (represents missing)
    
    if expected_python_type is None:
        return value is None, f"Expected None, got {type(value).__name__}"
    
    if isinstance(value, expected_python_type):
        return True, None
    
    # Try conversion
    try:
        convert_value(value, expected_knime_type)
        return True, None
    except Exception as e:
        return False, f"Cannot convert {type(value).__name__} to {expected_python_type.__name__}: {e}"


# ==================== Type Mapper Class ====================

class TypeMapper:
    """
    Centralized type mapping service.
    
    Usage:
        mapper = TypeMapper()
        python_value = mapper.convert("1000", "IntCell")  # Returns 1000 (int)
        dtype = mapper.get_pandas_dtype("DoubleCell")  # Returns "float64"
    """
    
    def __init__(self):
        self.type_map = KNIME_TO_PYTHON
        self.pandas_dtypes = KNIME_TO_PANDAS_DTYPE
        self.numpy_dtypes = KNIME_TO_NUMPY_DTYPE
    
    def convert(self, value: Any, knime_type: str) -> Any:
        """Convert value to proper Python type."""
        return convert_value(value, knime_type)
    
    def get_pandas_dtype(self, knime_type: str) -> str:
        """Get Pandas dtype for KNIME type."""
        return get_pandas_dtype(knime_type)
    
    def get_python_type(self, knime_type: str) -> Type:
        """Get Python type for KNIME type."""
        type_name = _normalize_type_name(knime_type)
        return self.type_map.get(type_name, str)
    
    def generate_cast(self, column: str, knime_type: str, df_var: str = "df") -> str:
        """Generate type cast code."""
        return generate_cast_code(column, knime_type, df_var)
    
    def validate(self, value: Any, knime_type: str) -> Tuple[bool, Optional[str]]:
        """Validate value matches expected type."""
        return validate_value_type(value, knime_type)
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported KNIME types."""
        return list(self.type_map.keys())


# Singleton instance
type_mapper = TypeMapper()
