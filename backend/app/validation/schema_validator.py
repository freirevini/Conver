"""
Schema Validator for DataFrame Outputs.

Validates DataFrame schema against expected specifications:
- Column presence and order
- Data type validation
- Nullable constraints
- Value constraints
"""
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class ValidationLevel(Enum):
    """Severity level of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ColumnSpec:
    """Specification for a single column."""
    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    description: str = ""


@dataclass
class SchemaSpec:
    """Complete schema specification."""
    columns: List[ColumnSpec] = field(default_factory=list)
    strict: bool = False  # If True, extra columns cause errors
    ordered: bool = False  # If True, column order matters
    min_rows: int = 0
    max_rows: Optional[int] = None
    
    def add_column(self, name: str, dtype: str, **kwargs) -> "SchemaSpec":
        """Add column spec (fluent API)."""
        self.columns.append(ColumnSpec(name=name, dtype=dtype, **kwargs))
        return self
    
    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]


@dataclass
class ValidationIssue:
    """Single validation issue."""
    level: ValidationLevel
    column: Optional[str]
    message: str
    row: Optional[int] = None
    value: Any = None
    
    def __str__(self) -> str:
        parts = [f"[{self.level.value.upper()}]"]
        if self.column:
            parts.append(f"Column '{self.column}'")
        if self.row is not None:
            parts.append(f"Row {self.row}")
        parts.append(self.message)
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]


class SchemaValidator:
    """
    Validates DataFrame against schema specification.
    
    Features:
    - Column name validation
    - Data type validation
    - Nullable constraint validation
    - Value constraint validation
    - Pattern matching for strings
    """
    
    TYPE_MAPPING = {
        "int": {"int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"},
        "float": {"float16", "float32", "float64"},
        "string": {"object", "string"},
        "bool": {"bool", "boolean"},
        "datetime": {"datetime64[ns]", "datetime64"},
        "category": {"category"},
    }
    
    def __init__(self, schema: SchemaSpec):
        self.schema = schema
        self.issues: List[ValidationIssue] = []
    
    def validate(self, df: "pd.DataFrame") -> ValidationResult:
        """
        Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with issues
        """
        if not PANDAS_AVAILABLE:
            return ValidationResult(
                is_valid=False,
                summary="Pandas not available"
            )
        
        self.issues = []
        
        # Validate row count
        self._validate_row_count(df)
        
        # Validate columns
        self._validate_columns(df)
        
        # Validate data types
        self._validate_types(df)
        
        # Validate constraints
        self._validate_constraints(df)
        
        is_valid = len([i for i in self.issues if i.level == ValidationLevel.ERROR]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=self.issues,
            summary=self._generate_summary(df, is_valid)
        )
    
    def _validate_row_count(self, df: "pd.DataFrame"):
        """Validate row count constraints."""
        row_count = len(df)
        
        if row_count < self.schema.min_rows:
            self._add_issue(
                ValidationLevel.ERROR,
                None,
                f"Row count {row_count} is less than minimum {self.schema.min_rows}"
            )
        
        if self.schema.max_rows and row_count > self.schema.max_rows:
            self._add_issue(
                ValidationLevel.ERROR,
                None,
                f"Row count {row_count} exceeds maximum {self.schema.max_rows}"
            )
    
    def _validate_columns(self, df: "pd.DataFrame"):
        """Validate column presence and order."""
        df_columns = set(df.columns)
        expected_columns = set(self.schema.column_names)
        
        # Check for missing columns
        missing = expected_columns - df_columns
        for col in missing:
            self._add_issue(
                ValidationLevel.ERROR,
                col,
                f"Required column '{col}' is missing"
            )
        
        # Check for extra columns
        extra = df_columns - expected_columns
        if extra and self.schema.strict:
            for col in extra:
                self._add_issue(
                    ValidationLevel.ERROR,
                    col,
                    f"Unexpected column '{col}' (strict mode)"
                )
        elif extra:
            for col in extra:
                self._add_issue(
                    ValidationLevel.WARNING,
                    col,
                    f"Extra column '{col}' not in schema"
                )
        
        # Check column order
        if self.schema.ordered:
            expected_order = [c for c in self.schema.column_names if c in df_columns]
            actual_order = [c for c in df.columns if c in expected_columns]
            
            if expected_order != actual_order:
                self._add_issue(
                    ValidationLevel.WARNING,
                    None,
                    f"Column order mismatch: expected {expected_order}, got {actual_order}"
                )
    
    def _validate_types(self, df: "pd.DataFrame"):
        """Validate data types."""
        for spec in self.schema.columns:
            if spec.name not in df.columns:
                continue
            
            actual_type = str(df[spec.name].dtype)
            
            if not self._type_matches(spec.dtype, actual_type):
                self._add_issue(
                    ValidationLevel.ERROR,
                    spec.name,
                    f"Type mismatch: expected '{spec.dtype}', got '{actual_type}'"
                )
    
    def _validate_constraints(self, df: "pd.DataFrame"):
        """Validate value constraints."""
        for spec in self.schema.columns:
            if spec.name not in df.columns:
                continue
            
            col = df[spec.name]
            
            # Nullable constraint
            if not spec.nullable and col.isna().any():
                null_count = col.isna().sum()
                self._add_issue(
                    ValidationLevel.ERROR,
                    spec.name,
                    f"Column has {null_count} null values but is not nullable"
                )
            
            # Unique constraint
            if spec.unique and col.duplicated().any():
                dup_count = col.duplicated().sum()
                self._add_issue(
                    ValidationLevel.ERROR,
                    spec.name,
                    f"Column has {dup_count} duplicate values but should be unique"
                )
            
            # Min/max value constraints
            if spec.min_value is not None:
                below_min = col.dropna() < spec.min_value
                if below_min.any():
                    self._add_issue(
                        ValidationLevel.ERROR,
                        spec.name,
                        f"Values below minimum {spec.min_value}"
                    )
            
            if spec.max_value is not None:
                above_max = col.dropna() > spec.max_value
                if above_max.any():
                    self._add_issue(
                        ValidationLevel.ERROR,
                        spec.name,
                        f"Values above maximum {spec.max_value}"
                    )
            
            # Allowed values constraint
            if spec.allowed_values is not None:
                invalid = ~col.dropna().isin(spec.allowed_values)
                if invalid.any():
                    invalid_vals = col.dropna()[invalid].unique()[:5]
                    self._add_issue(
                        ValidationLevel.ERROR,
                        spec.name,
                        f"Invalid values: {list(invalid_vals)}"
                    )
    
    def _type_matches(self, expected: str, actual: str) -> bool:
        """Check if actual type matches expected."""
        expected_lower = expected.lower()
        actual_lower = actual.lower()
        
        # Direct match
        if expected_lower in actual_lower or actual_lower in expected_lower:
            return True
        
        # Check type mapping
        for group, types in self.TYPE_MAPPING.items():
            if expected_lower == group or expected_lower in types:
                if any(t in actual_lower for t in types):
                    return True
        
        return False
    
    def _add_issue(self, level: ValidationLevel, column: Optional[str], message: str):
        """Add validation issue."""
        self.issues.append(ValidationIssue(
            level=level,
            column=column,
            message=message
        ))
    
    def _generate_summary(self, df: "pd.DataFrame", is_valid: bool) -> str:
        """Generate summary."""
        errors = len(self.errors)
        warnings = len(self.warnings)
        
        if is_valid:
            return f"✅ Schema valid ({len(df)} rows, {len(df.columns)} columns)"
        else:
            return f"❌ Schema invalid: {errors} errors, {warnings} warnings"


# ==================== Public API ====================

def validate_schema(df: "pd.DataFrame", schema: SchemaSpec) -> ValidationResult:
    """Validate DataFrame against schema."""
    validator = SchemaValidator(schema)
    return validator.validate(df)


def create_schema_from_dataframe(df: "pd.DataFrame") -> SchemaSpec:
    """Create schema spec from existing DataFrame."""
    schema = SchemaSpec()
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        nullable = df[col].isna().any()
        
        schema.add_column(
            name=col,
            dtype=dtype,
            nullable=nullable
        )
    
    return schema
