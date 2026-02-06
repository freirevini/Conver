"""
Validation Framework Module.

Provides:
- output_comparator: DataFrame comparison with tolerance
- schema_validator: Schema validation
- diff_reporter: Report generation
"""
from .output_comparator import (
    compare_dataframes,
    assert_dataframes_equal,
    DataFrameComparator,
    ComparisonConfig,
    ComparisonResult,
    Difference,
    DiffType
)

from .schema_validator import (
    validate_schema,
    create_schema_from_dataframe,
    SchemaValidator,
    SchemaSpec,
    ColumnSpec,
    ValidationResult,
    ValidationIssue,
    ValidationLevel
)

from .diff_reporter import (
    generate_comparison_report,
    generate_validation_report,
    DiffReporter,
    ValidationReporter,
    ReportConfig
)

__all__ = [
    # Comparator
    "compare_dataframes",
    "assert_dataframes_equal",
    "DataFrameComparator",
    "ComparisonConfig",
    "ComparisonResult",
    "Difference",
    "DiffType",
    # Schema
    "validate_schema",
    "create_schema_from_dataframe",
    "SchemaValidator",
    "SchemaSpec",
    "ColumnSpec",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    # Reporter
    "generate_comparison_report",
    "generate_validation_report",
    "DiffReporter",
    "ValidationReporter",
    "ReportConfig",
]
