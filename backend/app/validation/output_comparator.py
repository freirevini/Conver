"""
DataFrame Output Comparator.

Compares KNIME workflow outputs with Python-generated outputs:
- Schema comparison (columns, types)
- Value comparison (with tolerance for floats)
- Row count validation
- Detailed diff reporting
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class DiffType(Enum):
    """Types of differences found."""
    SCHEMA_COLUMN_MISSING = "column_missing"
    SCHEMA_COLUMN_EXTRA = "column_extra"
    SCHEMA_TYPE_MISMATCH = "type_mismatch"
    VALUE_MISMATCH = "value_mismatch"
    ROW_COUNT_MISMATCH = "row_count_mismatch"
    NULL_MISMATCH = "null_mismatch"


@dataclass
class Difference:
    """Single difference between two DataFrames."""
    diff_type: DiffType
    column: Optional[str] = None
    row: Optional[int] = None
    expected: Any = None
    actual: Any = None
    message: str = ""
    
    def __str__(self) -> str:
        if self.column and self.row is not None:
            return f"[{self.diff_type.value}] Column '{self.column}', Row {self.row}: expected={self.expected}, actual={self.actual}"
        elif self.column:
            return f"[{self.diff_type.value}] Column '{self.column}': {self.message or f'expected={self.expected}, actual={self.actual}'}"
        else:
            return f"[{self.diff_type.value}] {self.message}"


@dataclass
class ComparisonConfig:
    """Configuration for DataFrame comparison."""
    float_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5
    ignore_column_order: bool = True
    ignore_row_order: bool = False
    ignore_columns: List[str] = field(default_factory=list)
    case_sensitive_columns: bool = False
    max_differences: int = 100
    compare_nulls: bool = True
    string_strip: bool = True


@dataclass
class ComparisonResult:
    """Result of DataFrame comparison."""
    is_equal: bool
    differences: List[Difference] = field(default_factory=list)
    expected_rows: int = 0
    actual_rows: int = 0
    expected_columns: List[str] = field(default_factory=list)
    actual_columns: List[str] = field(default_factory=list)
    summary: str = ""
    
    @property
    def difference_count(self) -> int:
        return len(self.differences)
    
    @property
    def schema_differences(self) -> List[Difference]:
        schema_types = {DiffType.SCHEMA_COLUMN_MISSING, DiffType.SCHEMA_COLUMN_EXTRA, DiffType.SCHEMA_TYPE_MISMATCH}
        return [d for d in self.differences if d.diff_type in schema_types]
    
    @property
    def value_differences(self) -> List[Difference]:
        return [d for d in self.differences if d.diff_type == DiffType.VALUE_MISMATCH]


class DataFrameComparator:
    """
    Compares two Pandas DataFrames for equality.
    
    Features:
    - Schema comparison (columns and types)
    - Value comparison with configurable tolerance
    - Null value handling
    - Detailed difference reporting
    """
    
    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()
        self.differences: List[Difference] = []
    
    def compare(self, expected: "pd.DataFrame", actual: "pd.DataFrame") -> ComparisonResult:
        """
        Compare two DataFrames.
        
        Args:
            expected: Expected DataFrame (from KNIME)
            actual: Actual DataFrame (from Python)
            
        Returns:
            ComparisonResult with equality status and differences
        """
        if not PANDAS_AVAILABLE:
            return ComparisonResult(
                is_equal=False,
                summary="Pandas not available for comparison"
            )
        
        self.differences = []
        
        result = ComparisonResult(
            is_equal=True,
            expected_rows=len(expected),
            actual_rows=len(actual),
            expected_columns=list(expected.columns),
            actual_columns=list(actual.columns)
        )
        
        # 1. Compare row counts
        if len(expected) != len(actual):
            self._add_diff(Difference(
                diff_type=DiffType.ROW_COUNT_MISMATCH,
                expected=len(expected),
                actual=len(actual),
                message=f"Row count mismatch: expected {len(expected)}, got {len(actual)}"
            ))
        
        # 2. Compare schema
        self._compare_schema(expected, actual)
        
        # 3. Compare values (if schemas have common columns)
        common_columns = self._get_common_columns(expected, actual)
        if common_columns and len(expected) == len(actual):
            self._compare_values(expected, actual, common_columns)
        
        result.differences = self.differences[:self.config.max_differences]
        result.is_equal = len(self.differences) == 0
        result.summary = self._generate_summary(result)
        
        return result
    
    def _compare_schema(self, expected: "pd.DataFrame", actual: "pd.DataFrame"):
        """Compare DataFrame schemas."""
        expected_cols = set(expected.columns)
        actual_cols = set(actual.columns)
        
        # Handle case sensitivity
        if not self.config.case_sensitive_columns:
            expected_cols_lower = {c.lower(): c for c in expected_cols}
            actual_cols_lower = {c.lower(): c for c in actual_cols}
        else:
            expected_cols_lower = {c: c for c in expected_cols}
            actual_cols_lower = {c: c for c in actual_cols}
        
        # Remove ignored columns
        for col in self.config.ignore_columns:
            expected_cols_lower.pop(col.lower() if not self.config.case_sensitive_columns else col, None)
            actual_cols_lower.pop(col.lower() if not self.config.case_sensitive_columns else col, None)
        
        # Find missing columns
        for key, col in expected_cols_lower.items():
            if key not in actual_cols_lower:
                self._add_diff(Difference(
                    diff_type=DiffType.SCHEMA_COLUMN_MISSING,
                    column=col,
                    message=f"Column '{col}' missing from actual DataFrame"
                ))
        
        # Find extra columns
        for key, col in actual_cols_lower.items():
            if key not in expected_cols_lower:
                self._add_diff(Difference(
                    diff_type=DiffType.SCHEMA_COLUMN_EXTRA,
                    column=col,
                    message=f"Extra column '{col}' in actual DataFrame"
                ))
        
        # Compare types for common columns
        for key in set(expected_cols_lower.keys()) & set(actual_cols_lower.keys()):
            exp_col = expected_cols_lower[key]
            act_col = actual_cols_lower[key]
            
            exp_dtype = str(expected[exp_col].dtype)
            act_dtype = str(actual[act_col].dtype)
            
            if not self._types_compatible(exp_dtype, act_dtype):
                self._add_diff(Difference(
                    diff_type=DiffType.SCHEMA_TYPE_MISMATCH,
                    column=exp_col,
                    expected=exp_dtype,
                    actual=act_dtype,
                    message=f"Type mismatch for '{exp_col}': expected {exp_dtype}, got {act_dtype}"
                ))
    
    def _compare_values(self, expected: "pd.DataFrame", actual: "pd.DataFrame", columns: List[str]):
        """Compare DataFrame values for specified columns."""
        for col in columns:
            if col in self.config.ignore_columns:
                continue
            
            exp_col = expected[col]
            act_col = actual[col]
            
            for idx in range(len(expected)):
                if len(self.differences) >= self.config.max_differences:
                    return
                
                exp_val = exp_col.iloc[idx]
                act_val = act_col.iloc[idx]
                
                if not self._values_equal(exp_val, act_val):
                    self._add_diff(Difference(
                        diff_type=DiffType.VALUE_MISMATCH,
                        column=col,
                        row=idx,
                        expected=exp_val,
                        actual=act_val
                    ))
    
    def _values_equal(self, expected: Any, actual: Any) -> bool:
        """Check if two values are equal."""
        # Handle nulls
        exp_null = pd.isna(expected)
        act_null = pd.isna(actual)
        
        if exp_null and act_null:
            return True
        if exp_null != act_null:
            return False
        
        # Handle floats with tolerance
        if isinstance(expected, (float, np.floating)) and isinstance(actual, (float, np.floating)):
            if math.isnan(expected) and math.isnan(actual):
                return True
            if math.isinf(expected) or math.isinf(actual):
                return expected == actual
            
            # Absolute tolerance
            if abs(expected - actual) <= self.config.float_tolerance:
                return True
            
            # Relative tolerance
            if expected != 0:
                if abs((expected - actual) / expected) <= self.config.relative_tolerance:
                    return True
            
            return False
        
        # Handle strings
        if isinstance(expected, str) and isinstance(actual, str):
            if self.config.string_strip:
                return expected.strip() == actual.strip()
            return expected == actual
        
        # Default comparison
        return expected == actual
    
    def _types_compatible(self, expected_type: str, actual_type: str) -> bool:
        """Check if two types are compatible."""
        # Normalize types
        type_groups = {
            "int": {"int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64", "int"},
            "float": {"float16", "float32", "float64", "float"},
            "string": {"object", "string", "str"},
            "bool": {"bool", "boolean"},
            "datetime": {"datetime64", "datetime64[ns]"},
        }
        
        for group, types in type_groups.items():
            exp_in = any(t in expected_type.lower() for t in types)
            act_in = any(t in actual_type.lower() for t in types)
            if exp_in and act_in:
                return True
        
        return expected_type == actual_type
    
    def _get_common_columns(self, expected: "pd.DataFrame", actual: "pd.DataFrame") -> List[str]:
        """Get columns present in both DataFrames."""
        if self.config.case_sensitive_columns:
            return [c for c in expected.columns if c in actual.columns]
        else:
            actual_lower = {c.lower(): c for c in actual.columns}
            return [c for c in expected.columns if c.lower() in actual_lower]
    
    def _add_diff(self, diff: Difference):
        """Add difference if under limit."""
        if len(self.differences) < self.config.max_differences:
            self.differences.append(diff)
    
    def _generate_summary(self, result: ComparisonResult) -> str:
        """Generate summary text."""
        if result.is_equal:
            return f"✅ DataFrames are equal ({result.expected_rows} rows, {len(result.expected_columns)} columns)"
        
        lines = [
            f"❌ DataFrames differ",
            f"   Rows: expected={result.expected_rows}, actual={result.actual_rows}",
            f"   Columns: expected={len(result.expected_columns)}, actual={len(result.actual_columns)}",
            f"   Total differences: {result.difference_count}"
        ]
        
        schema_diffs = len(result.schema_differences)
        value_diffs = len(result.value_differences)
        
        if schema_diffs > 0:
            lines.append(f"   Schema differences: {schema_diffs}")
        if value_diffs > 0:
            lines.append(f"   Value differences: {value_diffs}")
        
        return "\n".join(lines)


# ==================== Public API ====================

def compare_dataframes(
    expected: "pd.DataFrame",
    actual: "pd.DataFrame",
    config: Optional[ComparisonConfig] = None
) -> ComparisonResult:
    """
    Compare two DataFrames.
    
    Args:
        expected: Expected DataFrame
        actual: Actual DataFrame
        config: Comparison configuration
        
    Returns:
        ComparisonResult with equality status and differences
    """
    comparator = DataFrameComparator(config)
    return comparator.compare(expected, actual)


def assert_dataframes_equal(
    expected: "pd.DataFrame",
    actual: "pd.DataFrame",
    config: Optional[ComparisonConfig] = None
) -> None:
    """
    Assert two DataFrames are equal, raising AssertionError if not.
    
    Args:
        expected: Expected DataFrame
        actual: Actual DataFrame
        config: Comparison configuration
        
    Raises:
        AssertionError: If DataFrames differ
    """
    result = compare_dataframes(expected, actual, config)
    
    if not result.is_equal:
        diff_details = "\n".join(str(d) for d in result.differences[:10])
        raise AssertionError(f"{result.summary}\n\nFirst 10 differences:\n{diff_details}")
