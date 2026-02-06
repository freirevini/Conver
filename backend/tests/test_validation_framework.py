"""
Test Suite for Validation Framework.

Tests:
- DataFrame comparison
- Schema validation
- Diff reporting
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    if not PANDAS_AVAILABLE:
        pytest.skip("Pandas not available")
    
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [85.5, 92.3, 78.1, 88.7, 95.0],
        "passed": [True, True, True, True, True]
    })


@pytest.fixture
def modified_df(sample_df):
    """Modified DataFrame for comparison."""
    df = sample_df.copy()
    df.loc[2, "score"] = 78.2  # Small change
    df.loc[4, "name"] = "Eva"  # String change
    return df


class TestDataFrameComparator:
    """Tests for output_comparator module."""
    
    def test_equal_dataframes(self, sample_df):
        """Equal DataFrames should compare as equal."""
        from app.validation.output_comparator import compare_dataframes
        
        result = compare_dataframes(sample_df, sample_df.copy())
        
        assert result.is_equal
        assert result.difference_count == 0
    
    def test_different_values(self, sample_df, modified_df):
        """Value differences should be detected."""
        from app.validation.output_comparator import compare_dataframes
        
        result = compare_dataframes(sample_df, modified_df)
        
        assert not result.is_equal
        assert result.difference_count > 0
    
    def test_float_tolerance(self, sample_df):
        """Float tolerance should be respected."""
        from app.validation.output_comparator import compare_dataframes, ComparisonConfig
        
        df2 = sample_df.copy()
        df2.loc[0, "score"] = 85.5000001  # Very small difference
        
        config = ComparisonConfig(float_tolerance=1e-5)
        result = compare_dataframes(sample_df, df2, config)
        
        assert result.is_equal
    
    def test_missing_column(self, sample_df):
        """Missing columns should be detected."""
        from app.validation.output_comparator import compare_dataframes
        
        df2 = sample_df.drop(columns=["passed"])
        result = compare_dataframes(sample_df, df2)
        
        assert not result.is_equal
        schema_diffs = result.schema_differences
        assert len(schema_diffs) > 0
    
    def test_extra_column(self, sample_df):
        """Extra columns should be detected."""
        from app.validation.output_comparator import compare_dataframes
        
        df2 = sample_df.copy()
        df2["extra"] = "value"
        result = compare_dataframes(sample_df, df2)
        
        schema_diffs = result.schema_differences
        assert len(schema_diffs) > 0
    
    def test_row_count_mismatch(self, sample_df):
        """Row count mismatches should be detected."""
        from app.validation.output_comparator import compare_dataframes
        
        df2 = sample_df.head(3)
        result = compare_dataframes(sample_df, df2)
        
        assert not result.is_equal
        assert result.expected_rows != result.actual_rows
    
    def test_null_handling(self):
        """Null values should be compared correctly."""
        if not PANDAS_AVAILABLE:
            pytest.skip("Pandas not available")
        
        from app.validation.output_comparator import compare_dataframes
        
        df1 = pd.DataFrame({"a": [1, None, 3]})
        df2 = pd.DataFrame({"a": [1, None, 3]})
        
        result = compare_dataframes(df1, df2)
        assert result.is_equal
    
    def test_assert_dataframes_equal(self, sample_df, modified_df):
        """Assert should raise on differences."""
        from app.validation.output_comparator import assert_dataframes_equal
        
        # Should pass
        assert_dataframes_equal(sample_df, sample_df.copy())
        
        # Should fail
        with pytest.raises(AssertionError):
            assert_dataframes_equal(sample_df, modified_df)


class TestSchemaValidator:
    """Tests for schema_validator module."""
    
    def test_valid_schema(self, sample_df):
        """Valid DataFrame should pass validation."""
        from app.validation.schema_validator import SchemaSpec, validate_schema
        
        schema = SchemaSpec()
        schema.add_column("id", "int")
        schema.add_column("name", "object")
        schema.add_column("score", "float")
        schema.add_column("passed", "bool")
        
        result = validate_schema(sample_df, schema)
        
        assert result.is_valid
    
    def test_missing_column(self, sample_df):
        """Missing required column should fail."""
        from app.validation.schema_validator import SchemaSpec, validate_schema
        
        schema = SchemaSpec()
        schema.add_column("id", "int")
        schema.add_column("nonexistent", "string")
        
        result = validate_schema(sample_df, schema)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_nullable_constraint(self):
        """Non-nullable column with nulls should fail."""
        if not PANDAS_AVAILABLE:
            pytest.skip("Pandas not available")
        
        from app.validation.schema_validator import SchemaSpec, validate_schema
        
        df = pd.DataFrame({"a": [1, None, 3]})
        
        schema = SchemaSpec()
        schema.add_column("a", "float", nullable=False)
        
        result = validate_schema(df, schema)
        
        assert not result.is_valid
    
    def test_unique_constraint(self):
        """Unique column with duplicates should fail."""
        if not PANDAS_AVAILABLE:
            pytest.skip("Pandas not available")
        
        from app.validation.schema_validator import SchemaSpec, validate_schema
        
        df = pd.DataFrame({"a": [1, 2, 2, 3]})
        
        schema = SchemaSpec()
        schema.add_column("a", "int", unique=True)
        
        result = validate_schema(df, schema)
        
        assert not result.is_valid
    
    def test_min_max_constraints(self):
        """Value constraints should be validated."""
        if not PANDAS_AVAILABLE:
            pytest.skip("Pandas not available")
        
        from app.validation.schema_validator import SchemaSpec, validate_schema
        
        df = pd.DataFrame({"score": [50, 75, 100, 150]})
        
        schema = SchemaSpec()
        schema.add_column("score", "int", min_value=0, max_value=100)
        
        result = validate_schema(df, schema)
        
        assert not result.is_valid
    
    def test_create_schema_from_dataframe(self, sample_df):
        """Schema should be created from DataFrame."""
        from app.validation.schema_validator import create_schema_from_dataframe
        
        schema = create_schema_from_dataframe(sample_df)
        
        assert len(schema.columns) == 4
        assert "id" in schema.column_names


class TestDiffReporter:
    """Tests for diff_reporter module."""
    
    def test_console_report(self, sample_df, modified_df):
        """Console report should be generated."""
        from app.validation.output_comparator import compare_dataframes
        from app.validation.diff_reporter import generate_comparison_report
        
        result = compare_dataframes(sample_df, modified_df)
        report = generate_comparison_report(result, format="console")
        
        assert "DataFrame Comparison Report" in report
        assert "DIFFERENT" in report
    
    def test_markdown_report(self, sample_df, modified_df):
        """Markdown report should be generated."""
        from app.validation.output_comparator import compare_dataframes
        from app.validation.diff_reporter import generate_comparison_report
        
        result = compare_dataframes(sample_df, modified_df)
        report = generate_comparison_report(result, format="markdown")
        
        assert "# DataFrame Comparison Report" in report
        assert "| Metric |" in report
    
    def test_html_report(self, sample_df, modified_df):
        """HTML report should be generated."""
        from app.validation.output_comparator import compare_dataframes
        from app.validation.diff_reporter import generate_comparison_report
        
        result = compare_dataframes(sample_df, modified_df)
        report = generate_comparison_report(result, format="html")
        
        assert "<!DOCTYPE html>" in report
        assert "<table>" in report
    
    def test_json_report(self, sample_df, modified_df):
        """JSON report should be generated."""
        from app.validation.output_comparator import compare_dataframes
        from app.validation.diff_reporter import DiffReporter
        
        result = compare_dataframes(sample_df, modified_df)
        reporter = DiffReporter()
        json_report = reporter.to_json(result)
        
        assert "is_equal" in json_report
        assert "differences" in json_report
        assert isinstance(json_report["differences"], list)
    
    def test_validation_report(self, sample_df):
        """Validation report should be generated."""
        from app.validation.schema_validator import SchemaSpec, validate_schema
        from app.validation.diff_reporter import generate_validation_report
        
        schema = SchemaSpec()
        schema.add_column("id", "int")
        schema.add_column("missing", "string")
        
        result = validate_schema(sample_df, schema)
        report = generate_validation_report(result, format="console")
        
        assert "Schema Validation Report" in report


class TestIntegration:
    """Integration tests for validation framework."""
    
    def test_full_validation_workflow(self, sample_df, modified_df):
        """Complete validation workflow should work."""
        from app.validation.output_comparator import compare_dataframes
        from app.validation.schema_validator import create_schema_from_dataframe, validate_schema
        from app.validation.diff_reporter import generate_comparison_report, generate_validation_report
        
        # Create schema from expected
        schema = create_schema_from_dataframe(sample_df)
        
        # Validate actual against schema
        validation = validate_schema(modified_df, schema)
        assert validation.is_valid
        
        # Compare values
        comparison = compare_dataframes(sample_df, modified_df)
        assert not comparison.is_equal
        
        # Generate reports
        comp_report = generate_comparison_report(comparison)
        val_report = generate_validation_report(validation)
        
        assert len(comp_report) > 0
        assert len(val_report) > 0
    
    def test_all_modules_import(self):
        """All validation modules should import."""
        from app.validation import output_comparator
        from app.validation import schema_validator
        from app.validation import diff_reporter
        
        assert output_comparator is not None
        assert schema_validator is not None
        assert diff_reporter is not None
