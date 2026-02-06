"""
Equivalence Validator - Validates functional equivalence between KNIME and Python outputs.

Handles:
- KNIME batch execution (optional)
- Python script execution
- DataFrame comparison with tolerances
- Diff report generation
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.models.ir_models import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two outputs."""
    is_equal: bool
    differences: List[str] = field(default_factory=list)
    tolerance_used: float = 1e-5
    rows_compared: int = 0
    columns_compared: int = 0


@dataclass  
class ValidationReport:
    """Complete validation report."""
    is_valid: bool
    total_nodes: int
    validated_nodes: int
    failed_nodes: int
    results: List[ValidationResult] = field(default_factory=list)
    execution_time_knime: Optional[float] = None
    execution_time_python: Optional[float] = None
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_nodes": self.total_nodes,
            "validated_nodes": self.validated_nodes,
            "failed_nodes": self.failed_nodes,
            "results": [r.to_dict() for r in self.results],
            "execution_time_knime": self.execution_time_knime,
            "execution_time_python": self.execution_time_python,
            "summary": self.summary,
        }


class EquivalenceValidator:
    """
    Validates that generated Python code produces equivalent output to KNIME.
    
    Supports multiple comparison strategies:
    - Exact match for categorical data
    - Tolerance-based comparison for numeric data
    - Hash comparison for large datasets
    - Structure comparison for complex types
    """
    
    DEFAULT_NUMERIC_TOLERANCE = 1e-5
    DEFAULT_DATETIME_TOLERANCE_SECONDS = 1
    
    def __init__(
        self,
        knime_path: Optional[Path] = None,
        numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
    ):
        """
        Initialize validator.
        
        Args:
            knime_path: Path to KNIME executable (for batch execution)
            numeric_tolerance: Tolerance for numeric comparisons
        """
        self.knime_path = knime_path
        self.numeric_tolerance = numeric_tolerance
    
    def compare_dataframes(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        *,
        check_column_order: bool = False,
        check_row_order: bool = False,
        numeric_tolerance: Optional[float] = None,
    ) -> ComparisonResult:
        """
        Compare two DataFrames for equivalence.
        
        Args:
            df1: First DataFrame (KNIME output)
            df2: Second DataFrame (Python output)
            check_column_order: Whether column order must match
            check_row_order: Whether row order must match
            numeric_tolerance: Override default numeric tolerance
            
        Returns:
            ComparisonResult with detailed differences
        """
        tolerance = numeric_tolerance or self.numeric_tolerance
        differences = []
        
        # Check shapes
        if df1.shape != df2.shape:
            differences.append(
                f"Shape mismatch: {df1.shape} vs {df2.shape}"
            )
            return ComparisonResult(
                is_equal=False,
                differences=differences,
                tolerance_used=tolerance,
            )
        
        # Check column names (sorted if order doesn't matter)
        cols1 = list(df1.columns)
        cols2 = list(df2.columns)
        
        if not check_column_order:
            cols1 = sorted(cols1)
            cols2 = sorted(cols2)
        
        if cols1 != cols2:
            differences.append(
                f"Column mismatch: {set(df1.columns) - set(df2.columns)} vs {set(df2.columns) - set(df1.columns)}"
            )
            return ComparisonResult(
                is_equal=False,
                differences=differences,
                tolerance_used=tolerance,
            )
        
        # Align columns
        if not check_column_order:
            df2 = df2[df1.columns]
        
        # Sort rows if order doesn't matter
        if not check_row_order:
            df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
            df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
        
        # Compare column by column
        for col in df1.columns:
            col_diffs = self._compare_columns(df1[col], df2[col], tolerance)
            if col_diffs:
                differences.extend([f"Column '{col}': {d}" for d in col_diffs])
        
        return ComparisonResult(
            is_equal=len(differences) == 0,
            differences=differences,
            tolerance_used=tolerance,
            rows_compared=len(df1),
            columns_compared=len(df1.columns),
        )
    
    def _compare_columns(
        self,
        col1: pd.Series,
        col2: pd.Series,
        tolerance: float,
    ) -> List[str]:
        """Compare two columns with appropriate strategy."""
        differences = []
        
        # Check dtypes compatibility
        dtype1 = col1.dtype
        dtype2 = col2.dtype
        
        # Numeric comparison
        if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
            # Use numpy allclose for numeric comparison
            if not np.allclose(
                col1.fillna(0).values,
                col2.fillna(0).values,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            ):
                # Find specific differences
                mask = ~np.isclose(
                    col1.fillna(0).values,
                    col2.fillna(0).values,
                    rtol=tolerance,
                    atol=tolerance,
                    equal_nan=True,
                )
                diff_count = mask.sum()
                if diff_count > 0:
                    differences.append(f"{diff_count} values differ beyond tolerance")
        
        # Datetime comparison
        elif pd.api.types.is_datetime64_any_dtype(dtype1) or pd.api.types.is_datetime64_any_dtype(dtype2):
            try:
                dt1 = pd.to_datetime(col1)
                dt2 = pd.to_datetime(col2)
                
                # Compare with 1 second tolerance
                diff = (dt1 - dt2).abs()
                max_diff = pd.Timedelta(seconds=self.DEFAULT_DATETIME_TOLERANCE_SECONDS)
                
                if (diff > max_diff).any():
                    diff_count = (diff > max_diff).sum()
                    differences.append(f"{diff_count} datetime values differ")
            except Exception:
                # Fall back to string comparison
                if not col1.astype(str).equals(col2.astype(str)):
                    differences.append("Values differ (datetime conversion failed)")
        
        # String/categorical comparison
        else:
            # Exact match after string conversion
            str1 = col1.fillna("").astype(str)
            str2 = col2.fillna("").astype(str)
            
            if not str1.equals(str2):
                mask = str1 != str2
                diff_count = mask.sum()
                if diff_count > 0:
                    differences.append(f"{diff_count} values differ")
        
        return differences
    
    def compute_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of a DataFrame for quick comparison."""
        # Sort to ensure consistent hashing
        df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        
        # Convert to bytes and hash
        data = df_sorted.to_csv(index=False).encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def run_python_script(
        self,
        script_path: Path,
        input_data: Optional[Dict[str, pd.DataFrame]] = None,
        timeout: int = 300,
    ) -> Tuple[bool, Dict[str, pd.DataFrame], str]:
        """
        Execute a Python script and capture its outputs.
        
        Args:
            script_path: Path to the Python script
            input_data: Optional input DataFrames to provide
            timeout: Execution timeout in seconds
            
        Returns:
            Tuple of (success, outputs_dict, error_message)
        """
        # Save input data to temp files if provided
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            if input_data:
                for name, df in input_data.items():
                    input_path = temp_dir / f"input_{name}.parquet"
                    df.to_parquet(input_path)
            
            # Run script
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir,
            )
            
            if result.returncode != 0:
                return False, {}, f"Script failed: {result.stderr}"
            
            # Load output files
            outputs = {}
            for output_file in temp_dir.glob("output_*.parquet"):
                name = output_file.stem.replace("output_", "")
                outputs[name] = pd.read_parquet(output_file)
            
            return True, outputs, ""
            
        except subprocess.TimeoutExpired:
            return False, {}, f"Script timed out after {timeout}s"
        except Exception as e:
            return False, {}, str(e)
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def run_knime_batch(
        self,
        workflow_path: Path,
        input_data: Optional[Dict[str, pd.DataFrame]] = None,
        timeout: int = 600,
    ) -> Tuple[bool, Dict[str, pd.DataFrame], str]:
        """
        Execute a KNIME workflow in batch mode.
        
        Requires KNIME to be installed and accessible.
        
        Args:
            workflow_path: Path to the .knwf file
            input_data: Optional input DataFrames
            timeout: Execution timeout
            
        Returns:
            Tuple of (success, outputs_dict, error_message)
        """
        if not self.knime_path:
            return False, {}, "KNIME path not configured"
        
        if not self.knime_path.exists():
            return False, {}, f"KNIME not found at {self.knime_path}"
        
        # This is a placeholder - actual KNIME batch execution would require
        # configuring workflow inputs/outputs and parsing KNIME output files
        logger.warning("KNIME batch execution not fully implemented")
        return False, {}, "KNIME batch execution not implemented"
    
    def validate(
        self,
        knime_outputs: Dict[str, pd.DataFrame],
        python_outputs: Dict[str, pd.DataFrame],
    ) -> ValidationReport:
        """
        Validate Python outputs against KNIME outputs.
        
        Args:
            knime_outputs: Dictionary of node_id -> DataFrame from KNIME
            python_outputs: Dictionary of node_id -> DataFrame from Python
            
        Returns:
            ValidationReport with detailed results
        """
        results = []
        failed_count = 0
        
        # Compare each output
        all_nodes = set(knime_outputs.keys()) | set(python_outputs.keys())
        
        for node_id in all_nodes:
            if node_id not in knime_outputs:
                results.append(ValidationResult(
                    is_valid=False,
                    node_id=node_id,
                    discrepancies=["Missing from KNIME outputs"],
                ))
                failed_count += 1
                continue
            
            if node_id not in python_outputs:
                results.append(ValidationResult(
                    is_valid=False,
                    node_id=node_id,
                    discrepancies=["Missing from Python outputs"],
                ))
                failed_count += 1
                continue
            
            # Compare DataFrames
            comparison = self.compare_dataframes(
                knime_outputs[node_id],
                python_outputs[node_id],
            )
            
            results.append(ValidationResult(
                is_valid=comparison.is_equal,
                node_id=node_id,
                knime_output_hash=self.compute_hash(knime_outputs[node_id]),
                python_output_hash=self.compute_hash(python_outputs[node_id]),
                discrepancies=comparison.differences,
                tolerance_used=comparison.tolerance_used,
            ))
            
            if not comparison.is_equal:
                failed_count += 1
        
        # Generate summary
        total = len(all_nodes)
        validated = len(results)
        passed = validated - failed_count
        
        summary = f"Validated {validated}/{total} nodes: {passed} passed, {failed_count} failed"
        
        return ValidationReport(
            is_valid=failed_count == 0,
            total_nodes=total,
            validated_nodes=validated,
            failed_nodes=failed_count,
            results=results,
            summary=summary,
        )
    
    def generate_diff_report(
        self,
        report: ValidationReport,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate a human-readable diff report."""
        lines = []
        lines.append("=" * 60)
        lines.append("VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Status: {'✅ PASSED' if report.is_valid else '❌ FAILED'}")
        lines.append(f"Summary: {report.summary}")
        lines.append("")
        
        if report.failed_nodes > 0:
            lines.append("-" * 60)
            lines.append("FAILED NODES:")
            lines.append("-" * 60)
            
            for result in report.results:
                if not result.is_valid:
                    lines.append(f"\n📍 Node: {result.node_id}")
                    for diff in result.discrepancies:
                        lines.append(f"   - {diff}")
        
        lines.append("")
        lines.append("=" * 60)
        
        report_text = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report_text, encoding="utf-8")
        
        return report_text
