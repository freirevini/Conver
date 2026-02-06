"""
Diff Reporter for Validation Results.

Generates human-readable and machine-readable diff reports:
- Console output
- HTML reports
- JSON export
- Markdown format
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .output_comparator import ComparisonResult, Difference, DiffType
from .schema_validator import ValidationResult, ValidationIssue, ValidationLevel

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for diff reports."""
    max_differences: int = 50
    show_values: bool = True
    show_row_samples: int = 5
    include_summary: bool = True
    include_statistics: bool = True


class DiffReporter:
    """
    Generates diff reports from comparison results.
    
    Formats:
    - Console (text)
    - Markdown
    - HTML
    - JSON
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
    
    def to_console(self, result: ComparisonResult) -> str:
        """Generate console-friendly text report."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("DataFrame Comparison Report")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary
        if result.is_equal:
            lines.append("✅ RESULT: DataFrames are EQUAL")
        else:
            lines.append("❌ RESULT: DataFrames are DIFFERENT")
        
        lines.append("")
        lines.append(f"Expected: {result.expected_rows} rows × {len(result.expected_columns)} columns")
        lines.append(f"Actual:   {result.actual_rows} rows × {len(result.actual_columns)} columns")
        lines.append("")
        
        if not result.is_equal:
            # Schema differences
            schema_diffs = result.schema_differences
            if schema_diffs:
                lines.append("-" * 40)
                lines.append("Schema Differences:")
                lines.append("-" * 40)
                for diff in schema_diffs[:self.config.max_differences]:
                    lines.append(f"  • {diff}")
                lines.append("")
            
            # Value differences
            value_diffs = result.value_differences
            if value_diffs:
                lines.append("-" * 40)
                lines.append(f"Value Differences ({len(value_diffs)} total):")
                lines.append("-" * 40)
                for diff in value_diffs[:self.config.max_differences]:
                    lines.append(f"  • {diff}")
                
                if len(value_diffs) > self.config.max_differences:
                    lines.append(f"  ... and {len(value_diffs) - self.config.max_differences} more")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_markdown(self, result: ComparisonResult) -> str:
        """Generate Markdown report."""
        lines = []
        
        # Header
        lines.append("# DataFrame Comparison Report")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        # Result badge
        if result.is_equal:
            lines.append("## ✅ Result: EQUAL")
        else:
            lines.append("## ❌ Result: DIFFERENT")
        
        lines.append("")
        
        # Summary table
        lines.append("### Summary")
        lines.append("")
        lines.append("| Metric | Expected | Actual |")
        lines.append("|--------|----------|--------|")
        lines.append(f"| Rows | {result.expected_rows} | {result.actual_rows} |")
        lines.append(f"| Columns | {len(result.expected_columns)} | {len(result.actual_columns)} |")
        lines.append("")
        
        if not result.is_equal:
            # Differences
            lines.append("### Differences")
            lines.append("")
            lines.append(f"Total: **{result.difference_count}**")
            lines.append("")
            
            # Schema
            schema_diffs = result.schema_differences
            if schema_diffs:
                lines.append("#### Schema Differences")
                lines.append("")
                lines.append("| Type | Column | Details |")
                lines.append("|------|--------|---------|")
                for diff in schema_diffs[:20]:
                    lines.append(f"| {diff.diff_type.value} | {diff.column or '-'} | {diff.message or '-'} |")
                lines.append("")
            
            # Values
            value_diffs = result.value_differences
            if value_diffs:
                lines.append("#### Value Differences")
                lines.append("")
                lines.append("| Column | Row | Expected | Actual |")
                lines.append("|--------|-----|----------|--------|")
                for diff in value_diffs[:30]:
                    exp = str(diff.expected)[:20]
                    act = str(diff.actual)[:20]
                    lines.append(f"| {diff.column} | {diff.row} | `{exp}` | `{act}` |")
                
                if len(value_diffs) > 30:
                    lines.append("")
                    lines.append(f"*... and {len(value_diffs) - 30} more differences*")
        
        return "\n".join(lines)
    
    def to_html(self, result: ComparisonResult) -> str:
        """Generate HTML report."""
        status_class = "success" if result.is_equal else "error"
        status_text = "EQUAL" if result.is_equal else "DIFFERENT"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>DataFrame Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .diff-row {{ background-color: #fff3cd; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>DataFrame Comparison Report</h1>
    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    
    <div class="summary">
        <h2 class="{status_class}">Result: {status_text}</h2>
        <p>Expected: {result.expected_rows} rows × {len(result.expected_columns)} columns</p>
        <p>Actual: {result.actual_rows} rows × {len(result.actual_columns)} columns</p>
        <p>Total differences: {result.difference_count}</p>
    </div>
"""
        
        if not result.is_equal:
            schema_diffs = result.schema_differences
            if schema_diffs:
                html += """
    <h3>Schema Differences</h3>
    <table>
        <tr><th>Type</th><th>Column</th><th>Details</th></tr>
"""
                for diff in schema_diffs[:20]:
                    html += f"        <tr class='diff-row'><td>{diff.diff_type.value}</td><td>{diff.column or '-'}</td><td>{diff.message or '-'}</td></tr>\n"
                html += "    </table>\n"
            
            value_diffs = result.value_differences
            if value_diffs:
                html += """
    <h3>Value Differences</h3>
    <table>
        <tr><th>Column</th><th>Row</th><th>Expected</th><th>Actual</th></tr>
"""
                for diff in value_diffs[:50]:
                    html += f"        <tr class='diff-row'><td>{diff.column}</td><td>{diff.row}</td><td>{diff.expected}</td><td>{diff.actual}</td></tr>\n"
                html += "    </table>\n"
        
        html += """
</body>
</html>"""
        
        return html
    
    def to_json(self, result: ComparisonResult) -> Dict[str, Any]:
        """Generate JSON report."""
        return {
            "is_equal": result.is_equal,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "expected_rows": result.expected_rows,
                "actual_rows": result.actual_rows,
                "expected_columns": result.expected_columns,
                "actual_columns": result.actual_columns,
                "difference_count": result.difference_count
            },
            "differences": [
                {
                    "type": d.diff_type.value,
                    "column": d.column,
                    "row": d.row,
                    "expected": str(d.expected) if d.expected is not None else None,
                    "actual": str(d.actual) if d.actual is not None else None,
                    "message": d.message
                }
                for d in result.differences[:self.config.max_differences]
            ]
        }
    
    def save_report(
        self,
        result: ComparisonResult,
        path: Path,
        format: str = "markdown"
    ) -> Path:
        """Save report to file."""
        if format == "markdown":
            content = self.to_markdown(result)
            path = path.with_suffix(".md")
        elif format == "html":
            content = self.to_html(result)
            path = path.with_suffix(".html")
        elif format == "json":
            import json
            content = json.dumps(self.to_json(result), indent=2)
            path = path.with_suffix(".json")
        else:
            content = self.to_console(result)
            path = path.with_suffix(".txt")
        
        path.write_text(content, encoding="utf-8")
        logger.info(f"Report saved to {path}")
        
        return path


class ValidationReporter:
    """Generates reports from validation results."""
    
    def to_console(self, result: ValidationResult) -> str:
        """Generate console text report."""
        lines = []
        
        lines.append("=" * 50)
        lines.append("Schema Validation Report")
        lines.append("=" * 50)
        lines.append("")
        
        if result.is_valid:
            lines.append("✅ Schema is VALID")
        else:
            lines.append("❌ Schema is INVALID")
        
        lines.append("")
        lines.append(result.summary)
        lines.append("")
        
        if result.errors:
            lines.append("Errors:")
            for issue in result.errors:
                lines.append(f"  ❌ {issue}")
        
        if result.warnings:
            lines.append("Warnings:")
            for issue in result.warnings:
                lines.append(f"  ⚠️ {issue}")
        
        return "\n".join(lines)
    
    def to_markdown(self, result: ValidationResult) -> str:
        """Generate Markdown report."""
        lines = []
        
        lines.append("# Schema Validation Report")
        lines.append("")
        
        if result.is_valid:
            lines.append("## ✅ Result: VALID")
        else:
            lines.append("## ❌ Result: INVALID")
        
        lines.append("")
        lines.append(result.summary)
        lines.append("")
        
        if result.errors:
            lines.append("### Errors")
            for issue in result.errors:
                lines.append(f"- ❌ {issue}")
            lines.append("")
        
        if result.warnings:
            lines.append("### Warnings")
            for issue in result.warnings:
                lines.append(f"- ⚠️ {issue}")
        
        return "\n".join(lines)


# ==================== Public API ====================

def generate_comparison_report(
    result: ComparisonResult,
    format: str = "console",
    config: Optional[ReportConfig] = None
) -> str:
    """Generate comparison report in specified format."""
    reporter = DiffReporter(config)
    
    if format == "markdown":
        return reporter.to_markdown(result)
    elif format == "html":
        return reporter.to_html(result)
    else:
        return reporter.to_console(result)


def generate_validation_report(
    result: ValidationResult,
    format: str = "console"
) -> str:
    """Generate validation report in specified format."""
    reporter = ValidationReporter()
    
    if format == "markdown":
        return reporter.to_markdown(result)
    else:
        return reporter.to_console(result)
