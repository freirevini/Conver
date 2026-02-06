"""
Validation Reporter for strategic schema validation.

Generates human-readable .txt reports with:
- Summary statistics
- Validated nodes (passed)
- Discrepancies found
- Unmapped nodes
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.services.validator.node_classifier import ClassificationResult, NodeCategory
from app.services.validator.schema_extractor import NodeOutputSchema

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of schema validation."""
    MATCH = "match"
    MISMATCH = "mismatch"
    SKIPPED = "skipped"
    NO_SCHEMA = "no_schema"


@dataclass
class ValidationResult:
    """Result of validating a single node."""
    node_id: str
    node_name: str
    category: Optional[NodeCategory]
    status: ValidationStatus
    expected_schema: Optional[NodeOutputSchema]
    actual_columns: Optional[int] = None
    actual_names: Optional[List[str]] = None
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class UnmappedNode:
    """Node without a handler template."""
    node_id: str
    node_name: str
    factory_class: str
    reason: str
    action: str


@dataclass
class ValidationReport:
    """Complete validation report for a workflow."""
    workflow_name: str
    timestamp: datetime
    total_nodes: int
    strategic_nodes: int
    validated: int
    passed: int
    failed: int
    results: List[ValidationResult] = field(default_factory=list)
    unmapped_nodes: List[UnmappedNode] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Percentage of passed validations."""
        if self.validated == 0:
            return 100.0
        return (self.passed / self.validated) * 100


class ValidationReporter:
    """
    Generates validation reports in human-readable TXT format.
    """
    
    SEPARATOR = "=" * 80
    THIN_SEP = "-" * 80
    
    def __init__(self):
        """Initialize reporter."""
        self.report: Optional[ValidationReport] = None
    
    def create_report(
        self,
        workflow_name: str,
        total_nodes: int,
        results: List[ValidationResult],
        unmapped_nodes: Optional[List[UnmappedNode]] = None
    ) -> ValidationReport:
        """
        Create a validation report from results.
        
        Args:
            workflow_name: Name of the workflow
            total_nodes: Total number of nodes in workflow
            results: List of validation results
            unmapped_nodes: List of nodes without handlers
            
        Returns:
            ValidationReport object
        """
        strategic = [r for r in results if r.category is not None]
        passed = [r for r in results if r.status == ValidationStatus.MATCH]
        failed = [r for r in results if r.status == ValidationStatus.MISMATCH]
        
        self.report = ValidationReport(
            workflow_name=workflow_name,
            timestamp=datetime.now(),
            total_nodes=total_nodes,
            strategic_nodes=len(strategic),
            validated=len([r for r in results if r.status != ValidationStatus.SKIPPED]),
            passed=len(passed),
            failed=len(failed),
            results=results,
            unmapped_nodes=unmapped_nodes or [],
        )
        
        return self.report
    
    def generate_txt(self, report: Optional[ValidationReport] = None) -> str:
        """
        Generate TXT report content.
        
        Args:
            report: ValidationReport (uses stored if not provided)
            
        Returns:
            Formatted TXT string
        """
        r = report or self.report
        if not r:
            return "No report available"
        
        lines = []
        
        # Header
        lines.append(self.SEPARATOR)
        lines.append("                    RELATÓRIO DE VALIDAÇÃO - ChatKnime")
        lines.append(self.SEPARATOR)
        lines.append(f"Workflow: {r.workflow_name}")
        lines.append(f"Data: {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(self.THIN_SEP)
        lines.append("")
        
        # Summary
        lines.append("📊 RESUMO")
        lines.append(self.THIN_SEP)
        lines.append(f"Total de nodes no workflow:          {r.total_nodes}")
        lines.append(f"Nodes estratégicos identificados:    {r.strategic_nodes}")
        lines.append(f"Validações realizadas:               {r.validated}")
        lines.append(f"Validações bem-sucedidas:            {r.passed}")
        lines.append(f"Validações com discrepância:         {r.failed}")
        lines.append(f"Taxa de sucesso:                     {r.success_rate:.1f}%")
        lines.append("")
        
        # Validated nodes (passed)
        passed_results = [res for res in r.results if res.status == ValidationStatus.MATCH]
        if passed_results:
            lines.append(self.SEPARATOR)
            lines.append("                         NODES ESTRATÉGICOS VALIDADOS")
            lines.append(self.SEPARATOR)
            lines.append("")
            
            for res in passed_results:
                lines.extend(self._format_passed_node(res))
        
        # Discrepancies
        failed_results = [res for res in r.results if res.status == ValidationStatus.MISMATCH]
        if failed_results:
            lines.append(self.SEPARATOR)
            lines.append("                         DISCREPÂNCIAS ENCONTRADAS")
            lines.append(self.SEPARATOR)
            lines.append("")
            
            for res in failed_results:
                lines.extend(self._format_failed_node(res))
        
        # Unmapped nodes
        if r.unmapped_nodes:
            lines.append(self.SEPARATOR)
            lines.append("                         NODES NÃO MAPEADOS (SEM HANDLER)")
            lines.append(self.SEPARATOR)
            lines.append("")
            
            for node in r.unmapped_nodes:
                lines.extend(self._format_unmapped_node(node))
        
        # Next steps
        lines.append(self.SEPARATOR)
        lines.append("                              PRÓXIMOS PASSOS")
        lines.append(self.SEPARATOR)
        lines.append("")
        
        step = 1
        if failed_results:
            lines.append(f"{step}. Revise as discrepâncias em 'DISCREPÂNCIAS ENCONTRADAS'")
            step += 1
        if r.unmapped_nodes:
            lines.append(f"{step}. Para nodes não mapeados, considere criar handlers específicos")
            step += 1
        lines.append(f"{step}. Execute o script Python gerado e compare os resultados manualmente")
        lines.append("")
        
        # Footer
        lines.append(self.THIN_SEP)
        lines.append("Gerado automaticamente por ChatKnime v1.0")
        lines.append(self.SEPARATOR)
        
        return "\n".join(lines)
    
    def _format_passed_node(self, res: ValidationResult) -> List[str]:
        """Format a passed validation result."""
        category_label = self._get_category_label(res.category)
        lines = [
            f"✅ NODE #{res.node_id} - {res.node_name}",
            f"   Categoria: {category_label}",
            f"   Status: MATCH",
        ]
        
        if res.expected_schema:
            lines.append(f"   Colunas esperadas: {res.expected_schema.column_count}")
            lines.append(f"   Colunas encontradas: {res.actual_columns or res.expected_schema.column_count}")
        
        lines.append(f"   Schema: OK")
        lines.append("")
        
        return lines
    
    def _format_failed_node(self, res: ValidationResult) -> List[str]:
        """Format a failed validation result."""
        category_label = self._get_category_label(res.category)
        lines = [
            f"❌ NODE #{res.node_id} - {res.node_name}",
            f"   Categoria: {category_label}",
            f"   Status: MISMATCH",
            "",
            f"   PROBLEMA DETECTADO:",
        ]
        
        for error in res.errors:
            lines.append(f"   - {error}")
        
        if res.suggestions:
            lines.append("")
            lines.append("   SUGESTÃO:")
            for suggestion in res.suggestions:
                lines.append(f"   {suggestion}")
        
        lines.append("")
        return lines
    
    def _format_unmapped_node(self, node: UnmappedNode) -> List[str]:
        """Format an unmapped node."""
        return [
            f"⚠️ NODE #{node.node_id} - {node.node_name}",
            f"   Tipo: {node.factory_class}",
            f"   Motivo: {node.reason}",
            f"   Ação: {node.action}",
            "",
        ]
    
    def _get_category_label(self, category: Optional[NodeCategory]) -> str:
        """Get human-readable category label."""
        labels = {
            NodeCategory.DB_QUERY: "DB Query",
            NodeCategory.OUTPUT: "Output/Sink",
            NodeCategory.FILE_LOADER: "File Loader",
            NodeCategory.PYTHON_EMBEDDED: "Python SQL/Excel",
        }
        return labels.get(category, "Unknown") if category else "N/A"
    
    def save_report(
        self,
        output_path: Path,
        report: Optional[ValidationReport] = None
    ) -> Path:
        """
        Save report to a .txt file.
        
        Args:
            output_path: Path to save the report
            report: ValidationReport (uses stored if not provided)
            
        Returns:
            Path to saved file
        """
        content = self.generate_txt(report)
        
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".txt")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        
        logger.info(f"Validation report saved to {output_path}")
        return output_path
