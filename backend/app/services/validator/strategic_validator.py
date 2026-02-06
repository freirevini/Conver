"""
Strategic Validator - Main orchestrator for schema validation.

Integrates node classification, schema extraction, and report generation
to validate strategic nodes in KNIME workflows during transpilation.

Non-blocking: Generates reports without preventing code generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.models.ir_models import NodeInstance, WorkflowIR
from app.services.validator.node_classifier import (
    StrategicNodeClassifier,
    ClassificationResult,
    NodeCategory,
)
from app.services.validator.schema_extractor import (
    SchemaExtractor,
    NodeOutputSchema,
)
from app.services.validator.validation_reporter import (
    ValidationReporter,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
    UnmappedNode,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategicValidationContext:
    """Context for strategic validation run."""
    workflow_name: str
    workflow_path: Optional[Path]
    knwf_path: Optional[Path]
    total_nodes: int
    unmapped_node_ids: List[str]


class StrategicValidator:
    """
    Main orchestrator for strategic schema validation.
    
    This validator is NON-BLOCKING: it generates reports for review
    but does not prevent code generation even if discrepancies are found.
    
    Usage:
        validator = StrategicValidator()
        report = validator.validate_workflow(ir, workflow_path)
        validator.save_report(report, output_path)
    """
    
    def __init__(self):
        """Initialize validator components."""
        self.classifier = StrategicNodeClassifier()
        self.extractor = SchemaExtractor()
        self.reporter = ValidationReporter()
    
    def validate_workflow(
        self,
        ir: WorkflowIR,
        workflow_path: Optional[Path] = None,
        knwf_path: Optional[Path] = None,
        unmapped_nodes: Optional[List[str]] = None,
    ) -> ValidationReport:
        """
        Validate strategic nodes in a workflow.
        
        Args:
            ir: Workflow Intermediate Representation
            workflow_path: Path to extracted workflow directory
            knwf_path: Path to original .knwf file
            unmapped_nodes: List of node IDs without handlers
            
        Returns:
            ValidationReport with all results
        """
        context = StrategicValidationContext(
            workflow_name=ir.metadata.get("name", "Unknown Workflow"),
            workflow_path=workflow_path,
            knwf_path=knwf_path,
            total_nodes=len(ir.nodes),
            unmapped_node_ids=unmapped_nodes or [],
        )
        
        logger.info(f"Starting strategic validation for {context.workflow_name}")
        
        # Step 1: Classify all nodes
        classifications = self._classify_nodes(ir.nodes)
        strategic = [c for c in classifications if c.is_strategic]
        
        logger.info(f"Found {len(strategic)} strategic nodes out of {len(ir.nodes)} total")
        
        # Step 2: Extract schemas if workflow path available
        schemas: Dict[str, List[NodeOutputSchema]] = {}
        if workflow_path and workflow_path.exists():
            schemas = self.extractor.extract_all_schemas(workflow_path)
            logger.info(f"Extracted schemas for {len(schemas)} nodes")
        elif knwf_path and knwf_path.exists():
            schemas = self.extractor.extract_from_knwf(knwf_path)
            logger.info(f"Extracted schemas from KNWF for {len(schemas)} nodes")
        
        # Step 3: Validate strategic nodes
        results = self._validate_strategic_nodes(strategic, schemas)
        
        # Step 4: Collect unmapped nodes info
        unmapped = self._collect_unmapped_nodes(ir.nodes, context.unmapped_node_ids)
        
        # Step 5: Generate report
        report = self.reporter.create_report(
            workflow_name=context.workflow_name,
            total_nodes=context.total_nodes,
            results=results,
            unmapped_nodes=unmapped,
        )
        
        logger.info(
            f"Validation complete: {report.passed}/{report.validated} passed, "
            f"{len(unmapped)} unmapped nodes"
        )
        
        return report
    
    def _classify_nodes(
        self,
        nodes: List[NodeInstance]
    ) -> List[ClassificationResult]:
        """Classify all nodes in the workflow."""
        results = []
        
        for node in nodes:
            # Get Python source if available
            python_source = None
            if node.settings:
                python_source = node.settings.get("sourceCode") or node.settings.get("script")
            
            result = self.classifier.classify(
                node_id=node.node_id,
                node_name=node.name,
                factory_class=node.factory_class,
                python_source=python_source,
            )
            results.append(result)
        
        return results
    
    def _validate_strategic_nodes(
        self,
        classifications: List[ClassificationResult],
        schemas: Dict[str, List[NodeOutputSchema]]
    ) -> List[ValidationResult]:
        """Validate all classified strategic nodes."""
        results = []
        
        for cls in classifications:
            node_schemas = schemas.get(cls.node_id, [])
            
            if not node_schemas:
                # No schema available - mark as no_schema
                result = ValidationResult(
                    node_id=cls.node_id,
                    node_name=cls.node_name,
                    category=cls.category,
                    status=ValidationStatus.NO_SCHEMA,
                    expected_schema=None,
                    errors=["Schema não disponível - workflow não salvo com dados"],
                    suggestions=["Execute o workflow no KNIME e salve com dados"],
                )
            else:
                # Use first output port schema (most common)
                schema = node_schemas[0]
                
                # For now, mark as MATCH since we can't run Python code
                # Actual validation will happen at runtime
                result = ValidationResult(
                    node_id=cls.node_id,
                    node_name=cls.node_name,
                    category=cls.category,
                    status=ValidationStatus.MATCH,
                    expected_schema=schema,
                    actual_columns=schema.column_count,
                    actual_names=schema.column_names,
                )
            
            results.append(result)
        
        return results
    
    def _collect_unmapped_nodes(
        self,
        nodes: List[NodeInstance],
        unmapped_ids: List[str]
    ) -> List[UnmappedNode]:
        """Collect information about unmapped nodes."""
        unmapped = []
        
        for node in nodes:
            if node.node_id in unmapped_ids:
                reason = self._determine_unmapped_reason(node)
                action = self._determine_unmapped_action(node)
                
                unmapped.append(UnmappedNode(
                    node_id=node.node_id,
                    node_name=node.name,
                    factory_class=node.factory_class,
                    reason=reason,
                    action=action,
                ))
        
        return unmapped
    
    def _determine_unmapped_reason(self, node: NodeInstance) -> str:
        """Determine why a node is unmapped."""
        if "MetaNode" in node.factory_class or node.is_meta_node:
            return "MetaNodes requerem expansão recursiva"
        if "Component" in node.factory_class:
            return "Componentes KNIME requerem tratamento especial"
        if "Loop" in node.name or "Loop" in node.factory_class:
            return "Estruturas de loop requerem handler específico"
        if "Switch" in node.name or "Switch" in node.factory_class:
            return "Estruturas condicionais requerem handler específico"
        return "Handler não implementado para este tipo de node"
    
    def _determine_unmapped_action(self, node: NodeInstance) -> str:
        """Determine recommended action for unmapped node."""
        if "MetaNode" in node.factory_class or node.is_meta_node:
            return "LLM fallback será utilizado ou stub gerado"
        if "Python" in node.name:
            return "Código Python será extraído e adaptado"
        return "LLM fallback (gemini-2.5-pro) ou stub NotImplementedError"
    
    def save_report(
        self,
        report: ValidationReport,
        output_path: Path
    ) -> Path:
        """
        Save validation report to file.
        
        Args:
            report: ValidationReport to save
            output_path: Destination path
            
        Returns:
            Path to saved file
        """
        return self.reporter.save_report(output_path, report)
    
    def generate_report_text(self, report: ValidationReport) -> str:
        """Generate report as text string."""
        return self.reporter.generate_txt(report)


# Convenience function for quick validation
def validate_workflow(
    ir: WorkflowIR,
    workflow_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> ValidationReport:
    """
    Quick validation function.
    
    Args:
        ir: Workflow IR
        workflow_path: Path to workflow directory
        output_path: Where to save report (optional)
        
    Returns:
        ValidationReport
    """
    validator = StrategicValidator()
    report = validator.validate_workflow(ir, workflow_path)
    
    if output_path:
        validator.save_report(report, output_path)
    
    return report
