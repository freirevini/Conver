"""
Fallback Handler - Handles unsupported KNIME nodes with multiple fallback strategies.

Levels:
1. Template Exact - 100% functional template
2. Template Approximate - Similar template with adaptations
3. LLM Generated - AI-generated code
4. Stub - TODO stub for manual implementation
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from textwrap import dedent, indent
from typing import Any, Dict, List, Optional

from app.models.ir_models import FallbackLevel, NodeInstance
from app.services.catalog.node_registry import get_node_registry

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """Result of fallback code generation."""
    code: str
    level: FallbackLevel
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 - 1.0


class FallbackHandler:
    """
    Handles code generation for nodes without exact templates.
    
    Provides multiple fallback strategies to maximize coverage
    while clearly indicating when manual intervention is needed.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize fallback handler.
        
        Args:
            llm_client: Optional LLM client for AI-generated fallbacks
        """
        self.registry = get_node_registry()
        self.llm_client = llm_client
        
        # Mapping of node patterns to approximate templates
        self.approximate_mappings = self._build_approximate_mappings()
    
    def _build_approximate_mappings(self) -> Dict[str, str]:
        """Build mappings for approximate template matching."""
        return {
            # Reader patterns -> base CSV reader
            r".*reader.*": "io/csv_reader.py.j2",
            r".*input.*": "io/csv_reader.py.j2",
            
            # Writer patterns -> base CSV writer
            r".*writer.*": "io/csv_writer.py.j2",
            r".*output.*": "io/csv_writer.py.j2",
            
            # Filter patterns
            r".*filter.*": "filter/row_filter.py.j2",
            r".*selector.*": "filter/row_filter.py.j2",
            
            # Join patterns
            r".*join.*": "joining/joiner.py.j2",
            r".*merge.*": "joining/joiner.py.j2",
            
            # Aggregation patterns
            r".*group.*": "aggregation/groupby.py.j2",
            r".*aggregate.*": "aggregation/groupby.py.j2",
            
            # Transform patterns
            r".*column.*rename.*": "transform/column_rename.py.j2",
            r".*column.*filter.*": "transform/column_filter.py.j2",
            r".*formula.*": "transform/math_formula.py.j2",
            r".*rule.*": "transform/rule_engine.py.j2",
        }
    
    def get_fallback(self, node: NodeInstance) -> FallbackResult:
        """
        Get the best available fallback for a node.
        
        Args:
            node: The unsupported node instance
            
        Returns:
            FallbackResult with generated code and metadata
        """
        # Try exact template first
        registry_node = self.registry.get_node(node.factory_class)
        if not registry_node:
            registry_node = self.registry.get_node(node.name)
        
        if registry_node and registry_node.mapping.template:
            return FallbackResult(
                code="",  # Template will be used
                level=FallbackLevel.TEMPLATE_EXACT,
                confidence=1.0,
            )
        
        # Try approximate template matching
        approx_template = self._find_approximate_template(node)
        if approx_template:
            code = self._generate_approximate_code(node, approx_template)
            return FallbackResult(
                code=code,
                level=FallbackLevel.TEMPLATE_APPROXIMATE,
                warnings=[
                    f"Using approximate template: {approx_template}",
                    "Review and adjust the generated code for correctness",
                ],
                suggestions=self._get_suggestions(node),
                confidence=0.6,
            )
        
        # Try LLM generation
        if self.llm_client:
            try:
                llm_result = self._generate_with_llm(node)
                if llm_result:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")
        
        # Final fallback: stub
        code = self._generate_stub(node)
        return FallbackResult(
            code=code,
            level=FallbackLevel.STUB,
            warnings=[
                "No template available for this node",
                "Manual implementation required",
            ],
            suggestions=self._get_suggestions(node),
            confidence=0.0,
        )
    
    def _find_approximate_template(self, node: NodeInstance) -> Optional[str]:
        """Find an approximate template based on node name patterns."""
        node_name_lower = node.name.lower()
        factory_lower = node.factory_class.lower()
        
        for pattern, template in self.approximate_mappings.items():
            if re.search(pattern, node_name_lower) or re.search(pattern, factory_lower):
                return template
        
        return None
    
    def _generate_approximate_code(
        self, 
        node: NodeInstance, 
        template_name: str
    ) -> str:
        """Generate code using an approximate template with adaptations."""
        func_name = self._clean_name(node)
        
        return dedent(f'''
            def {func_name}(
                input_0: pd.DataFrame,
                **kwargs
            ) -> pd.DataFrame:
                """
                ⚡ APPROXIMATE MATCH: {node.name}
                KNIME Node ID: {node.node_id}
                Factory: {node.factory_class}
                
                Generated using approximate template: {template_name}
                
                ⚠️ REVIEW REQUIRED:
                This code was generated from a similar template and may need adjustments.
                Original node settings are preserved below for reference.
                
                Original settings:
{self._format_settings(node.settings)}
                """
                logger.info(f"Executing approximate node: {node.name}")
                
                # TODO: Adjust this implementation based on actual KNIME node behavior
                df = input_0.copy()
                
                # Placeholder transformation
                # Modify based on the original KNIME node documentation
                
                return df
        ''').strip()
    
    def _generate_stub(self, node: NodeInstance) -> str:
        """Generate a stub function with comprehensive guidance."""
        func_name = self._clean_name(node)
        suggestions = self._get_suggestions(node)
        
        suggestions_text = "\n".join([f"                {i+1}. {s}" for i, s in enumerate(suggestions)])
        
        return dedent(f'''
            def {func_name}(
                input_0: pd.DataFrame,
                **kwargs
            ) -> pd.DataFrame:
                """
                ⚠️ UNSUPPORTED NODE: {node.name}
                KNIME Node ID: {node.node_id}
                Factory: {node.factory_class}
                
                This node type is not yet supported by the automatic transpiler.
                Manual implementation is required.
                
                Original settings:
{self._format_settings(node.settings)}
                
                Suggested approach:
{suggestions_text if suggestions else "                1. Review KNIME node documentation"}
                
                Documentation resources:
                - KNIME Hub: https://hub.knime.com/
                - KNIME Forum: https://forum.knime.com/
                """
                logger.warning(f"Stub function called: {func_name}")
                logger.warning("This node requires manual implementation")
                
                # TODO: Implement the logic for {node.name}
                # The original KNIME node settings are shown in the docstring above
                
                raise NotImplementedError(
                    f"Node '{node.name}' ({node.factory_class}) requires manual implementation. "
                    f"See docstring for original settings and suggestions."
                )
        ''').strip()
    
    def _generate_with_llm(self, node: NodeInstance) -> Optional[FallbackResult]:
        """Generate code using an LLM."""
        if not self.llm_client:
            return None
        
        prompt = self._build_llm_prompt(node)
        
        try:
            response = self.llm_client.generate(prompt)
            
            if response and len(response) > 50:
                return FallbackResult(
                    code=response,
                    level=FallbackLevel.LLM_GENERATED,
                    warnings=[
                        "This code was generated by AI",
                        "Thorough testing is recommended",
                    ],
                    suggestions=[
                        "Verify the generated logic matches KNIME behavior",
                        "Test with sample data from the original workflow",
                    ],
                    confidence=0.4,
                )
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
        
        return None
    
    def _build_llm_prompt(self, node: NodeInstance) -> str:
        """Build a prompt for LLM code generation."""
        settings_str = self._format_settings(node.settings)
        
        return dedent(f'''
            Generate a Python function that replicates the behavior of a KNIME node.

            KNIME Node Information:
            - Name: {node.name}
            - Factory Class: {node.factory_class}
            - Category: {node.category.value if node.category else 'unknown'}

            Node Settings:
            {settings_str}

            Requirements:
            1. Function should accept pd.DataFrame as input_0
            2. Function should return pd.DataFrame
            3. Use only pandas, numpy, and scikit-learn
            4. Include proper logging
            5. Handle edge cases (empty DataFrame, missing columns)
            6. Add descriptive docstring

            Generate ONLY the Python function code, no explanations.
        ''').strip()
    
    def _get_suggestions(self, node: NodeInstance) -> List[str]:
        """Get implementation suggestions based on node type."""
        suggestions = []
        
        name_lower = node.name.lower()
        factory_lower = node.factory_class.lower()
        
        # Database nodes
        if "database" in name_lower or "database" in factory_lower or "db" in name_lower:
            suggestions.extend([
                "Use sqlalchemy for database connections",
                "Use pd.read_sql() for reading and df.to_sql() for writing",
                "Check for required credentials in environment variables",
            ])
        
        # BigQuery nodes
        if "bigquery" in name_lower or "bigquery" in factory_lower:
            suggestions.extend([
                "Install google-cloud-bigquery: pip install google-cloud-bigquery",
                "Use bigquery.Client() for connection",
                "Ensure GOOGLE_APPLICATION_CREDENTIALS is set",
            ])
        
        # ML nodes
        if "learner" in name_lower or "predictor" in name_lower or "classifier" in name_lower:
            suggestions.extend([
                "Use scikit-learn for ML algorithms",
                "Separate model training (Learner) from prediction (Predictor)",
                "Consider using joblib for model serialization",
            ])
        
        # Loop nodes
        if "loop" in name_lower:
            suggestions.extend([
                "Use Python for/while loops",
                "Consider pandas apply() for row-wise operations",
                "Use itertools for complex iterations",
            ])
        
        # Text nodes
        if "string" in name_lower or "text" in name_lower or "regex" in name_lower:
            suggestions.extend([
                "Use pandas str accessor: df['col'].str.method()",
                "Use re module for regex operations",
                "Consider using str.replace() with regex=True",
            ])
        
        # Generic suggestions
        if not suggestions:
            suggestions.extend([
                "Review KNIME node documentation",
                "Search KNIME Hub for usage examples",
                "Check pandas documentation for equivalent operations",
            ])
        
        return suggestions
    
    def _clean_name(self, node: NodeInstance) -> str:
        """Clean node name for use as function name."""
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', node.name.lower())
        clean = re.sub(r'_+', '_', clean)  # Remove multiple underscores
        clean = clean.strip('_')
        return f"{node.node_id}_{clean}"
    
    def _format_settings(self, settings: Dict[str, Any]) -> str:
        """Format settings dictionary for docstring."""
        if not settings:
            return "                (no settings captured)"
        
        lines = []
        for key, value in list(settings.items())[:15]:
            value_str = str(value)[:50]
            if len(str(value)) > 50:
                value_str += "..."
            lines.append(f"                - {key}: {value_str}")
        
        if len(settings) > 15:
            lines.append(f"                ... and {len(settings) - 15} more settings")
        
        return "\n".join(lines)
    
    def get_fallback_summary(
        self, 
        nodes: List[NodeInstance]
    ) -> Dict[str, int]:
        """Get a summary of fallback levels for a list of nodes."""
        summary = {level.value: 0 for level in FallbackLevel}
        
        for node in nodes:
            result = self.get_fallback(node)
            summary[result.level.value] += 1
        
        return summary
