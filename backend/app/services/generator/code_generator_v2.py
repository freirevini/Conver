"""
Code Generator V2 - Advanced Python code generation from KNIME workflow IR.

Features:
- Jinja2 template engine integration
- AST-based validation
- Type hints generation
- Docstrings with KNIME provenance
- Parallel execution with ThreadPoolExecutor
- Loop and switch handling
"""
from __future__ import annotations

import ast
import logging
import re
from datetime import datetime
from pathlib import Path
from textwrap import dedent, indent
from typing import Any, Dict, List, Optional, Set

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from app.models.ir_models import (
    ExecutionLayer,
    FallbackLevel,
    GeneratedCode,
    LoopStructure,
    NodeInstance,
    ParallelGroup,
    SwitchStructure,
    WorkflowIR,
)
from app.services.catalog.node_registry import get_node_registry
from app.services.generator.template_mapper import TemplateMapper
from app.services.generator.extended_templates import (
    is_expression_node,
    generate_expression_code
)
from app.services.generator.gemini_client import GeminiClient
from app.services.generator.llm_quality_gate import LLMQualityGate
from app.services.validator.strategic_validator import StrategicValidator

logger = logging.getLogger(__name__)


class CodeGeneratorV2:
    """
    Advanced Python code generator for KNIME workflows.
    
    Uses Jinja2 templates for node-specific code and generates
    a complete, executable Python script with proper structure,
    logging, error handling, and parallelization support.
    """
    
    # Template directory location
    TEMPLATE_DIR = Path(__file__).parent.parent / "catalog" / "node_templates"
    
    def __init__(
        self,
        template_dir: Optional[Path] = None,
        use_llm_fallback: bool = True,
        enable_validation: bool = True,
    ):
        """
        Initialize the code generator.
        
        Args:
            template_dir: Custom template directory path
            use_llm_fallback: Whether to use LLM for unsupported nodes (default: True)
            enable_validation: Whether to run strategic validation (default: True)
        """
        self.template_dir = template_dir or self.TEMPLATE_DIR
        self.registry = get_node_registry()
        
        # LLM fallback client
        self.use_llm_fallback = use_llm_fallback
        self.llm_client = GeminiClient() if use_llm_fallback else None
        
        # LLM Quality Gate for syntax validation (P3.1)
        self.llm_quality_gate = LLMQualityGate(self.llm_client) if use_llm_fallback else None
        
        # Strategic validator (non-blocking)
        self.enable_validation = enable_validation
        self.strategic_validator = StrategicValidator() if enable_validation else None
        self.last_validation_report = None
        
        # Phase 1-4: Catalog-driven template mapper
        self.template_mapper = TemplateMapper()
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Add custom filters
        self.jinja_env.filters["tojson"] = self._to_json_filter
    
    def _to_json_filter(self, value: Any) -> str:
        """Jinja2 filter to convert Python objects to JSON."""
        import json
        return json.dumps(value, default=str)
    
    def generate_from_ir(
        self,
        ir: WorkflowIR,
        workflow_path: Optional[Path] = None,
    ) -> GeneratedCode:
        """
        Generate complete Python code from workflow IR.
        
        Args:
            ir: WorkflowIR containing the workflow structure
            workflow_path: Path to workflow directory for schema extraction
            
        Returns:
            GeneratedCode with complete script and metadata
        """
        # Run strategic validation (non-blocking)
        if self.strategic_validator:
            try:
                unmapped_ids = self._collect_unmapped_node_ids(ir.nodes)
                self.last_validation_report = self.strategic_validator.validate_workflow(
                    ir=ir,
                    workflow_path=workflow_path,
                    unmapped_nodes=unmapped_ids,
                )
                logger.info(
                    f"Strategic validation: {self.last_validation_report.passed}/"
                    f"{self.last_validation_report.validated} passed"
                )
            except Exception as e:
                logger.warning(f"Strategic validation failed (non-blocking): {e}")
        
        # Generate imports
        imports = self.generate_imports(ir.nodes)
        
        # Generate function for each node
        functions = {}
        for node in ir.nodes:
            func_code = self.generate_function(node)
            functions[node.node_id] = func_code
        
        # Generate main execution block
        main_block = self.generate_main(ir)
        
        # Combine into complete script
        script = self._assemble_script(ir, imports, functions, main_block)
        
        # Validate syntax
        is_valid, errors = self._validate_syntax(script)
        if not is_valid:
            logger.warning(f"Generated code has syntax issues: {errors}")
        
        return GeneratedCode(
            script=script,
            imports=imports,
            functions=functions,
            main_block=main_block,
            requirements=ir.environment.dependencies,
            source_ir=ir,
            generation_time=datetime.now(),
        )
    
    def generate_imports(self, nodes: List[NodeInstance]) -> List[str]:
        """Generate import statements based on nodes used."""
        imports: Set[str] = set()
        
        # Standard imports always included
        imports.add("from __future__ import annotations")
        imports.add("import os")
        imports.add("import logging")
        imports.add("from typing import Any, Dict, List, Optional, Tuple")
        imports.add("from concurrent.futures import ThreadPoolExecutor, as_completed")
        
        # Add imports based on nodes
        for node in nodes:
            registry_node = self.registry.get_node(node.factory_class)
            if registry_node and registry_node.mapping.imports:
                imports.update(registry_node.mapping.imports)
            
            # Also try by name
            if not registry_node:
                registry_node = self.registry.get_node(node.name)
                if registry_node and registry_node.mapping.imports:
                    imports.update(registry_node.mapping.imports)
        
        # Ensure pandas and numpy are always available
        imports.add("import pandas as pd")
        imports.add("import numpy as np")
        
        # Sort imports
        sorted_imports = sorted(imports)
        
        return sorted_imports
    
    def _collect_unmapped_node_ids(self, nodes: List[NodeInstance]) -> List[str]:
        """Collect IDs of nodes without registered handlers."""
        unmapped = []
        for node in nodes:
            registry_node = self.registry.get_node(node.factory_class)
            if not registry_node:
                registry_node = self.registry.get_node(node.name)
            
            if not registry_node or not registry_node.mapping.template:
                unmapped.append(node.node_id)
        
        return unmapped
    
    def get_validation_report_text(self) -> Optional[str]:
        """Get the last validation report as formatted text."""
        if self.last_validation_report and self.strategic_validator:
            return self.strategic_validator.generate_report_text(self.last_validation_report)
        return None
    
    def save_validation_report(self, output_path: Path) -> Optional[Path]:
        """Save the last validation report to a file."""
        if self.last_validation_report and self.strategic_validator:
            return self.strategic_validator.save_report(self.last_validation_report, output_path)
        return None
    
    def generate_function(self, node: NodeInstance) -> str:
        """
        Generate a Python function for a single node.
        
        Priority order:
        1. Expression parsers (String/Math/Rule/ColumnExpr) - 100% deterministic
        2. Catalog templates (template_mapper) - 100% deterministic
        3. Registry templates (Jinja2) - deterministic
        4. LLM fallback - best effort
        5. Stub - placeholder
        """
        factory_class = node.factory_class
        func_name = self._get_function_name(node)
        
        # === Phase 2: Expression Parsers (highest priority for expression nodes) ===
        if is_expression_node(factory_class):
            expression = node.settings.get("expression", "") or node.settings.get("script", "")
            output_col = node.settings.get("new_column_name", "result")
            
            if expression:
                try:
                    code, imports = generate_expression_code(
                        factory_class, expression, output_col, "df"
                    )
                    if code:
                        logger.info(f"[CATALOG] Expression parser used for: {node.name}")
                        return self._wrap_deterministic_code(node, code, imports)
                except Exception as e:
                    logger.warning(f"Expression parser failed for {node.name}: {e}")
        
        # === Phase 3: Extended Templates (loops, switches, DB connectors) ===
        extended_template = self.template_mapper.get_template(factory_class)
        if extended_template:
            try:
                code = self._apply_template(node, extended_template)
                if code:
                    logger.info(f"[CATALOG] Extended template used for: {node.name}")
                    imports = extended_template.get("imports", [])
                    return self._wrap_deterministic_code(node, code, imports)
            except Exception as e:
                logger.warning(f"Template application failed for {node.name}: {e}")
        
        # === Legacy: Registry Templates (Jinja2) ===
        registry_node = self.registry.get_node(factory_class)
        if not registry_node:
            registry_node = self.registry.get_node(node.name)
        
        if registry_node and registry_node.mapping.template:
            logger.info(f"[REGISTRY] Jinja2 template used for: {node.name}")
            return self._render_template(node, registry_node.mapping.template)
        
        # === Fallback: LLM or Stub ===
        return self._generate_with_fallback(node)
    
    def _wrap_deterministic_code(
        self, node: NodeInstance, code: str, imports: List[str]
    ) -> str:
        """Wrap catalog-generated code in proper function structure."""
        func_name = self._get_function_name(node)
        
        # Prepare code block with proper indentation (4 spaces for function body)
        code_lines = dedent(code).strip()
        indented_code = indent(code_lines, "    ")
        
        # Build function with docstring
        return f'''def {func_name}(df: pd.DataFrame) -> pd.DataFrame:
    """
    {node.name} (Node #{node.node_id})
    
    Factory: {node.factory_class}
    Generated: DETERMINISTIC (catalog)
    """
{indented_code}
    return df'''
    
    def _apply_template(self, node: NodeInstance, template: Dict[str, Any]) -> str:
        """Apply a catalog template to generate code."""
        code_template = template.get("code", "") or template.get("template", "")
        if not code_template:
            return ""
        
        # Replace placeholders with actual values from node settings
        settings = node.settings
        
        # Standard replacements
        replacements = {
            "{output_var}": f"df_{node.node_id}",
            "{input_var}": "df",
            "{df_var}": "df",
        }
        
        # Add settings-based replacements
        for key, value in settings.items():
            placeholder = "{" + key + "}"
            if isinstance(value, str):
                replacements[placeholder] = value
            elif isinstance(value, list):
                replacements[placeholder] = str(value)
            elif isinstance(value, dict):
                replacements[placeholder] = str(value)
            else:
                replacements[placeholder] = str(value) if value is not None else ""
        
        # Apply replacements
        result = code_template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result
    
    def _generate_with_fallback(self, node: NodeInstance) -> str:
        """Try LLM fallback with quality gate validation, then generate stub if unavailable."""
        # Try LLM if enabled and available
        if self.llm_client and self.llm_client.is_available():
            try:
                logger.info(f"Attempting LLM generation for: {node.name}")
                
                # Get dependencies for input variables
                input_vars = [f"input_{i}" for i in range(len(node.input_ports))]
                output_var = f"output_{node.node_id}"
                
                result = self.llm_client.generate_node_code(
                    node_type=node.factory_class,
                    node_config=node.settings,
                    input_vars=input_vars,
                    output_var=output_var,
                )
                
                if result and result.get('code'):
                    llm_code = result.get('code', '')
                    
                    # P3.1: Validate LLM code through quality gate
                    if self.llm_quality_gate:
                        validation_result = self.llm_quality_gate.validate_and_fix(
                            code=llm_code,
                            node_type=node.factory_class,
                            node_config=node.settings,
                            input_vars=input_vars,
                            output_var=output_var,
                        )
                        
                        if validation_result.is_valid:
                            # Use validated/corrected code
                            result['code'] = validation_result.code
                            if validation_result.corrections:
                                logger.info(
                                    f"LLM code corrected after {validation_result.attempts} attempts"
                                )
                            return self._wrap_llm_code(node, result)
                        else:
                            # Quality gate failed - fall through to stub
                            logger.warning(
                                f"LLM quality gate failed for {node.name}: "
                                f"{validation_result.error_message}"
                            )
                    else:
                        # No quality gate - use code directly (legacy behavior)
                        return self._wrap_llm_code(node, result)
                    
            except Exception as e:
                logger.warning(f"LLM fallback failed for {node.name}: {e}")
        
        # Final fallback: generate stub
        return self._generate_stub(node)
    
    def _wrap_llm_code(self, node: NodeInstance, llm_result: Dict[str, Any]) -> str:
        """Wrap LLM-generated code in a proper function structure."""
        func_name = self._get_function_name(node)
        
        code = llm_result.get('code', '')
        explanation = llm_result.get('explanation', 'Generated by LLM')
        
        # Prepare code block with proper indentation
        code_lines = dedent(code).strip()
        indented_code = indent(code_lines, "    ")
        
        return f'''def {func_name}(
    input_0: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    🤖 LLM-GENERATED: {node.name}
    KNIME Node ID: {node.node_id}
    Factory: {node.factory_class}
    
    {explanation}
    
    ⚠️ REVIEW RECOMMENDED:
    This code was generated by AI (gemini-2.5-pro).
    Verify results match original KNIME behavior.
    """
    logger.info(f"Executing LLM-generated node: {node.name}")
    
{indented_code}'''
    
    def _render_template(self, node: NodeInstance, template_path: str) -> str:
        """Render a Jinja2 template for a node."""
        try:
            template = self.jinja_env.get_template(template_path)
            
            # Prepare context
            context = {
                "node_id": node.node_id,
                "node_name": node.name,
                "factory_class": node.factory_class,
                "settings": node.settings,
                "input_ports": node.input_ports,
                "output_ports": node.output_ports,
            }
            
            return template.render(**context)
            
        except TemplateNotFound:
            logger.warning(f"Template not found: {template_path}")
            return self._generate_stub(node)
        except Exception as e:
            logger.error(f"Template rendering failed for {node.name}: {e}")
            return self._generate_stub(node)
    
    def _generate_stub(self, node: NodeInstance) -> str:
        """Generate a stub function for unsupported nodes."""
        func_name = self._get_function_name(node)
        
        # Format settings for docstring
        settings_str = ""
        if node.settings:
            settings_lines = []
            for key, value in list(node.settings.items())[:10]:  # Limit to 10 settings
                settings_lines.append(f"        - {key}: {value}")
            if len(node.settings) > 10:
                settings_lines.append(f"        ... and {len(node.settings) - 10} more settings")
            settings_str = "\n".join(settings_lines)
        else:
            settings_str = "        (no settings captured)"
        
        return f'''def {func_name}(
    input_0: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    ⚠️ UNSUPPORTED NODE: {node.name}
    KNIME Node ID: {node.node_id}
    Factory: {node.factory_class}
    
    TODO: Implement manually. Original settings:
{settings_str}
    
    Suggested approach:
    1. Review KNIME node documentation
    2. Identify equivalent pandas/numpy operations
    3. Implement the transformation logic below
    """
    logger.warning(f"Stub function called: {func_name}")
    
    # TODO: Replace with actual implementation
    raise NotImplementedError(
        f"Node '{node.name}' requires manual implementation. "
        f"See docstring for guidance."
    )'''
    
    def generate_main(self, ir: WorkflowIR) -> str:
        """Generate the main execution block."""
        lines = []
        
        # Add flow variables
        if ir.flow_variables:
            lines.append("# === Flow Variables ===")
            lines.append("FLOW_VARS: Dict[str, Any] = {")
            for fv in ir.flow_variables:
                lines.append(f'    "{fv.name}": {repr(fv.value)},')
            lines.append("}")
            lines.append("")
        
        lines.append("def main() -> None:")
        lines.append('    """Main execution function for the converted workflow."""')
        lines.append("    logger.info('Starting workflow execution...')")
        lines.append("")
        lines.append("    # Node outputs storage")
        lines.append("    outputs: Dict[str, pd.DataFrame] = {}")
        lines.append("")
        
        # Generate execution by layers
        if ir.parallel_groups:
            lines.extend(self._generate_parallel_execution(ir))
        else:
            lines.extend(self._generate_sequential_execution(ir))
        
        lines.append("")
        lines.append("    logger.info('Workflow execution completed!')")
        lines.append("    return outputs")
        
        return "\n".join(lines)
    
    def _generate_sequential_execution(self, ir: WorkflowIR) -> List[str]:
        """Generate sequential execution code."""
        lines = ["    # === Sequential Execution ==="]
        
        for node_id in ir.execution_order:
            node = ir.get_node_by_id(node_id)
            if not node:
                continue
            
            # Get input references
            deps = ir.get_node_dependencies(node_id)
            
            # Generate function call
            func_name = self._get_function_name(node)
            
            if deps:
                input_args = ", ".join([f"outputs['{d}']" for d in deps])
                lines.append(f"    outputs['{node_id}'] = {func_name}({input_args})")
            else:
                lines.append(f"    outputs['{node_id}'] = {func_name}()")
            
            lines.append(f"    logger.info('Completed: {node.name}')")
        
        return lines
    
    def _generate_parallel_execution(self, ir: WorkflowIR) -> List[str]:
        """Generate parallel execution code using ThreadPoolExecutor."""
        lines = ["    # === Parallel Execution with ThreadPoolExecutor ==="]
        lines.append("    with ThreadPoolExecutor(max_workers=4) as executor:")
        lines.append("")
        
        # Group nodes by execution layer
        for layer in ir.execution_layers:
            if layer.can_parallelize and len(layer.node_ids) > 1:
                # Parallel execution
                lines.append(f"        # Layer {layer.layer_index} (parallel)")
                lines.append("        futures = {}")
                
                for node_id in layer.node_ids:
                    node = ir.get_node_by_id(node_id)
                    if not node:
                        continue
                    
                    func_name = self._get_function_name(node)
                    deps = ir.get_node_dependencies(node_id)
                    
                    if deps:
                        input_args = ", ".join([f"outputs['{d}']" for d in deps])
                        lines.append(
                            f"        futures['{node_id}'] = executor.submit({func_name}, {input_args})"
                        )
                    else:
                        lines.append(
                            f"        futures['{node_id}'] = executor.submit({func_name})"
                        )
                
                lines.append("")
                lines.append("        # Wait for all parallel tasks")
                lines.append("        for node_id, future in futures.items():")
                lines.append("            outputs[node_id] = future.result()")
                lines.append(f"        logger.info('Completed layer {layer.layer_index}')")
                lines.append("")
            else:
                # Sequential execution
                lines.append(f"        # Layer {layer.layer_index} (sequential)")
                for node_id in layer.node_ids:
                    node = ir.get_node_by_id(node_id)
                    if not node:
                        continue
                    
                    func_name = self._get_function_name(node)
                    deps = ir.get_node_dependencies(node_id)
                    
                    if deps:
                        input_args = ", ".join([f"outputs['{d}']" for d in deps])
                        lines.append(f"        outputs['{node_id}'] = {func_name}({input_args})")
                    else:
                        lines.append(f"        outputs['{node_id}'] = {func_name}()")
                
                lines.append("")
        
        return lines
    
    def _get_function_name(self, node: NodeInstance) -> str:
        """Get the function name for a node."""
        # Sanitize node_id: remove #, spaces, and other invalid chars
        clean_id = re.sub(r'[^a-zA-Z0-9]', '', str(node.node_id))
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', node.name.lower())
        return f"node_{clean_id}_{clean_name}"
    
    def _assemble_script(
        self,
        ir: WorkflowIR,
        imports: List[str],
        functions: Dict[str, str],
        main_block: str,
    ) -> str:
        """Assemble all parts into a complete Python script."""
        parts = []
        
        # Shebang and docstring
        parts.append('#!/usr/bin/env python3')
        parts.append('"""')
        parts.append(f'KNIME Workflow: {ir.metadata.name}')
        parts.append(f'Generated: {datetime.now().isoformat()}')
        if ir.metadata.source_file:
            parts.append(f'Source: {ir.metadata.source_file}')
        if ir.metadata.knime_version:
            parts.append(f'KNIME Version: {ir.metadata.knime_version}')
        parts.append('')
        parts.append('This script was automatically generated by the KNIME to Python Transpiler.')
        if ir.unsupported_nodes:
            parts.append('')
            parts.append('⚠️ The following nodes require manual implementation:')
            for node in ir.unsupported_nodes[:10]:
                parts.append(f'  - {node}')
            if len(ir.unsupported_nodes) > 10:
                parts.append(f'  ... and {len(ir.unsupported_nodes) - 10} more')
        parts.append('"""')
        parts.append('')
        
        # Imports
        parts.append('# === Imports ===')
        parts.extend(imports)
        parts.append('')
        
        # Logger setup
        parts.append('# === Logging Setup ===')
        parts.append('logging.basicConfig(')
        parts.append('    level=logging.INFO,')
        parts.append('    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"')
        parts.append(')')
        parts.append('logger = logging.getLogger(__name__)')
        parts.append('')
        
        # Node functions
        parts.append('# === Node Functions ===')
        for node_id, func_code in functions.items():
            parts.append('')
            parts.append(func_code)
        parts.append('')
        
        # Main block
        parts.append('')
        parts.append('# === Main Execution ===')
        parts.append(main_block)
        parts.append('')
        
        # Entry point
        parts.append('')
        parts.append('if __name__ == "__main__":')
        parts.append('    main()')
        parts.append('')
        
        return '\n'.join(parts)
    
    def _validate_syntax(self, code: str) -> tuple[bool, List[str]]:
        """Validate Python syntax using AST."""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            return False, [error_msg]
    
    def save(
        self,
        generated: GeneratedCode,
        output_path: Path | str,
    ) -> Path:
        """Save generated code to file."""
        output_path = Path(output_path)
        output_path.write_text(generated.script, encoding="utf-8")
        logger.info(f"Saved generated code to {output_path}")
        return output_path
