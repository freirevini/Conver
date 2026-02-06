"""
Python Code Generator

Orchestrates the generation of Python code from KNIME workflows.
Combines template-based and LLM-based generation.
"""
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx

from app.services.generator.template_mapper import TemplateMapper
from app.services.generator.gemini_client import GeminiClient
from app.services.parser.topology_builder import TopologyBuilder

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    Generates Python code from parsed KNIME workflow.
    
    Uses a hybrid approach:
    1. Template-based generation for common nodes (80%)
    2. LLM-based generation for unsupported nodes (20%)
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize code generator.
        
        Args:
            use_llm: Whether to use LLM for unsupported nodes
        """
        self.template_mapper = TemplateMapper()
        self.gemini_client = GeminiClient() if use_llm else None
        self.topology_builder = TopologyBuilder()
    
    def generate_python_code(
        self,
        workflow_data: Dict[str, Any],
        dag: nx.DiGraph
    ) -> str:
        """
        Generate complete Python code for a KNIME workflow.
        
        Args:
            workflow_data: Parsed workflow data from WorkflowParser
            dag: Directed acyclic graph from TopologyBuilder
            
        Returns:
            Complete Python code as string
        """
        workflow_name = workflow_data.get('name', 'knime_workflow')
        nodes = workflow_data.get('nodes', [])
        
        logger.info(f"Generating Python code for '{workflow_name}'")
        
        # Get execution order
        execution_order = self.topology_builder.get_execution_order(dag)
        
        # Generate code for each node
        all_imports: Set[str] = set()
        node_codes: List[str] = []
        
        # Track statistics
        stats = {
            'total': len(execution_order),
            'template': 0,
            'llm': 0,
            'fallback': 0
        }
        
        for node_id in execution_order:
            node_data = dag.nodes.get(node_id, {})
            
            code, imports, method = self._generate_node_code(
                node_id=node_id,
                node_data=node_data,
                dag=dag
            )
            
            if code:
                node_codes.append(code)
                all_imports.update(imports)
                
                if method == 'template':
                    stats['template'] += 1
                elif method == 'llm':
                    stats['llm'] += 1
                else:
                    stats['fallback'] += 1
        
        logger.info(
            f"Code generation stats: {stats['template']} template, "
            f"{stats['llm']} LLM, {stats['fallback']} fallback"
        )
        
        # Assemble final code
        return self._assemble_code(
            workflow_name=workflow_name,
            imports=all_imports,
            node_codes=node_codes,
            stats=stats
        )
    
    def _generate_node_code(
        self,
        node_id: int,
        node_data: Dict,
        dag: nx.DiGraph
    ) -> Tuple[str, List[str], str]:
        """
        Generate code for a single node.
        
        Returns:
            Tuple of (code, imports, method_used)
        """
        node_name = node_data.get('name', f'Node_{node_id}')
        factory_class = node_data.get('factory_class', '')
        settings = node_data.get('settings', {})
        config = settings.get('configuration', {})
        
        # Get input/output variables
        input_var = self.topology_builder.get_input_variable_name(dag, node_id)
        output_var = self.topology_builder.get_output_variable_name(node_id)
        
        # Add node comment
        comment = f"# Node {node_id}: {node_name}\n"
        
        # Try template-based generation first
        if factory_class:
            result = self.template_mapper.generate_code(
                factory_class=factory_class,
                input_var=input_var,
                output_var=output_var,
                settings=config
            )
            
            if result:
                code, imports = result
                return comment + code, imports, 'template'
        
        # Try LLM-based generation
        if self.gemini_client and self.gemini_client.is_available():
            try:
                # Get all input variables for multi-input nodes
                predecessors = list(dag.predecessors(node_id))
                input_vars = [f"df_node_{p}" for p in predecessors] if predecessors else ['df_input']
                
                llm_result = self.gemini_client.generate_node_code(
                    node_type=factory_class or settings.get('factory', 'UnknownNode'),
                    node_config=config,
                    input_vars=input_vars,
                    output_var=output_var
                )
                
                code = llm_result.get('code', '')
                imports = llm_result.get('imports', ['import pandas as pd'])
                explanation = llm_result.get('explanation', '')
                
                if code:
                    if explanation:
                        comment += f"# {explanation}\n"
                    return comment + code, imports, 'llm'
                    
            except Exception as e:
                logger.warning(f"LLM generation failed for node {node_id}: {e}")
        
        # Fallback: passthrough
        fallback_code = f"{output_var} = {input_var}.copy()  # TODO: Implement {node_name}"
        return comment + fallback_code, ['import pandas as pd'], 'fallback'
    
    def _assemble_code(
        self,
        workflow_name: str,
        imports: Set[str],
        node_codes: List[str],
        stats: Dict
    ) -> str:
        """
        Assemble final Python file from components.
        
        Args:
            workflow_name: Name of the workflow
            imports: Set of import statements
            node_codes: List of code blocks for each node
            stats: Generation statistics
            
        Returns:
            Complete Python code
        """
        # Clean workflow name for use as filename/title
        clean_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in workflow_name)
        
        # Build header
        header = f'''"""
KNIME to Python Conversion
Workflow: {workflow_name}

Generated automatically by KNIME to Python Converter.
This code replicates the logic of the original KNIME workflow.

Generation Statistics:
- Total nodes: {stats['total']}
- Template-based: {stats['template']}
- LLM-generated: {stats['llm']}
- Fallback (TODO): {stats['fallback']}
"""
'''
        
        # Sort and format imports
        sorted_imports = sorted(imports)
        import_block = '\n'.join(sorted_imports)
        
        # Main function wrapper
        main_start = f'''

def run_workflow(input_data=None):
    """
    Execute the converted KNIME workflow.
    
    Args:
        input_data: Optional input DataFrame. If None, source nodes
                   will load data from their configured sources.
    
    Returns:
        Dictionary of output DataFrames keyed by node ID.
    """
    outputs = {{}}
    
    # Initialize input if provided
    if input_data is not None:
        df_input = input_data
    else:
        df_input = None
    
'''
        
        # Indent node codes for main function
        indented_codes = []
        for code in node_codes:
            indented = '\n'.join('    ' + line for line in code.split('\n'))
            indented_codes.append(indented)
        
        code_block = '\n\n'.join(indented_codes)
        
        # Add output collection
        output_collection = '''
    
    # Collect all node outputs
    import inspect
    for name, value in list(locals().items()):
        if name.startswith('df_node_'):
            node_id = name.replace('df_node_', '')
            outputs[node_id] = value
    
    return outputs
'''
        
        # Main execution block
        main_block = '''

if __name__ == "__main__":
    import sys
    
    # Run workflow
    print("Starting KNIME workflow execution...")
    results = run_workflow()
    
    # Print summary
    print(f"\\nWorkflow completed. Generated {len(results)} outputs:")
    for node_id, df in results.items():
        if hasattr(df, 'shape'):
            print(f"  - Node {node_id}: {df.shape[0]} rows x {df.shape[1]} columns")
        else:
            print(f"  - Node {node_id}: {type(df).__name__}")
'''
        
        # Combine all parts
        full_code = (
            header +
            import_block +
            main_start +
            code_block +
            output_collection +
            main_block
        )
        
        return full_code
    
    def generate_for_node(
        self,
        factory_class: str,
        settings: Dict[str, Any],
        input_var: str = 'df_input',
        output_var: str = 'df_output'
    ) -> Optional[str]:
        """
        Generate code for a single node (utility method).
        
        Useful for testing or interactive code generation.
        """
        result = self.template_mapper.generate_code(
            factory_class=factory_class,
            input_var=input_var,
            output_var=output_var,
            settings=settings
        )
        
        if result:
            code, imports = result
            return '\n'.join(imports) + '\n\n' + code
        
        return None
