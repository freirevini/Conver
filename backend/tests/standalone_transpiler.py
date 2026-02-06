"""
Standalone Complete Transpilation.

Uses pre-analyzed workflow data to generate complete Python code.
Maximizes coverage with templates + LLM fallback.
"""
import sys
import json
import ast
import re
import os
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class StandaloneTranspiler:
    """
    Standalone transpiler using pre-analyzed workflow data.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.stats = {
            'template': 0,
            'llm': 0, 
            'scripting': 0,
            'fallback': 0,
            'pattern': 0,  # Semantic patterns
            'total': 0
        }
        self.imports = set()
        self.code_blocks = []
        self.pattern_detected_nodes = set()  # Nodes handled by patterns
        
    def load_analysis(self, analysis_path: Path) -> Dict[str, Any]:
        """Load pre-analyzed workflow data."""
        with open(analysis_path, encoding='utf-8') as f:
            return json.load(f)
    
    def get_template_mapper(self):
        """Get template mapper instance."""
        from app.services.generator.template_mapper import TemplateMapper
        return TemplateMapper()
    
    def decode_knime_source(self, encoded: str) -> str:
        """Decode KNIME-encoded source code."""
        if not encoded:
            return ""
        decoded = encoded
        decoded = decoded.replace("%%00010", "\n")
        decoded = decoded.replace("%%00009", "\t")
        decoded = decoded.replace("%%00013", "\r")
        decoded = re.sub(r'%%(\d{5})', lambda m: chr(int(m.group(1))), decoded)
        return decoded
    
    def sanitize_var_name(self, name: str) -> str:
        """Convert node name to valid Python variable."""
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        if sanitized and sanitized[0].isdigit():
            sanitized = f'node_{sanitized}'
        return sanitized.lower() or 'df'
    
    def extract_settings(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Extract settings from node data."""
        return node.get('settings', node.get('model', {}))
    
    def generate_node_code(
        self, 
        node: Dict[str, Any], 
        input_var: str,
        output_var: str,
        mapper
    ) -> Tuple[str, List[str]]:
        """
        Generate Python code for a single node.
        
        Returns:
            Tuple of (code, imports)
        """
        factory = node.get('factory', '')
        node_type = node.get('node_type', '')
        node_name = node.get('node_name', 'unknown')
        settings = self.extract_settings(node)
        
        # Try template first
        result = mapper.generate_code(factory, input_var, output_var, settings)
        if result:
            self.stats['template'] += 1
            return result
        
        # Check for Python Script node
        if 'python' in factory.lower() or 'python' in node_name.lower():
            source_code = settings.get('sourceCode', '')
            if source_code:
                decoded = self.decode_knime_source(source_code)
                if decoded.strip():
                    # Adapt variable names
                    adapted = decoded.replace('input_table_1', input_var)
                    adapted = adapted.replace('output_table_1', output_var)
                    
                    # Extract imports
                    imports = []
                    code_lines = []
                    for line in adapted.split('\n'):
                        if line.strip().startswith(('import ', 'from ')):
                            imports.append(line.strip())
                        else:
                            code_lines.append(line)
                    
                    code = f"# Python Script: {node_name}\n" + '\n'.join(code_lines)
                    self.stats['scripting'] += 1
                    return code, imports
        
        # Use LLM for complex nodes
        if self.use_llm and self._should_use_llm(node):
            code, imports = self._generate_with_llm(node, input_var, output_var)
            if code:
                self.stats['llm'] += 1
                return code, imports
        
        # Fallback: passthrough
        self.stats['fallback'] += 1
        code = f"""# {node_type}: {node_name} (FALLBACK - needs manual implementation)
# Factory: {factory[:80]}
{output_var} = {input_var}.copy()
"""
        return code, []
    
    def _should_use_llm(self, node: Dict[str, Any]) -> bool:
        """Determine if LLM should be used for this node."""
        # Use LLM for complex/unknown nodes
        factory = node.get('factory', '').lower()
        skip_patterns = ['loopstart', 'loopend', 'metanode', 'component']
        return not any(p in factory for p in skip_patterns)
    
    def _generate_with_llm(
        self, 
        node: Dict[str, Any],
        input_var: str,
        output_var: str
    ) -> Tuple[Optional[str], List[str]]:
        """Generate code using LLM."""
        try:
            from app.services.generator.llm_generator import LLMGenerator
            
            llm = LLMGenerator()
            code = llm.generate_node_code(
                node_type=node.get('node_type', ''),
                node_name=node.get('node_name', ''),
                factory=node.get('factory', ''),
                settings=self.extract_settings(node),
                input_var=input_var,
                output_var=output_var
            )
            
            if code and code.strip():
                return f"# LLM Generated: {node.get('node_name', 'unknown')}\n{code}", ['import pandas as pd']
        except Exception as e:
            logger.warning(f"LLM generation failed for {node.get('node_name')}: {e}")
        
        return None, []
    
    def _detect_semantic_patterns(self, analysis: Dict[str, Any]):
        """
        Detect semantic patterns in metanodes before node-by-node processing.
        
        This allows complex node groups (Date Generators, Loops) to be
        transpiled as single semantic units.
        """
        try:
            from app.services.pattern import get_pattern_registry
            
            registry = get_pattern_registry()
            nodes = analysis.get('nodes', [])
            
            # Group nodes by potential metanode context
            metanodes = [n for n in nodes if n.get('node_is_meta', False)]
            
            for metanode in metanodes:
                metanode_name = metanode.get('node_name', 'Unknown')
                internal_nodes = metanode.get('internal_nodes', [])
                internal_connections = metanode.get('internal_connections', [])
                
                if not internal_nodes or len(internal_nodes) < 2:
                    continue
                
                metanode_info = {
                    'name': metanode_name,
                    'path': metanode.get('path', ''),
                    'output_type': metanode.get('output_type')
                }
                
                detected = registry.detect_all(internal_nodes, internal_connections, metanode_info)
                
                if detected:
                    pattern = detected[0]
                    logger.info(f"Pattern detected: {pattern.pattern_name} in '{metanode_name}'")
                    
                    # Generate pattern code
                    input_var = f"df_{self.sanitize_var_name(metanode_name)}_in"
                    output_var = f"df_{self.sanitize_var_name(metanode_name)}_out"
                    
                    result = registry.generate_code(pattern, input_var, output_var)
                    if result:
                        code, imports = result
                        self.imports.update(imports)
                        self.code_blocks.append({
                            'id': metanode.get('id', 'pattern'),
                            'name': f"[PATTERN] {metanode_name}",
                            'code': code,
                            'source': 'pattern'
                        })
                        
                        # Mark internal nodes as handled
                        for node_id in pattern.node_ids:
                            self.pattern_detected_nodes.add(str(node_id))
                        
                        self.stats['pattern'] += 1
                        logger.info(f"  → Generated semantic code ({len(pattern.node_ids)} nodes → 1 pattern)")
            
            if self.stats['pattern'] > 0:
                logger.info(f"Total patterns detected: {self.stats['pattern']}")
            
        except ImportError:
            logger.debug("Pattern detection not available")
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
    
    def transpile(self, analysis: Dict[str, Any]) -> str:
        """
        Transpile all nodes to Python code.
        
        Args:
            analysis: Pre-analyzed workflow data
            
        Returns:
            Complete Python code
        """
        nodes = analysis.get('nodes', [])
        self.stats['total'] = len(nodes)
        
        logger.info(f"Transpiling {len(nodes)} nodes...")
        
        # Get template mapper
        try:
            mapper = self.get_template_mapper()
        except Exception as e:
            logger.error(f"Failed to load template mapper: {e}")
            return ""
        
        # ========== PHASE 1: Pattern Detection ==========
        self._detect_semantic_patterns(analysis)
        
        # Standard imports
        self.imports.add('import pandas as pd')
        self.imports.add('import numpy as np')
        self.imports.add('from datetime import datetime, timedelta')
        self.imports.add('import os')
        
        # Generate code for each node
        for i, node in enumerate(nodes):
            node_id = node.get('id', str(i))
            node_name = node.get('node_name', f'node_{i}')
            
            input_var = f'df_{self.sanitize_var_name(node_name)}_in'
            output_var = f'df_{self.sanitize_var_name(node_name)}_out'
            
            try:
                code, imports = self.generate_node_code(node, input_var, output_var, mapper)
                self.imports.update(imports)
                self.code_blocks.append({
                    'id': node_id,
                    'name': node_name,
                    'code': code
                })
            except Exception as e:
                logger.warning(f"Failed to generate code for node {node_name}: {e}")
                self.code_blocks.append({
                    'id': node_id,
                    'name': node_name,
                    'code': f"# ERROR generating {node_name}: {str(e)[:50]}\n# Passthrough\ndf = df.copy()"
                })
        
        # Build final Python file
        return self._build_python_file()
    
    def _build_python_file(self) -> str:
        """Build the complete Python file."""
        header = f'''"""
Generated Python Pipeline from KNIME Workflow.

Workflow: Indicador_Calculo_CET_Rodas
Generated: {datetime.now().isoformat()}
Generator: ChatKnime Transpiler

Statistics:
- Total Nodes: {self.stats['total']}
- Templates: {self.stats['template']}
- Scripting: {self.stats['scripting']}
- LLM: {self.stats['llm']}
- Fallback: {self.stats['fallback']}
- Coverage: {(self.stats['template'] + self.stats['scripting'] + self.stats['llm']) / max(self.stats['total'], 1) * 100:.1f}%
"""
'''
        
        # Imports section
        sorted_imports = sorted(self.imports)
        imports_section = '\n'.join(sorted_imports)
        
        # Environment configuration
        env_section = '''
# =============================================================================
# Environment Configuration
# =============================================================================

DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_USER = os.environ.get('DB_USER', 'user')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_NAME = os.environ.get('DB_NAME', 'database')

'''
        
        # Helper functions
        helpers_section = '''
# =============================================================================
# Helper Functions
# =============================================================================

def safe_divide(numerator, denominator, default=0):
    """Safe division avoiding ZeroDivisionError."""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def coalesce(*args):
    """Return first non-null value."""
    for arg in args:
        if arg is not None and not (isinstance(arg, float) and pd.isna(arg)):
            return arg
    return None

'''
        
        # Node code sections
        nodes_section = '# =============================================================================\n'
        nodes_section += '# Node Implementations\n'
        nodes_section += '# =============================================================================\n\n'
        
        for block in self.code_blocks:
            nodes_section += f"# --- Node: {block['name']} (ID: {block['id']}) ---\n"
            nodes_section += block['code']
            nodes_section += '\n\n'
        
        # Main function
        main_section = '''
# =============================================================================
# Main Pipeline Entry Point
# =============================================================================

def run_pipeline(input_data=None):
    """
    Execute the complete data processing pipeline.
    
    Args:
        input_data: Optional input DataFrame (for testing)
        
    Returns:
        Processed DataFrame
    """
    if input_data is None:
        # Load from database or file
        print("Pipeline initialized - waiting for input data")
        return None
    
    df = input_data.copy()
    
    # TODO: Implement node execution order based on DAG
    # For now, this is a placeholder
    
    print(f"Processed {len(df)} rows")
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='KNIME Workflow Pipeline')
    parser.add_argument('--input', '-i', help='Input CSV file')
    parser.add_argument('--output', '-o', help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.input:
        df = pd.read_csv(args.input)
        result = run_pipeline(df)
        
        if args.output and result is not None:
            result.to_csv(args.output, index=False)
            print(f"Output saved to: {args.output}")
    else:
        print("Usage: python generated_pipeline.py --input data.csv --output result.csv")
'''
        
        # Combine all sections
        return header + imports_section + env_section + helpers_section + nodes_section + main_section


def main():
    """Run standalone transpilation."""
    # Paths
    analysis_path = Path(__file__).parent.parent.parent / 'workflow_analysis_v2.json'
    output_path = Path(__file__).parent.parent.parent / 'generated_pipeline.py'
    report_path = Path(__file__).parent.parent.parent / 'transpilation_report.md'
    
    if not analysis_path.exists():
        logger.error(f"Analysis file not found: {analysis_path}")
        sys.exit(1)
    
    # Run transpilation
    transpiler = StandaloneTranspiler(use_llm=True)  # Enable LLM fallback
    
    logger.info(f"Loading analysis from: {analysis_path}")
    analysis = transpiler.load_analysis(analysis_path)
    
    logger.info("Starting transpilation...")
    python_code = transpiler.transpile(analysis)
    
    # Validate syntax
    logger.info("Validating syntax...")
    syntax_valid = False
    syntax_errors = []
    try:
        ast.parse(python_code)
        syntax_valid = True
        logger.info("Syntax: VALID")
    except SyntaxError as e:
        syntax_errors.append(f"Line {e.lineno}: {e.msg}")
        logger.error(f"Syntax error at line {e.lineno}: {e.msg}")
    
    # Save Python file
    logger.info(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(python_code)
    
    # Generate report
    stats = transpiler.stats
    coverage = (stats['template'] + stats['scripting'] + stats['llm']) / max(stats['total'], 1) * 100
    
    report = f"""# Relatório de Transpilação Completa

## Sumário

| Métrica | Valor |
|---------|-------|
| **Status** | {'✅ SUCESSO' if syntax_valid else '⚠️ PARCIAL'} |
| **Total de Nós** | {stats['total']} |
| **Cobertura** | {coverage:.1f}% |

---

## Estatísticas por Método

| Método | Quantidade | % |
|--------|------------|---|
| Template | {stats['template']} | {stats['template']/max(stats['total'],1)*100:.1f}% |
| Scripting | {stats['scripting']} | {stats['scripting']/max(stats['total'],1)*100:.1f}% |
| LLM | {stats['llm']} | {stats['llm']/max(stats['total'],1)*100:.1f}% |
| Fallback | {stats['fallback']} | {stats['fallback']/max(stats['total'],1)*100:.1f}% |

---

## Validação de Sintaxe

**Status:** {'✅ VÁLIDO' if syntax_valid else '❌ INVÁLIDO'}

"""
    if syntax_errors:
        report += "### Erros\n\n"
        for err in syntax_errors:
            report += f"- `{err}`\n"
    
    report += f"""
---

## Arquivos Gerados

| Arquivo | Caminho |
|---------|---------|
| Python | `{output_path}` |
| Relatório | `{report_path}` |

---

## Preview do Código

```python
{python_code[:2000]}...
```

---

*Gerado em {datetime.now().isoformat()}*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"Status: {'SUCCESS' if syntax_valid else 'PARTIAL'}")
    print(f"Coverage: {coverage:.1f}%")
    print(f"  Templates: {stats['template']}")
    print(f"  Scripting: {stats['scripting']}")
    print(f"  LLM: {stats['llm']}")
    print(f"  Fallback: {stats['fallback']}")
    print(f"Python file: {output_path}")
    print(f"Report: {report_path}")
    print("="*60)


if __name__ == '__main__':
    main()
