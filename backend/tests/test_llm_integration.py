"""
LLM Integration Test - 50% Template Coverage.

This script tests the LLM fallback by using only 50% of templates
and forcing the remaining nodes to be processed by the LLM.
"""
import sys
import json
import ast
import re
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LLMTestTranspiler:
    """
    Transpiler that limits templates to 50% to force LLM usage.
    """
    
    def __init__(self, template_limit_percent: float = 0.5):
        self.template_limit_percent = template_limit_percent
        self.stats = {
            'template': 0,
            'llm': 0, 
            'scripting': 0,
            'fallback': 0,
            'total': 0
        }
        self.imports = set()
        self.code_blocks = []
        self.llm_generated_samples = []
        self.allowed_templates = set()
        
    def load_analysis(self, analysis_path: Path) -> Dict[str, Any]:
        """Load pre-analyzed workflow data."""
        with open(analysis_path, encoding='utf-8') as f:
            return json.load(f)
    
    def get_template_mapper(self):
        """Get template mapper instance."""
        from app.services.generator.template_mapper import TemplateMapper
        mapper = TemplateMapper()
        
        # Limit templates to 50%
        all_factories = list(mapper.TEMPLATES.keys())
        limit = int(len(all_factories) * self.template_limit_percent)
        self.allowed_templates = set(all_factories[:limit])
        
        logger.info(f"Using {len(self.allowed_templates)}/{len(all_factories)} templates ({self.template_limit_percent*100:.0f}%)")
        return mapper
    
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
        """
        factory = node.get('factory', '')
        node_type = node.get('node_type', '')
        node_name = node.get('node_name', 'unknown')
        settings = self.extract_settings(node)
        
        # Try template ONLY if in allowed list (50%)
        if factory in self.allowed_templates:
            result = mapper.generate_code(factory, input_var, output_var, settings)
            if result:
                self.stats['template'] += 1
                return result
        
        # Use LLM for all other nodes
        code, imports = self._generate_with_llm(node, input_var, output_var)
        if code:
            self.stats['llm'] += 1
            return code, imports
        
        # Fallback: passthrough
        self.stats['fallback'] += 1
        code = f"""# {node_type}: {node_name} (FALLBACK)
# Factory: {factory[:80]}
{output_var} = {input_var}.copy()
"""
        return code, []
    
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
                # Save sample for analysis
                if len(self.llm_generated_samples) < 10:
                    self.llm_generated_samples.append({
                        'node_name': node.get('node_name', 'unknown'),
                        'factory': node.get('factory', ''),
                        'code': code[:500]
                    })
                return f"# LLM Generated: {node.get('node_name', 'unknown')}\n{code}", ['import pandas as pd']
        except Exception as e:
            logger.warning(f"LLM generation failed for {node.get('node_name')}: {e}")
        
        return None, []
    
    def transpile(self, analysis: Dict[str, Any]) -> str:
        """Transpile all nodes to Python code."""
        nodes = analysis.get('nodes', [])
        self.stats['total'] = len(nodes)
        
        logger.info(f"Transpiling {len(nodes)} nodes...")
        
        # Get template mapper with limited templates
        try:
            mapper = self.get_template_mapper()
        except Exception as e:
            logger.error(f"Failed to load template mapper: {e}")
            return ""
        
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
                    'code': f"# ERROR generating {node_name}: {str(e)[:50]}\ndf = df.copy()"
                })
        
        # Build final Python file
        return self._build_python_file()
    
    def _build_python_file(self) -> str:
        """Build the complete Python file."""
        header = f'''"""
Generated Python Pipeline - LLM Integration Test

Workflow: Indicador_Calculo_CET_Rodas
Generated: {datetime.now().isoformat()}
Generator: ChatKnime Transpiler (LLM Test Mode)

Statistics:
- Total Nodes: {self.stats['total']}
- Templates (50%): {self.stats['template']}
- LLM Generated: {self.stats['llm']}
- Fallback: {self.stats['fallback']}
- Coverage: {(self.stats['template'] + self.stats['llm']) / max(self.stats['total'], 1) * 100:.1f}%
"""
'''
        
        sorted_imports = sorted(self.imports)
        imports_section = '\n'.join(sorted_imports)
        
        nodes_section = '# ' + '='*77 + '\n'
        nodes_section += '# Node Implementations\n'
        nodes_section += '# ' + '='*77 + '\n\n'
        
        for block in self.code_blocks:
            nodes_section += f"# --- Node: {block['name']} (ID: {block['id']}) ---\n"
            nodes_section += block['code']
            nodes_section += '\n\n'
        
        return header + imports_section + '\n\n' + nodes_section


def main():
    """Run LLM integration test."""
    # Paths
    analysis_path = Path(__file__).parent.parent.parent / 'workflow_analysis_v2.json'
    output_path = Path(__file__).parent.parent.parent / 'generated_pipeline_llm_test.py'
    report_path = Path(__file__).parent.parent.parent / 'llm_test_report.md'
    
    if not analysis_path.exists():
        logger.error(f"Analysis file not found: {analysis_path}")
        sys.exit(1)
    
    # Run transpilation with 50% template limit
    transpiler = LLMTestTranspiler(template_limit_percent=0.5)
    
    logger.info(f"Loading analysis from: {analysis_path}")
    analysis = transpiler.load_analysis(analysis_path)
    
    logger.info("Starting transpilation with 50% template limit...")
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
    coverage = (stats['template'] + stats['llm']) / max(stats['total'], 1) * 100
    
    report = f"""# LLM Integration Test Report

## Configuration
- Template Limit: 50%
- LLM Fallback: Enabled

## Results

| Metric | Value |
|--------|-------|
| **Status** | {'✅ SUCCESS' if syntax_valid else '⚠️ PARTIAL'} |
| **Total Nodes** | {stats['total']} |
| **Coverage** | {coverage:.1f}% |

### Method Distribution

| Method | Count | % |
|--------|-------|---|
| Template | {stats['template']} | {stats['template']/max(stats['total'],1)*100:.1f}% |
| LLM | {stats['llm']} | {stats['llm']/max(stats['total'],1)*100:.1f}% |
| Fallback | {stats['fallback']} | {stats['fallback']/max(stats['total'],1)*100:.1f}% |

## LLM Generated Samples

"""
    
    for i, sample in enumerate(transpiler.llm_generated_samples, 1):
        report += f"""### Sample {i}: {sample['node_name']}
- Factory: `{sample['factory'][:60]}`

```python
{sample['code']}
```

"""
    
    if syntax_errors:
        report += "## Syntax Errors\n\n"
        for err in syntax_errors:
            report += f"- `{err}`\n"
    
    report += f"""
## Files Generated

| File | Path |
|------|------|
| Python | `{output_path}` |
| Report | `{report_path}` |

---

*Generated at {datetime.now().isoformat()}*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("LLM INTEGRATION TEST RESULTS")
    print("="*60)
    print(f"Status: {'SUCCESS' if syntax_valid else 'PARTIAL'}")
    print(f"Coverage: {coverage:.1f}%")
    print(f"  Templates (50% limit): {stats['template']}")
    print(f"  LLM Generated: {stats['llm']}")
    print(f"  Fallback: {stats['fallback']}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print("="*60)


if __name__ == '__main__':
    main()
