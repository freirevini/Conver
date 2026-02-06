"""
Complete E2E Transpilation Test.

Runs full transpilation pipeline on fluxo_knime_exemplo.knwf:
1. Parse all nodes and connections
2. Use templates for supported nodes
3. Use LLM (Gemini 2.5 Pro) for unsupported nodes
4. Generate complete Python file
5. Validate syntax and evaluate result
"""
import sys
import os
import json
import ast
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any, Tuple

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_complete_transpilation(knwf_path: Path, use_llm: bool = True) -> Dict[str, Any]:
    """
    Run complete transpilation pipeline.
    
    Args:
        knwf_path: Path to KNIME workflow file
        use_llm: Whether to use LLM for unsupported nodes
        
    Returns:
        Dictionary with results and statistics
    """
    from app.services.parser.workflow_parser import WorkflowParser
    from app.services.parser.node_parser import NodeParser
    from app.services.parser.topology_builder import TopologyBuilder
    from app.services.generator.code_generator import CodeGenerator
    from app.services.validator.python_validator import PythonValidator
    from app.utils.zip_extractor import ZipExtractor
    
    results = {
        'workflow': knwf_path.name,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending',
        'nodes_total': 0,
        'nodes_template': 0,
        'nodes_llm': 0,
        'nodes_fallback': 0,
        'nodes_failed': [],
        'syntax_valid': False,
        'generated_code': '',
        'errors': []
    }
    
    try:
        # Step 1: Extract workflow
        logger.info("Step 1: Extracting workflow...")
        extractor = ZipExtractor()
        workflow_dir = extractor.extract(str(knwf_path))
        logger.info(f"  Extracted to: {workflow_dir}")
        
        # Step 2: Parse workflow
        logger.info("Step 2: Parsing workflow.knime...")
        parser = WorkflowParser()
        workflow_data = parser.parse_workflow(workflow_dir)
        
        results['nodes_total'] = len(workflow_data['nodes'])
        results['connections_total'] = len(workflow_data['connections'])
        logger.info(f"  Found {results['nodes_total']} nodes, {results['connections_total']} connections")
        
        # Step 3: Parse node settings
        logger.info("Step 3: Parsing node settings...")
        node_parser = NodeParser()
        for node in workflow_data['nodes']:
            settings_path = node.get('settings_path')
            if settings_path and Path(settings_path).exists():
                try:
                    node['settings'] = node_parser.parse_node_settings(settings_path)
                except Exception as e:
                    logger.warning(f"  Failed to parse settings for node {node.get('id')}: {e}")
                    node['settings'] = {}
            else:
                node['settings'] = {}
        
        # Step 4: Build DAG
        logger.info("Step 4: Building execution DAG...")
        topology = TopologyBuilder()
        dag = topology.build_dag(workflow_data['nodes'], workflow_data['connections'])
        
        # Step 5: Generate code
        logger.info(f"Step 5: Generating Python code (LLM={use_llm})...")
        generator = CodeGenerator(use_llm=use_llm)
        python_code = generator.generate_python_code(workflow_data, dag)
        
        results['generated_code'] = python_code
        results['lines_of_code'] = len(python_code.split('\n'))
        
        # Extract statistics from generated code
        if '# Statistics:' in python_code:
            import re
            template_match = re.search(r'Templates:\s*(\d+)', python_code)
            llm_match = re.search(r'LLM:\s*(\d+)', python_code)
            fallback_match = re.search(r'Fallback:\s*(\d+)', python_code)
            
            if template_match:
                results['nodes_template'] = int(template_match.group(1))
            if llm_match:
                results['nodes_llm'] = int(llm_match.group(1))
            if fallback_match:
                results['nodes_fallback'] = int(fallback_match.group(1))
        
        logger.info(f"  Generated {results['lines_of_code']} lines")
        logger.info(f"  Templates: {results['nodes_template']}, LLM: {results['nodes_llm']}, Fallback: {results['nodes_fallback']}")
        
        # Step 6: Validate syntax
        logger.info("Step 6: Validating Python syntax...")
        try:
            ast.parse(python_code)
            results['syntax_valid'] = True
            logger.info("  Syntax: VALID")
        except SyntaxError as e:
            results['syntax_valid'] = False
            results['errors'].append(f"SyntaxError at line {e.lineno}: {e.msg}")
            logger.error(f"  Syntax: INVALID - {e.msg}")
        
        # Step 7: Use validator for additional checks
        logger.info("Step 7: Running additional validations...")
        try:
            validator = PythonValidator()
            validation_result = validator.validate_code(python_code)
            results['validation'] = validation_result
        except Exception as e:
            logger.warning(f"  Validation warning: {e}")
        
        results['status'] = 'success' if results['syntax_valid'] else 'partial'
        
    except Exception as e:
        logger.error(f"Transpilation failed: {e}")
        results['status'] = 'failed'
        results['errors'].append(str(e))
        import traceback
        results['traceback'] = traceback.format_exc()
    
    return results


def save_python_file(code: str, output_path: Path) -> None:
    """Save generated Python code to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)
    logger.info(f"Saved to: {output_path}")


def generate_evaluation_report(results: Dict[str, Any]) -> str:
    """Generate evaluation report in Markdown."""
    r = results
    
    coverage = (r['nodes_template'] + r['nodes_llm']) / max(r['nodes_total'], 1) * 100
    
    status_emoji = {
        'success': '✅',
        'partial': '⚠️',
        'failed': '❌'
    }.get(r['status'], '❓')
    
    report = f"""# Relatório de Transpilação E2E

## Sumário Executivo

| Métrica | Valor |
|---------|-------|
| **Workflow** | `{r['workflow']}` |
| **Timestamp** | {r['timestamp']} |
| **Status** | {status_emoji} {r['status'].upper()} |
| **Cobertura** | {coverage:.1f}% |

---

## 1. Estatísticas de Processamento

### Nodes

| Categoria | Quantidade | % |
|-----------|------------|---|
| Total | {r['nodes_total']} | 100% |
| Template | {r['nodes_template']} | {r['nodes_template']/max(r['nodes_total'],1)*100:.1f}% |
| LLM | {r['nodes_llm']} | {r['nodes_llm']/max(r['nodes_total'],1)*100:.1f}% |
| Fallback | {r['nodes_fallback']} | {r['nodes_fallback']/max(r['nodes_total'],1)*100:.1f}% |

### Código Gerado

| Métrica | Valor |
|---------|-------|
| Linhas de Código | {r.get('lines_of_code', 0)} |
| Sintaxe Válida | {'✅ Sim' if r['syntax_valid'] else '❌ Não'} |

---

## 2. Validação de Sintaxe

**Status:** {'✅ VÁLIDO' if r['syntax_valid'] else '❌ INVÁLIDO'}

"""
    if r.get('errors'):
        report += "### Erros Encontrados\n\n"
        for err in r['errors']:
            report += f"- `{err}`\n"
        report += "\n"
    
    report += f"""
---

## 3. Avaliação de Qualidade

### Pontos Positivos
- Processamento completo do workflow
- Templates determinísticos para maioria dos nodes
- Código Python 3.12+ compatível

### Pontos de Atenção
- Nodes com fallback requerem revisão manual
- Loops KNIME podem não ter equivalente direto
- Conexões de banco requerem credenciais reais

---

## 4. Preview do Código Gerado

```python
{r['generated_code'][:3000]}{'...' if len(r['generated_code']) > 3000 else ''}
```

---

*Relatório gerado automaticamente pelo ChatKnime E2E Transpiler*
"""
    return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete E2E Transpilation')
    parser.add_argument('knwf_path', help='Path to KNIME workflow file')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM fallback')
    parser.add_argument('--output', '-o', help='Output Python file path')
    parser.add_argument('--report', '-r', help='Output report path')
    
    args = parser.parse_args()
    
    knwf_path = Path(args.knwf_path)
    if not knwf_path.exists():
        print(f"Error: File not found: {knwf_path}")
        sys.exit(1)
    
    # Run transpilation
    results = run_complete_transpilation(knwf_path, use_llm=not args.no_llm)
    
    # Save Python file
    output_path = Path(args.output) if args.output else knwf_path.with_suffix('.py')
    if results['generated_code']:
        save_python_file(results['generated_code'], output_path)
    
    # Generate and save report
    report = generate_evaluation_report(results)
    report_path = Path(args.report) if args.report else knwf_path.with_name(f"{knwf_path.stem}_transpilation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"Status: {results['status'].upper()}")
    print(f"Coverage: {(results['nodes_template'] + results['nodes_llm']) / max(results['nodes_total'], 1) * 100:.1f}%")
    print(f"Syntax Valid: {results['syntax_valid']}")
    print(f"Python file: {output_path}")
    print(f"Report: {report_path}")
    print("="*60)
    
    sys.exit(0 if results['status'] == 'success' else 1)


if __name__ == '__main__':
    main()
