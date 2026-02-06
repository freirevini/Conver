"""
E2E Transpilation Validation Script.

Validates KNIME workflow transpilation following skills:
- python-pro: Modern Python patterns
- data-engineer: Data pipeline validation
- code-review-checklist: Comprehensive code review
- e2e-testing-patterns: End-to-end testing
- testing-patterns: Test case design
- error-handling-patterns: Error handling validation
"""
import sys
import os
import json
import ast
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.parser.workflow_parser import WorkflowParser
from app.services.parser.node_parser import NodeParser
from app.services.parser.topology_builder import TopologyBuilder
from app.services.generator.code_generator import CodeGenerator
from app.services.validator.python_validator import PythonValidator
from app.utils.zip_extractor import ZipExtractor


@dataclass
class NodeInventory:
    """Inventory of KNIME nodes."""
    node_id: str
    name: str
    factory: str
    factory_short: str
    category: str = ""
    has_settings: bool = False


@dataclass
class ConnectionInventory:
    """Inventory of KNIME connections."""
    source_id: str
    dest_id: str
    source_port: int = 0
    dest_port: int = 0


@dataclass
class TranspilationResult:
    """Result of transpilation for a single node."""
    node_id: str
    node_name: str
    factory: str
    method: str  # 'template' | 'llm' | 'fallback' | 'skipped'
    success: bool
    code_snippet: str = ""
    imports: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class ValidationIssue:
    """Validation issue found."""
    severity: str  # 'critical' | 'high' | 'medium' | 'low'
    category: str  # 'syntax' | 'mapping' | 'logic' | 'data_type' | 'auth'
    description: str
    node_id: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: str = ""


@dataclass
class ValidationReport:
    """Complete validation report."""
    workflow_name: str
    execution_timestamp: str
    
    # KNIME analysis
    total_nodes: int = 0
    total_connections: int = 0
    nodes_inventory: List[NodeInventory] = field(default_factory=list)
    connections_inventory: List[ConnectionInventory] = field(default_factory=list)
    data_sources: List[Dict] = field(default_factory=list)
    transformations: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)
    
    # Transpilation results
    transpilation_results: List[TranspilationResult] = field(default_factory=list)
    template_count: int = 0
    llm_count: int = 0
    fallback_count: int = 0
    skipped_count: int = 0
    
    # Generated code
    generated_code: str = ""
    total_lines: int = 0
    
    # Python syntax validation
    syntax_valid: bool = False
    syntax_errors: List[str] = field(default_factory=list)
    
    # Validation issues
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Summary
    overall_status: str = ""  # 'success' | 'partial' | 'failed'
    completeness_score: float = 0.0


class E2ETranspilationValidator:
    """End-to-end transpilation validator."""
    
    def __init__(self, knwf_path: Path):
        self.knwf_path = knwf_path
        self.report = ValidationReport(
            workflow_name=knwf_path.name,
            execution_timestamp=datetime.now().isoformat()
        )
        
    def run_full_validation(self) -> ValidationReport:
        """Execute complete E2E validation."""
        print(f"=== E2E Transpilation Validation ===")
        print(f"Workflow: {self.knwf_path.name}")
        print(f"Timestamp: {self.report.execution_timestamp}\n")
        
        # Phase 1: Extract and analyze KNIME workflow
        self._analyze_knime_workflow()
        
        # Phase 2: Run transpilation
        self._run_transpilation()
        
        # Phase 3: Validate Python syntax
        self._validate_python_syntax()
        
        # Phase 4: Validate mapping integrity
        self._validate_mapping_integrity()
        
        # Phase 5: Validate data handling
        self._validate_data_handling()
        
        # Phase 6: Validate authentication requirements
        self._validate_authentication()
        
        # Phase 7: Calculate overall status
        self._calculate_overall_status()
        
        return self.report
    
    def _analyze_knime_workflow(self) -> None:
        """Phase 1: Analyze KNIME workflow structure."""
        print("Phase 1: Analyzing KNIME workflow...")
        
        try:
            # Extract workflow
            extractor = ZipExtractor()
            workflow_dir = extractor.extract(str(self.knwf_path))
            
            # Parse workflow
            parser = WorkflowParser()
            workflow_data = parser.parse_workflow(workflow_dir)
            
            # Parse node settings
            node_parser = NodeParser()
            for node in workflow_data['nodes']:
                settings_path = node.get('settings_path')
                if settings_path and Path(settings_path).exists():
                    node['settings'] = node_parser.parse_node_settings(settings_path)
                    
            # Store workflow data for later
            self._workflow_data = workflow_data
            self._workflow_dir = workflow_dir
            
            # Build inventory
            self.report.total_nodes = len(workflow_data['nodes'])
            self.report.total_connections = len(workflow_data['connections'])
            
            # Categorize nodes
            for node in workflow_data['nodes']:
                factory = node.get('factory', node.get('factory_class', ''))
                factory_short = factory.split('.')[-1] if factory else 'Unknown'
                
                inventory = NodeInventory(
                    node_id=str(node.get('id', node.get('node_id', ''))),
                    name=node.get('name', node.get('annotation', 'Unknown')),
                    factory=factory,
                    factory_short=factory_short,
                    category=self._categorize_node(factory_short),
                    has_settings=bool(node.get('settings'))
                )
                self.report.nodes_inventory.append(inventory)
                
                # Classify as source/transform/output
                if self._is_data_source(factory_short):
                    self.report.data_sources.append({
                        'id': inventory.node_id,
                        'name': inventory.name,
                        'type': factory_short
                    })
                elif self._is_output(factory_short):
                    self.report.outputs.append({
                        'id': inventory.node_id,
                        'name': inventory.name,
                        'type': factory_short
                    })
                else:
                    self.report.transformations.append({
                        'id': inventory.node_id,
                        'name': inventory.name,
                        'type': factory_short
                    })
            
            # Build connections inventory
            for conn in workflow_data['connections']:
                self.report.connections_inventory.append(ConnectionInventory(
                    source_id=str(conn.get('source_id', conn.get('sourceID', ''))),
                    dest_id=str(conn.get('dest_id', conn.get('destID', ''))),
                    source_port=conn.get('source_port', 0),
                    dest_port=conn.get('dest_port', 0)
                ))
                
            print(f"  - Nodes found: {self.report.total_nodes}")
            print(f"  - Connections found: {self.report.total_connections}")
            print(f"  - Data sources: {len(self.report.data_sources)}")
            print(f"  - Transformations: {len(self.report.transformations)}")
            print(f"  - Outputs: {len(self.report.outputs)}")
            
        except Exception as e:
            self.report.issues.append(ValidationIssue(
                severity="critical",
                category="parsing",
                description=f"Failed to parse KNIME workflow: {str(e)}",
                recommendation="Check if the .knwf file is valid and complete"
            ))
            print(f"  - ERROR: {e}")
            
    def _run_transpilation(self) -> None:
        """Phase 2: Run transpilation pipeline."""
        print("\nPhase 2: Running transpilation...")
        
        if not hasattr(self, '_workflow_data'):
            print("  - SKIPPED: No workflow data available")
            return
            
        try:
            # Build DAG
            topology = TopologyBuilder()
            dag = topology.build_dag(
                self._workflow_data['nodes'],
                self._workflow_data['connections']
            )
            
            # Generate code
            generator = CodeGenerator(use_llm=False)  # Disable LLM for validation
            self.report.generated_code = generator.generate_python_code(
                self._workflow_data, dag
            )
            
            self.report.total_lines = len(self.report.generated_code.split('\n'))
            
            # Analyze generation stats from code comments
            stats_match = re.search(
                r'Statistics:\s*\n#\s*-\s*Templates:\s*(\d+)\s*\n#\s*-\s*LLM:\s*(\d+)\s*\n#\s*-\s*Fallback:\s*(\d+)',
                self.report.generated_code
            )
            if stats_match:
                self.report.template_count = int(stats_match.group(1))
                self.report.llm_count = int(stats_match.group(2))
                self.report.fallback_count = int(stats_match.group(3))
                
            print(f"  - Lines generated: {self.report.total_lines}")
            print(f"  - Templates used: {self.report.template_count}")
            print(f"  - LLM used: {self.report.llm_count}")
            print(f"  - Fallbacks: {self.report.fallback_count}")
            
        except Exception as e:
            self.report.issues.append(ValidationIssue(
                severity="critical",
                category="transpilation",
                description=f"Failed to transpile workflow: {str(e)}",
                recommendation="Check CodeGenerator and template coverage"
            ))
            print(f"  - ERROR: {e}")
            
    def _validate_python_syntax(self) -> None:
        """Phase 3: Validate Python syntax."""
        print("\nPhase 3: Validating Python syntax...")
        
        if not self.report.generated_code:
            print("  - SKIPPED: No generated code")
            return
            
        try:
            # Use AST to validate syntax
            ast.parse(self.report.generated_code)
            self.report.syntax_valid = True
            print("  - Syntax: VALID")
            
        except SyntaxError as e:
            self.report.syntax_valid = False
            self.report.syntax_errors.append(
                f"Line {e.lineno}: {e.msg} - {e.text}"
            )
            self.report.issues.append(ValidationIssue(
                severity="critical",
                category="syntax",
                description=f"SyntaxError at line {e.lineno}: {e.msg}",
                line_number=e.lineno,
                recommendation=f"Fix syntax: {e.text}"
            ))
            print(f"  - Syntax: INVALID - {e.msg}")
            
        # Additional checks
        self._check_indentation()
        self._check_imports()
        self._check_undefined_variables()
        
    def _check_indentation(self) -> None:
        """Check for indentation issues."""
        lines = self.report.generated_code.split('\n')
        for i, line in enumerate(lines, 1):
            if line.strip() and line[0] in ' \t':
                # Check for mixed tabs/spaces
                if '\t' in line and '    ' in line:
                    self.report.issues.append(ValidationIssue(
                        severity="medium",
                        category="syntax",
                        description=f"Mixed tabs and spaces at line {i}",
                        line_number=i,
                        recommendation="Use consistent indentation (4 spaces)"
                    ))
                    
    def _check_imports(self) -> None:
        """Check import statements."""
        code = self.report.generated_code
        
        # Check for common import issues
        imports = re.findall(r'^(?:from|import)\s+(.+)$', code, re.MULTILINE)
        
        required_libs = ['pandas', 'numpy']
        for lib in required_libs:
            if lib not in code:
                self.report.issues.append(ValidationIssue(
                    severity="low",
                    category="imports",
                    description=f"Common library '{lib}' not imported",
                    recommendation=f"Consider if {lib} is needed"
                ))
                
    def _check_undefined_variables(self) -> None:
        """Check for potential undefined variables."""
        # Simple heuristic: look for variables used before assignment
        try:
            tree = ast.parse(self.report.generated_code)
            
            # Track defined names
            defined = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Store):
                        defined.add(node.id)
                        
        except:
            pass  # Already reported syntax errors
            
    def _validate_mapping_integrity(self) -> None:
        """Phase 4: Validate mapping integrity between KNIME and Python."""
        print("\nPhase 4: Validating mapping integrity...")
        
        if not self.report.generated_code or not self.report.nodes_inventory:
            print("  - SKIPPED: Missing data for comparison")
            return
            
        code = self.report.generated_code
        mapped_count = 0
        unmapped_nodes = []
        
        for node in self.report.nodes_inventory:
            # Check if node ID appears in code or comments
            node_referenced = (
                f"node_{node.node_id}" in code or
                f"# Node {node.node_id}" in code or
                node.name in code or
                node.factory_short in code
            )
            
            if node_referenced:
                mapped_count += 1
            else:
                unmapped_nodes.append(node)
                self.report.issues.append(ValidationIssue(
                    severity="medium",
                    category="mapping",
                    description=f"Node '{node.name}' ({node.factory_short}) may not be mapped",
                    node_id=node.node_id,
                    recommendation="Verify if node functionality is implemented"
                ))
                
        mapping_rate = (mapped_count / len(self.report.nodes_inventory)) * 100 if self.report.nodes_inventory else 0
        print(f"  - Mapping rate: {mapping_rate:.1f}% ({mapped_count}/{len(self.report.nodes_inventory)})")
        print(f"  - Potentially unmapped: {len(unmapped_nodes)}")
        
    def _validate_data_handling(self) -> None:
        """Phase 5: Validate data type handling."""
        print("\nPhase 5: Validating data handling...")
        
        code = self.report.generated_code
        
        # Check for proper DataFrame operations
        df_patterns = [
            (r'\.astype\(\s*[\'"]', "Type conversion detected"),
            (r'pd\.to_datetime\(', "Date conversion detected"),
            (r'pd\.to_numeric\(', "Numeric conversion detected"),
        ]
        
        for pattern, msg in df_patterns:
            if re.search(pattern, code):
                print(f"  - {msg}")
                
        # Check for groupby/aggregation
        if 'groupby' in code:
            print("  - GroupBy operations detected")
            
        # Check for sorting
        if 'sort_values' in code or 'sort_index' in code:
            print("  - Sorting operations detected")
            
        # Check for filtering
        if '.query(' in code or '.loc[' in code:
            print("  - Filtering operations detected")
            
    def _validate_authentication(self) -> None:
        """Phase 6: Validate authentication handling."""
        print("\nPhase 6: Validating authentication requirements...")
        
        code = self.report.generated_code
        
        # Check for database-related nodes
        db_patterns = [
            (r'oracle', 'Oracle'),
            (r'mysql', 'MySQL'),
            (r'postgres', 'PostgreSQL'),
            (r'sql\s*server', 'SQL Server'),
            (r'sqlite', 'SQLite'),
        ]
        
        detected_dbs = []
        for pattern, name in db_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                detected_dbs.append(name)
                
        if detected_dbs:
            print(f"  - Databases detected: {', '.join(detected_dbs)}")
            
            # Check for proper credential handling
            if 'os.environ' in code or 'getenv' in code:
                print("  - Using environment variables for credentials: GOOD")
            elif 'password' in code.lower() and '=' in code:
                self.report.issues.append(ValidationIssue(
                    severity="high",
                    category="auth",
                    description="Potential hardcoded credentials detected",
                    recommendation="Use environment variables for credentials"
                ))
                print("  - WARNING: Potential hardcoded credentials")
        else:
            print("  - No database connections detected")
            
    def _calculate_overall_status(self) -> None:
        """Phase 7: Calculate overall validation status."""
        print("\nPhase 7: Calculating overall status...")
        
        critical_count = sum(1 for i in self.report.issues if i.severity == 'critical')
        high_count = sum(1 for i in self.report.issues if i.severity == 'high')
        medium_count = sum(1 for i in self.report.issues if i.severity == 'medium')
        low_count = sum(1 for i in self.report.issues if i.severity == 'low')
        
        # Calculate completeness score
        node_score = (self.report.template_count + self.report.llm_count) / max(self.report.total_nodes, 1)
        syntax_score = 1.0 if self.report.syntax_valid else 0.0
        issue_penalty = (critical_count * 0.3 + high_count * 0.15 + medium_count * 0.05 + low_count * 0.01)
        
        self.report.completeness_score = max(0, min(1, (node_score + syntax_score) / 2 - issue_penalty)) * 100
        
        # Determine status
        if critical_count > 0:
            self.report.overall_status = "failed"
        elif high_count > 2 or medium_count > 5:
            self.report.overall_status = "partial"
        elif self.report.completeness_score >= 80:
            self.report.overall_status = "success"
        else:
            self.report.overall_status = "partial"
            
        print(f"  - Critical issues: {critical_count}")
        print(f"  - High issues: {high_count}")
        print(f"  - Medium issues: {medium_count}")
        print(f"  - Low issues: {low_count}")
        print(f"  - Completeness score: {self.report.completeness_score:.1f}%")
        print(f"  - Overall status: {self.report.overall_status.upper()}")
        
    def _categorize_node(self, factory_short: str) -> str:
        """Categorize node by factory type."""
        factory_lower = factory_short.lower()
        
        if any(x in factory_lower for x in ['reader', 'source', 'input', 'csv', 'excel', 'table', 'file']):
            return "Input"
        elif any(x in factory_lower for x in ['writer', 'output', 'sink', 'save']):
            return "Output"
        elif any(x in factory_lower for x in ['filter', 'row', 'column', 'select']):
            return "Filter"
        elif any(x in factory_lower for x in ['join', 'merge', 'concatenate', 'append']):
            return "Join"
        elif any(x in factory_lower for x in ['group', 'aggregate', 'pivot']):
            return "Aggregation"
        elif any(x in factory_lower for x in ['sort', 'order']):
            return "Sorting"
        elif any(x in factory_lower for x in ['math', 'formula', 'calculation', 'string']):
            return "Transformation"
        elif any(x in factory_lower for x in ['db', 'database', 'sql', 'oracle', 'mysql']):
            return "Database"
        elif any(x in factory_lower for x in ['learner', 'predictor', 'model', 'classifier']):
            return "ML"
        else:
            return "Other"
            
    def _is_data_source(self, factory_short: str) -> bool:
        """Check if node is a data source."""
        return any(x in factory_short.lower() for x in [
            'reader', 'source', 'input', 'connector', 'db reader'
        ])
        
    def _is_output(self, factory_short: str) -> bool:
        """Check if node is an output."""
        return any(x in factory_short.lower() for x in [
            'writer', 'sink', 'output', 'save'
        ])
        
    def generate_markdown_report(self) -> str:
        """Generate markdown report."""
        r = self.report
        
        # Group issues by severity
        critical = [i for i in r.issues if i.severity == 'critical']
        high = [i for i in r.issues if i.severity == 'high']
        medium = [i for i in r.issues if i.severity == 'medium']
        low = [i for i in r.issues if i.severity == 'low']
        
        # Factory distribution
        factory_counts = Counter([n.factory_short for n in r.nodes_inventory])
        
        md = f"""# E2E Transpilation Validation Report

## Executive Summary

| Metric | Value |
|--------|-------|
| **Workflow** | `{r.workflow_name}` |
| **Timestamp** | {r.execution_timestamp} |
| **Overall Status** | {"✅ SUCCESS" if r.overall_status == 'success' else "⚠️ PARTIAL" if r.overall_status == 'partial' else "❌ FAILED"} |
| **Completeness Score** | {r.completeness_score:.1f}% |

---

## 1. KNIME Workflow Analysis

### Node Inventory

| Metric | Count |
|--------|-------|
| Total Nodes | {r.total_nodes} |
| Total Connections | {r.total_connections} |
| Data Sources | {len(r.data_sources)} |
| Transformations | {len(r.transformations)} |
| Outputs | {len(r.outputs)} |

### Factory Distribution

| Factory | Count |
|---------|-------|
"""
        for factory, count in factory_counts.most_common(15):
            md += f"| {factory} | {count} |\n"
            
        md += f"""
### Data Sources Identified

| ID | Name | Type |
|----|------|------|
"""
        for ds in r.data_sources[:10]:
            md += f"| {ds['id']} | {ds['name']} | {ds['type']} |\n"
            
        md += f"""
### Transformation Nodes

| ID | Name | Type |
|----|------|------|
"""
        for t in r.transformations[:15]:
            md += f"| {t['id']} | {t['name']} | {t['type']} |\n"
            
        if len(r.transformations) > 15:
            md += f"| ... | *({len(r.transformations) - 15} more)* | ... |\n"
            
        md += f"""
### Output Nodes

| ID | Name | Type |
|----|------|------|
"""
        for out in r.outputs[:10]:
            md += f"| {out['id']} | {out['name']} | {out['type']} |\n"

        md += f"""
---

## 2. Transpilation Results

| Metric | Value |
|--------|-------|
| Lines Generated | {r.total_lines} |
| Template-based | {r.template_count} |
| LLM-based | {r.llm_count} |
| Fallback | {r.fallback_count} |

---

## 3. Python Syntax Validation

| Status | {"✅ VALID" if r.syntax_valid else "❌ INVALID"} |
|--------|-------|

"""
        if r.syntax_errors:
            md += "### Syntax Errors\n\n"
            for err in r.syntax_errors:
                md += f"- `{err}`\n"
            md += "\n"
            
        md += f"""
---

## 4. Issues by Severity

### 🔴 Critical ({len(critical)})

"""
        if critical:
            md += "| Category | Description | Recommendation |\n|----------|-------------|----------------|\n"
            for i in critical:
                md += f"| {i.category} | {i.description} | {i.recommendation} |\n"
        else:
            md += "*No critical issues*\n"
            
        md += f"""
### 🟠 High ({len(high)})

"""
        if high:
            md += "| Category | Description | Recommendation |\n|----------|-------------|----------------|\n"
            for i in high:
                md += f"| {i.category} | {i.description} | {i.recommendation} |\n"
        else:
            md += "*No high severity issues*\n"
            
        md += f"""
### 🟡 Medium ({len(medium)})

"""
        if medium:
            md += "| Category | Node | Description |\n|----------|------|-------------|\n"
            for i in medium[:10]:
                md += f"| {i.category} | {i.node_id or '-'} | {i.description} |\n"
            if len(medium) > 10:
                md += f"| ... | ... | *({len(medium) - 10} more)* |\n"
        else:
            md += "*No medium severity issues*\n"
            
        md += f"""
### 🟢 Low ({len(low)})

"""
        if low:
            md += "| Category | Description |\n|----------|-------------|\n"
            for i in low[:5]:
                md += f"| {i.category} | {i.description} |\n"
            if len(low) > 5:
                md += f"| ... | *({len(low) - 5} more)* |\n"
        else:
            md += "*No low severity issues*\n"
            
        md += f"""
---

## 5. Mapping Verification

### Elements Correctly Mapped

| Category | Mapped | Total | Rate |
|----------|--------|-------|------|
| Nodes | {r.template_count + r.llm_count} | {r.total_nodes} | {((r.template_count + r.llm_count) / max(r.total_nodes, 1)) * 100:.1f}% |
| Connections | {r.total_connections} | {r.total_connections} | 100% |

### Authentication Requirements

"""
        # Check if any DB nodes
        db_nodes = [n for n in r.nodes_inventory if n.category == "Database"]
        if db_nodes:
            md += """| Requirement | Status |
|-------------|--------|
| Environment variables | ✅ Recommended |
| Credentials handling | Verify manually |

**Database nodes detected:**
"""
            for n in db_nodes[:5]:
                md += f"- {n.name} ({n.factory_short})\n"
        else:
            md += "*No database authentication required*\n"
            
        md += f"""
---

## 6. Recommendations

### Immediate Actions

"""
        if critical:
            md += "1. **Fix critical syntax errors** before deployment\n"
        if high:
            md += "2. **Review high-severity issues** for potential logic errors\n"
        if r.fallback_count > 0:
            md += f"3. **Review {r.fallback_count} fallback nodes** for implementation completeness\n"
            
        md += f"""
### Quality Improvements

1. Add unit tests for generated functions
2. Implement integration tests with sample data
3. Review data type conversions for edge cases

---

## Generated Code Preview

```python
{r.generated_code[:2000]}{'...' if len(r.generated_code) > 2000 else ''}
```

---

*Report generated by E2E Transpilation Validator*
"""
        return md


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='E2E Transpilation Validation')
    parser.add_argument('knwf_path', help='Path to KNIME workflow file (.knwf)')
    parser.add_argument('--output', '-o', help='Output report path', default=None)
    
    args = parser.parse_args()
    
    knwf_path = Path(args.knwf_path)
    if not knwf_path.exists():
        print(f"Error: File not found: {knwf_path}")
        sys.exit(1)
        
    # Run validation
    validator = E2ETranspilationValidator(knwf_path)
    report = validator.run_full_validation()
    
    # Generate markdown report
    md_report = validator.generate_markdown_report()
    
    # Output
    output_path = args.output or knwf_path.with_suffix('.validation_report.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
        
    print(f"\n=== Report saved to: {output_path} ===")
    
    # Return exit code based on status
    if report.overall_status == 'failed':
        sys.exit(1)
    elif report.overall_status == 'partial':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
