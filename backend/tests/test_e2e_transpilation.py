"""
E2E Transpilation Validation Test
===================================
Comprehensive test of KNIME to Python transpilation with:
- LLM fallback for unsupported nodes
- Syntax validation
- Quantitative metrics
- Qualitative analysis
- Detailed comparison report
"""
import sys
import os
import ast
import zipfile
import tempfile
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import re

sys.path.insert(0, '.')

from lxml import etree
from app.services.parser.workflow_parser import WorkflowParser
from app.services.generator.code_generator_v2 import CodeGeneratorV2
from app.services.catalog.catalog_service import catalog_service
from app.services.generator.template_mapper import TemplateMapper
from app.services.generator.extended_templates import is_expression_node
from app.models.ir_models import NodeInstance, NodeCategory, WorkflowIR, Connection


@dataclass
class TranspilationResult:
    """Result of transpiling a single node."""
    node_id: str
    node_name: str
    factory_class: str
    method: str  # deterministic_parser, deterministic_template, llm, stub
    code: str
    syntax_valid: bool
    syntax_error: str = ""
    llm_used: bool = False
    
    
@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    workflow_file: str = ""
    
    # Quantitative metrics
    total_nodes: int = 0
    total_connections: int = 0
    nodes_transpiled: int = 0
    nodes_with_valid_syntax: int = 0
    nodes_with_llm_fallback: int = 0
    nodes_deterministic: int = 0
    
    # Coverage metrics
    node_coverage_pct: float = 0.0
    connection_coverage_pct: float = 0.0
    parameter_fidelity_pct: float = 0.0
    
    # By method
    by_method: Dict[str, int] = field(default_factory=dict)
    
    # Qualitative
    logic_preservation: str = ""
    transformation_correctness: str = ""
    control_flow_equivalence: str = ""
    
    # Issues
    syntax_errors: List[Dict] = field(default_factory=list)
    llm_interventions: List[Dict] = field(default_factory=list)
    discrepancies: List[Dict] = field(default_factory=list)
    
    # Generated code
    full_code: str = ""
    code_lines: int = 0
    functions_generated: int = 0


def extract_knwf(knwf_path: Path) -> Path:
    """Extract KNIME workflow ZIP."""
    temp_dir = Path(tempfile.mkdtemp(prefix="knime_e2e_"))
    with zipfile.ZipFile(knwf_path, 'r') as zf:
        zf.extractall(temp_dir)
    return temp_dir


def extract_nodes_from_settings(workflow_dir: Path) -> List[Dict]:
    """Extract node info from settings.xml files."""
    nodes = []
    for sf in workflow_dir.rglob("settings.xml"):
        try:
            tree = etree.parse(str(sf))
            parent_name = sf.parent.name
            
            # Extract node name and ID from folder name
            match = re.match(r"(.+?)\s*\(#(\d+)\)", parent_name)
            if match:
                node_name = match.group(1).strip()
                node_id = match.group(2)
            else:
                node_name = parent_name
                node_id = "0"
            
            # Try to get factory class from workflow.knime
            factory_class = ""
            
            nodes.append({
                "id": node_id,
                "name": node_name,
                "factory_class": factory_class,
                "settings_path": str(sf),
                "settings": extract_settings(tree)
            })
        except Exception as e:
            pass
    
    return nodes


def extract_settings(tree) -> Dict:
    """Extract settings from XML tree."""
    settings = {}
    for entry in tree.xpath("//entry[@key]"):
        key = entry.get("key")
        value = entry.get("value", "")
        settings[key] = value
    return settings


def validate_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def analyze_code_quality(code: str) -> Dict:
    """Analyze generated code quality."""
    metrics = {
        "total_lines": len(code.splitlines()),
        "function_count": code.count("def "),
        "import_count": code.count("import "),
        "deterministic_count": code.count("DETERMINISTIC"),
        "llm_count": code.count("LLM-GENERATED"),
        "stub_count": code.count("UNSUPPORTED") + code.count("TODO"),
        "comment_lines": len([l for l in code.splitlines() if l.strip().startswith("#")]),
        "docstring_count": code.count('"""'),
    }
    return metrics


def run_e2e_transpilation(workflow_path: Path) -> ValidationReport:
    """Execute complete E2E transpilation with validation."""
    report = ValidationReport()
    report.workflow_file = str(workflow_path)
    
    start_time = time.time()
    
    print("=" * 70)
    print("E2E TRANSPILATION VALIDATION TEST")
    print("=" * 70)
    print(f"Workflow: {workflow_path.name}")
    print(f"Started: {report.timestamp}")
    print()
    
    # 1. Extract workflow
    print("[1/6] Extracting workflow...")
    extract_dir = extract_knwf(workflow_path)
    print(f"   ✅ Extracted to temp directory")
    
    # 2. Extract nodes
    print("\n[2/6] Extracting nodes from workflow...")
    nodes = extract_nodes_from_settings(extract_dir)
    report.total_nodes = len(nodes)
    print(f"   ✅ Found {report.total_nodes} nodes")
    
    # 3. Initialize generator with LLM fallback
    print("\n[3/6] Initializing transpiler with LLM fallback...")
    generator = CodeGeneratorV2(use_llm_fallback=True, enable_validation=False)
    template_mapper = TemplateMapper()
    
    llm_available = generator.llm_client.is_available() if generator.llm_client else False
    print(f"   ✅ Generator initialized")
    print(f"   🤖 LLM fallback: {'ENABLED' if llm_available else 'DISABLED'}")
    
    # 4. Transpile each node
    print("\n[4/6] Transpiling nodes...")
    results: List[TranspilationResult] = []
    generated_functions = []
    
    for i, node in enumerate(nodes):
        node_name = node["name"]
        node_id = node["id"]
        
        # Map node name to factory class (best effort)
        factory_class = map_name_to_factory(node_name)
        
        # Create NodeInstance
        node_instance = NodeInstance(
            node_id=node_id,
            node_type=node_name,
            factory_class=factory_class,
            name=node_name,
            settings=node.get("settings", {}),
            category=NodeCategory.UNKNOWN
        )
        
        try:
            code = generator.generate_function(node_instance)
            
            # Determine method used
            if "DETERMINISTIC" in code:
                method = "deterministic"
                report.nodes_deterministic += 1
            elif "LLM-GENERATED" in code:
                method = "llm"
                report.nodes_with_llm_fallback += 1
                report.llm_interventions.append({
                    "node_id": node_id,
                    "node_name": node_name,
                    "reason": "No template available"
                })
            else:
                method = "stub"
            
            # Validate syntax
            syntax_ok, syntax_err = validate_syntax(code)
            
            result = TranspilationResult(
                node_id=node_id,
                node_name=node_name,
                factory_class=factory_class,
                method=method,
                code=code,
                syntax_valid=syntax_ok,
                syntax_error=syntax_err,
                llm_used=(method == "llm")
            )
            
            results.append(result)
            generated_functions.append(code)
            
            if syntax_ok:
                report.nodes_with_valid_syntax += 1
            else:
                report.syntax_errors.append({
                    "node_id": node_id,
                    "node_name": node_name,
                    "error": syntax_err,
                    "severity": "HIGH"
                })
            
            report.nodes_transpiled += 1
            
        except Exception as e:
            results.append(TranspilationResult(
                node_id=node_id,
                node_name=node_name,
                factory_class=factory_class,
                method="error",
                code="",
                syntax_valid=False,
                syntax_error=str(e)
            ))
            report.discrepancies.append({
                "node_id": node_id,
                "node_name": node_name,
                "issue": f"Transpilation failed: {e}",
                "severity": "CRITICAL"
            })
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"   ... {i + 1}/{report.total_nodes} nodes processed")
    
    print(f"   ✅ Transpiled {report.nodes_transpiled}/{report.total_nodes} nodes")
    
    # 5. Aggregate results
    print("\n[5/6] Aggregating results...")
    
    # Count by method
    for result in results:
        report.by_method[result.method] = report.by_method.get(result.method, 0) + 1
    
    # Calculate coverage
    report.node_coverage_pct = (report.nodes_transpiled / report.total_nodes * 100) if report.total_nodes > 0 else 0
    report.parameter_fidelity_pct = (report.nodes_with_valid_syntax / report.total_nodes * 100) if report.total_nodes > 0 else 0
    
    # Combine all functions
    report.full_code = "\n\n".join(generated_functions)
    
    # Analyze code quality
    quality = analyze_code_quality(report.full_code)
    report.code_lines = quality["total_lines"]
    report.functions_generated = quality["function_count"]
    
    print(f"   ✅ Node coverage: {report.node_coverage_pct:.1f}%")
    print(f"   ✅ Syntax validity: {report.parameter_fidelity_pct:.1f}%")
    
    # 6. Qualitative analysis
    print("\n[6/6] Qualitative analysis...")
    
    report.logic_preservation = assess_logic_preservation(results)
    report.transformation_correctness = assess_transformation_correctness(results)
    report.control_flow_equivalence = assess_control_flow(results)
    
    print(f"   ✅ Logic preservation: {report.logic_preservation}")
    print(f"   ✅ Transformations: {report.transformation_correctness}")
    print(f"   ✅ Control flow: {report.control_flow_equivalence}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed:.2f}s")
    
    return report


def map_name_to_factory(node_name: str) -> str:
    """Map node display name to factory class (best effort)."""
    KNOWN_MAPPINGS = {
        "Math Formula": "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory",
        "Rule Engine": "org.knime.base.node.rules.engine.RuleEngineNodeFactory",
        "String Manipulation": "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
        "Column Filter": "org.knime.base.node.preproc.filter.column.FilterColumnNodeFactory",
        "Row Filter": "org.knime.base.node.preproc.filter.row.RowFilterNodeFactory",
        "GroupBy": "org.knime.base.node.preproc.groupby.GroupByNodeFactory",
        "Joiner": "org.knime.base.node.preproc.joiner.JoinerNodeFactory",
        "Concatenate": "org.knime.base.node.append.ConcatenateNodeFactory",
        "Column Rename": "org.knime.base.node.preproc.rename.ColumnRenameNodeFactory",
        "Pivot": "org.knime.base.node.preproc.pivot.PivotNodeFactory",
        "End IF": "org.knime.base.node.switches.endcase.EndCaseNodeFactory",
        "Empty Table Switch": "org.knime.base.node.switches.emptytableswitch.EmptyTableSwitchNodeFactory",
        "Loop End": "org.knime.base.node.meta.looper.LoopEndNodeFactory",
        "Group Loop Start": "org.knime.base.node.meta.looper.group.GroupLoopStartNodeFactory",
    }
    return KNOWN_MAPPINGS.get(node_name, f"org.knime.unknown.{node_name.replace(' ', '')}NodeFactory")


def assess_logic_preservation(results: List[TranspilationResult]) -> str:
    """Assess if business logic is preserved."""
    deterministic = sum(1 for r in results if r.method == "deterministic")
    total = len(results)
    ratio = deterministic / total if total > 0 else 0
    
    if ratio >= 0.95:
        return "EXCELLENT (95%+ deterministic)"
    elif ratio >= 0.80:
        return "GOOD (80%+ deterministic)"
    elif ratio >= 0.60:
        return "MODERATE (60%+ deterministic)"
    else:
        return "NEEDS REVIEW (< 60% deterministic)"


def assess_transformation_correctness(results: List[TranspilationResult]) -> str:
    """Assess transformation correctness."""
    valid = sum(1 for r in results if r.syntax_valid)
    total = len(results)
    ratio = valid / total if total > 0 else 0
    
    if ratio >= 0.98:
        return "EXCELLENT (98%+ valid syntax)"
    elif ratio >= 0.90:
        return "GOOD (90%+ valid syntax)"
    elif ratio >= 0.75:
        return "MODERATE (75%+ valid syntax)"
    else:
        return "NEEDS FIXING (< 75% valid syntax)"


def assess_control_flow(results: List[TranspilationResult]) -> str:
    """Assess control flow equivalence."""
    control_nodes = ["Loop", "IF", "Switch", "End"]
    control_results = [r for r in results if any(cn in r.node_name for cn in control_nodes)]
    
    if not control_results:
        return "N/A (no control flow nodes)"
    
    valid = sum(1 for r in control_results if r.syntax_valid)
    total = len(control_results)
    ratio = valid / total if total > 0 else 0
    
    if ratio >= 0.95:
        return f"EXCELLENT ({valid}/{total} control nodes valid)"
    elif ratio >= 0.80:
        return f"GOOD ({valid}/{total} control nodes valid)"
    else:
        return f"NEEDS REVIEW ({valid}/{total} control nodes valid)"


def generate_markdown_report(report: ValidationReport) -> str:
    """Generate comprehensive markdown report."""
    md = []
    
    md.append("# E2E Transpilation Validation Report")
    md.append(f"\n**Generated:** {report.timestamp}")
    md.append(f"\n**Workflow:** `{Path(report.workflow_file).name}`")
    
    # Executive Summary
    md.append("\n---\n## 📊 Executive Summary\n")
    
    overall_score = (
        report.node_coverage_pct * 0.3 +
        report.parameter_fidelity_pct * 0.4 +
        (report.nodes_deterministic / report.total_nodes * 100 if report.total_nodes > 0 else 0) * 0.3
    )
    
    if overall_score >= 95:
        status = "✅ EXCELLENT"
    elif overall_score >= 80:
        status = "🟢 GOOD"
    elif overall_score >= 60:
        status = "🟡 MODERATE"
    else:
        status = "🔴 NEEDS WORK"
    
    md.append(f"**Overall Score:** {overall_score:.1f}% ({status})")
    
    # Quantitative Metrics
    md.append("\n---\n## 📈 Quantitative Metrics\n")
    md.append("### Node Coverage\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total KNIME Nodes | {report.total_nodes} |")
    md.append(f"| Nodes Transpiled | {report.nodes_transpiled} |")
    md.append(f"| **Coverage Rate** | **{report.node_coverage_pct:.1f}%** |")
    
    md.append("\n### Syntax Validation\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Valid Syntax | {report.nodes_with_valid_syntax} |")
    md.append(f"| Syntax Errors | {len(report.syntax_errors)} |")
    md.append(f"| **Validity Rate** | **{report.parameter_fidelity_pct:.1f}%** |")
    
    md.append("\n### Generation Methods\n")
    md.append("| Method | Count | % |")
    md.append("|--------|-------|---|")
    for method, count in sorted(report.by_method.items(), key=lambda x: -x[1]):
        pct = count / report.total_nodes * 100 if report.total_nodes > 0 else 0
        icon = "✅" if method == "deterministic" else "🤖" if method == "llm" else "⚠️"
        md.append(f"| {icon} {method.upper()} | {count} | {pct:.1f}% |")
    
    md.append("\n### Code Quality\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Lines | {report.code_lines} |")
    md.append(f"| Functions Generated | {report.functions_generated} |")
    md.append(f"| LLM Interventions | {report.nodes_with_llm_fallback} |")
    
    # Qualitative Analysis
    md.append("\n---\n## 🔍 Qualitative Analysis\n")
    md.append("| Dimension | Assessment |")
    md.append("|-----------|------------|")
    md.append(f"| Logic Preservation | {report.logic_preservation} |")
    md.append(f"| Transformation Correctness | {report.transformation_correctness} |")
    md.append(f"| Control Flow Equivalence | {report.control_flow_equivalence} |")
    
    # LLM Interventions
    if report.llm_interventions:
        md.append("\n---\n## 🤖 LLM Fallback Interventions\n")
        md.append("| Node ID | Node Name | Reason |")
        md.append("|---------|-----------|--------|")
        for intervention in report.llm_interventions[:15]:
            md.append(f"| {intervention['node_id']} | {intervention['node_name'][:25]} | {intervention['reason']} |")
        if len(report.llm_interventions) > 15:
            md.append(f"\n*...and {len(report.llm_interventions) - 15} more*")
    
    # Syntax Errors
    if report.syntax_errors:
        md.append("\n---\n## ❌ Syntax Errors\n")
        md.append("| Node | Error | Severity |")
        md.append("|------|-------|----------|")
        for err in report.syntax_errors[:10]:
            md.append(f"| {err['node_name'][:20]} | {err['error'][:40]} | {err['severity']} |")
        if len(report.syntax_errors) > 10:
            md.append(f"\n*...and {len(report.syntax_errors) - 10} more*")
    
    # Discrepancies
    if report.discrepancies:
        md.append("\n---\n## ⚠️ Discrepancies Found\n")
        md.append("| Node | Issue | Severity |")
        md.append("|------|-------|----------|")
        for disc in report.discrepancies[:10]:
            md.append(f"| {disc['node_name'][:20]} | {disc['issue'][:40]} | {disc['severity']} |")
    
    # Recommendations
    md.append("\n---\n## 💡 Recommendations\n")
    
    if report.nodes_with_llm_fallback > 0:
        md.append(f"- **Add templates** for {report.nodes_with_llm_fallback} nodes using LLM fallback")
    
    if report.syntax_errors:
        md.append(f"- **Fix syntax errors** in {len(report.syntax_errors)} generated functions")
    
    if report.discrepancies:
        md.append(f"- **Review discrepancies** in {len(report.discrepancies)} nodes")
    
    if overall_score >= 95:
        md.append("- ✅ **Ready for production** - transpilation quality is excellent")
    else:
        md.append("- ⚠️ **Manual review recommended** before production use")
    
    return "\n".join(md)


if __name__ == "__main__":
    workflow_path = Path(r"c:\Users\vinic\Documents\Projetos\ChatKnime\fluxo_knime_exemplo.knwf")
    
    if not workflow_path.exists():
        print(f"❌ Workflow not found: {workflow_path}")
        sys.exit(1)
    
    # Run validation
    report = run_e2e_transpilation(workflow_path)
    
    # Generate report
    markdown = generate_markdown_report(report)
    
    # Save report
    report_path = Path("e2e_transpilation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"\n📄 Report saved: {report_path}")
    print("\n" + "=" * 70)
    print(markdown)
