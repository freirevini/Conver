"""
E2E Test: Complete KNIME Workflow Transpilation
=================================================
Tests the full pipeline with fluxo_knime_exemplo.knwf
Generates comprehensive report of results.
"""
import sys
import os
import zipfile
import tempfile
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple

sys.path.insert(0, '.')

from app.services.parser.workflow_parser import WorkflowParser
from app.services.generator.code_generator_v2 import CodeGeneratorV2
from app.services.catalog.catalog_service import catalog_service
from app.services.generator.template_mapper import TemplateMapper
from app.services.generator.extended_templates import (
    is_expression_node,
    get_all_extended_templates,
)
from app.models.ir_models import NodeInstance, NodeCategory

report = {
    "timestamp": datetime.now().isoformat(),
    "workflow_file": "",
    "summary": {},
    "nodes_analyzed": [],
    "generation_results": {"deterministic": 0, "llm_generated": 0, "stub": 0},
    "catalog_coverage": {},
    "recommendations": []
}


def extract_knwf(knwf_path: Path) -> Path:
    """Extract .knwf file to temp directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="knime_e2e_"))
    with zipfile.ZipFile(knwf_path, 'r') as zf:
        zf.extractall(temp_dir)
    return temp_dir


def analyze_node(node: Dict, template_mapper: TemplateMapper) -> Dict:
    """Analyze a single node for catalog coverage."""
    factory_class = node.get("factory_class", "")
    
    analysis = {
        "node_id": node.get("id", "?"),
        "name": node.get("name", "Unknown"),
        "factory_class": factory_class,
        "category": "",
        "has_template": False,
        "has_expression_parser": False,
        "in_catalog": False,
        "generation_method": "llm_or_stub",
        "complexity": "HIGH"
    }
    
    catalog_node = catalog_service.get_node(factory_class)
    if catalog_node:
        analysis["in_catalog"] = True
        analysis["category"] = catalog_node.get("category", "")
        analysis["complexity"] = catalog_node.get("complexity", "HIGH")
    
    template = template_mapper.get_template(factory_class)
    if template:
        analysis["has_template"] = True
        analysis["generation_method"] = "deterministic_template"
    
    if is_expression_node(factory_class):
        analysis["has_expression_parser"] = True
        analysis["generation_method"] = "deterministic_parser"
    
    return analysis


def run_e2e_test(workflow_path: Path):
    """Run complete E2E test."""
    print("=" * 70)
    print("E2E TEST: KNIME Workflow Transpilation")
    print("=" * 70)
    
    report["workflow_file"] = str(workflow_path)
    
    # 1. Extract
    print("\n[1/5] Extracting workflow...")
    try:
        extract_dir = extract_knwf(workflow_path) if workflow_path.suffix == ".knwf" else workflow_path
        
        # Find the actual workflow directory (contains workflow.knime at top level)
        # The ZIP has nested structure like: fluxo_knime_exemplo/document/...
        workflow_dir = None
        for wf_file in extract_dir.rglob("workflow.knime"):
            # Get parent directory and check if it's the main workflow (not metanode)
            candidate = wf_file.parent
            # Prefer the shortest path (main workflow, not nested metanodes)
            if workflow_dir is None or len(str(candidate)) < len(str(workflow_dir)):
                workflow_dir = candidate
        
        if workflow_dir is None:
            workflow_dir = extract_dir
            print(f"   ⚠️ No workflow.knime found, using extract root")
        else:
            print(f"   ✅ Found workflow at: {workflow_dir.name}")
        
        report["summary"]["extraction"] = "SUCCESS"
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        report["summary"]["extraction"] = f"FAILED: {e}"
        return report
    
    # 2. Parse
    print("\n[2/5] Parsing workflow...")
    try:
        parser = WorkflowParser()
        workflow_data = parser.parse_workflow(str(workflow_dir))
        nodes = workflow_data.get("nodes", [])
        workflow_name = workflow_data.get("name", "Unknown")
        
        print(f"   ✅ Parsed {len(nodes)} nodes")
        print(f"   ✅ Workflow: {workflow_name}")
        report["summary"]["parsing"] = "SUCCESS"
        report["summary"]["node_count"] = len(nodes)
        report["summary"]["workflow_name"] = workflow_name
    except Exception as e:
        print(f"   ❌ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        report["summary"]["parsing"] = f"FAILED: {e}"
        return report
    
    # 3. Analyze coverage
    print("\n[3/5] Analyzing catalog coverage...")
    template_mapper = TemplateMapper()
    category_stats = defaultdict(int)
    method_stats = defaultdict(int)
    
    for node in nodes:
        analysis = analyze_node(node, template_mapper)
        report["nodes_analyzed"].append(analysis)
        category_stats[analysis.get("category", "Unknown")] += 1
        method_stats[analysis["generation_method"]] += 1
    
    report["catalog_coverage"] = {
        "by_category": dict(category_stats),
        "by_method": dict(method_stats)
    }
    
    deterministic = method_stats.get("deterministic_template", 0) + \
                   method_stats.get("deterministic_parser", 0)
    coverage_pct = (deterministic / len(nodes) * 100) if nodes else 0
    print(f"   ✅ Deterministic: {deterministic}/{len(nodes)} ({coverage_pct:.1f}%)")
    report["summary"]["deterministic_coverage"] = f"{coverage_pct:.1f}%"
    
    # 4. Generate code
    print("\n[4/5] Generating Python code...")
    generator = CodeGeneratorV2(use_llm_fallback=False, enable_validation=False)
    det_count = 0
    stub_count = 0
    
    for node in nodes:
        try:
            node_instance = NodeInstance(
                node_id=str(node.get("id", "0")),
                node_type=node.get("type", "Unknown"),
                factory_class=node.get("factory_class", ""),
                name=node.get("name", "Unknown"),
                settings=node.get("settings", {}),
                category=NodeCategory.UNKNOWN
            )
            code = generator.generate_function(node_instance)
            
            if "DETERMINISTIC" in code:
                det_count += 1
            else:
                stub_count += 1
        except Exception as e:
            stub_count += 1
    
    print(f"   📊 DETERMINISTIC: {det_count}")
    print(f"   ⚠️ STUB/OTHER: {stub_count}")
    
    report["generation_results"]["deterministic"] = det_count
    report["generation_results"]["stub"] = stub_count
    report["summary"]["code_generation"] = "SUCCESS"
    
    # 5. Recommendations
    print("\n[5/5] Generating recommendations...")
    uncovered = [n for n in report["nodes_analyzed"] 
                 if n["generation_method"] == "llm_or_stub"]
    
    if uncovered:
        print(f"   ⚠️ {len(uncovered)} nodes need templates:")
        for node in uncovered[:10]:
            short_factory = node["factory_class"].split(".")[-1]
            print(f"      - {node['name']}: {short_factory}")
            report["recommendations"].append({
                "type": "add_template",
                "node": node["name"],
                "factory": short_factory,
                "priority": "HIGH"
            })
    
    return report


def generate_markdown(report: Dict) -> str:
    """Generate markdown report."""
    md = [
        "# E2E Test Report: KNIME Workflow Transpilation",
        f"\n**Generated:** {report['timestamp']}",
        f"\n**Workflow:** `{Path(report['workflow_file']).name}`",
        "\n---\n## Summary\n",
        "| Metric | Result |",
        "|--------|--------|"
    ]
    
    for key, value in report.get("summary", {}).items():
        status = "✅" if "SUCCESS" in str(value) else "⚠️"
        md.append(f"| {key.replace('_', ' ').title()} | {status} {value} |")
    
    md.extend([
        "\n---\n## Generation Results\n",
        "| Method | Count |",
        "|--------|-------|",
        f"| ✅ DETERMINISTIC | {report['generation_results']['deterministic']} |",
        f"| ⚠️ STUB/OTHER | {report['generation_results']['stub']} |"
    ])
    
    md.extend(["\n---\n## Nodes Analyzed\n",
        "| ID | Name | Factory | Method |",
        "|----|------|---------|--------|"
    ])
    
    for node in report.get("nodes_analyzed", [])[:25]:
        icon = "✅" if "deterministic" in node["generation_method"] else "⚠️"
        factory = node["factory_class"].split(".")[-1][:25]
        md.append(f"| {node['node_id']} | {node['name'][:20]} | {factory} | {icon} {node['generation_method'][:15]} |")
    
    if len(report.get("nodes_analyzed", [])) > 25:
        md.append(f"\n*... and {len(report['nodes_analyzed']) - 25} more*")
    
    md.extend(["\n---\n## Recommendations\n"])
    if report.get("recommendations"):
        for rec in report["recommendations"][:15]:
            md.append(f"- 🔴 **Add template:** `{rec['node']}` (`{rec['factory']}`)")
    else:
        md.append("✅ No critical recommendations!")
    
    return "\n".join(md)


if __name__ == "__main__":
    workflow_path = Path(r"c:\Users\vinic\Documents\Projetos\ChatKnime\fluxo_knime_exemplo.knwf")
    
    if not workflow_path.exists():
        print(f"❌ File not found: {workflow_path}")
        sys.exit(1)
    
    result = run_e2e_test(workflow_path)
    markdown = generate_markdown(result)
    
    report_path = Path("e2e_test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"\n📄 Report saved: {report_path}")
    print("\n" + "=" * 70)
    print(markdown)
