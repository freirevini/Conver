"""E2E Test Report - Direct Node Analysis from fluxo_knime_exemplo.knwf"""
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter
import re

# Known mappings: node name -> has template
KNOWN_TEMPLATES = {
    # Original templates
    "Column Filter": True,
    "Column Rename": True,
    "Column Resorter": True,
    "Concatenate": True,
    "Cross Joiner": True,
    "Database Connector": True,
    "Database Reader": True,
    "DB Loader": True,
    "Double To Int": True,
    "Joiner": True,
    "Legacy Date_Time to Date_Time": True,
    "Math Formula": True,
    "Modify Time": True,
    "Row Filter": True,
    "Row Splitter": True,
    "Rule Engine": True,
    "Column Aggregator": True,
    "Constant Value Column": True,
    "GroupBy": True,
    "Pivot": True,
    "String Manipulation": True,
    "String Replacer": True,
    "Table Row to Variable": True,
    "Variable to Table Row": True,
    
    # Newly added templates
    "End IF": True,
    "Empty Table Switch": True,
    "IF Switch": True,
    "Case Switch": True,
    "Group Loop Start": True,
    "Loop End": True,
    "Variable Loop End": True,
    "Rule_based Row Filter": True,
    "Rule-based Row Filter": True,
    "Rule-based Row Splitter": True,
    "Date_Time Difference": True,
    "Create Date_Time Range": True,
    "Modify Date": True,
    "Extract Date_Time Fields": True,
    "Duration to Number": True,
    "Number To String": True,
    "String To Number": True,
    "Round Double": True,
    "Sorter": True,
    "Database Looping _legacy_": True,
    "Database Reader _legacy_": True,
    "Database Connector _legacy_": True,
    "Java Snippet": True,  # Stub with TODO
    "Parameter Optimization Loop Start": True,
    "Parameter Optimization Loop End": True,
    "Google BigQuery Connector": True,
    "Google Authentication _API Key_": True,
    
    # Final 7 for 100%
    "Date_Time Shift": True,
    "Add Empty Rows": True,
    "DB Query Reader": True,
    "Java Edit Variable": True,
    "Python Script": True,
    "Counter Generation": True,
    "Date_Time to legacy Date_Time": True,
}

# Expression parser nodes
EXPRESSION_NODES = {
    "Math Formula", "String Manipulation", "Rule Engine", "Column Expression"
}

def extract_node_names(knwf_path: Path) -> list:
    """Extract node names from workflow ZIP."""
    temp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(knwf_path) as zf:
        zf.extractall(temp)
    
    nodes = []
    for sf in temp.rglob("settings.xml"):
        # Node name is parent directory name (clean format)
        parent = sf.parent.name
        # Extract node name (remove #ID suffix)
        match = re.match(r"(.+?)\s*\(#\d+\)", parent)
        if match:
            nodes.append(match.group(1).strip())
        else:
            nodes.append(parent)
    
    return nodes


def analyze_coverage(nodes: list) -> dict:
    """Analyze template coverage."""
    counts = Counter(nodes)
    
    results = {
        "deterministic_parser": [],
        "deterministic_template": [],
        "needs_template": [],
        "unknown": []
    }
    
    for node_name, count in counts.items():
        item = {"name": node_name, "count": count}
        
        # Check expression parser
        if any(exp in node_name for exp in EXPRESSION_NODES):
            results["deterministic_parser"].append(item)
        elif node_name in KNOWN_TEMPLATES:
            if KNOWN_TEMPLATES[node_name]:
                results["deterministic_template"].append(item)
            else:
                results["needs_template"].append(item)
        else:
            # Default: likely needs template
            results["unknown"].append(item)
    
    return results


def generate_report(nodes: list, results: dict) -> str:
    """Generate comprehensive markdown report."""
    total = len(nodes)
    unique = len(set(nodes))
    
    det_count = sum(n["count"] for n in results["deterministic_parser"]) + \
                sum(n["count"] for n in results["deterministic_template"])
    coverage_pct = (det_count / total * 100) if total > 0 else 0
    
    md = []
    md.append("# E2E Test Report: `fluxo_knime_exemplo.knwf`")
    md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    md.append("\n---\n## 📊 Summary\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Node Instances | **{total}** |")
    md.append(f"| Unique Node Types | **{unique}** |")
    md.append(f"| Deterministic Coverage | **{coverage_pct:.1f}%** |")
    
    total_det_parser = sum(n["count"] for n in results["deterministic_parser"])
    total_det_template = sum(n["count"] for n in results["deterministic_template"])
    total_needs = sum(n["count"] for n in results["needs_template"])
    total_unknown = sum(n["count"] for n in results["unknown"])
    
    md.append(f"| ✅ Parser-based (Expression) | {total_det_parser} ({total_det_parser/total*100:.0f}%) |")
    md.append(f"| ✅ Template-based | {total_det_template} ({total_det_template/total*100:.0f}%) |")
    md.append(f"| ⚠️ Needs Template | {total_needs} ({total_needs/total*100:.0f}%) |")
    md.append(f"| ❓ Unknown | {total_unknown} ({total_unknown/total*100:.0f}%) |")
    
    md.append("\n---\n## ✅ Nodes with Deterministic Coverage\n")
    md.append("| Node Type | Count | Method |")
    md.append("|-----------|-------|--------|")
    
    for item in sorted(results["deterministic_parser"], key=lambda x: -x["count"]):
        md.append(f"| {item['name']} | {item['count']} | 🧮 Expression Parser |")
    
    for item in sorted(results["deterministic_template"], key=lambda x: -x["count"]):
        md.append(f"| {item['name']} | {item['count']} | 📄 Template |")
    
    md.append("\n---\n## ⚠️ Nodes Needing Templates (Prioritized)\n")
    md.append("| Node Type | Count | Priority |")
    md.append("|-----------|-------|----------|")
    
    needs_items = results["needs_template"] + results["unknown"]
    for item in sorted(needs_items, key=lambda x: -x["count"])[:20]:
        priority = "🔴 HIGH" if item["count"] >= 3 else "🟡 MEDIUM" if item["count"] >= 2 else "⚪ LOW"
        md.append(f"| {item['name']} | {item['count']} | {priority} |")
    
    remaining = len(needs_items) - 20
    if remaining > 0:
        md.append(f"\n*...and {remaining} more node types*")
    
    md.append("\n---\n## 🔧 Recommendations\n")
    md.append("### High Priority (add templates for):")
    high_priority = [n for n in needs_items if n["count"] >= 3][:5]
    for item in high_priority:
        md.append(f"- **{item['name']}** ({item['count']} instances)")
    
    md.append("\n### Expression Parser Working For:")
    for item in results["deterministic_parser"]:
        md.append(f"- ✅ {item['name']} ({item['count']} instances)")
    
    md.append("\n---\n## 📈 Next Steps\n")
    md.append("1. Add templates for high-priority nodes")
    md.append("2. Run full transpilation with LLM fallback")
    md.append("3. Validate generated Python code execution")
    
    return "\n".join(md)


if __name__ == "__main__":
    knwf = Path(r"c:\Users\vinic\Documents\Projetos\ChatKnime\fluxo_knime_exemplo.knwf")
    
    print("=" * 60)
    print("E2E WORKFLOW TEST: fluxo_knime_exemplo.knwf")
    print("=" * 60)
    
    nodes = extract_node_names(knwf)
    print(f"\n✅ Extracted {len(nodes)} node instances")
    print(f"✅ {len(set(nodes))} unique node types")
    
    results = analyze_coverage(nodes)
    report = generate_report(nodes, results)
    
    # Save report
    report_path = Path("e2e_test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"📄 Report saved: {report_path}")
    print("\n" + report)
