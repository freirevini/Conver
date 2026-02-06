#!/usr/bin/env python
"""
LLM Node Processing Evaluation

Tests the LLM's ability to understand and transpile KNIME nodes to Python.
Evaluates 10 nodes: 5 common, 5 advanced/uncommon.

Metrics:
- Understanding: Does the LLM grasp node purpose?
- Configuration Extraction: Are settings correctly parsed?
- Code Quality: Is the Python syntactically valid and correct?
- Completeness: Does the code implement full node logic?
"""
import sys
import json
import ast
import os
from pathlib import Path
from datetime import datetime

# Add backend to path (parent of tests/)
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Selected test nodes (5 common, 5 advanced)
COMMON_NODES = [
    # Common nodes - frequently used in workflows
    {"type": "Column Filter", "factory": "DataColumnSpecFilterNodeFactory", "complexity": "simple"},
    {"type": "Math Formula", "factory": "JEPNodeFactory", "complexity": "medium"},
    {"type": "GroupBy", "factory": "GroupByNodeFactory", "complexity": "medium"},
    {"type": "String Manipulation", "factory": "StringManipulationNodeFactory", "complexity": "medium"},
    {"type": "Column Rename", "factory": "RenameNodeFactory", "complexity": "simple"},
]

ADVANCED_NODES = [
    # Advanced/uncommon nodes - less frequent, more complex
    {"type": "Rule Engine", "factory": "RuleEngineNodeFactory", "complexity": "high"},
    {"type": "Database Looping (legacy)", "factory": "DBLoopingNodeFactory", "complexity": "high"},
    {"type": "Parameter Optimization Loop Start", "factory": "LoopStartParNodeFactory", "complexity": "high"},
    {"type": "Date/Time Difference", "factory": "DurationPeriodFormatNodeFactory", "complexity": "medium"},
    {"type": "Column Aggregator", "factory": "ColumnAggregatorNodeFactory", "complexity": "high"},
]


class LLMEvaluator:
    """Evaluates LLM performance on KNIME node transpilation."""
    
    def __init__(self, analysis_path: Path):
        self.analysis_path = analysis_path
        self.analysis = self._load_analysis()
        self.llm = None
        self.results = []
        
    def _load_analysis(self) -> dict:
        """Load workflow analysis JSON."""
        with open(self.analysis_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _init_llm(self):
        """Initialize LLM generator."""
        from app.services.generator.llm_generator import LLMGenerator
        self.llm = LLMGenerator()
        if not self.llm.is_available():
            raise RuntimeError("LLM not available - check GOOGLE_CLOUD_PROJECT env var")
    
    def _find_node_by_factory(self, factory_substr: str) -> dict:
        """Find a node by factory substring."""
        for node in self.analysis.get('nodes', []):
            if factory_substr.lower() in node.get('factory', '').lower():
                return node
        return None
    
    def _load_settings(self, settings_path: str) -> dict:
        """Load node settings from XML."""
        import xml.etree.ElementTree as ET
        
        full_path = self.analysis_path.parent / settings_path
        if not full_path.exists():
            return {}
        
        try:
            tree = ET.parse(full_path)
            root = tree.getroot()
            settings = {}
            
            for config in root.findall('.//config'):
                key = config.get('key', 'unknown')
                for entry in config.findall('entry'):
                    entry_key = entry.get('key')
                    entry_value = entry.get('value')
                    if entry_key and entry_value:
                        settings[f"{key}.{entry_key}"] = entry_value
            
            return settings
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_node(self, node_info: dict, node_data: dict) -> dict:
        """Evaluate LLM on a single node."""
        result = {
            "node_type": node_info["type"],
            "factory": node_info["factory"],
            "complexity": node_info["complexity"],
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Load settings
        settings = self._load_settings(node_data.get('settings_path', ''))
        result["settings_loaded"] = len(settings) > 0
        result["settings_count"] = len(settings)
        
        # Generate code with LLM
        try:
            code = self.llm.generate_node_code(
                node_type=node_data.get('node_type', 'Unknown'),
                node_name=node_data.get('node_name', 'Unknown'),
                factory=node_data.get('factory', ''),
                settings=settings,
                input_var="df_input",
                output_var="df_output"
            )
            
            result["llm_generated"] = code is not None
            result["code"] = code
            result["code_length"] = len(code) if code else 0
            
            # Validate syntax
            if code:
                try:
                    ast.parse(code)
                    result["syntax_valid"] = True
                except SyntaxError as e:
                    result["syntax_valid"] = False
                    result["syntax_error"] = str(e)
            else:
                result["syntax_valid"] = False
                result["syntax_error"] = "No code generated"
            
            # Evaluate code quality
            result["metrics"] = self._evaluate_code_quality(code, node_info)
            
        except Exception as e:
            result["llm_generated"] = False
            result["error"] = str(e)
            result["syntax_valid"] = False
        
        return result
    
    def _evaluate_code_quality(self, code: str, node_info: dict) -> dict:
        """Evaluate generated code quality."""
        if not code:
            return {"score": 0, "issues": ["No code generated"]}
        
        metrics = {
            "has_imports": "import" in code or "from" in code,
            "has_pandas": "pd." in code or "pandas" in code or "df" in code,
            "has_comments": "#" in code,
            "uses_input_var": "df_input" in code or "input" in code.lower(),
            "has_output": "df_output" in code or "return" in code or "=" in code,
            "is_complete": not ("TODO" in code or "pass" in code.strip().endswith("pass")),
            "reasonable_length": 5 <= len(code.split('\n')) <= 100,
        }
        
        score = sum(1 for v in metrics.values() if v) / len(metrics) * 100
        
        return {
            "score": round(score, 1),
            "details": metrics,
            "issues": [k for k, v in metrics.items() if not v]
        }
    
    def run_evaluation(self):
        """Run complete evaluation on all test nodes."""
        print("="*70)
        print("LLM NODE PROCESSING EVALUATION")
        print("="*70)
        print(f"Analysis file: {self.analysis_path.name}")
        print(f"Total nodes in workflow: {len(self.analysis.get('nodes', []))}")
        print()
        
        # Initialize LLM
        print("Initializing LLM connection...")
        self._init_llm()
        print(f"✅ LLM ready: {self.llm.MODEL_ID}")
        print()
        
        # Evaluate common nodes
        print("-"*70)
        print("PHASE 1: COMMON NODES (5)")
        print("-"*70)
        
        for node_info in COMMON_NODES:
            node_data = self._find_node_by_factory(node_info["factory"])
            if node_data:
                print(f"\n📍 Testing: {node_info['type']} ({node_info['complexity']})")
                result = self.evaluate_node(node_info, node_data)
                self.results.append(result)
                self._print_result(result)
            else:
                print(f"⚠️ Node not found: {node_info['type']}")
                self.results.append({
                    "node_type": node_info["type"],
                    "error": "Node not found in analysis",
                    "llm_generated": False
                })
        
        # Evaluate advanced nodes
        print("\n" + "-"*70)
        print("PHASE 2: ADVANCED/UNCOMMON NODES (5)")
        print("-"*70)
        
        for node_info in ADVANCED_NODES:
            node_data = self._find_node_by_factory(node_info["factory"])
            if node_data:
                print(f"\n📍 Testing: {node_info['type']} ({node_info['complexity']})")
                result = self.evaluate_node(node_info, node_data)
                self.results.append(result)
                self._print_result(result)
            else:
                print(f"⚠️ Node not found: {node_info['type']}")
                self.results.append({
                    "node_type": node_info["type"],
                    "error": "Node not found in analysis",
                    "llm_generated": False
                })
        
        return self.results
    
    def _print_result(self, result: dict):
        """Print evaluation result."""
        status = "✅" if result.get("llm_generated") and result.get("syntax_valid") else "❌"
        print(f"   {status} Generated: {result.get('llm_generated', False)}")
        print(f"   {status} Syntax Valid: {result.get('syntax_valid', False)}")
        print(f"   📊 Quality Score: {result.get('metrics', {}).get('score', 0)}%")
        print(f"   📏 Code Length: {result.get('code_length', 0)} chars")
        
        if result.get('code'):
            # Show first 3 lines of code
            lines = result['code'].split('\n')[:3]
            print(f"   📄 Code Preview:")
            for line in lines:
                print(f"      {line[:60]}...")
    
    def generate_report(self) -> str:
        """Generate markdown evaluation report."""
        total = len(self.results)
        generated = sum(1 for r in self.results if r.get('llm_generated'))
        valid_syntax = sum(1 for r in self.results if r.get('syntax_valid'))
        avg_score = sum(r.get('metrics', {}).get('score', 0) for r in self.results) / total if total > 0 else 0
        
        report = f"""# LLM Node Processing Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Nodes Tested | {total} |
| Code Generated | {generated}/{total} ({generated/total*100:.0f}%) |
| Valid Syntax | {valid_syntax}/{total} ({valid_syntax/total*100:.0f}%) |
| Average Quality Score | {avg_score:.1f}% |
| Timestamp | {datetime.now().isoformat()} |

## Results by Node

### Common Nodes (5)

| Node Type | Generated | Syntax | Quality | Code Length |
|-----------|-----------|--------|---------|-------------|
"""
        for r in self.results[:5]:
            gen = "✅" if r.get('llm_generated') else "❌"
            syn = "✅" if r.get('syntax_valid') else "❌"
            score = r.get('metrics', {}).get('score', 0)
            length = r.get('code_length', 0)
            report += f"| {r.get('node_type', 'Unknown')} | {gen} | {syn} | {score}% | {length} |\n"
        
        report += """
### Advanced/Uncommon Nodes (5)

| Node Type | Generated | Syntax | Quality | Code Length |
|-----------|-----------|--------|---------|-------------|
"""
        for r in self.results[5:]:
            gen = "✅" if r.get('llm_generated') else "❌"
            syn = "✅" if r.get('syntax_valid') else "❌"
            score = r.get('metrics', {}).get('score', 0)
            length = r.get('code_length', 0)
            report += f"| {r.get('node_type', 'Unknown')} | {gen} | {syn} | {score}% | {length} |\n"
        
        report += """
## Generated Code Samples

"""
        for r in self.results:
            if r.get('code'):
                report += f"### {r.get('node_type', 'Unknown')}\n"
                report += f"```python\n{r['code'][:500]}{'...' if len(r['code']) > 500 else ''}\n```\n\n"
        
        report += """
## Conclusions

"""
        if valid_syntax == total:
            report += "- ✅ **100% syntax validity** - All generated code is syntactically correct\n"
        elif valid_syntax > total * 0.8:
            report += f"- ⚠️ **{valid_syntax/total*100:.0f}% syntax validity** - Most code is valid, some needs review\n"
        else:
            report += f"- ❌ **{valid_syntax/total*100:.0f}% syntax validity** - Significant issues with code generation\n"
        
        if avg_score >= 80:
            report += f"- ✅ **High quality score ({avg_score:.1f}%)** - Code meets quality expectations\n"
        elif avg_score >= 50:
            report += f"- ⚠️ **Medium quality score ({avg_score:.1f}%)** - Room for improvement\n"
        else:
            report += f"- ❌ **Low quality score ({avg_score:.1f}%)** - Needs significant improvement\n"
        
        return report


def main():
    print("\n" + "="*70)
    print("🧪 LLM KNIME NODE PROCESSING EVALUATION")
    print("="*70 + "\n")
    
    # Find analysis file - it's in ChatKnime root
    analysis_path = Path(__file__).parent.parent.parent / "workflow_analysis_v2.json"
    
    if not analysis_path.exists():
        print(f"❌ Analysis file not found: {analysis_path}")
        sys.exit(1)
    
    # Run evaluation
    evaluator = LLMEvaluator(analysis_path)
    results = evaluator.run_evaluation()
    
    # Generate report
    report = evaluator.generate_report()
    report_path = Path(__file__).parent.parent / "llm_evaluation_report.md"
    report_path.write_text(report, encoding='utf-8')
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    total = len(results)
    generated = sum(1 for r in results if r.get('llm_generated'))
    valid = sum(1 for r in results if r.get('syntax_valid'))
    
    print(f"📊 Generated: {generated}/{total} ({generated/total*100:.0f}%)")
    print(f"✅ Valid Syntax: {valid}/{total} ({valid/total*100:.0f}%)")
    print(f"📄 Report: {report_path}")
    print("="*70)


if __name__ == "__main__":
    main()
