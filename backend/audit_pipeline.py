#!/usr/bin/env python
"""
KNIME Transpiler Pipeline - Comprehensive Audit Script

Validates all critical stages of the transpilation pipeline:
1. KNWF Extraction
2. Mapping File Reading  
3. Node Interpretation
4. LLM Activation
5. Output Generation

Usage:
    python audit_pipeline.py <arquivo.knwf>
"""
import sys
import zipfile
import tempfile
import shutil
import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class AuditResult:
    """Result of a single audit check."""
    stage: str
    check: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class PipelineAudit:
    """Comprehensive pipeline audit results."""
    input_file: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    results: List[AuditResult] = field(default_factory=list)
    
    def add(self, stage: str, check: str, passed: bool, message: str, **details):
        self.results.append(AuditResult(stage, check, passed, message, details))
    
    @property
    def total_checks(self) -> int:
        return len(self.results)
    
    @property
    def passed_checks(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_checks(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def success_rate(self) -> float:
        return (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0


class PipelineAuditor:
    """Audits all stages of the KNIME transpiler pipeline."""
    
    def __init__(self, knwf_path: Path):
        self.knwf_path = knwf_path
        self.audit = PipelineAudit(input_file=str(knwf_path))
        self.extract_dir = None
        self.nodes = []
        self.output_py = None
        self.output_md = None
    
    def run_full_audit(self) -> PipelineAudit:
        """Execute all audit stages."""
        print("\n" + "="*70)
        print("🔍 KNIME TRANSPILER PIPELINE AUDIT")
        print("="*70)
        print(f"Input: {self.knwf_path}")
        print(f"Time:  {self.audit.timestamp}")
        print("="*70 + "\n")
        
        # Stage 1: Extraction
        self._audit_extraction()
        
        # Stage 2: Mapping Files
        self._audit_mapping_files()
        
        # Stage 3: Node Interpretation
        self._audit_node_interpretation()
        
        # Stage 4: LLM Activation
        self._audit_llm_activation()
        
        # Stage 5: Output Generation
        self._audit_output_generation()
        
        # Cleanup
        if self.extract_dir and self.extract_dir.exists():
            shutil.rmtree(self.extract_dir, ignore_errors=True)
        
        return self.audit
    
    def _audit_extraction(self):
        """Stage 1: Validate KNWF extraction."""
        stage = "1. EXTRACTION"
        print(f"\n{'='*50}")
        print(f"📦 STAGE 1: KNWF EXTRACTION")
        print(f"{'='*50}")
        
        # Check 1.1: File exists
        file_exists = self.knwf_path.exists()
        self.audit.add(stage, "File exists", file_exists,
            f"Input file {'found' if file_exists else 'NOT FOUND'}: {self.knwf_path.name}")
        print(f"  {'✅' if file_exists else '❌'} File exists: {file_exists}")
        
        if not file_exists:
            return
        
        # Check 1.2: Valid ZIP format
        try:
            with zipfile.ZipFile(self.knwf_path, 'r') as zf:
                is_valid_zip = True
                file_list = zf.namelist()
                file_count = len(file_list)
        except zipfile.BadZipFile:
            is_valid_zip = False
            file_count = 0
            file_list = []
        
        self.audit.add(stage, "Valid ZIP format", is_valid_zip,
            f"ZIP format {'valid' if is_valid_zip else 'INVALID'}, {file_count} files",
            file_count=file_count)
        print(f"  {'✅' if is_valid_zip else '❌'} Valid ZIP: {is_valid_zip} ({file_count} files)")
        
        if not is_valid_zip:
            return
        
        # Check 1.3: Contains workflow.knime
        has_workflow = any('workflow.knime' in f for f in file_list)
        self.audit.add(stage, "Contains workflow.knime", has_workflow,
            f"workflow.knime {'found' if has_workflow else 'NOT FOUND'}")
        print(f"  {'✅' if has_workflow else '❌'} Has workflow.knime: {has_workflow}")
        
        # Check 1.4: Contains settings.xml files
        settings_files = [f for f in file_list if f.endswith('settings.xml')]
        has_settings = len(settings_files) > 0
        self.audit.add(stage, "Contains settings.xml", has_settings,
            f"Found {len(settings_files)} settings.xml files",
            settings_count=len(settings_files))
        print(f"  {'✅' if has_settings else '❌'} Settings files: {len(settings_files)}")
        
        # Check 1.5: Extract successfully
        try:
            self.extract_dir = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(self.knwf_path, 'r') as zf:
                zf.extractall(self.extract_dir)
            
            extracted_count = len(list(self.extract_dir.rglob('*')))
            extract_success = True
        except Exception as e:
            extract_success = False
            extracted_count = 0
        
        self.audit.add(stage, "Extraction complete", extract_success,
            f"Extracted {extracted_count} items to temp directory",
            extracted_items=extracted_count)
        print(f"  {'✅' if extract_success else '❌'} Extraction: {extracted_count} items")
        
        # Check 1.6: No corruption (all files readable)
        corrupted = 0
        for f in self.extract_dir.rglob('*'):
            if f.is_file():
                try:
                    _ = f.read_bytes()
                except:
                    corrupted += 1
        
        no_corruption = corrupted == 0
        self.audit.add(stage, "No file corruption", no_corruption,
            f"{'All files readable' if no_corruption else f'{corrupted} files corrupted'}",
            corrupted_files=corrupted)
        print(f"  {'✅' if no_corruption else '❌'} Data integrity: {corrupted} corrupted")
    
    def _audit_mapping_files(self):
        """Stage 2: Validate mapping file reading."""
        stage = "2. MAPPING FILES"
        print(f"\n{'='*50}")
        print(f"📄 STAGE 2: MAPPING FILE ANALYSIS")
        print(f"{'='*50}")
        
        if not self.extract_dir or not self.extract_dir.exists():
            self.audit.add(stage, "Extract dir available", False, "No extraction directory")
            return
        
        # Check 2.1: Find all workflow.knime files
        workflow_files = list(self.extract_dir.rglob('workflow.knime'))
        has_workflows = len(workflow_files) > 0
        self.audit.add(stage, "Workflow files found", has_workflows,
            f"Found {len(workflow_files)} workflow.knime files (main + metanodes)",
            workflow_count=len(workflow_files))
        print(f"  {'✅' if has_workflows else '❌'} Workflow files: {len(workflow_files)}")
        
        # Check 2.2: Find all settings.xml files
        settings_files = list(self.extract_dir.rglob('settings.xml'))
        has_settings = len(settings_files) > 0
        self.audit.add(stage, "Settings files found", has_settings,
            f"Found {len(settings_files)} settings.xml files",
            settings_count=len(settings_files))
        print(f"  {'✅' if has_settings else '❌'} Settings files: {len(settings_files)}")
        
        # Check 2.3: Parse workflow.knime structure
        parsed_workflows = 0
        for wf in workflow_files:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(wf)
                parsed_workflows += 1
            except:
                pass
        
        all_parsed = parsed_workflows == len(workflow_files)
        self.audit.add(stage, "Workflow XML parsing", all_parsed,
            f"Parsed {parsed_workflows}/{len(workflow_files)} workflow files",
            parsed_count=parsed_workflows)
        print(f"  {'✅' if all_parsed else '⚠️'} XML parsing: {parsed_workflows}/{len(workflow_files)}")
        
        # Check 2.4: Identify connections
        connections_found = 0
        for wf in workflow_files:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(wf)
                root = tree.getroot()
                # Look for connection elements
                connections = root.findall('.//{http://www.knime.org/2008/09/XMLConfig}config[@key="connections"]')
                if connections:
                    connections_found += 1
            except:
                pass
        
        has_connections = connections_found > 0
        self.audit.add(stage, "Connection mapping", has_connections,
            f"Found connections in {connections_found}/{len(workflow_files)} workflows",
            connection_files=connections_found)
        print(f"  {'✅' if has_connections else '⚠️'} Connections: {connections_found} workflows")
        
        # Check 2.5: Check for pre-analyzed JSON
        json_path = self.knwf_path.parent / "workflow_analysis_v2.json"
        json_path_alt = Path(__file__).parent.parent / "workflow_analysis_v2.json"
        
        has_json = json_path.exists() or json_path_alt.exists()
        if has_json:
            try:
                path = json_path if json_path.exists() else json_path_alt
                with open(path, 'r') as f:
                    data = json.load(f)
                    node_count = len(data.get('nodes', []))
                json_valid = True
            except:
                json_valid = False
                node_count = 0
        else:
            json_valid = False
            node_count = 0
        
        self.audit.add(stage, "Pre-analyzed JSON", has_json and json_valid,
            f"{'Found' if has_json else 'Not found'} ({node_count} nodes)" if has_json else "Using extraction fallback",
            json_nodes=node_count)
        print(f"  {'✅' if has_json else '⚠️'} Pre-analyzed JSON: {node_count} nodes" if has_json else "  ⚠️ Using extraction")
    
    def _audit_node_interpretation(self):
        """Stage 3: Validate node interpretation."""
        stage = "3. NODE INTERPRETATION"
        print(f"\n{'='*50}")
        print(f"🔧 STAGE 3: NODE INTERPRETATION")
        print(f"{'='*50}")
        
        # Find nodes via settings.xml
        self.nodes = []
        settings_files = list(self.extract_dir.rglob('settings.xml')) if self.extract_dir else []
        
        for sf in settings_files:
            try:
                content = sf.read_text(encoding='utf-8', errors='ignore')
                factory_match = re.search(r'org\.knime\.[a-zA-Z0-9_.]+NodeFactory', content)
                if factory_match:
                    self.nodes.append({
                        'factory': factory_match.group(0),
                        'name': sf.parent.name,
                        'path': str(sf)
                    })
            except:
                pass
        
        # Fallback to JSON if available
        json_path = self.knwf_path.parent / "workflow_analysis_v2.json"
        json_path_alt = Path(__file__).parent.parent / "workflow_analysis_v2.json"
        
        if json_path.exists() or json_path_alt.exists():
            try:
                path = json_path if json_path.exists() else json_path_alt
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.nodes = data.get('nodes', [])
            except:
                pass
        
        # Check 3.1: Nodes detected
        has_nodes = len(self.nodes) > 0
        self.audit.add(stage, "Nodes detected", has_nodes,
            f"Found {len(self.nodes)} nodes",
            node_count=len(self.nodes))
        print(f"  {'✅' if has_nodes else '❌'} Nodes found: {len(self.nodes)}")
        
        if not has_nodes:
            return
        
        # Check 3.2: Factory classes identified
        factories = set(n.get('factory', '') for n in self.nodes if n.get('factory'))
        unique_factories = len(factories)
        self.audit.add(stage, "Factory identification", unique_factories > 0,
            f"Identified {unique_factories} unique factory types",
            unique_factories=unique_factories)
        print(f"  {'✅' if unique_factories > 0 else '❌'} Unique factories: {unique_factories}")
        
        # Check 3.3: Template matches
        template_patterns = [
            'DataColumnSpecFilterNodeFactory', 'RenameNodeFactory', 'GroupByNodeFactory',
            'JoinerNodeFactory', 'RowFilterNodeFactory', 'SorterNodeFactory',
            'JEPNodeFactory', 'RuleEngineNodeFactory', 'StringManipulationNodeFactory',
            'CrossJoinerNodeFactory', 'ColumnAggregatorNodeFactory', 'RoundDoubleNodeFactory'
        ]
        
        matches = 0
        for node in self.nodes:
            factory = node.get('factory', '')
            if any(p in factory for p in template_patterns):
                matches += 1
        
        match_rate = (matches / len(self.nodes) * 100) if self.nodes else 0
        self.audit.add(stage, "Template coverage", match_rate > 0,
            f"{matches}/{len(self.nodes)} nodes match templates ({match_rate:.1f}%)",
            template_matches=matches, match_rate=match_rate)
        print(f"  {'✅' if match_rate > 40 else '⚠️'} Template coverage: {match_rate:.1f}%")
        
        # Check 3.4: Node categories
        categories = {
            'IO': ['Reader', 'Writer', 'Loader'],
            'Transform': ['Filter', 'Sorter', 'Rename', 'Joiner'],
            'Manipulation': ['Math', 'String', 'Rule', 'Aggregator'],
            'Flow': ['Loop', 'If', 'Switch'],
            'Database': ['DB', 'Database', 'JDBC', 'BigQuery']
        }
        
        category_counts = {cat: 0 for cat in categories}
        for node in self.nodes:
            factory = node.get('factory', '')
            for cat, patterns in categories.items():
                if any(p in factory for p in patterns):
                    category_counts[cat] += 1
                    break
        
        categorized = sum(category_counts.values())
        self.audit.add(stage, "Node categorization", categorized > 0,
            f"Categorized {categorized} nodes across {len([c for c in category_counts.values() if c > 0])} categories",
            categories=category_counts)
        print(f"  {'✅' if categorized > 0 else '⚠️'} Categorized: {category_counts}")
    
    def _audit_llm_activation(self):
        """Stage 4: Validate LLM module activation."""
        stage = "4. LLM ACTIVATION"
        print(f"\n{'='*50}")
        print(f"🤖 STAGE 4: LLM MODULE")
        print(f"{'='*50}")
        
        # Check 4.1: LLM module exists
        llm_path = Path(__file__).parent / "app" / "services" / "generator" / "llm_generator.py"
        llm_exists = llm_path.exists()
        self.audit.add(stage, "LLM module exists", llm_exists,
            f"LLM generator {'found' if llm_exists else 'NOT FOUND'}")
        print(f"  {'✅' if llm_exists else '❌'} LLM module: {llm_exists}")
        
        # Check 4.2: LLM can be imported
        try:
            from app.services.generator.llm_generator import LLMGenerator
            llm_importable = True
        except ImportError as e:
            llm_importable = False
            import_error = str(e)
        
        self.audit.add(stage, "LLM importable", llm_importable,
            f"LLMGenerator {'can be imported' if llm_importable else 'import failed'}")
        print(f"  {'✅' if llm_importable else '❌'} Import: {llm_importable}")
        
        # Check 4.3: Environment variables
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
        location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
        has_config = bool(project_id)
        self.audit.add(stage, "GCP configuration", has_config,
            f"Project: {'set' if project_id else 'NOT SET'}, Location: {location}",
            project_set=bool(project_id), location=location)
        print(f"  {'✅' if has_config else '⚠️'} GCP config: project={'set' if project_id else 'NOT SET'}")
        
        # Check 4.4: LLM initialization
        if llm_importable:
            try:
                llm = LLMGenerator()
                llm_available = llm.is_available()
            except Exception as e:
                llm_available = False
        else:
            llm_available = False
        
        self.audit.add(stage, "LLM available", llm_available,
            f"LLM {'ready for inference' if llm_available else 'not available'}")
        print(f"  {'✅' if llm_available else '⚠️'} LLM ready: {llm_available}")
        
        # Check 4.5: Behavior catalog
        catalog_path = Path(__file__).parent / "app" / "services" / "generator" / "node_behavior_catalog.py"
        catalog_exists = catalog_path.exists()
        self.audit.add(stage, "Behavior catalog", catalog_exists,
            f"Node behavior catalog {'found' if catalog_exists else 'NOT FOUND'}")
        print(f"  {'✅' if catalog_exists else '⚠️'} Behavior catalog: {catalog_exists}")
    
    def _audit_output_generation(self):
        """Stage 5: Validate output file generation."""
        stage = "5. OUTPUT GENERATION"
        print(f"\n{'='*50}")
        print(f"📝 STAGE 5: OUTPUT GENERATION")
        print(f"{'='*50}")
        
        # Run transpile.py
        output_py = self.knwf_path.with_suffix('.py')
        output_log = self.knwf_path.parent / f"{self.knwf_path.stem}_log.md"
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, 'transpile.py', str(self.knwf_path)],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=120
            )
            transpile_success = result.returncode == 0
            transpile_output = result.stdout + result.stderr
        except Exception as e:
            transpile_success = False
            transpile_output = str(e)
        
        self.audit.add(stage, "Transpilation executed", transpile_success,
            f"transpile.py {'completed' if transpile_success else 'FAILED'}",
            output_preview=transpile_output[:500])
        print(f"  {'✅' if transpile_success else '❌'} Transpilation: {'OK' if transpile_success else 'FAILED'}")
        
        # Check 5.1: Python file created
        py_exists = output_py.exists()
        py_size = output_py.stat().st_size if py_exists else 0
        self.audit.add(stage, "Python file created", py_exists,
            f"{output_py.name}: {'created' if py_exists else 'NOT CREATED'} ({py_size} bytes)",
            file_size=py_size)
        print(f"  {'✅' if py_exists else '❌'} {output_py.name}: {py_size} bytes")
        
        # Check 5.2: Log file created  
        log_exists = output_log.exists()
        log_size = output_log.stat().st_size if log_exists else 0
        self.audit.add(stage, "Log file created", log_exists,
            f"{output_log.name}: {'created' if log_exists else 'NOT CREATED'} ({log_size} bytes)",
            file_size=log_size)
        print(f"  {'✅' if log_exists else '❌'} {output_log.name}: {log_size} bytes")
        
        if py_exists:
            # Check 5.3: Python syntax valid
            try:
                import ast
                code = output_py.read_text(encoding='utf-8')
                ast.parse(code)
                syntax_valid = True
            except SyntaxError as e:
                syntax_valid = False
                syntax_error = str(e)
            
            self.audit.add(stage, "Python syntax valid", syntax_valid,
                f"Syntax {'valid' if syntax_valid else 'INVALID'}")
            print(f"  {'✅' if syntax_valid else '❌'} Syntax: {'valid' if syntax_valid else 'INVALID'}")
            
            # Check 5.4: Contains expected structure
            has_imports = 'import pandas' in code
            has_function = 'def run_pipeline' in code
            has_structure = has_imports and has_function
            self.audit.add(stage, "Code structure", has_structure,
                f"Imports: {has_imports}, Function: {has_function}")
            print(f"  {'✅' if has_structure else '❌'} Structure: imports={has_imports}, function={has_function}")
    
    def generate_report(self) -> str:
        """Generate markdown audit report."""
        lines = [
            "# Pipeline Audit Report",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Input | `{Path(self.audit.input_file).name}` |",
            f"| Timestamp | {self.audit.timestamp} |",
            f"| Total Checks | {self.audit.total_checks} |",
            f"| Passed | {self.audit.passed_checks} |",
            f"| Failed | {self.audit.failed_checks} |",
            f"| Success Rate | {self.audit.success_rate:.1f}% |",
            "",
        ]
        
        # Group by stage
        stages = {}
        for r in self.audit.results:
            if r.stage not in stages:
                stages[r.stage] = []
            stages[r.stage].append(r)
        
        for stage, results in stages.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
            
            lines.extend([
                f"## {status} {stage}",
                "",
                "| Check | Status | Details |",
                "|-------|--------|---------|",
            ])
            
            for r in results:
                status = "✅" if r.passed else "❌"
                lines.append(f"| {r.check} | {status} | {r.message} |")
            
            lines.append("")
        
        # Recommendations
        failed = [r for r in self.audit.results if not r.passed]
        if failed:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for r in failed:
                lines.append(f"- **{r.check}**: {r.message}")
            lines.append("")
        
        return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_pipeline.py <arquivo.knwf>")
        sys.exit(1)
    
    knwf_path = Path(sys.argv[1]).resolve()
    
    if not knwf_path.exists():
        print(f"Error: File not found: {knwf_path}")
        sys.exit(1)
    
    # Run audit
    auditor = PipelineAuditor(knwf_path)
    audit = auditor.run_full_audit()
    
    # Generate report
    report = auditor.generate_report()
    report_path = knwf_path.parent / f"{knwf_path.stem}_audit.md"
    report_path.write_text(report, encoding='utf-8')
    
    # Print summary
    print("\n" + "="*70)
    print("📊 AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks:  {audit.total_checks}")
    print(f"Passed:        {audit.passed_checks}")
    print(f"Failed:        {audit.failed_checks}")
    print(f"Success Rate:  {audit.success_rate:.1f}%")
    print("="*70)
    print(f"Report: {report_path}")
    print("="*70)
    
    sys.exit(0 if audit.failed_checks == 0 else 1)


if __name__ == "__main__":
    main()
