#!/usr/bin/env python
"""
KNIME to Python Transpiler - Standalone Version

Usage:
    python transpile.py <arquivo.knwf>
    
Output:
    Creates <arquivo>.py and <arquivo>_log.md in the same directory
"""
import sys
import zipfile
import tempfile
import json
import xml.etree.ElementTree as ET
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class TranspilerLog:
    """Collects diagnostic information during transpilation."""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.start_time = datetime.now()
        self.extraction_method = ""
        self.nodes_found = []
        self.template_matches = []
        self.fallback_nodes = []
        self.errors = []
        self.warnings = []
        self._unique_warnings = set()  # Track unique warnings
        self.factory_counts = Counter()
        self.settings_samples = []
    
    def add_node(self, name: str, factory: str, matched: bool, template_name: str = ""):
        """Record a processed node."""
        self.nodes_found.append({
            'name': name,
            'factory': factory,
            'matched': matched,
            'template': template_name
        })
        
        simple_factory = factory.split('.')[-1] if factory else 'Unknown'
        self.factory_counts[simple_factory] += 1
        
        if matched:
            self.template_matches.append({'name': name, 'factory': simple_factory, 'template': template_name})
        else:
            self.fallback_nodes.append({'name': name, 'factory': simple_factory})
    
    def add_error(self, msg: str):
        self.errors.append(msg)
    
    def add_warning(self, msg: str):
        # Deduplicate warnings  
        if msg not in self._unique_warnings:
            self._unique_warnings.add(msg)
            self.warnings.append(msg)
    
    def add_settings_sample(self, name: str, factory: str, settings_preview: str):
        """Store a sample of node settings for analysis."""
        if len(self.settings_samples) < 10:  # Keep first 10 samples
            self.settings_samples.append({
                'name': name,
                'factory': factory.split('.')[-1] if factory else 'Unknown',
                'preview': settings_preview[:500]
            })
    
    def generate_markdown(self) -> str:
        """Generate diagnostic markdown log."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        total = len(self.nodes_found)
        matched = len(self.template_matches)
        fallback = len(self.fallback_nodes)
        coverage = (matched / total * 100) if total > 0 else 0
        
        lines = [
            "# Transpilation Log",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Input | `{Path(self.input_path).name}` |",
            f"| Timestamp | {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| Duration | {duration:.2f}s |",
            f"| Method | {self.extraction_method} |",
            f"| Total Nodes | {total} |",
            f"| Template Matches | {matched} |",
            f"| Fallback | {fallback} |",
            f"| Coverage | {coverage:.1f}% |",
            "",
        ]
        
        # Errors section
        if self.errors:
            lines.extend([
                "## Errors",
                "",
            ])
            for err in self.errors:
                lines.append(f"- `{err}`")
            lines.append("")
        
        # Warnings section
        if self.warnings:
            lines.extend([
                "## Warnings",
                "",
            ])
            for warn in self.warnings:
                lines.append(f"- {warn}")
            lines.append("")
        
        # Factory distribution (top 20)
        lines.extend([
            "## Factory Distribution (Top 20)",
            "",
            "| Factory | Count |",
            "|---------|-------|",
        ])
        for factory, count in self.factory_counts.most_common(20):
            lines.append(f"| {factory} | {count} |")
        lines.append("")
        
        # Template matches (first 15)
        if self.template_matches:
            lines.extend([
                "## Template Matches (Sample)",
                "",
                "| Node | Factory | Template |",
                "|------|---------|----------|",
            ])
            for item in self.template_matches[:15]:
                lines.append(f"| {item['name'][:30]} | {item['factory']} | {item.get('template', 'default')[:20]} |")
            if len(self.template_matches) > 15:
                lines.append(f"| ... | +{len(self.template_matches) - 15} more | |")
            lines.append("")
        
        # Fallback nodes (all - important for analysis)
        if self.fallback_nodes:
            lines.extend([
                "## Fallback Nodes (No Template)",
                "",
                "| Node | Factory |",
                "|------|---------|",
            ])
            # Group by factory to reduce noise
            fallback_by_factory = {}
            for item in self.fallback_nodes:
                factory = item['factory']
                if factory not in fallback_by_factory:
                    fallback_by_factory[factory] = []
                fallback_by_factory[factory].append(item['name'])
            
            for factory, names in sorted(fallback_by_factory.items(), key=lambda x: -len(x[1])):
                sample = names[0][:25] if names else ""
                if len(names) > 1:
                    lines.append(f"| {sample}... ({len(names)}x) | {factory} |")
                else:
                    lines.append(f"| {sample} | {factory} |")
            lines.append("")
        
        # Settings samples (for debugging node detection)
        if self.settings_samples:
            lines.extend([
                "## Settings Samples",
                "",
            ])
            for i, sample in enumerate(self.settings_samples[:5]):
                lines.extend([
                    f"### {i+1}. {sample['name']} ({sample['factory']})",
                    "",
                    "```xml",
                    sample['preview'],
                    "```",
                    "",
                ])
        
        # All unique factories (for template development)
        lines.extend([
            "## All Unique Factories",
            "",
            "```",
        ])
        for factory in sorted(set(n.get('factory', '') for n in self.nodes_found if n.get('factory'))):
            lines.append(factory.split('.')[-1])
        lines.extend([
            "```",
            "",
        ])
        
        return '\n'.join(lines)


def extract_knwf(knwf_path: Path, log: TranspilerLog) -> Path:
    """Extract .knwf file to temp directory."""
    try:
        temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(knwf_path, 'r') as zf:
            zf.extractall(temp_dir)
        return temp_dir
    except Exception as e:
        log.add_error(f"Extraction failed: {e}")
        raise


def find_nodes_from_settings(extract_dir: Path, log: TranspilerLog) -> List[Dict[str, Any]]:
    """Find nodes by scanning settings.xml files."""
    nodes = []
    node_id = 0
    
    for settings_file in extract_dir.rglob('settings.xml'):
        try:
            content = settings_file.read_text(encoding='utf-8', errors='ignore')
            
            # Try to find factory class using regex
            factory_match = re.search(r'org\.knime\.[a-zA-Z0-9_.]+NodeFactory', content)
            
            if factory_match:
                factory = factory_match.group(0)
                name = settings_file.parent.name
                name_clean = re.sub(r'\s*\(#\d+\)$', '', name)
                
                nodes.append({
                    'id': node_id,
                    'name': name_clean,
                    'factory': factory,
                    'path': str(settings_file)
                })
                
                # Add settings sample for first few nodes
                log.add_settings_sample(name_clean, factory, content[:500])
                node_id += 1
                
        except Exception as e:
            log.add_warning(f"Error reading {settings_file.name}: {e}")
    
    return nodes


def load_analysis_json(knwf_path: Path, log: TranspilerLog) -> Optional[List[Dict]]:
    """Try to load pre-analyzed workflow_analysis_v2.json."""
    paths_to_try = [
        knwf_path.parent / "workflow_analysis_v2.json",
        Path(__file__).parent.parent / "workflow_analysis_v2.json",
        Path(__file__).parent / "workflow_analysis_v2.json",
    ]
    
    for analysis_path in paths_to_try:
        if analysis_path.exists():
            try:
                with open(analysis_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    nodes = data.get('nodes', [])
                    log.extraction_method = f"JSON ({analysis_path.name})"
                    return nodes
            except Exception as e:
                log.add_warning(f"Failed to load {analysis_path}: {e}")
    
    return None


def get_template(factory: str, log: TranspilerLog) -> Tuple[Optional[str], str]:
    """Get Python template for a KNIME factory. Returns (code, template_name)."""
    template_name = ""
    
    # Try using the full template mapper
    try:
        from app.services.generator.template_mapper import TemplateMapper
        mapper = TemplateMapper()
        if mapper.has_template(factory):
            code = mapper.generate_code(factory, {}, 'df', 'df')
            return code, "TemplateMapper"
    except Exception as e:
        log.add_warning(f"TemplateMapper unavailable: {e}")
    
    # Fallback templates
    TEMPLATES = {
        'DataColumnSpecFilterNodeFactory': ('df = df.copy()', 'ColumnFilter'),
        'RenameNodeFactory': ('df = df.rename(columns={})', 'Rename'),
        'GroupByNodeFactory': ('df = df.groupby([]).agg({}).reset_index()', 'GroupBy'),
        'JoinerNodeFactory': ('df = df.merge(df_right, how="inner")', 'Joiner'),
        'RowFilterNodeFactory': ('df = df[df["col"] == val].copy()', 'RowFilter'),
        'SorterNodeFactory': ('df = df.sort_values(by=[])', 'Sorter'),
        'JEPNodeFactory': ('df["result"] = df["a"] + df["b"]', 'MathFormula'),
        'RuleEngineNodeFactory': ('df["result"] = np.where(df["c"] > 0, "A", "B")', 'RuleEngine'),
        'StringManipulationNodeFactory': ('df["col"] = df["col"].str.upper()', 'StringManip'),
        'CrossJoinerNodeFactory': ('df = df.merge(df2, how="cross")', 'CrossJoin'),
        'ColumnAggregatorNodeFactory': ('df = df.agg("sum")', 'Aggregator'),
        'RoundDoubleNodeFactory': ('df["col"] = df["col"].round(2)', 'Round'),
        'NumberToString2NodeFactory': ('df["col"] = df["col"].astype(str)', 'NumToStr'),
        'CellSplitterNodeFactory': ('df = df["col"].str.split(",", expand=True)', 'CellSplit'),
        'MissingValueNodeFactory': ('df = df.fillna(0)', 'MissingValue'),
        'DuplicateRowFilterNodeFactory': ('df = df.drop_duplicates()', 'DupFilter'),
        'ConcatenateNodeFactory': ('df = pd.concat([df1, df2])', 'Concat'),
        'ColumnResorterNodeFactory': ('df = df[sorted_cols]', 'Resorter'),
        'EmptyTableSwitchNodeFactory': ('pass  # flow control', 'FlowCtrl'),
        'EndifNodeFactory': ('pass  # end if', 'FlowCtrl'),
    }
    
    for key, (template, name) in TEMPLATES.items():
        if key in factory:
            return template, name
    
    return None, ""


def generate_code(nodes: List[Dict], log: TranspilerLog) -> str:
    """Generate Python code from nodes."""
    lines = [
        '"""',
        'Auto-generated Python pipeline from KNIME workflow',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Nodes: {len(nodes)}',
        '"""',
        'import pandas as pd',
        'import numpy as np',
        'from datetime import datetime',
        '',
        '',
        'def run_pipeline(df_input: pd.DataFrame) -> pd.DataFrame:',
        '    """Execute the transpiled KNIME workflow."""',
        '    df = df_input.copy()',
        '',
    ]
    
    for i, node in enumerate(nodes):
        factory = node.get('factory', '')
        name = node.get('name', node.get('node_name', f'Node_{i}'))
        
        template, template_name = get_template(factory, log)
        
        if template:
            log.add_node(name, factory, True, template_name)
            lines.append(f'    # {name}')
            for tline in template.split('\n'):
                if tline.strip():
                    lines.append(f'    {tline}')
            lines.append('')
        else:
            log.add_node(name, factory, False)
            simple = factory.split('.')[-1].replace('NodeFactory', '') if factory else 'Unknown'
            lines.append(f'    # {name} ({simple})')
            lines.append(f'    pass  # TODO: Implement')
            lines.append('')
    
    lines.extend([
        '    return df',
        '',
        '',
        'if __name__ == "__main__":',
        '    print("Pipeline loaded. Call run_pipeline(df) with your DataFrame.")',
    ])
    
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python transpile.py <arquivo.knwf>")
        sys.exit(1)
    
    knwf_path = Path(sys.argv[1]).resolve()
    
    if not knwf_path.exists():
        print(f"Error: File not found: {knwf_path}")
        sys.exit(1)
    
    output_path = knwf_path.with_suffix('.py')
    log_path = knwf_path.parent / f"{knwf_path.stem}_log.md"
    
    # Initialize log
    log = TranspilerLog(str(knwf_path))
    
    print("="*60)
    print("KNIME to Python Transpiler")
    print("="*60)
    print(f"Input:  {knwf_path}")
    print(f"Output: {output_path}")
    print(f"Log:    {log_path}")
    print("="*60)
    
    try:
        # Try pre-analyzed JSON first
        print("\n[1/4] Loading nodes...")
        nodes = load_analysis_json(knwf_path, log)
        
        if nodes:
            print(f"      Found {len(nodes)} nodes (from JSON)")
        else:
            # Extract and analyze
            log.extraction_method = "KNWF Extraction"
            extract_dir = extract_knwf(knwf_path, log)
            nodes = find_nodes_from_settings(extract_dir, log)
            print(f"      Found {len(nodes)} nodes (from extraction)")
            
            # Cleanup
            import shutil
            shutil.rmtree(extract_dir, ignore_errors=True)
        
        if not nodes:
            log.add_error("No nodes found in workflow")
            print("\n[!] No nodes found!")
        else:
            print("[2/4] Generating code...")
            code = generate_code(nodes, log)
            
            print("[3/4] Writing output...")
            output_path.write_text(code, encoding='utf-8')
        
        print("[4/4] Writing log...")
        log_content = log.generate_markdown()
        log_path.write_text(log_content, encoding='utf-8')
        
        # Print summary
        matched = len(log.template_matches)
        fallback = len(log.fallback_nodes)
        total = len(log.nodes_found)
        coverage = (matched / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)
        print(f"Nodes:    {total}")
        print(f"Matched:  {matched}")
        print(f"Fallback: {fallback}")
        print(f"Coverage: {coverage:.1f}%")
        print("="*60)
        print(f"Output:   {output_path}")
        print(f"Log:      {log_path}")
        print("="*60)
        
    except Exception as e:
        log.add_error(str(e))
        log_content = log.generate_markdown()
        log_path.write_text(log_content, encoding='utf-8')
        
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nLog written to: {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
