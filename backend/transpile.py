#!/usr/bin/env python
"""
KNIME to Python Transpiler - Standalone Minimal Version

NÃO requer instalação de dependências externas.
Usa apenas bibliotecas padrão do Python.

Usage:
    python transpile.py arquivo.knwf
    
Output:
    - arquivo.py     (código Python gerado)
    - arquivo_log.md (log de diagnóstico)
"""
import sys
import os
import zipfile
import tempfile
import shutil
import json
import re
from datetime import datetime
from pathlib import Path
from collections import Counter


class TranspilerLog:
    """Collects diagnostic information."""
    
    def __init__(self, input_path):
        self.input_path = input_path
        self.start_time = datetime.now()
        self.extraction_method = ""
        self.nodes_found = []
        self.template_matches = []
        self.fallback_nodes = []
        self.errors = []
        self.warnings = []
        self._unique_warnings = set()
        self.factory_counts = Counter()
    
    def add_node(self, name, factory, matched, template_name=""):
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
    
    def add_error(self, msg):
        self.errors.append(msg)
    
    def add_warning(self, msg):
        if msg not in self._unique_warnings:
            self._unique_warnings.add(msg)
            self.warnings.append(msg)
    
    def generate_markdown(self):
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
            "| Metric | Value |",
            "|--------|-------|",
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
        
        if self.errors:
            lines.extend(["## Errors", ""])
            for err in self.errors:
                lines.append(f"- `{err}`")
            lines.append("")
        
        if self.warnings:
            lines.extend(["## Warnings", ""])
            for warn in self.warnings:
                lines.append(f"- {warn}")
            lines.append("")
        
        lines.extend([
            "## Factory Distribution (Top 20)",
            "",
            "| Factory | Count |",
            "|---------|-------|",
        ])
        for factory, count in self.factory_counts.most_common(20):
            lines.append(f"| {factory} | {count} |")
        lines.append("")
        
        if self.fallback_nodes:
            lines.extend([
                "## Fallback Nodes (No Template)",
                "",
                "| Node | Factory |",
                "|------|---------|",
            ])
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
        
        lines.extend([
            "## All Unique Factories",
            "",
            "```",
        ])
        for factory in sorted(set(n.get('factory', '') for n in self.nodes_found if n.get('factory'))):
            lines.append(factory.split('.')[-1])
        lines.extend(["```", ""])
        
        return '\n'.join(lines)


def extract_knwf(knwf_path, log):
    """Extract .knwf to temp directory."""
    try:
        temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(knwf_path, 'r') as zf:
            zf.extractall(temp_dir)
        return temp_dir
    except Exception as e:
        log.add_error(f"Extraction failed: {e}")
        raise


def find_nodes_from_settings(extract_dir, log):
    """Find nodes by scanning settings.xml files."""
    nodes = []
    node_id = 0
    
    for settings_file in extract_dir.rglob('settings.xml'):
        try:
            content = settings_file.read_text(encoding='utf-8', errors='ignore')
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
                node_id += 1
                
        except Exception as e:
            log.add_warning(f"Error reading {settings_file.name}: {e}")
    
    return nodes


def load_analysis_json(knwf_path, log):
    """Try to load pre-analyzed JSON."""
    paths_to_try = [
        knwf_path.parent / "workflow_analysis_v2.json",
        Path(__file__).parent / "workflow_analysis_v2.json",
    ]
    
    for path in paths_to_try:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    nodes = data.get('nodes', [])
                    log.extraction_method = f"JSON ({path.name})"
                    return nodes
            except Exception as e:
                log.add_warning(f"Failed to load {path}: {e}")
    
    return None


# Template mappings (built-in, no external dependencies)
TEMPLATES = {
    # Column Operations
    'DataColumnSpecFilterNodeFactory': 'df = df.copy()  # Column Filter',
    'ColumnFilterNodeFactory': 'df = df.copy()  # Column Filter',
    'RenameNodeFactory': 'df = df.rename(columns={})  # Rename columns',
    'ColumnResorterNodeFactory': 'df = df[sorted_columns]  # Reorder columns',
    
    # Data Transformation
    'GroupByNodeFactory': 'df = df.groupby([]).agg({}).reset_index()',
    'SorterNodeFactory': 'df = df.sort_values(by=[])',
    'RowFilterNodeFactory': 'df = df[condition].copy()  # Row Filter',
    'DuplicateRowFilterNodeFactory': 'df = df.drop_duplicates()',
    
    # Joins
    'JoinerNodeFactory': 'df = df.merge(df_right, how="inner", on=[])',
    'Joiner3NodeFactory': 'df = df.merge(df_right, how="inner", on=[])',
    'CrossJoinerNodeFactory': 'df = df.merge(df_right, how="cross")',
    'ConcatenateNodeFactory': 'df = pd.concat([df1, df2], ignore_index=True)',
    'AppendedRowsNodeFactory': 'df = pd.concat([df1, df2], ignore_index=True)',
    
    # Math & Formulas
    'JEPNodeFactory': 'df["result"] = df["a"] + df["b"]  # Math Formula',
    'MathFormulaNodeFactory': 'df["result"] = df["a"] + df["b"]',
    'RoundDoubleNodeFactory': 'df["col"] = df["col"].round(2)',
    'ColumnAggregatorNodeFactory': 'result = df.agg("sum")',
    
    # String Operations
    'StringManipulationNodeFactory': 'df["col"] = df["col"].str.upper()',
    'CellSplitterNodeFactory': 'df = df["col"].str.split(",", expand=True)',
    'StringReplacerNodeFactory': 'df["col"] = df["col"].str.replace("old", "new")',
    
    # Type Conversion
    'NumberToString2NodeFactory': 'df["col"] = df["col"].astype(str)',
    'StringToNumber2NodeFactory': 'df["col"] = pd.to_numeric(df["col"], errors="coerce")',
    'DoubleToIntNodeFactory': 'df["col"] = df["col"].astype(int)',
    
    # Rule Engine
    'RuleEngineNodeFactory': 'df["result"] = np.where(condition, "A", "B")',
    'RuleEngineFilterNodeFactory': 'df = df[rule_condition]  # Rule Filter',
    
    # Flow Control
    'EmptyTableSwitchNodeFactory': 'pass  # Empty Table Switch',
    'EndifNodeFactory': 'pass  # End IF',
    'IfSwitchNodeFactory': 'pass  # IF Switch',
    
    # Loops
    'GroupLoopStartNodeFactory': 'for group in groups:  # Group Loop',
    'LoopEndDynamicNodeFactory': 'pass  # Loop End',
    'LoopEndNodeFactory': 'pass  # Loop End',
    
    # Variables
    'TableToVariable3NodeFactory': 'var = df.iloc[0]["col"]  # Table to Variable',
    'VariableToTable4NodeFactory': 'df = pd.DataFrame([{"var": value}])',
    'ConstantValueColumnNodeFactory': 'df["new_col"] = constant_value',
    
    # Date/Time
    'OldToNewTimeNodeFactory': 'df["col"] = pd.to_datetime(df["col"])',
    'DateTimeDifferenceNodeFactory': 'df["diff"] = df["end"] - df["start"]',
    'CreateDateTimeNodeFactory': 'df["date"] = pd.Timestamp.now()',
    'ExtractDateTimeFieldsNodeFactory2': 'df["year"] = df["date"].dt.year',
    'DateTimeShiftNodeFactory': 'df["date"] = df["date"] + pd.Timedelta(days=1)',
    'ModifyTimeNodeFactory': 'df["time"] = df["time"].apply(modify_func)',
    
    # Missing Values
    'MissingValueNodeFactory': 'df = df.fillna(0)',
    
    # Database (placeholder)
    'DatabaseLoopingNodeFactory': 'pass  # Database Loop - requires connection',
    'DBReaderNodeFactory': 'df = pd.read_sql(query, conn)',
    'DBLoaderNodeFactory2': 'df.to_sql("table", conn)',
    'DBQueryReaderNodeFactory': 'df = pd.read_sql(query, conn)',
    
    # Counter
    'CounterGenerationNodeFactory': 'df["counter"] = range(1, len(df) + 1)',
    
    # Add Empty Rows
    'AddEmptyRowsNodeFactory': 'df = pd.concat([df, pd.DataFrame([{}])])',
}


def get_template(factory, log):
    """Get Python template for a KNIME factory."""
    for key, template in TEMPLATES.items():
        if key in factory:
            return template, key.replace('NodeFactory', '')
    return None, ""


def generate_code(nodes, log):
    """Generate Python code from nodes."""
    lines = [
        '"""',
        'Auto-generated Python pipeline from KNIME workflow',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Nodes: {len(nodes)}',
        '"""',
        'import pandas as pd',
        'import numpy as np',
        'from datetime import datetime, timedelta',
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
            lines.append(f'    {template}')
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
        print("")
        print("Output:")
        print("  - arquivo.py      (Python code)")
        print("  - arquivo_log.md  (diagnostic log)")
        sys.exit(1)
    
    knwf_path = Path(sys.argv[1]).resolve()
    
    if not knwf_path.exists():
        print(f"Error: File not found: {knwf_path}")
        sys.exit(1)
    
    output_path = knwf_path.with_suffix('.py')
    log_path = knwf_path.parent / f"{knwf_path.stem}_log.md"
    
    log = TranspilerLog(str(knwf_path))
    
    print("="*60)
    print("KNIME to Python Transpiler")
    print("="*60)
    print(f"Input:  {knwf_path}")
    print(f"Output: {output_path}")
    print(f"Log:    {log_path}")
    print("="*60)
    
    try:
        print("\n[1/4] Loading nodes...")
        nodes = load_analysis_json(knwf_path, log)
        
        if nodes:
            print(f"      Found {len(nodes)} nodes (from JSON)")
        else:
            log.extraction_method = "KNWF Extraction"
            extract_dir = extract_knwf(knwf_path, log)
            nodes = find_nodes_from_settings(extract_dir, log)
            print(f"      Found {len(nodes)} nodes (from extraction)")
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
