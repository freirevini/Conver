#!/usr/bin/env python
"""
KNIME to Python Transpiler - Standalone Version

Usage:
    python transpile.py <arquivo.knwf>
    
Output:
    Creates <arquivo>.py in the same directory as the input file
"""
import sys
import zipfile
import tempfile
import json
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def extract_knwf(knwf_path: Path) -> Path:
    """Extract .knwf file to temp directory."""
    temp_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(knwf_path, 'r') as zf:
        zf.extractall(temp_dir)
    return temp_dir


def find_nodes_from_workflow(extract_dir: Path) -> List[Dict[str, Any]]:
    """Find all nodes by parsing workflow.knime files."""
    nodes = []
    
    # Find all workflow.knime files (main and metanodes)
    for workflow_file in extract_dir.rglob('workflow.knime'):
        try:
            tree = ET.parse(workflow_file)
            root = tree.getroot()
            
            # Find all node entries
            for node_elem in root.findall('.//{http://www.knime.org/2008/09/XMLConfig}config[@key="nodes"]'):
                for entry in node_elem.findall('.//{http://www.knime.org/2008/09/XMLConfig}config'):
                    node_info = {'path': str(workflow_file), 'factory': '', 'name': '', 'id': ''}
                    
                    for e in entry.findall('.//{http://www.knime.org/2008/09/XMLConfig}entry'):
                        key = e.get('key', '')
                        value = e.get('value', '')
                        
                        if key == 'node_type':
                            node_info['factory'] = value
                        elif key == 'node_settings_file':
                            node_info['settings_file'] = value
                        elif key == 'id':
                            node_info['id'] = value
                        elif key == 'node_is_meta':
                            node_info['is_meta'] = value == 'true'
                    
                    # Extract node name from id pattern like "Column Filter (#1609)"  
                    for e in entry.findall('.//{http://www.knime.org/2008/09/XMLConfig}config[@key="node_settings"]'):
                        for sub in e.findall('.//{http://www.knime.org/2008/09/XMLConfig}entry[@key="name"]'):
                            node_info['name'] = sub.get('value', '')
                    
                    if node_info.get('factory'):
                        nodes.append(node_info)
                        
        except Exception as e:
            continue
    
    return nodes


def find_nodes_from_settings(extract_dir: Path) -> List[Dict[str, Any]]:
    """Find nodes by scanning settings.xml files and looking for factory patterns."""
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
                
                # Clean up name - remove (#ID) pattern
                name_clean = re.sub(r'\s*\(#\d+\)$', '', name)
                
                nodes.append({
                    'id': node_id,
                    'name': name_clean,
                    'factory': factory,
                    'path': str(settings_file)
                })
                node_id += 1
                
        except Exception:
            continue
    
    return nodes


def load_analysis_json(knwf_path: Path) -> Optional[List[Dict]]:
    """Try to load pre-analyzed workflow_analysis_v2.json if it exists."""
    # Check next to knwf file
    analysis_path = knwf_path.parent / "workflow_analysis_v2.json"
    if not analysis_path.exists():
        # Check in script directory parent
        analysis_path = Path(__file__).parent.parent / "workflow_analysis_v2.json"
    
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('nodes', [])
        except Exception:
            pass
    
    return None


def get_template(factory: str) -> Optional[str]:
    """Get Python template for a KNIME factory."""
    # Try using the full template mapper if available
    try:
        from app.services.generator.template_mapper import TemplateMapper
        mapper = TemplateMapper()
        if mapper.has_template(factory):
            code = mapper.generate_code(factory, {}, 'df', 'df')
            return code
    except Exception:
        pass
    
    # Fallback basic templates for common nodes
    TEMPLATES = {
        'DataColumnSpecFilterNodeFactory': 'df = df.copy()  # Column Filter - passthrough',
        'RenameNodeFactory': 'df = df.rename(columns={})  # Column Rename',
        'GroupByNodeFactory': 'df = df.groupby([]).agg({}).reset_index()  # GroupBy',
        'JoinerNodeFactory': 'df = df.merge(df_right, how="inner")  # Joiner',
        'RowFilterNodeFactory': 'df = df[df["column"] == value].copy()  # Row Filter',
        'SorterNodeFactory': 'df = df.sort_values(by=[])  # Sorter',
        'JEPNodeFactory': 'df["result"] = df["col1"] + df["col2"]  # Math Formula',
        'RuleEngineNodeFactory': 'df["result"] = np.where(df["col"] > 0, "A", "B")  # Rule Engine',
        'StringManipulationNodeFactory': 'df["result"] = df["col"].str.upper()  # String Manipulation',
        'CrossJoinerNodeFactory': 'df = df.merge(df_right, how="cross")  # Cross Joiner',
        'ColumnAggregatorNodeFactory': 'df = df.agg("sum")  # Column Aggregator',
        'RoundDoubleNodeFactory': 'df["col"] = df["col"].round(2)  # Round Double',
        'NumberToString2NodeFactory': 'df["col"] = df["col"].astype(str)  # Number To String',
        'EmptyTableSwitchNodeFactory': '# Empty Table Switch - flow control',
        'EndifNodeFactory': '# End IF - flow control',
    }
    
    for key, template in TEMPLATES.items():
        if key in factory:
            return template
    
    return None


def generate_code(nodes: List[Dict]) -> str:
    """Generate Python code from nodes."""
    lines = [
        '# Transpiled: {} nodes'.format(len(nodes)),
        '"""',
        'Auto-generated Python pipeline from KNIME workflow',
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
    
    stats = {'total': 0, 'template': 0, 'fallback': 0}
    
    for i, node in enumerate(nodes):
        factory = node.get('factory', '')
        name = node.get('name', node.get('node_name', f'Node_{i}'))
        
        template = get_template(factory)
        stats['total'] += 1
        
        if template:
            stats['template'] += 1
            lines.append(f'    # {name}')
            # Ensure template is properly indented
            for tline in template.split('\n'):
                if tline.strip():
                    lines.append(f'    {tline}')
            lines.append('')
        else:
            stats['fallback'] += 1
            simple_factory = factory.split('.')[-1].replace('NodeFactory', '') if factory else 'Unknown'
            lines.append(f'    # {name} ({simple_factory})')
            lines.append(f'    pass  # TODO: Implement')
            lines.append('')
    
    lines.extend([
        '    return df',
        '',
        '',
        'if __name__ == "__main__":',
        '    # Example usage',
        '    # df = pd.read_csv("input.csv")',
        '    # result = run_pipeline(df)',
        '    # result.to_csv("output.csv", index=False)',
        f'    print("Pipeline loaded: {stats["total"]} nodes ({stats["template"]} templates, {stats["fallback"]} fallback)")',
        '    print("Call run_pipeline(df) with your DataFrame.")',
    ])
    
    # Update first line with stats
    lines[0] = f'# Transpiled: {stats["total"]} nodes (Template: {stats["template"]}, Fallback: {stats["fallback"]})'
    
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python transpile.py <arquivo.knwf>")
        print("\nExample:")
        print("  python transpile.py meu_fluxo.knwf")
        sys.exit(1)
    
    knwf_path = Path(sys.argv[1]).resolve()
    
    if not knwf_path.exists():
        print(f"Error: File not found: {knwf_path}")
        sys.exit(1)
    
    output_path = knwf_path.with_suffix('.py')
    
    print("="*60)
    print("KNIME to Python Transpiler")
    print("="*60)
    print(f"Input:  {knwf_path}")
    print(f"Output: {output_path}")
    print("="*60)
    
    try:
        # Try to load pre-analyzed JSON first (faster, more accurate)
        print("\n1. Checking for pre-analyzed data...")
        nodes = load_analysis_json(knwf_path)
        
        if nodes:
            print(f"   Found {len(nodes)} nodes from analysis JSON")
        else:
            # Extract and analyze manually
            print("   No pre-analyzed data found, extracting workflow...")
            extract_dir = extract_knwf(knwf_path)
            
            print("2. Finding nodes...")
            nodes = find_nodes_from_settings(extract_dir)
            print(f"   Found {len(nodes)} nodes")
            
            # Cleanup temp directory
            import shutil
            shutil.rmtree(extract_dir, ignore_errors=True)
        
        if not nodes:
            print("\n⚠️ No nodes found in workflow!")
            print("The workflow may be empty or use an unsupported structure.")
            sys.exit(1)
        
        print("3. Generating Python code...")
        code = generate_code(nodes)
        
        print("4. Writing output file...")
        output_path.write_text(code, encoding='utf-8')
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"Output: {output_path}")
        print(f"Nodes:  {len(nodes)}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
