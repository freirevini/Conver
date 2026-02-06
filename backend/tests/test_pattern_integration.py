"""
Integration Test - Real Metanode Pattern Detection.

Tests pattern detection with actual metanode data from
fluxo_knime_exemplo workflow.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.pattern import WorkflowPatternAnalyzer, get_pattern_registry


def parse_metanode_workflow(metanode_path: Path) -> dict:
    """Parse workflow.knime from a metanode directory."""
    workflow_file = metanode_path / "workflow.knime"
    
    if not workflow_file.exists():
        return {'nodes': [], 'connections': []}
    
    tree = ET.parse(workflow_file)
    root = tree.getroot()
    
    # Namespace handling
    ns = {'k': 'http://www.knime.org/2008/09/XMLConfig'}
    
    nodes = []
    connections = []
    
    # Parse nodes
    nodes_config = root.find(".//k:config[@key='nodes']", ns)
    if nodes_config:
        for node_config in nodes_config.findall("k:config", ns):
            node_id = node_config.find("k:entry[@key='id']", ns)
            settings_file = node_config.find("k:entry[@key='node_settings_file']", ns)
            
            if node_id is not None and settings_file is not None:
                node_id_val = node_id.get('value', '')
                settings_path = settings_file.get('value', '')
                
                # Extract factory from settings file
                factory = extract_factory_from_settings(
                    metanode_path / Path(settings_path).parent / 'settings.xml'
                )
                
                node_name = Path(settings_path).parent.name.split(' (#')[0] if settings_path else "Unknown"
                
                nodes.append({
                    'id': node_id_val,
                    'factory': factory,
                    'node_name': node_name,
                    'settings': {}
                })
    
    # Parse connections
    connections_config = root.find(".//k:config[@key='connections']", ns)
    if connections_config:
        for conn_config in connections_config.findall("k:config", ns):
            source_id = conn_config.find("k:entry[@key='sourceID']", ns)
            dest_id = conn_config.find("k:entry[@key='destID']", ns)
            
            if source_id is not None and dest_id is not None:
                connections.append({
                    'source': source_id.get('value', ''),
                    'dest': dest_id.get('value', '')
                })
    
    return {'nodes': nodes, 'connections': connections}


def extract_factory_from_settings(settings_path: Path) -> str:
    """Extract factory class from node settings.xml."""
    if not settings_path.exists():
        return ""
    
    try:
        tree = ET.parse(settings_path)
        root = tree.getroot()
        ns = {'k': 'http://www.knime.org/2008/09/XMLConfig'}
        
        factory_entry = root.find(".//k:entry[@key='factory']", ns)
        if factory_entry is not None:
            return factory_entry.get('value', '')
    except Exception:
        pass
    
    return ""


def test_real_date_metanode():
    """Test with the actual CRIA DATA DE (#1385) metanode."""
    
    print("\n" + "="*70)
    print("INTEGRATION TEST: Real Metanode Pattern Detection")
    print("="*70)
    
    metanode_path = Path(r"C:\Users\vinic\Documents\Projetos\ChatKnime\fluxo_knime_exemplo\document\Indicador_Calculo_CET_Rodas\CRIA DATA DE (#1385)")
    
    if not metanode_path.exists():
        print(f"❌ Metanode not found: {metanode_path}")
        return False
    
    print(f"\nMetanode: {metanode_path.name}")
    print(f"Path: {metanode_path}")
    
    # Parse the metanode
    workflow_data = parse_metanode_workflow(metanode_path)
    nodes = workflow_data['nodes']
    connections = workflow_data['connections']
    
    print(f"\nParsed:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Connections: {len(connections)}")
    
    if len(nodes) == 0:
        print("⚠️ No nodes parsed - checking factories directly...")
        # Fallback: list directories
        subdirs = [d for d in metanode_path.iterdir() if d.is_dir()]
        print(f"  Subdirectories: {len(subdirs)}")
        for sd in subdirs[:5]:
            print(f"    - {sd.name}")
    else:
        print("\n  Node factories:")
        for node in nodes[:5]:
            factory_short = node['factory'].split('.')[-1] if node['factory'] else "unknown"
            print(f"    - {node['node_name']}: {factory_short}")
    
    # Detect patterns
    registry = get_pattern_registry()
    
    metanode_info = {
        'name': 'CRIA DATA DE REFERÊNCIA',
        'path': str(metanode_path),
        'output_type': 'FlowVariablePortObject'
    }
    
    detected = registry.detect_all(nodes, connections, metanode_info)
    
    print(f"\n{'─'*70}")
    print("DETECTION RESULTS")
    print(f"{'─'*70}")
    
    if detected:
        pattern = detected[0]
        print(f"\n✅ Pattern Detected: {pattern.pattern_name}")
        print(f"   Confidence: {pattern.confidence:.0%}")
        print(f"   Type: {pattern.pattern_type.value}")
        
        # Generate code
        result = registry.generate_code(pattern, 'df', 'date_vars')
        if result:
            code, imports = result
            print(f"\n📄 Generated Python Code:")
            print("─" * 50)
            for line in code.split('\n')[:20]:
                print(line)
            print("─" * 50)
            
            # Validate syntax
            import ast
            try:
                ast.parse(code)
                print("✅ Syntax: VALID")
                return True
            except SyntaxError as e:
                print(f"❌ Syntax Error: {e}")
                return False
    else:
        print("\n⚠️ No pattern detected")
        print("   This may be due to factory extraction issues.")
        return False


def main():
    """Run integration tests."""
    print("\n" + "="*70)
    print("PATTERN DETECTION - INTEGRATION TEST SUITE")
    print("="*70)
    
    results = []
    
    try:
        results.append(('Real Date Metanode', test_real_date_metanode()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(('Real Date Metanode', False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, result in results:
        icon = "✅" if result else "❌"
        print(f"  {icon} {name}")
    
    print("="*70)


if __name__ == '__main__':
    main()
