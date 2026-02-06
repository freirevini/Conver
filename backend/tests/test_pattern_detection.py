"""
Test Pattern Detection with Real KNIME Metanodes.

Tests the semantic pattern detection using actual metanode data
from fluxo_knime_exemplo.knwf.
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.pattern import get_pattern_registry, PatternType


def test_date_generator_pattern():
    """Test DateGeneratorPattern detection with CRIA DATA DE metanode."""
    
    print("\n" + "="*60)
    print("TEST: DateGeneratorPattern Detection")
    print("="*60)
    
    # Simulated node data from CRIA DATA DE (#1385) metanode
    nodes = [
        {
            'id': '1367',
            'factory': 'org.knime.time.node.create.createdatetime.CreateDateTimeNodeFactory',
            'node_name': 'Create Date_Time Range',
            'settings': {}
        },
        {
            'id': '1368',
            'factory': 'org.knime.base.node.preproc.groupby.GroupByNodeFactory',
            'node_name': 'GroupBy',
            'settings': {}
        },
        {
            'id': '1369',
            'factory': 'org.knime.time.node.extract.datetime.ExtractDateTimeFieldsNodeFactory',
            'node_name': 'Extract Date_Time Fields',
            'settings': {}
        },
        {
            'id': '1370',
            'factory': 'org.knime.ext.jep.JEPNodeFactory',
            'node_name': 'Math Formula',
            'settings': {}
        },
        {
            'id': '1371',
            'factory': 'org.knime.time.node.manipulate.datetimeshift.DateTimeShiftNodeFactory',
            'node_name': 'Date_Time Shift',
            'settings': {'shiftvalue': -1, 'granularity': 'MONTHS'}
        },
        {
            'id': '1385',
            'factory': 'org.knime.base.node.flowvariable.tablerowtovariable.TableRowToVariableNodeFactory',
            'node_name': 'Table Row to Variable',
            'settings': {'variableName': 'data_referencia'}
        }
    ]
    
    connections = []  # Not used in current detection
    
    metanode_info = {
        'name': 'CRIA DATA DE REFERÊNCIA',
        'output_type': 'FlowVariablePortObject'
    }
    
    # Get registry and detect
    registry = get_pattern_registry()
    detected = registry.detect_all(nodes, connections, metanode_info)
    
    print(f"\nNodes analyzed: {len(nodes)}")
    print(f"Metanode: {metanode_info['name']}")
    print(f"\nDetected patterns: {len(detected)}")
    
    if detected:
        pattern = detected[0]
        print(f"\n✅ Pattern: {pattern.pattern_name}")
        print(f"   Type: {pattern.pattern_type.value}")
        print(f"   Confidence: {pattern.confidence:.0%}")
        print(f"   Nodes: {len(pattern.node_ids)}")
        
        # Generate Python code
        result = registry.generate_code(pattern, 'df_input', 'date_vars')
        if result:
            code, imports = result
            print(f"\n📄 Generated Python Code ({len(code)} chars):")
            print("-" * 40)
            print(code[:500])
            print("-" * 40)
            
        return True
    else:
        print("\n❌ No pattern detected!")
        return False


def test_loop_pattern():
    """Test LoopPattern detection."""
    
    print("\n" + "="*60)
    print("TEST: LoopPattern Detection")
    print("="*60)
    
    # Simulated loop node data
    nodes = [
        {
            'id': '1001',
            'factory': 'org.knime.base.node.meta.looper.columnlist2.ColumnListLoopStartNodeFactory',
            'node_name': 'Column List Loop Start',
            'settings': {}
        },
        {
            'id': '1002',
            'factory': 'org.knime.ext.jep.JEPNodeFactory',
            'node_name': 'Math Formula',
            'settings': {}
        },
        {
            'id': '1003',
            'factory': 'org.knime.base.node.meta.looper.LoopEndNodeFactory',
            'node_name': 'Loop End',
            'settings': {}
        }
    ]
    
    connections = []
    metanode_info = {'name': 'CALCULA A DI'}
    
    registry = get_pattern_registry()
    detected = registry.detect_all(nodes, connections, metanode_info)
    
    print(f"\nNodes analyzed: {len(nodes)}")
    print(f"Metanode: {metanode_info['name']}")
    print(f"\nDetected patterns: {len(detected)}")
    
    if detected:
        pattern = detected[0]
        print(f"\n✅ Pattern: {pattern.pattern_name}")
        print(f"   Type: {pattern.pattern_type.value}")
        print(f"   Confidence: {pattern.confidence:.0%}")
        print(f"   Loop Type: {pattern.extracted_data.get('loop_type', 'unknown')}")
        
        # Generate Python code
        result = registry.generate_code(pattern, 'df_input', 'df_output')
        if result:
            code, imports = result
            print(f"\n📄 Generated Python Code ({len(code)} chars):")
            print("-" * 40)
            print(code[:400])
            print("-" * 40)
            
        return True
    else:
        print("\n❌ No pattern detected!")
        return False


def test_no_pattern():
    """Test that non-pattern nodes return no detection."""
    
    print("\n" + "="*60)
    print("TEST: Non-Pattern Detection (should return empty)")
    print("="*60)
    
    # Regular nodes that don't form a pattern
    nodes = [
        {
            'id': '1',
            'factory': 'org.knime.base.node.io.filereader.FileReaderNodeFactory',
            'node_name': 'CSV Reader',
            'settings': {}
        },
        {
            'id': '2',
            'factory': 'org.knime.base.node.preproc.filter.column.FilterColumnNodeFactory',
            'node_name': 'Column Filter',
            'settings': {}
        }
    ]
    
    registry = get_pattern_registry()
    detected = registry.detect_all(nodes, [], None)
    
    print(f"\nNodes analyzed: {len(nodes)}")
    print(f"Detected patterns: {len(detected)}")
    
    if len(detected) == 0:
        print("\n✅ Correctly returned no patterns")
        return True
    else:
        print(f"\n⚠️ Unexpected pattern detected: {detected[0].pattern_name}")
        return False


def main():
    """Run all pattern detection tests."""
    
    print("\n" + "="*60)
    print("PATTERN DETECTION TEST SUITE")
    print("="*60)
    
    results = []
    
    results.append(('DateGenerator', test_date_generator_pattern()))
    results.append(('Loop', test_loop_pattern()))
    results.append(('NoPattern', test_no_pattern()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        icon = "✅" if result else "❌"
        print(f"  {icon} {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    print("="*60)


if __name__ == '__main__':
    main()
