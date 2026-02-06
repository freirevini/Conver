"""
Focused LLM Test - Test specific node types.

Tests a small set of KNIME nodes to verify LLM interpretation.
Includes both common and rare node types.
"""
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env explicitly
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded .env from {env_path}")
except ImportError:
    logger.warning("python-dotenv not installed")


# Test nodes - mix of common and rare types
TEST_NODES = [
    {
        "node_name": "Column Filter",
        "node_type": "ColumnFilter",
        "factory": "org.knime.base.node.preproc.filter.column.FilterColumnNodeFactory",
        "settings": {
            "include_columns": ["id", "name", "value"],
            "exclude_pattern": "tmp_*"
        },
        "description": "Common node - filter specific columns from DataFrame"
    },
    {
        "node_name": "Row Filter",
        "node_type": "RowFilter",
        "factory": "org.knime.base.node.preproc.filter.row.RowFilterNodeFactory",
        "settings": {
            "condition": "value > 100",
            "column": "amount"
        },
        "description": "Common node - filter rows by condition"
    },
    {
        "node_name": "Math Formula",
        "node_type": "MathFormula",
        "factory": "org.knime.ext.jep.JEPNodeFactory",
        "settings": {
            "expression": "abs($price$ - $cost$) / $cost$ * 100",
            "new_column": "margin_percent"
        },
        "description": "Common node - mathematical expression with column references"
    },
    {
        "node_name": "Date&Time Difference",
        "node_type": "DateTimeDifference",
        "factory": "org.knime.time.node.calculate.datetimedifference.DateTimeDifferenceNodeFactory",
        "settings": {
            "start_column": "start_date",
            "end_column": "end_date",
            "output_column": "days_diff",
            "granularity": "days"
        },
        "description": "Rare node - calculate difference between dates"
    },
    {
        "node_name": "Pivoting",
        "node_type": "Pivot",
        "factory": "org.knime.base.node.preproc.pivot.Pivot2NodeFactory",
        "settings": {
            "group_columns": ["category", "region"],
            "pivot_column": "month",
            "agg_column": "sales",
            "agg_method": "sum"
        },
        "description": "Complex node - pivot table with aggregation"
    }
]


def test_llm_generator():
    """Test LLM generator with specific nodes."""
    
    print("\n" + "="*70)
    print("FOCUSED LLM TEST - Testing 5 Specific Node Types")
    print("="*70)
    
    # Check environment
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
    
    print(f"\nConfiguration:")
    print(f"  GOOGLE_CLOUD_PROJECT: {project_id or 'NOT SET'}")
    print(f"  GOOGLE_CLOUD_LOCATION: {location}")
    
    if not project_id:
        print("\n❌ ERROR: GOOGLE_CLOUD_PROJECT not set!")
        print("Please check .env file or set environment variable.")
        return
    
    # Initialize LLM Generator
    print("\nInitializing LLM Generator...")
    try:
        from app.services.generator.llm_generator import LLMGenerator
        llm = LLMGenerator()
        
        if not llm.is_available():
            print("❌ LLM Generator not available!")
            return
        
        print(f"✅ LLM Generator initialized (model: {llm.MODEL_ID})")
        
    except Exception as e:
        print(f"❌ Failed to initialize LLM Generator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test each node
    results = []
    
    for i, node in enumerate(TEST_NODES, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/5: {node['node_name']}")
        print(f"Description: {node['description']}")
        print(f"Factory: {node['factory']}")
        print(f"{'─'*70}")
        
        input_var = f"df_{node['node_type'].lower()}_in"
        output_var = f"df_{node['node_type'].lower()}_out"
        
        try:
            code = llm.generate_node_code(
                node_type=node['node_type'],
                node_name=node['node_name'],
                factory=node['factory'],
                settings=node['settings'],
                input_var=input_var,
                output_var=output_var
            )
            
            if code:
                print(f"\n✅ LLM GENERATED CODE:")
                print(f"{'─'*40}")
                # Show first 500 chars
                preview = code[:500] + "..." if len(code) > 500 else code
                print(preview)
                
                # Validate syntax
                import ast
                try:
                    ast.parse(code)
                    print(f"\n✅ Syntax: VALID")
                    results.append({
                        'node': node['node_name'],
                        'status': 'SUCCESS',
                        'code_length': len(code)
                    })
                except SyntaxError as e:
                    print(f"\n⚠️ Syntax Error: {e}")
                    results.append({
                        'node': node['node_name'],
                        'status': 'SYNTAX_ERROR',
                        'error': str(e)
                    })
            else:
                print("\n❌ LLM returned no code")
                results.append({
                    'node': node['node_name'],
                    'status': 'NO_CODE',
                    'error': 'Empty response'
                })
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({
                'node': node['node_name'],
                'status': 'ERROR',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    success = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nTotal: {len(results)} tests")
    print(f"Success: {success}/{len(results)} ({success/len(results)*100:.0f}%)")
    
    print("\nDetails:")
    for r in results:
        icon = "✅" if r['status'] == 'SUCCESS' else "❌"
        print(f"  {icon} {r['node']}: {r['status']}")
        if 'error' in r:
            print(f"      Error: {r['error'][:50]}")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    test_llm_generator()
