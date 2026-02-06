"""Test integration of catalog into code_generator_v2."""
import sys
sys.path.insert(0, '.')

from app.models.ir_models import NodeInstance, NodeCategory
from app.services.generator.code_generator_v2 import CodeGeneratorV2

print("=" * 60)
print("CODE GENERATOR V2 INTEGRATION TEST")
print("=" * 60)

# Initialize generator
generator = CodeGeneratorV2(use_llm_fallback=False, enable_validation=False)

# Test 1: Expression node (String Manipulation)
print("\n=== TEST 1: Expression Node (String Manipulation) ===")
string_node = NodeInstance(
    node_id="101",
    node_type="StringManipulation",
    factory_class="org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
    name="String Manipulation",
    settings={
        "expression": "upperCase($Name$)",
        "new_column_name": "upper_name"
    },
    category=NodeCategory.TRANSFORM
)

code = generator.generate_function(string_node)
if "DETERMINISTIC" in code and "str.upper" in code:
    print("✅ String Manipulation: PASS")
    print(f"   Code preview: ...{code[100:200]}...")
else:
    print("❌ String Manipulation: FAIL")
    print(code[:300])

# Test 2: Expression node (Math Formula)
print("\n=== TEST 2: Expression Node (Math Formula) ===")
math_node = NodeInstance(
    node_id="102",
    node_type="MathFormula",
    factory_class="org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory",
    name="Math Formula",
    settings={
        "expression": "$Price$ * 1.1 + ABS($Discount$)",
        "new_column_name": "final_price"
    },
    category=NodeCategory.TRANSFORM
)

code = generator.generate_function(math_node)
if "DETERMINISTIC" in code and "np.abs" in code:
    print("✅ Math Formula: PASS")
    print(f"   Code preview: ...{code[100:200]}...")
else:
    print("❌ Math Formula: FAIL")
    print(code[:300])

# Test 3: Template node (CSV Reader)
print("\n=== TEST 3: Template Node (CSV Reader) ===")
csv_node = NodeInstance(
    node_id="103",
    node_type="CSVReader",
    factory_class="org.knime.base.node.io.csvreader.CSVReaderNodeFactory",
    name="CSV Reader",
    settings={"file_path": "data.csv", "separator": ","},
    category=NodeCategory.IO
)

code = generator.generate_function(csv_node)
if code:
    deterministic = "DETERMINISTIC" in code
    print(f"{'✅' if deterministic else '⚠️'} CSV Reader: {'DETERMINISTIC' if deterministic else 'FALLBACK'}")
    print(f"   Code preview: ...{code[50:150]}...")
else:
    print("❌ CSV Reader: FAIL - No code generated")

# Test 4: Extended template node (Pivot)
print("\n=== TEST 4: Extended Template Node (Pivot) ===")
pivot_node = NodeInstance(
    node_id="104",
    node_type="Pivoting",
    factory_class="org.knime.base.node.preproc.pivot.PivotNodeFactory",
    name="Pivoting",
    settings={
        "group_columns": ["Region"],
        "pivot_column": "Category",
        "value_columns": ["Sales"],
        "aggfunc": "sum"
    },
    category=NodeCategory.TRANSFORM
)

code = generator.generate_function(pivot_node)
if "DETERMINISTIC" in code:
    print("✅ Pivot: PASS (DETERMINISTIC)")
    print(f"   Code preview: ...{code[100:200]}...")
else:
    print("⚠️ Pivot: Uses fallback (expected for complex nodes)")
    print(f"   Code preview: ...{code[50:150]}...")

# Test 5: Unknown node (should generate stub)
print("\n=== TEST 5: Unknown Node (Stub) ===")
unknown_node = NodeInstance(
    node_id="105",
    node_type="CustomNode",
    factory_class="com.example.CustomNodeFactory",
    name="Custom Node",
    settings={},
    category=NodeCategory.UNKNOWN
)

code = generator.generate_function(unknown_node)
if "TODO" in code or "stub" in code.lower() or "NotImplementedError" in code:
    print("✅ Unknown Node: PASS (generates stub)")
else:
    print("⚠️ Unknown Node: Generated something (LLM or other)")
print(f"   Code preview: ...{code[50:150]}...")

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)
