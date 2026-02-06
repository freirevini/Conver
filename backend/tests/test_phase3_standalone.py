"""Standalone test for Phase 3 - avoids import chain issues."""
import sys
sys.path.insert(0, '.')

# Direct imports without triggering the full module chain
import json
from pathlib import Path

print("=" * 60)
print("PHASE 3: Standalone Test")
print("=" * 60)

# Test 1: Load extended_templates directly
print("\n=== TESTING EXTENDED TEMPLATES ===")
exec(open("app/services/generator/extended_templates.py").read())

# Since we exec'd the file, the variables are now in scope
extended = get_all_extended_templates()
stats = get_template_stats()

print(f"Grupo A nodes: {stats['grupo_a_nodes']}")
print(f"Grupo B DB connectors: {stats['grupo_b_db_connectors']}")
print(f"Total new templates: {stats['total_new']}")

# Test 2: Try a few specific templates
print("\n=== TEMPLATE SAMPLES ===")
for name, tmpl in list(extended.items())[:3]:
    short_name = name.split(".")[-1].replace("NodeFactory", "")
    print(f"- {short_name}: {tmpl['description']}")

# Test 3: Test expression code generation
print("\n=== EXPRESSION CODE GENERATION ===")
code, imports = generate_expression_code(
    "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory",
    "$Price$ * 1.1 + ABS($Discount$)",
    "final_price",
    "df"
)
print(f"Input: $Price$ * 1.1 + ABS($Discount$)")
print(f"Output: {code}")

# Test string parser
code2, imports2 = generate_expression_code(
    "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
    "upperCase($Name$)",
    "upper_name",
    "df"
)
print(f"\nInput: upperCase($Name$)")
print(f"Output: {code2}")

print("\n" + "=" * 60)
print("PHASE 3 STANDALONE TEST PASSED!")
print("=" * 60)
