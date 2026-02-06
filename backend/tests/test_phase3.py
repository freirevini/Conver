"""Test integration of Phase 3: Extended TemplateMapper."""
from app.services.generator.template_mapper import TemplateMapper
from app.services.generator.extended_templates import (
    get_template_stats,
    is_expression_node,
    generate_expression_code
)

print("=" * 60)
print("PHASE 3: Extended TemplateMapper Integration Test")
print("=" * 60)

# Test 1: Template stats
print("\n=== TEMPLATE STATS ===")
stats = get_template_stats()
print(f"Grupo A nodes: {stats['grupo_a_nodes']}")
print(f"Grupo B DB connectors: {stats['grupo_b_db_connectors']}")
print(f"Expression nodes: {stats['expression_nodes']}")
print(f"Total new templates: {stats['total_new']}")

# Test 2: TemplateMapper integration
print("\n=== TEMPLATE MAPPER INTEGRATION ===")
mapper = TemplateMapper()

# Test base template
base_template = mapper.get_template("org.knime.base.node.io.csvreader.CSVReaderNodeFactory")
print(f"Base CSV Reader: {'✅ Found' if base_template else '❌ Not Found'}")

# Test extended templates
extended_nodes = [
    "org.knime.base.node.io.table.read.TableReaderNodeFactory",
    "org.knime.base.node.preproc.pivot.PivotNodeFactory",
    "org.knime.base.node.switches.ifswitch.IFSwitchNodeFactory",
    "org.knime.database.connector.MySQLConnectorNodeFactory",
]

for node in extended_nodes:
    template = mapper.get_template(node)
    status = "✅" if template else "❌"
    name = node.split(".")[-1].replace("NodeFactory", "")
    print(f"Extended {name}: {status}")

# Test total count
base_count = len(mapper.TEMPLATES)
extended = mapper._get_extended_templates()
extended_count = len(extended) if extended else 0
print(f"\nTotal templates: {base_count} base + {extended_count} extended = {base_count + extended_count}")

# Test 3: Expression node detection
print("\n=== EXPRESSION NODE DETECTION ===")
expr_nodes = [
    "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
    "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory",
    "org.knime.base.node.rules.engine.RuleEngineNodeFactory",
]

for node in expr_nodes:
    is_expr = is_expression_node(node)
    name = node.split(".")[-1].replace("NodeFactory", "")
    print(f"{name}: {'✅ Expression' if is_expr else '❌ Template'}")

# Test 4: Expression code generation
print("\n=== EXPRESSION CODE GENERATION ===")
code, imports = generate_expression_code(
    "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory",
    "$Price$ * 1.1 + ABS($Discount$)",
    "final_price",
    "df"
)
print(f"Math Formula output:\n{code}")

print("\n" + "=" * 60)
print("PHASE 3 INTEGRATION TEST COMPLETE!")
print("=" * 60)
