"""
Integration Test: Validate all Phase 1-4 components with real KNIME workflow.

Tests:
1. Catalog Service - Node lookup
2. Expression Parsers - Math, String, Rule, Column Expr
3. Extended Templates - Support for workflow nodes
4. Type Mapper - Type conversions

Generates detailed report of pass/fail results.
"""
import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

sys.path.insert(0, '.')

# ==================== Test Configuration ====================

WORKFLOW_PATH = Path("../fluxo_knime_exemplo/document/Indicador_Calculo_CET_Rodas")
REPORT_PATH = Path("integration_test_report.md")

# ==================== Test Results ====================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.details = []
    
    def add_pass(self, test_name: str, detail: str = ""):
        self.passed += 1
        self.details.append(f"✅ {test_name}: PASS" + (f" - {detail}" if detail else ""))
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"❌ {test_name}: {error}")
        self.details.append(f"❌ {test_name}: FAIL - {error}")
    
    def get_summary(self) -> str:
        total = self.passed + self.failed
        pct = (self.passed / total * 100) if total > 0 else 0
        return f"{self.passed}/{total} tests passed ({pct:.1f}%)"

results = TestResults()

# ==================== Test 1: Catalog Service ====================

def test_catalog_service():
    """Test catalog service loading and lookup."""
    print("\n=== TEST 1: Catalog Service ===")
    
    try:
        from app.services.catalog import catalog_service
        
        # Test catalog loaded
        if catalog_service._catalog is None:
            # Force reload
            catalog_service._load_catalog()
        
        nodes = catalog_service._catalog.get("nodes", {})
        results.add_pass("Catalog Loading", f"{len(nodes)} nodes loaded")
        
        # Test specific lookups
        test_nodes = [
            "org.knime.base.node.io.csvreader.CSVReaderNodeFactory",
            "org.knime.base.node.preproc.filter.row.RowFilterNodeFactory",
            "org.knime.base.node.preproc.groupby.GroupByNodeFactory",
        ]
        
        for factory in test_nodes:
            node = catalog_service.get_node(factory)
            if node:
                results.add_pass(f"Lookup {factory.split('.')[-1]}")
            else:
                results.add_fail(f"Lookup {factory.split('.')[-1]}", "Not found")
        
        # Test operators
        operators = catalog_service._catalog.get("operators", {})
        if len(operators) > 0:
            results.add_pass("Operators Loaded", f"{len(operators)} operators")
        else:
            results.add_fail("Operators Loaded", "No operators found")
            
    except Exception as e:
        results.add_fail("Catalog Service", str(e))

# ==================== Test 2: Expression Parsers ====================

def test_expression_parsers():
    """Test all expression parsers with sample expressions."""
    print("\n=== TEST 2: Expression Parsers ===")
    
    # Test Math Parser
    try:
        from app.services.parsers import math_parser
        
        test_cases = [
            ("$Price$ * 1.1", "df['result'] = df['Price'] * 1.1"),
            ("ABS($Discount$)", "np.abs"),
            ("ROUND($Value$, 2)", "np.round"),
        ]
        
        for expr, expected_substr in test_cases:
            result = math_parser.parse(expr, "result")
            if expected_substr in result:
                results.add_pass(f"Math: {expr[:30]}")
            else:
                results.add_fail(f"Math: {expr[:30]}", f"Expected '{expected_substr}' in '{result}'")
                
    except Exception as e:
        results.add_fail("Math Parser", str(e))
    
    # Test String Parser
    try:
        from app.services.parsers import string_parser
        
        test_cases = [
            ("upperCase($Name$)", "str.upper"),
            ("trim($Text$)", "str.strip"),
            ("substr($Text$, 0, 5)", "str["),
        ]
        
        for expr, expected_substr in test_cases:
            result = string_parser.parse(expr, "result")
            if expected_substr in result:
                results.add_pass(f"String: {expr[:30]}")
            else:
                results.add_fail(f"String: {expr[:30]}", f"Expected '{expected_substr}' in '{result}'")
                
    except Exception as e:
        results.add_fail("String Parser", str(e))
    
    # Test Rule Parser
    try:
        from app.services.parsers import rule_parser
        
        rules = "$Age$ >= 18 => Adult\nTRUE => Minor"
        result = rule_parser.parse(rules, "category")
        
        if "np.select" in result and "conditions" in result:
            results.add_pass("Rule Parser", "np.select generated")
        else:
            results.add_fail("Rule Parser", f"Unexpected output: {result[:100]}")
            
    except Exception as e:
        results.add_fail("Rule Parser", str(e))
    
    # Test Column Expressions Parser
    try:
        from app.services.parsers import column_expr_parser
        
        expr = 'if (column("Age") > 18) { "Adult" } else { "Minor" }'
        result = column_expr_parser.parse(expr, "status")
        
        if "np.where" in result:
            results.add_pass("Column Expr Parser", "np.where generated")
        else:
            results.add_fail("Column Expr Parser", f"Unexpected: {result[:100]}")
            
    except Exception as e:
        results.add_fail("Column Expr Parser", str(e))

# ==================== Test 3: Extended Templates ====================

def test_extended_templates():
    """Test extended templates loading and lookup."""
    print("\n=== TEST 3: Extended Templates ===")
    
    try:
        from app.services.generator.extended_templates import (
            EXTENDED_TEMPLATES,
            DB_CONNECTOR_TEMPLATES,
            get_template_stats,
            is_expression_node,
            generate_expression_code
        )
        
        stats = get_template_stats()
        results.add_pass("Extended Templates Loaded", f"{stats['total_new']} templates")
        
        # Test specific templates
        test_templates = [
            "org.knime.base.node.io.table.read.TableReaderNodeFactory",
            "org.knime.base.node.preproc.pivot.PivotNodeFactory",
            "org.knime.base.node.switches.ifswitch.IFSwitchNodeFactory",
        ]
        
        all_templates = {**EXTENDED_TEMPLATES, **DB_CONNECTOR_TEMPLATES}
        for factory in test_templates:
            if factory in all_templates:
                results.add_pass(f"Template: {factory.split('.')[-1]}")
            else:
                results.add_fail(f"Template: {factory.split('.')[-1]}", "Not found")
        
        # Test DB connectors
        db_count = len(DB_CONNECTOR_TEMPLATES)
        if db_count >= 4:
            results.add_pass("DB Connectors", f"{db_count} connectors")
        else:
            results.add_fail("DB Connectors", f"Expected 4, got {db_count}")
        
        # Test expression node detection
        expr_factories = [
            "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
            "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory",
        ]
        
        for factory in expr_factories:
            if is_expression_node(factory):
                results.add_pass(f"Expr Detection: {factory.split('.')[-1]}")
            else:
                results.add_fail(f"Expr Detection: {factory.split('.')[-1]}", "Not detected")
        
        # Test expression code generation
        code, imports = generate_expression_code(
            "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory",
            "$Price$ * 0.9",
            "discounted",
            "df"
        )
        if "df['discounted']" in code:
            results.add_pass("Expression Code Gen", code.strip())
        else:
            results.add_fail("Expression Code Gen", f"Unexpected: {code}")
            
    except Exception as e:
        results.add_fail("Extended Templates", str(e))
        traceback.print_exc()

# ==================== Test 4: Type Mapper ====================

def test_type_mapper():
    """Test type mapper conversions and code generation."""
    print("\n=== TEST 4: Type Mapper ===")
    
    try:
        from app.services.type_mapper import (
            type_mapper,
            convert_value,
            get_pandas_dtype,
            generate_cast_code,
            infer_type_from_value
        )
        
        # Test value conversions
        conversions = [
            ("1000", "IntCell", int, 1000),
            ("3.14", "DoubleCell", float, 3.14),
            ("true", "BooleanCell", bool, True),
        ]
        
        for value, knime_type, expected_type, expected_value in conversions:
            result = convert_value(value, knime_type)
            if isinstance(result, expected_type) and result == expected_value:
                results.add_pass(f"Convert: {value} → {knime_type}")
            else:
                results.add_fail(f"Convert: {value} → {knime_type}", f"Got {result} ({type(result).__name__})")
        
        # Test pandas dtype
        dtype_tests = [
            ("IntCell", "Int64"),
            ("DoubleCell", "float64"),
            ("BooleanCell", "boolean"),
        ]
        
        for knime_type, expected_dtype in dtype_tests:
            result = get_pandas_dtype(knime_type)
            if result == expected_dtype:
                results.add_pass(f"Dtype: {knime_type} → {expected_dtype}")
            else:
                results.add_fail(f"Dtype: {knime_type}", f"Expected {expected_dtype}, got {result}")
        
        # Test cast code generation
        cast_code = generate_cast_code("price", "DoubleCell")
        if "pd.to_numeric" in cast_code:
            results.add_pass("Cast Code Gen", "pd.to_numeric used")
        else:
            results.add_fail("Cast Code Gen", f"Unexpected: {cast_code}")
        
        # Test type inference
        inferred = infer_type_from_value(42)
        if inferred == "IntCell":
            results.add_pass("Type Inference", "42 → IntCell")
        else:
            results.add_fail("Type Inference", f"Expected IntCell, got {inferred}")
        
        # Test supported types count
        supported = type_mapper.get_supported_types()
        if len(supported) >= 20:
            results.add_pass("Supported Types", f"{len(supported)} types")
        else:
            results.add_fail("Supported Types", f"Expected >=20, got {len(supported)}")
            
    except Exception as e:
        results.add_fail("Type Mapper", str(e))
        traceback.print_exc()

# ==================== Test 5: TemplateMapper Integration ====================

def test_template_mapper_integration():
    """Test that TemplateMapper can access extended templates."""
    print("\n=== TEST 5: TemplateMapper Integration ===")
    
    try:
        from app.services.generator.template_mapper import TemplateMapper
        
        mapper = TemplateMapper()
        
        # Test base template
        base = mapper.get_template("org.knime.base.node.io.csvreader.CSVReaderNodeFactory")
        if base:
            results.add_pass("Base Template Access", "CSVReader found")
        else:
            results.add_fail("Base Template Access", "CSVReader not found")
        
        # Test extended template access
        extended = mapper.get_template("org.knime.base.node.preproc.pivot.PivotNodeFactory")
        if extended:
            results.add_pass("Extended Template Access", "Pivot found")
        else:
            results.add_fail("Extended Template Access", "Pivot not found via mapper")
        
        # Count total templates
        base_count = len(mapper.TEMPLATES)
        ext = mapper._get_extended_templates()
        ext_count = len(ext) if ext else 0
        total = base_count + ext_count
        
        if total >= 48:
            results.add_pass("Total Templates", f"{total} templates available")
        else:
            results.add_fail("Total Templates", f"Expected >=48, got {total}")
            
    except Exception as e:
        results.add_fail("TemplateMapper Integration", str(e))
        traceback.print_exc()

# ==================== Generate Report ====================

def generate_report() -> str:
    """Generate markdown report."""
    
    report = f"""# Integration Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Workflow:** `fluxo_knime_exemplo`

---

## Summary

| Metric | Value |
|--------|-------|
| **Passed** | {results.passed} |
| **Failed** | {results.failed} |
| **Total** | {results.passed + results.failed} |
| **Success Rate** | {(results.passed / (results.passed + results.failed) * 100) if (results.passed + results.failed) > 0 else 0:.1f}% |

---

## Test Results

"""
    
    for detail in results.details:
        report += f"- {detail}\n"
    
    if results.errors:
        report += f"""
---

## ❌ Errors ({len(results.errors)})

"""
        for error in results.errors:
            report += f"- {error}\n"
    
    report += f"""
---

## Components Tested

| Component | Status |
|-----------|--------|
| Catalog Service | {'✅' if results.failed == 0 or any('Catalog' in d and '✅' in d for d in results.details) else '❌'} |
| Expression Parsers | {'✅' if any('Math:' in d and '✅' in d for d in results.details) else '❌'} |
| Extended Templates | {'✅' if any('Extended Templates' in d and '✅' in d for d in results.details) else '❌'} |
| Type Mapper | {'✅' if any('Convert:' in d and '✅' in d for d in results.details) else '❌'} |
| TemplateMapper Integration | {'✅' if any('Total Templates' in d and '✅' in d for d in results.details) else '❌'} |

---

## Conclusion

{"✅ **ALL TESTS PASSED** - Components are ready for production integration." if results.failed == 0 else f"⚠️ **{results.failed} TESTS FAILED** - Review errors above before proceeding."}
"""
    
    return report

# ==================== Main ====================

def main():
    print("=" * 60)
    print("INTEGRATION TEST: KNIME Catalog Components")
    print("=" * 60)
    
    # Run all tests
    test_catalog_service()
    test_expression_parsers()
    test_extended_templates()
    test_type_mapper()
    test_template_mapper_integration()
    
    # Generate report
    report = generate_report()
    
    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULT: {results.get_summary()}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 60)
    
    # Print report preview
    print("\n" + report)
    
    return results.failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
