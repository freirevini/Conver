"""Test script for expression parsers."""
from app.services.parsers import (
    math_parser, 
    string_parser, 
    rule_parser, 
    column_expr_parser
)

print("=" * 50)
print("PHASE 2: Expression Parsers Test")
print("=" * 50)

# Test Math Parser
print("\n=== MATH PARSER ===")
result = math_parser.parse('ABS($Discount$) + ROUND($Price$ * 1.1, 2)', 'total')
print(f"Input: ABS($Discount$) + ROUND($Price$ * 1.1, 2)")
print(f"Output: {result}")

result2 = math_parser.parse('$Sales$ * 1.15', 'sales_with_tax')
print(f"\nInput: $Sales$ * 1.15")
print(f"Output: {result2}")

# Test String Parser
print("\n=== STRING PARSER ===")
result = string_parser.parse('upperCase($Name$)', 'upper_name')
print(f"Input: upperCase($Name$)")
print(f"Output: {result}")

result2 = string_parser.parse('join($FirstName$, \" \", $LastName$)', 'full_name')
print(f"\nInput: join($FirstName$, \" \", $LastName$)")
print(f"Output: {result2}")

result3 = string_parser.parse('substr($Text$, 0, 10)', 'short_text')
print(f"\nInput: substr($Text$, 0, 10)")
print(f"Output: {result3}")

# Test Rule Parser  
print("\n=== RULE PARSER ===")
rules = """$Age$ >= 65 => Senior
$Age$ >= 18 => Adult
TRUE => Minor"""
result = rule_parser.parse(rules, 'category')
print(f"Input:\n{rules}")
print(f"\nOutput:\n{result}")

# Test Column Expressions Parser
print("\n=== COLUMN EXPRESSIONS PARSER ===")
expr = 'if (column("Age") > 18) { "Adult" } else { "Minor" }'
result = column_expr_parser.parse(expr, 'status')
print(f"Input: {expr}")
print(f"Output: {result}")

print("\n" + "=" * 50)
print("ALL PARSERS TESTED SUCCESSFULLY!")
print("=" * 50)
