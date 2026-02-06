"""Test Phase 4: Type Mapper."""
import sys
sys.path.insert(0, '.')

from app.services.type_mapper import (
    type_mapper,
    convert_value,
    get_pandas_dtype,
    generate_cast_code,
    generate_column_type_conversions,
    infer_type_from_value
)

print("=" * 60)
print("PHASE 4: Type Mapper Test")
print("=" * 60)

# Test 1: Value Conversion
print("\n=== VALUE CONVERSION ===")
test_cases = [
    ("1000", "IntCell"),
    ("3.14159", "DoubleCell"),
    ("true", "BooleanCell"),
    ("false", "BooleanCell"),
    ("2024-01-15", "LocalDateCell"),
    ("2024-01-15T10:30:00", "DateAndTimeCell"),
    ("Hello World", "StringCell"),
]

for value, knime_type in test_cases:
    result = convert_value(value, knime_type)
    print(f"  '{value}' ({knime_type}) → {result} ({type(result).__name__})")

# Test 2: Pandas dtype mapping
print("\n=== PANDAS DTYPE MAPPING ===")
types = ["IntCell", "DoubleCell", "BooleanCell", "DateAndTimeCell", "StringCell"]
for t in types:
    dtype = get_pandas_dtype(t)
    print(f"  {t} → {dtype}")

# Test 3: Cast code generation
print("\n=== CAST CODE GENERATION ===")
columns = {
    "price": "DoubleCell",
    "quantity": "IntCell",
    "date": "DateAndTimeCell",
    "active": "BooleanCell",
}

for col, knime_type in columns.items():
    code = generate_cast_code(col, knime_type)
    print(f"  {col}: {code}")

# Test 4: Batch conversion code
print("\n=== BATCH CONVERSION ===")
code, imports = generate_column_type_conversions(columns)
print(code)

# Test 5: Type inference
print("\n=== TYPE INFERENCE ===")
values = [42, 3.14, True, "hello", None]
for v in values:
    inferred = infer_type_from_value(v)
    print(f"  {v} ({type(v).__name__}) → {inferred}")

# Test 6: TypeMapper class
print("\n=== TYPE MAPPER CLASS ===")
print(f"Supported types: {len(type_mapper.get_supported_types())}")

result = type_mapper.convert("999", "IntCell")
print(f"type_mapper.convert('999', 'IntCell') = {result} ({type(result).__name__})")

is_valid, error = type_mapper.validate(42, "IntCell")
print(f"Validate 42 as IntCell: {'✅ Valid' if is_valid else f'❌ {error}'}")

is_valid, error = type_mapper.validate("hello", "IntCell")
print(f"Validate 'hello' as IntCell: {'✅ Valid' if is_valid else f'❌ {error}'}")

print("\n" + "=" * 60)
print("PHASE 4 TEST COMPLETE!")
print("=" * 60)
