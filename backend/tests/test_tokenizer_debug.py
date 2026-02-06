"""Test tokenizer and parser."""
import sys
sys.path.insert(0, '.')

from app.services.expression.jep_tokenizer import tokenize
from app.services.expression import convert

# Test tokenization
expr = "$TxCet$"
tokens = tokenize(expr)
print(f"Expression: {expr}")
print(f"Tokens: {[(t.type.name, t.value) for t in tokens]}")

# Test conversion
expr2 = "abs($TxCetAnualContrato$ - $VrCetCalculada$)"
print(f"\nExpression: {expr2}")
result = convert(expr2)
print(f"Result: {result}")

# Test simple addition
expr3 = "$a$ + $b$"
print(f"\nExpression: {expr3}")
result3 = convert(expr3)
print(f"Result: {result3}")
