"""
JEP Expression Parser Tests.

Tests for the KNIME JEP to Pandas/NumPy converter.
"""
import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.expression import convert
from app.services.expression.jep_tokenizer import tokenize, TokenType
from app.services.expression.jep_parser import parse
from app.services.expression.jep_ast import ColumnRef, Number, BinaryOp, FunctionCall


class TestTokenizer:
    """Tests for JEP tokenizer."""
    
    def test_column_reference(self):
        """Test $column$ tokenization."""
        tokens = tokenize("$myColumn$")
        assert tokens[0].type == TokenType.COLUMN_REF
        assert tokens[0].value == "myColumn"
    
    def test_number(self):
        """Test numeric literals."""
        tokens = tokenize("123.45")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "123.45"
    
    def test_operators(self):
        """Test operator tokenization."""
        tokens = tokenize("+ - * /")
        ops = [t for t in tokens if t.type == TokenType.OPERATOR]
        assert len(ops) == 4
        assert [t.value for t in ops] == ['+', '-', '*', '/']
    
    def test_function(self):
        """Test function tokenization."""
        tokens = tokenize("abs(")
        assert tokens[0].type == TokenType.FUNCTION
        assert tokens[0].value == "abs"
    
    def test_complex_expression(self):
        """Test complex expression."""
        tokens = tokenize("abs($col1$ - $col2$)")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        expected = [
            TokenType.FUNCTION,
            TokenType.LPAREN,
            TokenType.COLUMN_REF,
            TokenType.OPERATOR,
            TokenType.COLUMN_REF,
            TokenType.RPAREN
        ]
        assert types == expected


class TestParser:
    """Tests for JEP parser."""
    
    def test_parse_column(self):
        """Test parsing column reference."""
        ast = parse("$myCol$")
        assert isinstance(ast, ColumnRef)
        assert ast.name == "myCol"
    
    def test_parse_number(self):
        """Test parsing number."""
        ast = parse("42")
        assert isinstance(ast, Number)
        assert ast.value == 42.0
    
    def test_parse_binary_op(self):
        """Test parsing binary operation."""
        ast = parse("$a$ + $b$")
        assert isinstance(ast, BinaryOp)
        assert ast.op == '+'
        assert isinstance(ast.left, ColumnRef)
        assert isinstance(ast.right, ColumnRef)
    
    def test_parse_function(self):
        """Test parsing function call."""
        ast = parse("abs($x$)")
        assert isinstance(ast, FunctionCall)
        assert ast.name == "abs"
        assert len(ast.args) == 1


class TestConversion:
    """Tests for JEP to Python conversion."""
    
    @pytest.mark.parametrize("jep,expected", [
        # Simple column reference
        ("$col$", "df['col']"),
        
        # Binary operations
        ("$a$ + $b$", "(df['a'] + df['b'])"),
        ("$a$ - $b$", "(df['a'] - df['b'])"),
        ("$a$ * $b$", "(df['a'] * df['b'])"),
        ("$x$ / 100", "(df['x'] / 100)"),
        
        # Power operator
        ("$x$ ^ 2", "(df['x'] ** 2)"),
        
        # Math functions
        ("abs($x$)", "np.abs(df['x'])"),
        ("sqrt($x$)", "np.sqrt(df['x'])"),
        ("round($x$)", "np.round(df['x'])"),
        
        # Nested expression
        ("abs($a$ - $b$)", "np.abs((df['a'] - df['b']))"),
    ])
    def test_conversions(self, jep: str, expected: str):
        """Test various JEP conversions."""
        result = convert(jep)
        assert result == expected
    
    def test_custom_df_var(self):
        """Test custom DataFrame variable name."""
        result = convert("$col$", df_var="data")
        assert result == "data['col']"
    
    def test_empty_expression(self):
        """Test empty expression."""
        result = convert("")
        assert result == "''"
    
    def test_real_knime_expression(self):
        """Test real KNIME expression from workflow."""
        expr = "abs($TxCetAnualContrato$ - $VrCetCalculada$)"
        result = convert(expr)
        expected = "np.abs((df['TxCetAnualContrato'] - df['VrCetCalculada']))"
        assert result == expected


class TestConditional:
    """Tests for conditional expressions."""
    
    def test_if_function(self):
        """Test if(condition, then, else)."""
        expr = "if($x$ > 0, $y$, 0)"
        result = convert(expr)
        assert "np.where" in result
        assert "df['x']" in result
        assert "df['y']" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
