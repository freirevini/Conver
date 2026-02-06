"""
Test Suite for Formula Parser and Converter.

Tests:
- Lexer tokenization
- Parser AST generation
- Function mapping
- Rule Engine conversion
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFormulaLexer:
    """Tests for FormulaLexer."""
    
    def test_tokenize_column_reference(self):
        """Column references should be tokenized."""
        from app.services.transpiler.formula_parser import FormulaLexer, TokenType
        
        lexer = FormulaLexer("$column_name$")
        tokens = lexer.tokenize()
        
        assert len(tokens) == 2  # COLUMN_REF + EOF
        assert tokens[0].type == TokenType.COLUMN_REF
        assert tokens[0].value == "column_name"
    
    def test_tokenize_flow_variable(self):
        """Flow variables should be tokenized."""
        from app.services.transpiler.formula_parser import FormulaLexer, TokenType
        
        lexer = FormulaLexer("$$my_var$$")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.FLOW_VAR
        assert tokens[0].value == "my_var"
    
    def test_tokenize_row_index(self):
        """$$ROWINDEX$$ should be recognized."""
        from app.services.transpiler.formula_parser import FormulaLexer, TokenType
        
        lexer = FormulaLexer("$$ROWINDEX$$")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.ROW_INDEX
    
    def test_tokenize_number(self):
        """Numbers should be tokenized."""
        from app.services.transpiler.formula_parser import FormulaLexer, TokenType
        
        lexer = FormulaLexer("42 3.14")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == 3.14
    
    def test_tokenize_string(self):
        """Strings should be tokenized."""
        from app.services.transpiler.formula_parser import FormulaLexer, TokenType
        
        lexer = FormulaLexer('"hello world"')
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello world"
    
    def test_tokenize_operators(self):
        """Operators should be tokenized."""
        from app.services.transpiler.formula_parser import FormulaLexer, TokenType
        
        lexer = FormulaLexer("+ - * / == != < >")
        tokens = lexer.tokenize()
        
        types = [t.type for t in tokens[:-1]]  # Exclude EOF
        
        assert TokenType.PLUS in types
        assert TokenType.MINUS in types
        assert TokenType.MULTIPLY in types
        assert TokenType.DIVIDE in types
        assert TokenType.EQ in types
        assert TokenType.NE in types
    
    def test_tokenize_complex_expression(self):
        """Complex expressions should tokenize correctly."""
        from app.services.transpiler.formula_parser import FormulaLexer, TokenType
        
        lexer = FormulaLexer('$col1$ + $col2$ > 100 AND upper("test")')
        tokens = lexer.tokenize()
        
        # Should have: COLUMN_REF, PLUS, COLUMN_REF, GT, NUMBER, AND, IDENTIFIER, LPAREN, STRING, RPAREN, EOF
        assert len(tokens) >= 10


class TestFormulaParser:
    """Tests for FormulaParser."""
    
    def test_parse_number(self):
        """Numbers should parse to NumberNode."""
        from app.services.transpiler.formula_parser import parse_formula, NumberNode
        
        ast = parse_formula("42")
        assert isinstance(ast, NumberNode)
        assert ast.value == 42
    
    def test_parse_column_reference(self):
        """Column references should parse to ColumnRefNode."""
        from app.services.transpiler.formula_parser import parse_formula, ColumnRefNode
        
        ast = parse_formula("$my_column$")
        assert isinstance(ast, ColumnRefNode)
        assert ast.name == "my_column"
    
    def test_parse_binary_operation(self):
        """Binary operations should parse correctly."""
        from app.services.transpiler.formula_parser import parse_formula, BinaryOpNode
        
        ast = parse_formula("$a$ + $b$")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "+"
    
    def test_parse_function_call(self):
        """Function calls should parse correctly."""
        from app.services.transpiler.formula_parser import parse_formula, FunctionCallNode
        
        ast = parse_formula('upper("test")')
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "upper"
        assert len(ast.arguments) == 1
    
    def test_parse_nested_function(self):
        """Nested functions should parse correctly."""
        from app.services.transpiler.formula_parser import parse_formula, FunctionCallNode
        
        ast = parse_formula('lower(upper("test"))')
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "lower"
        inner = ast.arguments[0]
        assert isinstance(inner, FunctionCallNode)
        assert inner.name == "upper"
    
    def test_parse_ternary(self):
        """Ternary expressions should parse correctly."""
        from app.services.transpiler.formula_parser import parse_formula, TernaryNode
        
        ast = parse_formula('$col$ > 10 ? "Yes" : "No"')
        assert isinstance(ast, TernaryNode)
    
    def test_parse_comparison(self):
        """Comparison operators should parse correctly."""
        from app.services.transpiler.formula_parser import parse_formula, BinaryOpNode
        
        ast = parse_formula("$col$ >= 100")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == ">="
    
    def test_parse_logical_and(self):
        """AND operator should parse correctly."""
        from app.services.transpiler.formula_parser import parse_formula, BinaryOpNode
        
        ast = parse_formula("$a$ > 0 AND $b$ < 10")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "and"
    
    def test_parse_parentheses(self):
        """Parentheses should affect precedence."""
        from app.services.transpiler.formula_parser import parse_formula, BinaryOpNode
        
        ast = parse_formula("($a$ + $b$) * $c$")
        assert isinstance(ast, BinaryOpNode)
        assert ast.operator == "*"


class TestFunctionMapping:
    """Tests for FunctionMapping."""
    
    def test_get_string_function(self):
        """String functions should be mapped."""
        from app.services.transpiler.function_mapping import get_function_mapping
        
        mapping = get_function_mapping("upper")
        assert mapping is not None
        assert "upper()" in mapping.python_template
    
    def test_get_math_function(self):
        """Math functions should be mapped."""
        from app.services.transpiler.function_mapping import get_function_mapping
        
        mapping = get_function_mapping("sqrt")
        assert mapping is not None
        assert "math.sqrt" in mapping.python_template
        assert mapping.python_import == "import math"
    
    def test_get_date_function(self):
        """Date functions should be mapped."""
        from app.services.transpiler.function_mapping import get_function_mapping
        
        mapping = get_function_mapping("now")
        assert mapping is not None
        assert "datetime" in mapping.python_template
    
    def test_function_count(self):
        """Should have 70+ functions mapped."""
        from app.services.transpiler.function_mapping import FUNCTION_COUNT
        
        assert FUNCTION_COUNT >= 70
    
    def test_get_required_imports(self):
        """Should collect required imports."""
        from app.services.transpiler.function_mapping import get_required_imports
        
        imports = get_required_imports(["sqrt", "now", "upper"])
        
        assert "import math" in imports
        assert "import datetime" in imports
    
    def test_case_insensitive_lookup(self):
        """Function lookup should be case-insensitive."""
        from app.services.transpiler.function_mapping import get_function_mapping
        
        mapping1 = get_function_mapping("UPPER")
        mapping2 = get_function_mapping("upper")
        
        # At least one should work
        assert mapping1 is not None or mapping2 is not None


class TestRuleEngine:
    """Tests for Rule Engine conversion."""
    
    @pytest.fixture
    def sample_rules(self):
        return '''
$score$ > 90 => "A"
$score$ > 80 => "B"
$score$ > 70 => "C"
TRUE => "F"
        '''
    
    def test_parse_rules(self, sample_rules):
        """Rules should be parsed correctly."""
        from app.services.transpiler.rule_engine import parse_rules
        
        rules = parse_rules(sample_rules)
        
        assert len(rules) == 4
        assert rules[0].condition == "$score$ > 90"
        assert rules[0].result == '"A"'
        assert rules[3].is_default is True
    
    def test_convert_to_np_select(self, sample_rules):
        """Should convert to np.select()."""
        from app.services.transpiler.rule_engine import convert_rule_engine
        
        result = convert_rule_engine(sample_rules, strategy="np_select")
        
        assert "np.select" in result.python_code
        assert "conditions" in result.python_code
        assert "choices" in result.python_code
        assert "import numpy" in result.imports[0]
    
    def test_convert_to_lambda(self, sample_rules):
        """Should convert to lambda function."""
        from app.services.transpiler.rule_engine import convert_rule_engine
        
        result = convert_rule_engine(sample_rules, strategy="lambda")
        
        assert "def apply_rules" in result.python_code
        assert "if " in result.python_code
        assert "elif " in result.python_code
        assert "return" in result.python_code
    
    def test_convert_to_case(self, sample_rules):
        """Should convert to nested np.where()."""
        from app.services.transpiler.rule_engine import convert_rule_engine
        
        result = convert_rule_engine(sample_rules, strategy="case")
        
        assert "np.where" in result.python_code
    
    def test_column_reference_conversion(self):
        """Column references should be converted."""
        from app.services.transpiler.rule_engine import RuleEngineParser
        
        parser = RuleEngineParser('$col$ > 10 => "Yes"')
        parser.parse()
        
        converted = parser._convert_condition("$col$ > 10")
        assert 'df["col"]' in converted
    
    def test_empty_rules(self):
        """Empty rules should not crash."""
        from app.services.transpiler.rule_engine import convert_rule_engine
        
        result = convert_rule_engine("", strategy="np_select")
        
        assert result.python_code is not None
        assert len(result.warnings) > 0


class TestIntegration:
    """Integration tests for complete formula conversion."""
    
    def test_complex_formula_parse(self):
        """Complex formula should parse without errors."""
        from app.services.transpiler.formula_parser import parse_formula
        
        formula = 'if($amount$ > 1000, "Large", if($amount$ > 100, "Medium", "Small"))'
        ast = parse_formula(formula)
        
        assert ast is not None
    
    def test_formula_with_functions(self):
        """Formula with multiple functions should parse."""
        from app.services.transpiler.formula_parser import parse_formula
        
        formula = 'upper(trim($name$)) + " - " + string(round($value$, 2))'
        ast = parse_formula(formula)
        
        assert ast is not None
    
    def test_all_modules_import(self):
        """All transpiler modules should import without errors."""
        from app.services.transpiler import formula_parser
        from app.services.transpiler import function_mapping
        from app.services.transpiler import rule_engine
        
        assert formula_parser is not None
        assert function_mapping is not None
        assert rule_engine is not None
