"""
JEP Expression Parser.

Main entry point for converting KNIME JEP expressions to Pandas/NumPy code.
"""
import re
from typing import List, Optional, Set, Tuple

from .jep_tokenizer import Token, TokenType, tokenize
from .jep_ast import (
    Node, ColumnRef, Number, String, BinaryOp, UnaryOp,
    FunctionCall, Comparison, Conditional
)
from .jep_codegen import generate_code, JEPCodeGenerator


class JEPParser:
    """
    Recursive descent parser for JEP expressions.
    
    Grammar:
        expression -> term (('+' | '-') term)*
        term -> factor (('*' | '/') factor)*
        factor -> unary ('^' unary)*
        unary -> ('-' | 'NOT')? comparison
        comparison -> primary (('<' | '>' | '<=' | '>=' | '==' | '!=') primary)?
        primary -> NUMBER | STRING | COLUMN_REF | function_call | '(' expression ')'
        function_call -> FUNCTION '(' args ')'
        args -> expression (',' expression)*
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
    
    def current(self) -> Token:
        """Get current token."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return self.tokens[-1]  # EOF
    
    def advance(self) -> Token:
        """Advance and return previous token."""
        token = self.current()
        self.position += 1
        return token
    
    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self.current().type in types
    
    def expect(self, token_type: TokenType) -> Token:
        """Expect current token to be of given type."""
        if self.match(token_type):
            return self.advance()
        raise SyntaxError(
            f"Expected {token_type.name}, got {self.current().type.name} "
            f"at position {self.current().position}"
        )
    
    def parse(self) -> Node:
        """Parse the expression."""
        return self.expression()
    
    def expression(self) -> Node:
        """Parse expression: term (('+' | '-') term)*"""
        left = self.term()
        
        while self.match(TokenType.OPERATOR):
            op = self.current().value
            if op in ('+', '-'):
                self.advance()
                right = self.term()
                left = BinaryOp(left, op, right)
            else:
                break
        
        return left
    
    def term(self) -> Node:
        """Parse term: factor (('*' | '/') factor)*"""
        left = self.factor()
        
        while self.match(TokenType.OPERATOR):
            op = self.current().value
            if op in ('*', '/', '%'):
                self.advance()
                right = self.factor()
                left = BinaryOp(left, op, right)
            else:
                break
        
        return left
    
    def factor(self) -> Node:
        """Parse factor: unary ('^' unary)*"""
        left = self.unary()
        
        while self.match(TokenType.OPERATOR):
            op = self.current().value
            if op == '^':
                self.advance()
                right = self.unary()
                left = BinaryOp(left, op, right)
            else:
                break
        
        return left
    
    def unary(self) -> Node:
        """Parse unary: ('-' | 'NOT')? comparison"""
        if self.match(TokenType.OPERATOR) and self.current().value == '-':
            self.advance()
            operand = self.unary()
            return UnaryOp('-', operand)
        
        if self.match(TokenType.LOGICAL) and self.current().value.upper() == 'NOT':
            self.advance()
            operand = self.unary()
            return UnaryOp('NOT', operand)
        
        return self.comparison()
    
    def comparison(self) -> Node:
        """Parse comparison: primary (op primary)?"""
        left = self.primary()
        
        if self.match(TokenType.COMPARISON, TokenType.OPERATOR):
            op = self.current().value
            if op in ('=', '==', '!=', '<>', '<', '>', '<=', '>='):
                self.advance()
                right = self.primary()
                return Comparison(left, op, right)
        
        return left
    
    def primary(self) -> Node:
        """Parse primary: NUMBER | STRING | COLUMN_REF | function | '(' expr ')'"""
        token = self.current()
        
        # Number
        if self.match(TokenType.NUMBER):
            self.advance()
            return Number(float(token.value))
        
        # String
        if self.match(TokenType.STRING):
            self.advance()
            return String(token.value)
        
        # Column reference
        if self.match(TokenType.COLUMN_REF):
            self.advance()
            return ColumnRef(token.value)
        
        # Function call
        if self.match(TokenType.FUNCTION):
            return self.function_call()
        
        # Parentheses
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        raise SyntaxError(
            f"Unexpected token {token.type.name}: '{token.value}' "
            f"at position {token.position}"
        )
    
    def function_call(self) -> Node:
        """Parse function call: FUNCTION '(' args ')'"""
        name_token = self.expect(TokenType.FUNCTION)
        self.expect(TokenType.LPAREN)
        
        args = []
        if not self.match(TokenType.RPAREN):
            args.append(self.expression())
            while self.match(TokenType.COMMA):
                self.advance()
                args.append(self.expression())
        
        self.expect(TokenType.RPAREN)
        
        # Special handling for if()
        if name_token.value.lower() == 'if' and len(args) == 3:
            return Conditional(args[0], args[1], args[2])
        
        return FunctionCall(name_token.value, args)


def parse(expression: str) -> Node:
    """Parse a JEP expression into AST."""
    tokens = tokenize(expression)
    parser = JEPParser(tokens)
    return parser.parse()


def convert(expression: str, df_var: str = 'df') -> str:
    """
    Convert JEP expression to Pandas/NumPy code.
    
    Args:
        expression: JEP expression string (e.g., "abs($col1$ - $col2$)")
        df_var: DataFrame variable name (default: 'df')
        
    Returns:
        Pandas/NumPy code string (e.g., "np.abs(df['col1'] - df['col2'])")
    
    Examples:
        >>> convert("$col$")
        "df['col']"
        
        >>> convert("abs($a$ - $b$)")
        "np.abs((df['a'] - df['b']))"
        
        >>> convert("if($x$ > 0, $y$, 0)")
        "np.where((df['x'] > 0), df['y'], 0)"
    """
    if not expression or not expression.strip():
        return "''"
    
    try:
        ast = parse(expression)
        return generate_code(ast, df_var)
    except Exception as e:
        # Fallback: basic column reference replacement
        return _fallback_convert(expression, df_var)


def _fallback_convert(expression: str, df_var: str) -> str:
    """
    Simple fallback conversion using regex.
    
    Used when full parsing fails.
    """
    # Replace $column$ with df['column']
    result = re.sub(
        r'\$([A-Za-z_][A-Za-z0-9_]*)\$',
        rf"{df_var}['\1']",
        expression
    )
    
    # Replace common functions
    result = re.sub(r'\babs\s*\(', 'np.abs(', result)
    result = re.sub(r'\bsqrt\s*\(', 'np.sqrt(', result)
    result = re.sub(r'\bround\s*\(', 'np.round(', result)
    result = re.sub(r'\bfloor\s*\(', 'np.floor(', result)
    result = re.sub(r'\bceil\s*\(', 'np.ceil(', result)
    result = re.sub(r'\bmin\s*\(', 'np.minimum(', result)
    result = re.sub(r'\bmax\s*\(', 'np.maximum(', result)
    
    # Replace ^ with **
    result = result.replace('^', '**')
    
    return result


def get_imports() -> Set[str]:
    """Get required imports for generated code."""
    return {'import pandas as pd', 'import numpy as np'}
