"""
JEP Expression Tokenizer.

Tokenizes KNIME JEP expressions into tokens for parsing.
"""
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class TokenType(Enum):
    """Token types for JEP expressions."""
    COLUMN_REF = auto()    # $columnName$
    NUMBER = auto()        # 123, 45.67
    STRING = auto()        # "string"
    FUNCTION = auto()      # abs, sqrt, if
    OPERATOR = auto()      # +, -, *, /, ^, <, >, =, !
    COMPARISON = auto()    # ==, !=, <=, >=, <, >
    LOGICAL = auto()       # AND, OR, NOT
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    COMMA = auto()         # ,
    EOF = auto()           # End of expression


@dataclass
class Token:
    """Represents a single token."""
    type: TokenType
    value: str
    position: int


# JEP Functions mapping
JEP_FUNCTIONS = {
    # Math functions
    'abs', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sqrt', 'log', 'log10', 'exp', 'pow', 'round', 'floor', 'ceil',
    'min', 'max', 'mod',
    # String functions  
    'length', 'substr', 'upper', 'lower', 'trim', 'replace',
    'indexOf', 'startsWith', 'endsWith', 'contains',
    # Conditional
    'if', 'isnan', 'isnull', 'missing',
    # Logical (as functions)
    'and', 'or', 'not',
}

# Token patterns in order of precedence
TOKEN_PATTERNS = [
    (TokenType.COLUMN_REF, r'\$([A-Za-z_][A-Za-z0-9_]*)\$'),
    (TokenType.STRING, r'"([^"]*)"'),
    (TokenType.STRING, r"'([^']*)'"),
    (TokenType.NUMBER, r'\d+\.?\d*'),
    (TokenType.COMPARISON, r'(==|!=|<=|>=|<>)'),
    (TokenType.OPERATOR, r'[+\-*/^%<>=!]'),
    (TokenType.LPAREN, r'\('),
    (TokenType.RPAREN, r'\)'),
    (TokenType.COMMA, r','),
]


class JEPTokenizer:
    """Tokenizes JEP expressions."""
    
    def __init__(self, expression: str):
        self.expression = expression
        self.position = 0
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        """Convert expression to list of tokens."""
        while self.position < len(self.expression):
            # Skip whitespace
            if self.expression[self.position].isspace():
                self.position += 1
                continue
            
            matched = False
            remaining = self.expression[self.position:]
            
            # Check $column$ pattern FIRST (highest priority)
            col_match = re.match(r'\$([A-Za-z_][A-Za-z0-9_]*)\$', remaining)
            if col_match:
                self.tokens.append(Token(
                    TokenType.COLUMN_REF,
                    col_match.group(1),
                    self.position
                ))
                self.position += len(col_match.group(0))
                continue
            
            # Try other patterns (numbers, strings, operators)
            for token_type, pattern in TOKEN_PATTERNS:
                if token_type == TokenType.COLUMN_REF:
                    continue  # Already handled above
                match = re.match(pattern, remaining)
                if match:
                    value = match.group(1) if match.lastindex else match.group(0)
                    self.tokens.append(Token(token_type, value, self.position))
                    self.position += len(match.group(0))
                    matched = True
                    break
            
            if matched:
                continue
            
            # Check for function/identifier LAST
            func_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', remaining)
            if func_match:
                name = func_match.group(1).lower()
                if name in JEP_FUNCTIONS:
                    self.tokens.append(Token(
                        TokenType.FUNCTION,
                        func_match.group(1),
                        self.position
                    ))
                elif name in ('and', 'or', 'not'):
                    self.tokens.append(Token(
                        TokenType.LOGICAL,
                        name.upper(),
                        self.position
                    ))
                else:
                    # Unknown identifier - treat as column ref without $
                    self.tokens.append(Token(
                        TokenType.COLUMN_REF,
                        func_match.group(1),
                        self.position
                    ))
                self.position += len(func_match.group(1))
                continue
            
            # Skip unknown character (like standalone $)
            self.position += 1
        
        self.tokens.append(Token(TokenType.EOF, '', self.position))
        return self.tokens


def tokenize(expression: str) -> List[Token]:
    """Tokenize a JEP expression."""
    return JEPTokenizer(expression).tokenize()
