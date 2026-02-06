"""
KNIME Formula Parser - AST-based Expression Converter.

Converts KNIME expressions to Python code:
- $column$ → df["column"]
- $$ROWINDEX$$ → df.index
- Functions (join, substr, etc.) → Python equivalents

Architecture:
    Tokenizer → Parser → AST → Transformer → Python Code
"""
import re
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==================== Token Types ====================

class TokenType(Enum):
    """Token types for KNIME formula lexer."""
    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Identifiers
    COLUMN_REF = auto()      # $column$
    FLOW_VAR = auto()        # $$var$$
    ROW_INDEX = auto()       # $$ROWINDEX$$
    ROW_COUNT = auto()       # $$ROWCOUNT$$
    ROW_ID = auto()          # $$ROWID$$
    IDENTIFIER = auto()      # function names
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    
    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    SEMICOLON = auto()
    QUESTION = auto()  # Ternary
    COLON = auto()
    
    # Special
    EOF = auto()
    NEWLINE = auto()


@dataclass
class Token:
    """Single token from lexer."""
    type: TokenType
    value: Any
    line: int = 1
    column: int = 0
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r})"


# ==================== Lexer ====================

class FormulaLexer:
    """
    Tokenizer for KNIME formulas.
    
    Handles:
    - Column references: $column_name$
    - Flow variables: $$variable$$
    - Row specials: $$ROWINDEX$$, $$ROWCOUNT$$, $$ROWID$$
    - Literals: numbers, strings, booleans
    - Operators and functions
    """
    
    KEYWORDS = {
        "AND": TokenType.AND,
        "OR": TokenType.OR,
        "NOT": TokenType.NOT,
        "TRUE": TokenType.BOOLEAN,
        "FALSE": TokenType.BOOLEAN,
    }
    
    OPERATORS = {
        "+": TokenType.PLUS,
        "-": TokenType.MINUS,
        "*": TokenType.MULTIPLY,
        "/": TokenType.DIVIDE,
        "%": TokenType.MODULO,
        "^": TokenType.POWER,
        "(": TokenType.LPAREN,
        ")": TokenType.RPAREN,
        ",": TokenType.COMMA,
        ";": TokenType.SEMICOLON,
        "?": TokenType.QUESTION,
        ":": TokenType.COLON,
    }
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire input."""
        while self.pos < len(self.text):
            self._scan_token()
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens
    
    def _scan_token(self):
        """Scan next token."""
        self._skip_whitespace()
        
        if self.pos >= len(self.text):
            return
        
        char = self.text[self.pos]
        
        # Column reference: $column$
        if char == "$":
            self._scan_reference()
        # String literal
        elif char == '"':
            self._scan_string()
        # Number
        elif char.isdigit() or (char == "." and self._peek_is_digit()):
            self._scan_number()
        # Identifier or keyword
        elif char.isalpha() or char == "_":
            self._scan_identifier()
        # Comparison operators
        elif char in "=!<>":
            self._scan_comparison()
        # Single-char operators
        elif char in self.OPERATORS:
            self._add_token(self.OPERATORS[char], char)
            self._advance()
        # Newline
        elif char == "\n":
            self.line += 1
            self.column = 1
            self._advance()
        else:
            self._advance()  # Skip unknown
    
    def _scan_reference(self):
        """Scan column reference or flow variable."""
        start = self.pos
        self._advance()  # Skip first $
        
        if self.pos < len(self.text) and self.text[self.pos] == "$":
            # Flow variable: $$var$$
            self._advance()  # Skip second $
            name_start = self.pos
            
            while self.pos < len(self.text) and self.text[self.pos] not in "$\n":
                self._advance()
            
            name = self.text[name_start:self.pos]
            
            # Skip closing $$
            if self.pos < len(self.text) and self.text[self.pos] == "$":
                self._advance()
            if self.pos < len(self.text) and self.text[self.pos] == "$":
                self._advance()
            
            # Check for special row variables
            if name == "ROWINDEX":
                self._add_token(TokenType.ROW_INDEX, name)
            elif name == "ROWCOUNT":
                self._add_token(TokenType.ROW_COUNT, name)
            elif name == "ROWID":
                self._add_token(TokenType.ROW_ID, name)
            else:
                self._add_token(TokenType.FLOW_VAR, name)
        else:
            # Column reference: $column$
            name_start = self.pos
            
            while self.pos < len(self.text) and self.text[self.pos] != "$":
                self._advance()
            
            name = self.text[name_start:self.pos]
            
            if self.pos < len(self.text) and self.text[self.pos] == "$":
                self._advance()  # Skip closing $
            
            self._add_token(TokenType.COLUMN_REF, name)
    
    def _scan_string(self):
        """Scan string literal."""
        self._advance()  # Skip opening quote
        start = self.pos
        
        while self.pos < len(self.text) and self.text[self.pos] != '"':
            if self.text[self.pos] == "\\":
                self._advance()  # Skip escape
            self._advance()
        
        value = self.text[start:self.pos]
        
        if self.pos < len(self.text):
            self._advance()  # Skip closing quote
        
        self._add_token(TokenType.STRING, value)
    
    def _scan_number(self):
        """Scan numeric literal."""
        start = self.pos
        
        while self.pos < len(self.text) and (
            self.text[self.pos].isdigit() or self.text[self.pos] == "."
        ):
            self._advance()
        
        value = self.text[start:self.pos]
        
        if "." in value:
            self._add_token(TokenType.NUMBER, float(value))
        else:
            self._add_token(TokenType.NUMBER, int(value))
    
    def _scan_identifier(self):
        """Scan identifier or keyword."""
        start = self.pos
        
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self._advance()
        
        value = self.text[start:self.pos]
        upper = value.upper()
        
        if upper in self.KEYWORDS:
            token_type = self.KEYWORDS[upper]
            if token_type == TokenType.BOOLEAN:
                self._add_token(token_type, upper == "TRUE")
            else:
                self._add_token(token_type, value)
        else:
            self._add_token(TokenType.IDENTIFIER, value)
    
    def _scan_comparison(self):
        """Scan comparison operators."""
        char = self.text[self.pos]
        next_char = self.text[self.pos + 1] if self.pos + 1 < len(self.text) else ""
        
        if char == "=" and next_char == "=":
            self._add_token(TokenType.EQ, "==")
            self._advance()
            self._advance()
        elif char == "!" and next_char == "=":
            self._add_token(TokenType.NE, "!=")
            self._advance()
            self._advance()
        elif char == "<" and next_char == "=":
            self._add_token(TokenType.LE, "<=")
            self._advance()
            self._advance()
        elif char == ">" and next_char == "=":
            self._add_token(TokenType.GE, ">=")
            self._advance()
            self._advance()
        elif char == "<":
            self._add_token(TokenType.LT, "<")
            self._advance()
        elif char == ">":
            self._add_token(TokenType.GT, ">")
            self._advance()
        elif char == "=":
            self._add_token(TokenType.EQ, "=")
            self._advance()
        else:
            self._advance()
    
    def _skip_whitespace(self):
        """Skip whitespace characters."""
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r":
            self._advance()
    
    def _advance(self):
        """Advance position by one."""
        self.pos += 1
        self.column += 1
    
    def _add_token(self, token_type: TokenType, value: Any):
        """Add token to list."""
        self.tokens.append(Token(token_type, value, self.line, self.column))
    
    def _peek_is_digit(self) -> bool:
        """Check if next char is digit."""
        return (
            self.pos + 1 < len(self.text) and 
            self.text[self.pos + 1].isdigit()
        )


# ==================== AST Nodes ====================

class ASTNode(ABC):
    """Base class for AST nodes."""
    
    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> Any:
        pass


@dataclass
class NumberNode(ASTNode):
    """Numeric literal."""
    value: Union[int, float]
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_number(self)


@dataclass
class StringNode(ASTNode):
    """String literal."""
    value: str
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_string(self)


@dataclass
class BooleanNode(ASTNode):
    """Boolean literal."""
    value: bool
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_boolean(self)


@dataclass
class ColumnRefNode(ASTNode):
    """Column reference: $column$."""
    name: str
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_column_ref(self)


@dataclass
class FlowVarNode(ASTNode):
    """Flow variable: $$var$$."""
    name: str
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_flow_var(self)


@dataclass
class RowSpecialNode(ASTNode):
    """Row special: $$ROWINDEX$$, etc."""
    name: str  # ROWINDEX, ROWCOUNT, ROWID
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_row_special(self)


@dataclass
class BinaryOpNode(ASTNode):
    """Binary operation."""
    left: ASTNode
    operator: str
    right: ASTNode
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOpNode(ASTNode):
    """Unary operation."""
    operator: str
    operand: ASTNode
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_unary_op(self)


@dataclass
class FunctionCallNode(ASTNode):
    """Function call."""
    name: str
    arguments: List[ASTNode]
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_function_call(self)


@dataclass
class TernaryNode(ASTNode):
    """Ternary conditional: cond ? true : false."""
    condition: ASTNode
    true_value: ASTNode
    false_value: ASTNode
    
    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_ternary(self)


# ==================== Parser ====================

class FormulaParser:
    """
    Recursive descent parser for KNIME formulas.
    
    Grammar:
        expression     → ternary
        ternary        → or ("?" expression ":" expression)?
        or             → and ("OR" and)*
        and            → equality ("AND" equality)*
        equality       → comparison (("==" | "!=") comparison)*
        comparison     → term (("<" | ">" | "<=" | ">=") term)*
        term           → factor (("+" | "-") factor)*
        factor         → unary (("*" | "/" | "%") unary)*
        unary          → ("NOT" | "-") unary | power
        power          → call ("^" unary)?
        call           → primary ("(" arguments? ")")?
        primary        → NUMBER | STRING | BOOLEAN | COLUMN_REF | ...
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def parse(self) -> ASTNode:
        """Parse tokens into AST."""
        return self._expression()
    
    def _expression(self) -> ASTNode:
        return self._ternary()
    
    def _ternary(self) -> ASTNode:
        expr = self._or()
        
        if self._match(TokenType.QUESTION):
            true_val = self._expression()
            self._consume(TokenType.COLON, "Expected ':' in ternary")
            false_val = self._expression()
            return TernaryNode(expr, true_val, false_val)
        
        return expr
    
    def _or(self) -> ASTNode:
        left = self._and()
        
        while self._match(TokenType.OR):
            right = self._and()
            left = BinaryOpNode(left, "or", right)
        
        return left
    
    def _and(self) -> ASTNode:
        left = self._equality()
        
        while self._match(TokenType.AND):
            right = self._equality()
            left = BinaryOpNode(left, "and", right)
        
        return left
    
    def _equality(self) -> ASTNode:
        left = self._comparison()
        
        while self._match(TokenType.EQ, TokenType.NE):
            op = "==" if self._previous().type == TokenType.EQ else "!="
            right = self._comparison()
            left = BinaryOpNode(left, op, right)
        
        return left
    
    def _comparison(self) -> ASTNode:
        left = self._term()
        
        while self._match(TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            op = self._previous().value
            right = self._term()
            left = BinaryOpNode(left, op, right)
        
        return left
    
    def _term(self) -> ASTNode:
        left = self._factor()
        
        while self._match(TokenType.PLUS, TokenType.MINUS):
            op = "+" if self._previous().type == TokenType.PLUS else "-"
            right = self._factor()
            left = BinaryOpNode(left, op, right)
        
        return left
    
    def _factor(self) -> ASTNode:
        left = self._unary()
        
        while self._match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            prev = self._previous().type
            if prev == TokenType.MULTIPLY:
                op = "*"
            elif prev == TokenType.DIVIDE:
                op = "/"
            else:
                op = "%"
            right = self._unary()
            left = BinaryOpNode(left, op, right)
        
        return left
    
    def _unary(self) -> ASTNode:
        if self._match(TokenType.NOT):
            operand = self._unary()
            return UnaryOpNode("not", operand)
        
        if self._match(TokenType.MINUS):
            operand = self._unary()
            return UnaryOpNode("-", operand)
        
        return self._power()
    
    def _power(self) -> ASTNode:
        left = self._call()
        
        if self._match(TokenType.POWER):
            right = self._unary()
            left = BinaryOpNode(left, "**", right)
        
        return left
    
    def _call(self) -> ASTNode:
        expr = self._primary()
        
        if isinstance(expr, ColumnRefNode):
            # Column refs don't become function calls
            return expr
        
        # Check for function call
        if hasattr(expr, 'name') and self._match(TokenType.LPAREN):
            args = self._arguments()
            self._consume(TokenType.RPAREN, "Expected ')' after arguments")
            return FunctionCallNode(expr.name if hasattr(expr, 'name') else str(expr), args)
        
        return expr
    
    def _arguments(self) -> List[ASTNode]:
        args = []
        
        if not self._check(TokenType.RPAREN):
            args.append(self._expression())
            
            while self._match(TokenType.COMMA, TokenType.SEMICOLON):
                args.append(self._expression())
        
        return args
    
    def _primary(self) -> ASTNode:
        if self._match(TokenType.NUMBER):
            return NumberNode(self._previous().value)
        
        if self._match(TokenType.STRING):
            return StringNode(self._previous().value)
        
        if self._match(TokenType.BOOLEAN):
            return BooleanNode(self._previous().value)
        
        if self._match(TokenType.COLUMN_REF):
            return ColumnRefNode(self._previous().value)
        
        if self._match(TokenType.FLOW_VAR):
            return FlowVarNode(self._previous().value)
        
        if self._match(TokenType.ROW_INDEX, TokenType.ROW_COUNT, TokenType.ROW_ID):
            return RowSpecialNode(self._previous().value)
        
        if self._match(TokenType.IDENTIFIER):
            name = self._previous().value
            
            # Check if function call
            if self._match(TokenType.LPAREN):
                args = self._arguments()
                self._consume(TokenType.RPAREN, "Expected ')' after arguments")
                return FunctionCallNode(name, args)
            
            # Just identifier (could be constant)
            return FunctionCallNode(name, [])
        
        if self._match(TokenType.LPAREN):
            expr = self._expression()
            self._consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        
        # Default: return empty
        return NumberNode(0)
    
    # Helper methods
    def _match(self, *types: TokenType) -> bool:
        for t in types:
            if self._check(t):
                self._advance()
                return True
        return False
    
    def _check(self, token_type: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type == token_type
    
    def _advance(self) -> Token:
        if not self._is_at_end():
            self.pos += 1
        return self._previous()
    
    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF
    
    def _peek(self) -> Token:
        return self.tokens[self.pos]
    
    def _previous(self) -> Token:
        return self.tokens[self.pos - 1]
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        if self._check(token_type):
            return self._advance()
        raise SyntaxError(f"{message} at position {self.pos}")


# ==================== Visitor ====================

class ASTVisitor(ABC):
    """Base visitor for AST transformation."""
    
    @abstractmethod
    def visit_number(self, node: NumberNode) -> Any:
        pass
    
    @abstractmethod
    def visit_string(self, node: StringNode) -> Any:
        pass
    
    @abstractmethod
    def visit_boolean(self, node: BooleanNode) -> Any:
        pass
    
    @abstractmethod
    def visit_column_ref(self, node: ColumnRefNode) -> Any:
        pass
    
    @abstractmethod
    def visit_flow_var(self, node: FlowVarNode) -> Any:
        pass
    
    @abstractmethod
    def visit_row_special(self, node: RowSpecialNode) -> Any:
        pass
    
    @abstractmethod
    def visit_binary_op(self, node: BinaryOpNode) -> Any:
        pass
    
    @abstractmethod
    def visit_unary_op(self, node: UnaryOpNode) -> Any:
        pass
    
    @abstractmethod
    def visit_function_call(self, node: FunctionCallNode) -> Any:
        pass
    
    @abstractmethod
    def visit_ternary(self, node: TernaryNode) -> Any:
        pass


# ==================== Public API ====================

def parse_formula(formula: str) -> ASTNode:
    """
    Parse a KNIME formula string into an AST.
    
    Args:
        formula: KNIME formula string
        
    Returns:
        Root AST node
    """
    lexer = FormulaLexer(formula)
    tokens = lexer.tokenize()
    parser = FormulaParser(tokens)
    return parser.parse()
