"""
JEP Expression AST Nodes.

Defines the Abstract Syntax Tree nodes for parsed JEP expressions.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class ColumnRef:
    """Reference to a DataFrame column: $columnName$"""
    name: str
    
    def __repr__(self) -> str:
        return f"ColumnRef({self.name!r})"


@dataclass
class Number:
    """Numeric literal."""
    value: float
    
    def __repr__(self) -> str:
        return f"Number({self.value})"


@dataclass
class String:
    """String literal."""
    value: str
    
    def __repr__(self) -> str:
        return f"String({self.value!r})"


@dataclass
class BinaryOp:
    """Binary operation: left op right"""
    left: Any  # Node
    op: str
    right: Any  # Node
    
    def __repr__(self) -> str:
        return f"BinaryOp({self.left}, {self.op!r}, {self.right})"


@dataclass
class UnaryOp:
    """Unary operation: op operand"""
    op: str
    operand: Any  # Node
    
    def __repr__(self) -> str:
        return f"UnaryOp({self.op!r}, {self.operand})"


@dataclass
class FunctionCall:
    """Function call: name(arg1, arg2, ...)"""
    name: str
    args: List[Any]  # List[Node]
    
    def __repr__(self) -> str:
        return f"FunctionCall({self.name!r}, {self.args})"


@dataclass
class Comparison:
    """Comparison: left op right"""
    left: Any  # Node
    op: str  # ==, !=, <, >, <=, >=
    right: Any  # Node
    
    def __repr__(self) -> str:
        return f"Comparison({self.left}, {self.op!r}, {self.right})"


@dataclass 
class Conditional:
    """Conditional expression: if(condition, then, else)"""
    condition: Any  # Node
    then_expr: Any  # Node
    else_expr: Any  # Node
    
    def __repr__(self) -> str:
        return f"Conditional({self.condition}, {self.then_expr}, {self.else_expr})"


# Type alias for any AST node
Node = Union[ColumnRef, Number, String, BinaryOp, UnaryOp, FunctionCall, Comparison, Conditional]
