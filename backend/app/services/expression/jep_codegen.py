"""
JEP Expression Code Generator.

Converts AST nodes to Pandas/NumPy expressions.
"""
from typing import Any, Set

from .jep_ast import (
    Node, ColumnRef, Number, String, BinaryOp, UnaryOp,
    FunctionCall, Comparison, Conditional
)


# JEP to NumPy function mapping
FUNCTION_MAP = {
    # Math functions
    'abs': 'np.abs',
    'sin': 'np.sin',
    'cos': 'np.cos', 
    'tan': 'np.tan',
    'asin': 'np.arcsin',
    'acos': 'np.arccos',
    'atan': 'np.arctan',
    'sqrt': 'np.sqrt',
    'log': 'np.log',
    'log10': 'np.log10',
    'exp': 'np.exp',
    'pow': 'np.power',
    'round': 'np.round',
    'floor': 'np.floor',
    'ceil': 'np.ceil',
    'min': 'np.minimum',
    'max': 'np.maximum',
    'mod': 'np.mod',
    # Check functions
    'isnan': 'pd.isna',
    'isnull': 'pd.isna',
    'missing': 'pd.isna',
}

# Operator mapping
OPERATOR_MAP = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    '^': '**',
    '%': '%',
}

# Comparison mapping
COMPARISON_MAP = {
    '=': '==',
    '==': '==',
    '!=': '!=',
    '<>': '!=',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
}


class JEPCodeGenerator:
    """Generates Pandas/NumPy code from AST."""
    
    def __init__(self, df_var: str = 'df'):
        self.df_var = df_var
        self.imports: Set[str] = {'import pandas as pd', 'import numpy as np'}
    
    def generate(self, node: Node) -> str:
        """Generate Python code from AST node."""
        if isinstance(node, ColumnRef):
            return self._gen_column_ref(node)
        elif isinstance(node, Number):
            return self._gen_number(node)
        elif isinstance(node, String):
            return self._gen_string(node)
        elif isinstance(node, BinaryOp):
            return self._gen_binary_op(node)
        elif isinstance(node, UnaryOp):
            return self._gen_unary_op(node)
        elif isinstance(node, FunctionCall):
            return self._gen_function_call(node)
        elif isinstance(node, Comparison):
            return self._gen_comparison(node)
        elif isinstance(node, Conditional):
            return self._gen_conditional(node)
        else:
            return str(node)
    
    def _gen_column_ref(self, node: ColumnRef) -> str:
        """Generate column reference: df['column']"""
        return f"{self.df_var}['{node.name}']"
    
    def _gen_number(self, node: Number) -> str:
        """Generate numeric literal."""
        # Check if it's an integer
        if node.value == int(node.value):
            return str(int(node.value))
        return str(node.value)
    
    def _gen_string(self, node: String) -> str:
        """Generate string literal."""
        return repr(node.value)
    
    def _gen_binary_op(self, node: BinaryOp) -> str:
        """Generate binary operation."""
        left = self.generate(node.left)
        right = self.generate(node.right)
        op = OPERATOR_MAP.get(node.op, node.op)
        return f"({left} {op} {right})"
    
    def _gen_unary_op(self, node: UnaryOp) -> str:
        """Generate unary operation."""
        operand = self.generate(node.operand)
        if node.op == 'NOT' or node.op == '!':
            return f"~({operand})"
        elif node.op == '-':
            return f"-({operand})"
        return f"{node.op}({operand})"
    
    def _gen_function_call(self, node: FunctionCall) -> str:
        """Generate function call."""
        func_name = node.name.lower()
        args = [self.generate(arg) for arg in node.args]
        
        # Special case: if() -> np.where()
        if func_name == 'if' and len(args) == 3:
            return f"np.where({args[0]}, {args[1]}, {args[2]})"
        
        # Logical functions
        if func_name == 'and' and len(args) == 2:
            return f"(({args[0]}) & ({args[1]}))"
        if func_name == 'or' and len(args) == 2:
            return f"(({args[0]}) | ({args[1]}))"
        if func_name == 'not' and len(args) == 1:
            return f"~({args[0]})"
        
        # Map to NumPy/Pandas function
        np_func = FUNCTION_MAP.get(func_name, f'np.{func_name}')
        
        return f"{np_func}({', '.join(args)})"
    
    def _gen_comparison(self, node: Comparison) -> str:
        """Generate comparison expression."""
        left = self.generate(node.left)
        right = self.generate(node.right)
        op = COMPARISON_MAP.get(node.op, node.op)
        return f"({left} {op} {right})"
    
    def _gen_conditional(self, node: Conditional) -> str:
        """Generate conditional: np.where(cond, then, else)"""
        condition = self.generate(node.condition)
        then_expr = self.generate(node.then_expr)
        else_expr = self.generate(node.else_expr)
        return f"np.where({condition}, {then_expr}, {else_expr})"
    
    def get_imports(self) -> Set[str]:
        """Get required imports."""
        return self.imports


def generate_code(node: Node, df_var: str = 'df') -> str:
    """Generate Pandas/NumPy code from AST node."""
    generator = JEPCodeGenerator(df_var)
    return generator.generate(node)
