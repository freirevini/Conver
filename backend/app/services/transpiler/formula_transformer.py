"""
Formula Transformer - AST to Python Code.

Transforms parsed KNIME formula AST into Python code.
Integrates with function mapping for correct translation.
"""
import logging
from typing import List, Set
from dataclasses import dataclass, field

from .formula_parser import (
    ASTNode, ASTVisitor, NumberNode, StringNode, BooleanNode,
    ColumnRefNode, FlowVarNode, RowSpecialNode, BinaryOpNode,
    UnaryOpNode, FunctionCallNode, TernaryNode, parse_formula
)
from .function_mapping import get_function_mapping, get_required_imports

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Result of formula transformation."""
    python_code: str
    imports: List[str] = field(default_factory=list)
    columns_used: List[str] = field(default_factory=list)
    flow_vars_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PythonTransformer(ASTVisitor):
    """
    Transforms KNIME formula AST to Python code.
    
    Implements the Visitor pattern to traverse AST nodes
    and generate corresponding Python code.
    """
    
    def __init__(self, df_name: str = "df"):
        self.df_name = df_name
        self.columns_used: Set[str] = set()
        self.flow_vars_used: Set[str] = set()
        self.functions_used: Set[str] = set()
        self.warnings: List[str] = []
    
    def transform(self, node: ASTNode) -> str:
        """Transform AST node to Python code."""
        return node.accept(self)
    
    def visit_number(self, node: NumberNode) -> str:
        return str(node.value)
    
    def visit_string(self, node: StringNode) -> str:
        # Escape quotes in string
        escaped = node.value.replace('"', '\\"')
        return f'"{escaped}"'
    
    def visit_boolean(self, node: BooleanNode) -> str:
        return "True" if node.value else "False"
    
    def visit_column_ref(self, node: ColumnRefNode) -> str:
        self.columns_used.add(node.name)
        return f'{self.df_name}["{node.name}"]'
    
    def visit_flow_var(self, node: FlowVarNode) -> str:
        self.flow_vars_used.add(node.name)
        return f'flow_vars["{node.name}"]'
    
    def visit_row_special(self, node: RowSpecialNode) -> str:
        if node.name == "ROWINDEX":
            return f"{self.df_name}.index"
        elif node.name == "ROWCOUNT":
            return f"len({self.df_name})"
        elif node.name == "ROWID":
            return f"{self.df_name}.index.astype(str)"
        else:
            self.warnings.append(f"Unknown row special: {node.name}")
            return f"# Unknown: $${node.name}$$"
    
    def visit_binary_op(self, node: BinaryOpNode) -> str:
        left = self.transform(node.left)
        right = self.transform(node.right)
        
        # Handle string concatenation
        if node.operator == "+":
            # Could be string concat - wrap in parentheses
            return f"({left} {node.operator} {right})"
        
        return f"({left} {node.operator} {right})"
    
    def visit_unary_op(self, node: UnaryOpNode) -> str:
        operand = self.transform(node.operand)
        
        if node.operator == "not":
            return f"(not {operand})"
        elif node.operator == "-":
            return f"(-{operand})"
        else:
            return f"({node.operator}{operand})"
    
    def visit_function_call(self, node: FunctionCallNode) -> str:
        self.functions_used.add(node.name)
        
        # Get function mapping
        mapping = get_function_mapping(node.name)
        
        if mapping:
            # Transform arguments
            args = [self.transform(arg) for arg in node.arguments]
            
            # Apply template
            try:
                # Replace {0}, {1}, etc. with actual arguments
                result = mapping.python_template
                for i, arg in enumerate(args):
                    result = result.replace(f"{{{i}}}", arg)
                return result
            except Exception as e:
                self.warnings.append(f"Error mapping {node.name}: {e}")
                # Fallback
                return f"# {node.name}({', '.join(args)})"
        else:
            # Unknown function - generate best-effort Python
            args = [self.transform(arg) for arg in node.arguments]
            self.warnings.append(f"Unknown function: {node.name}")
            return f"{node.name}({', '.join(args)})"
    
    def visit_ternary(self, node: TernaryNode) -> str:
        condition = self.transform(node.condition)
        true_val = self.transform(node.true_value)
        false_val = self.transform(node.false_value)
        
        return f"({true_val} if {condition} else {false_val})"


def transform_formula(
    formula: str,
    df_name: str = "df",
    output_var: str = None
) -> TransformResult:
    """
    Transform a KNIME formula to Python code.
    
    Args:
        formula: KNIME formula string
        df_name: DataFrame variable name
        output_var: Optional output variable name
        
    Returns:
        TransformResult with Python code and metadata
    """
    try:
        # Parse formula to AST
        ast = parse_formula(formula)
        
        # Transform to Python
        transformer = PythonTransformer(df_name)
        python_code = transformer.transform(ast)
        
        # Get required imports
        imports = get_required_imports(list(transformer.functions_used))
        
        # Add output assignment if specified
        if output_var:
            python_code = f'{output_var} = {python_code}'
        
        return TransformResult(
            python_code=python_code,
            imports=imports,
            columns_used=list(transformer.columns_used),
            flow_vars_used=list(transformer.flow_vars_used),
            warnings=transformer.warnings
        )
    
    except Exception as e:
        logger.error(f"Formula transformation error: {e}")
        return TransformResult(
            python_code=f"# Error: {formula}\n# {e}",
            warnings=[str(e)]
        )


def transform_column_expression(
    formula: str,
    df_name: str = "df",
    output_column: str = "result"
) -> TransformResult:
    """
    Transform a KNIME formula for DataFrame column assignment.
    
    Args:
        formula: KNIME formula string
        df_name: DataFrame variable name
        output_column: Output column name
        
    Returns:
        TransformResult with assignment code
    """
    result = transform_formula(formula, df_name)
    
    # Create column assignment
    result.python_code = f'{df_name}["{output_column}"] = {result.python_code}'
    
    return result


# ==================== Module Initialization ====================

def create_transpiler_init():
    """Create __init__.py for transpiler module."""
    return '''"""
KNIME Formula Transpiler Module.

Provides:
- formula_parser: Lexer and Parser for KNIME expressions
- function_mapping: KNIME to Python function mappings
- rule_engine: Rule Engine syntax conversion
- formula_transformer: AST to Python code generation
"""
from .formula_parser import parse_formula, FormulaLexer, FormulaParser
from .function_mapping import get_function_mapping, get_all_function_names
from .rule_engine import convert_rule_engine, parse_rules
from .formula_transformer import transform_formula, transform_column_expression

__all__ = [
    "parse_formula",
    "FormulaLexer", 
    "FormulaParser",
    "get_function_mapping",
    "get_all_function_names",
    "convert_rule_engine",
    "parse_rules",
    "transform_formula",
    "transform_column_expression",
]
'''
