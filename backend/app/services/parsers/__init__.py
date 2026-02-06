"""
KNIME Expression Parsers Module

Provides deterministic parsing of KNIME expression nodes:
- Math Formula
- String Manipulation  
- Rule Engine
- Column Expressions
"""
from app.services.parsers.base_parser import BaseExpressionParser
from app.services.parsers.math_parser import MathFormulaParser, math_parser
from app.services.parsers.string_parser import StringManipulationParser, string_parser
from app.services.parsers.rule_parser import RuleEngineParser, rule_parser
from app.services.parsers.column_expr_parser import ColumnExpressionsParser, column_expr_parser

__all__ = [
    # Base
    "BaseExpressionParser",
    
    # Parsers
    "MathFormulaParser",
    "StringManipulationParser", 
    "RuleEngineParser",
    "ColumnExpressionsParser",
    
    # Singleton instances
    "math_parser",
    "string_parser",
    "rule_parser",
    "column_expr_parser",
]


def get_parser_for_node(factory_class: str) -> BaseExpressionParser:
    """
    Get the appropriate parser for a KNIME node.
    
    Args:
        factory_class: KNIME node factory class
        
    Returns:
        Parser instance or None
    """
    parser_map = {
        "org.knime.base.node.preproc.javasnippet.JavaSnippetNodeFactory": math_parser,
        "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory": string_parser,
        "org.knime.base.node.rules.engine.RuleEngineNodeFactory2": rule_parser,
        "org.knime.base.expressions.node.ExpressionNodeFactory": column_expr_parser,
    }
    
    return parser_map.get(factory_class)
