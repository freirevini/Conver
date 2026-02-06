"""
Math Formula Parser

Parses KNIME Math Formula expressions into Python/NumPy code.
Handles mathematical functions, column references, and operators.
"""
import re
from typing import Dict, List, Optional
from app.services.parsers.base_parser import BaseExpressionParser
from app.services.catalog import catalog_service


class MathFormulaParser(BaseExpressionParser):
    """
    Parser for KNIME Math Formula node expressions.
    
    Converts expressions like:
    - $Sales$ * 1.1 → df['Sales'] * 1.1
    - ABS($Value$) → np.abs(df['Value'])
    - ROUND($Price$, 2) → np.round(df['Price'], 2)
    - IF($Age$ > 18, "Adult", "Minor") → np.where(df['Age'] > 18, "Adult", "Minor")
    """
    
    def __init__(self, df_var: str = "df"):
        super().__init__(df_var)
        self.add_import("import numpy as np")
        self._math_functions = self._load_math_functions()
    
    def _load_math_functions(self) -> Dict[str, str]:
        """Load math function mappings."""
        return {
            "ABS": "np.abs",
            "ROUND": "np.round",
            "FLOOR": "np.floor",
            "CEIL": "np.ceil",
            "SQRT": "np.sqrt",
            "POW": "np.power",
            "LOG": "np.log",
            "LOG10": "np.log10",
            "LOG2": "np.log2",
            "EXP": "np.exp",
            "SIN": "np.sin",
            "COS": "np.cos",
            "TAN": "np.tan",
            "ASIN": "np.arcsin",
            "ACOS": "np.arccos",
            "ATAN": "np.arctan",
            "MIN": "np.minimum",
            "MAX": "np.maximum",
            "SIGN": "np.sign",
            "IF": "np.where",
        }
    
    def parse(self, expression: str, output_column: str = "result", **kwargs) -> str:
        """
        Parse Math Formula expression into Python code.
        
        Args:
            expression: KNIME Math Formula expression
            output_column: Name for the output column
            
        Returns:
            Python code string
        """
        # Validate
        is_valid, error = self.validate_expression(expression)
        if not is_valid:
            return f"# Error: {error}\n# Original: {expression}"
        
        # Step 1: Replace column references FIRST
        result = self.replace_column_refs(expression)
        
        # Step 2: Replace function names
        result = self._replace_functions(result)
        
        # Step 3: Handle operators
        result = self._handle_operators(result)
        
        return f"{self.df_var}['{output_column}'] = {result}"
    
    def _replace_functions(self, expression: str) -> str:
        """
        Replace KNIME math function names with Python equivalents.
        
        Simple approach: replace function names directly.
        """
        result = expression
        
        for knime_func, python_func in self._math_functions.items():
            # Match function name followed by (
            pattern = rf'\b{knime_func}\s*\('
            replacement = f'{python_func}('
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Handle MOD specially (infix operator)
        mod_pattern = r'\bMOD\s*\(\s*([^,]+),\s*([^)]+)\)'
        result = re.sub(mod_pattern, r'(\1 % \2)', result, flags=re.IGNORECASE)
        
        return result
    
    def _handle_operators(self, expression: str) -> str:
        """Handle special KNIME operators."""
        result = expression
        
        # Power operator
        result = result.replace("^", "**")
        
        # Boolean operators
        result = re.sub(r'\bAND\b', '&', result, flags=re.IGNORECASE)
        result = re.sub(r'\bOR\b', '|', result, flags=re.IGNORECASE)
        result = re.sub(r'\bNOT\b', '~', result, flags=re.IGNORECASE)
        
        # Comparison
        result = result.replace("<>", "!=")
        
        return result


# Singleton instance
math_parser = MathFormulaParser()
