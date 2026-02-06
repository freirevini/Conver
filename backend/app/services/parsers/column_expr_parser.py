"""
Column Expressions Parser

Parses KNIME Column Expressions syntax into Python code.
Modern expression syntax used in KNIME 4.5+.
"""
import re
from typing import Dict, List, Optional
from app.services.parsers.base_parser import BaseExpressionParser
from app.services.catalog import catalog_service


class ColumnExpressionsParser(BaseExpressionParser):
    """
    Parser for KNIME Column Expressions node.
    
    Handles modern KNIME expression syntax:
    - column("Name") → df['Name']
    - if (condition) { result } else { other }
    - rowIndex → df.index
    - rowNumber → range(len(df))
    """
    
    # Pattern to match column() function
    COLUMN_FUNC_PATTERN = re.compile(r'column\s*\(\s*["\']([^"\']+)["\']\s*\)')
    
    # Pattern to match if-else blocks
    IF_PATTERN = re.compile(
        r'if\s*\(([^)]+)\)\s*\{([^}]+)\}\s*else\s*\{([^}]+)\}',
        re.IGNORECASE | re.DOTALL
    )
    
    def __init__(self, df_var: str = "df"):
        super().__init__(df_var)
        self.add_import("import numpy as np")
    
    def parse(self, expression: str, output_column: str = "result", **kwargs) -> str:
        """
        Parse Column Expression into Python code.
        
        Args:
            expression: KNIME Column Expression
            output_column: Name for output column
            
        Returns:
            Python code string
        """
        # Validate
        is_valid, error = self.validate_expression(expression)
        if not is_valid:
            return f"# Error: {error}\n# Original: {expression}"
        
        # Parse column() function calls
        parsed = self._parse_column_funcs(expression)
        
        # Parse if-else blocks
        parsed = self._parse_if_else(parsed)
        
        # Replace $column$ references (legacy support)
        parsed = self.replace_column_refs(parsed)
        
        # Handle special variables
        parsed = self._handle_special_vars(parsed)
        
        # Handle operators
        parsed = self._handle_operators(parsed)
        
        return f"{self.df_var}['{output_column}'] = {parsed}"
    
    def _parse_column_funcs(self, expression: str) -> str:
        """
        Replace column("Name") with df['Name'].
        """
        def replacer(match):
            col_name = match.group(1)
            return f"{self.df_var}['{col_name}']"
        
        return self.COLUMN_FUNC_PATTERN.sub(replacer, expression)
    
    def _parse_if_else(self, expression: str) -> str:
        """
        Convert if-else blocks to np.where().
        
        if (condition) { result } else { other }
        → np.where(condition, result, other)
        """
        def replacer(match):
            condition = match.group(1).strip()
            true_val = match.group(2).strip()
            false_val = match.group(3).strip()
            
            # Handle nested column references in condition
            condition = self._parse_column_funcs(condition)
            condition = self.replace_column_refs(condition)
            
            true_val = self._parse_column_funcs(true_val)
            true_val = self.replace_column_refs(true_val)
            
            false_val = self._parse_column_funcs(false_val)
            false_val = self.replace_column_refs(false_val)
            
            return f"np.where({condition}, {true_val}, {false_val})"
        
        result = expression
        # Iteratively replace until no more matches (handles nested)
        for _ in range(10):  # Max 10 nesting levels
            new_result = self.IF_PATTERN.sub(replacer, result)
            if new_result == result:
                break
            result = new_result
        
        return result
    
    def _handle_special_vars(self, expression: str) -> str:
        """
        Handle special KNIME variables.
        
        - rowIndex → df.index
        - rowNumber → range(len(df))
        - ROWCOUNT → len(df)
        """
        result = expression
        
        # rowIndex and ROWINDEX
        result = re.sub(r'\browIndex\b', f'{self.df_var}.index', result, flags=re.IGNORECASE)
        
        # rowNumber (1-based)
        result = re.sub(r'\browNumber\b', f'range(1, len({self.df_var}) + 1)', result)
        
        # ROWCOUNT
        result = re.sub(r'\bROWCOUNT\b', f'len({self.df_var})', result, flags=re.IGNORECASE)
        
        # MISSING special handling
        result = re.sub(
            rf'MISSING\s*\(({self.df_var}\[[^\]]+\])\)',
            r'\1.isna()',
            result,
            flags=re.IGNORECASE
        )
        
        return result
    
    def _handle_operators(self, expression: str) -> str:
        """
        Handle KNIME operators.
        """
        result = expression
        
        # Logical operators
        result = re.sub(r'\b&&\b', '&', result)
        result = re.sub(r'\b\|\|\b', '|', result)
        result = re.sub(r'\b!\b', '~', result)
        
        # String operators
        result = re.sub(r'\+\s*"', '+ "', result)  # Normalize string concat
        
        # Comparison
        result = result.replace("===", "==")
        result = result.replace("!==", "!=")
        
        return result
    
    def parse_with_context(
        self, 
        expression: str, 
        output_column: str,
        input_columns: List[str]
    ) -> str:
        """
        Parse expression with column context for validation.
        
        Args:
            expression: KNIME expression
            output_column: Output column name
            input_columns: Available input columns
            
        Returns:
            Python code or error message
        """
        # Extract referenced columns
        referenced = self.extract_all_columns(expression)
        
        # Validate all columns exist
        missing = [c for c in referenced if c not in input_columns]
        if missing:
            return f"# Error: Columns not found: {missing}"
        
        return self.parse(expression, output_column)
    
    def extract_all_columns(self, expression: str) -> List[str]:
        """
        Extract all column references from expression.
        
        Handles both column("Name") and $Name$ syntax.
        """
        columns = []
        
        # From column() functions
        columns.extend(self.COLUMN_FUNC_PATTERN.findall(expression))
        
        # From $column$ references
        columns.extend(self.COLUMN_PATTERN.findall(expression))
        
        return list(set(columns))


# Singleton instance
column_expr_parser = ColumnExpressionsParser()
