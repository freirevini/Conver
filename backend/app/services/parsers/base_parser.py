"""
Base Expression Parser

Abstract base class for all KNIME expression parsers.
Provides common functionality for parsing $column$ references.
"""
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseExpressionParser(ABC):
    """
    Abstract base class for KNIME expression parsers.
    
    Handles common patterns like:
    - $column_name$ → df['column_name']
    - Column references in expressions
    - Function call parsing
    """
    
    # Pattern to match $column_name$ references
    COLUMN_PATTERN = re.compile(r'\$([^$]+)\$')
    
    def __init__(self, df_var: str = "df"):
        """
        Initialize parser.
        
        Args:
            df_var: Variable name for the DataFrame (default: "df")
        """
        self.df_var = df_var
        self._imports: List[str] = ["import pandas as pd"]
    
    @property
    def imports(self) -> List[str]:
        """Get required imports for generated code."""
        return list(set(self._imports))
    
    @abstractmethod
    def parse(self, expression: str, **kwargs) -> str:
        """
        Parse KNIME expression and return Python code.
        
        Args:
            expression: KNIME expression string
            **kwargs: Additional context (output_column, etc.)
            
        Returns:
            Python code string
        """
        pass
    
    def replace_column_refs(self, expression: str) -> str:
        """
        Replace $column$ references with df['column'] syntax.
        
        Args:
            expression: Expression containing $column$ references
            
        Returns:
            Expression with Python column access syntax
        """
        def replacer(match):
            col_name = match.group(1)
            return f"{self.df_var}['{col_name}']"
        
        return self.COLUMN_PATTERN.sub(replacer, expression)
    
    def extract_columns(self, expression: str) -> List[str]:
        """
        Extract all column names from an expression.
        
        Args:
            expression: Expression containing $column$ references
            
        Returns:
            List of column names
        """
        return self.COLUMN_PATTERN.findall(expression)
    
    def add_import(self, import_statement: str) -> None:
        """Add an import statement to the required imports."""
        if import_statement not in self._imports:
            self._imports.append(import_statement)
    
    def validate_expression(self, expression: str) -> Tuple[bool, Optional[str]]:
        """
        Validate expression syntax.
        
        Args:
            expression: Expression to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not expression or not expression.strip():
            return False, "Empty expression"
        
        # Check balanced parentheses
        paren_count = 0
        for char in expression:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if paren_count < 0:
                return False, "Unbalanced parentheses"
        
        if paren_count != 0:
            return False, "Unbalanced parentheses"
        
        return True, None
