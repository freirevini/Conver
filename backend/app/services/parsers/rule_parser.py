"""
Rule Engine Parser

Parses KNIME Rule Engine expressions into Python/NumPy code.
Handles rule-based conditional logic with => syntax.
"""
import re
from typing import Dict, List, Optional, Tuple
from app.services.parsers.base_parser import BaseExpressionParser
from app.services.catalog import catalog_service


class RuleEngineParser(BaseExpressionParser):
    """
    Parser for KNIME Rule Engine node expressions.
    
    Converts rules like:
    - $Age$ >= 18 AND $Age$ < 65 => "Adult"
    - $Age$ >= 65 => "Senior"  
    - TRUE => "Minor"
    
    Into np.select() Python code.
    """
    
    # Pattern to match rule: condition => result
    RULE_PATTERN = re.compile(r'^(.+?)\s*=>\s*(.+)$', re.MULTILINE)
    
    # Pattern to match operators
    OPERATORS = {
        "AND": "&",
        "OR": "|",
        "NOT": "~",
        "=": "==",
        "<>": "!=",
        "LIKE": ".str.contains",
        "MATCHES": ".str.match",
        "IN": ".isin",
        "MISSING": ".isna()",
    }
    
    def __init__(self, df_var: str = "df"):
        super().__init__(df_var)
        self.add_import("import numpy as np")
    
    def parse(self, expression: str, output_column: str = "result", **kwargs) -> str:
        """
        Parse Rule Engine expression into Python code.
        
        Args:
            expression: KNIME Rule Engine expression (multiple lines)
            output_column: Name for the output column
            
        Returns:
            Python code using np.select()
        """
        # Parse all rules
        rules = self._extract_rules(expression)
        
        if not rules:
            return f"# No valid rules found in expression"
        
        # Separate default rule (TRUE => value)
        conditions = []
        choices = []
        default = "'Unknown'"
        
        for condition, result in rules:
            condition_clean = condition.strip()
            result_clean = result.strip()
            
            if condition_clean.upper() == "TRUE":
                default = self._format_result(result_clean)
            else:
                py_condition = self._parse_condition(condition_clean)
                conditions.append(py_condition)
                choices.append(self._format_result(result_clean))
        
        # Generate np.select code
        return self._generate_select_code(conditions, choices, default, output_column)
    
    def _extract_rules(self, expression: str) -> List[Tuple[str, str]]:
        """
        Extract condition => result pairs from expression.
        
        Args:
            expression: Multi-line rule expression
            
        Returns:
            List of (condition, result) tuples
        """
        rules = []
        
        # Split by lines and parse each
        for line in expression.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            
            match = self.RULE_PATTERN.match(line)
            if match:
                condition = match.group(1).strip()
                result = match.group(2).strip()
                rules.append((condition, result))
        
        return rules
    
    def _parse_condition(self, condition: str) -> str:
        """
        Parse a single condition into Python code.
        
        Args:
            condition: KNIME rule condition
            
        Returns:
            Python condition expression
        """
        result = condition
        
        # Replace column references
        result = self.replace_column_refs(result)
        
        # Replace operators
        for knime_op, py_op in self.OPERATORS.items():
            if knime_op in ("MISSING",):
                # Handle as method
                pattern = rf'MISSING\s*\$([^$]+)\$'
                result = re.sub(pattern, rf"{self.df_var}['\1'].isna()", result, flags=re.IGNORECASE)
            elif knime_op in ("LIKE", "MATCHES"):
                # Handle pattern matching
                result = self._handle_pattern_operator(result, knime_op, py_op)
            elif knime_op == "IN":
                result = self._handle_in_operator(result)
            else:
                # Simple replacement
                result = re.sub(rf'\b{knime_op}\b', py_op, result, flags=re.IGNORECASE)
        
        # Wrap conditions in parentheses for proper operator precedence
        if "&" in result or "|" in result:
            # Split by & and | and wrap each sub-condition
            result = self._wrap_conditions(result)
        
        return result
    
    def _wrap_conditions(self, condition: str) -> str:
        """
        Wrap sub-conditions in parentheses for pandas.
        
        df['A'] > 1 & df['B'] < 2 → (df['A'] > 1) & (df['B'] < 2)
        """
        # Simple heuristic: wrap comparisons
        # More robust parsing would use a proper tokenizer
        result = condition
        
        # Find comparison patterns and wrap them
        comparison_pattern = re.compile(
            r"(\w+\[['\"]?\w+['\"]?\])\s*([<>=!]+)\s*([^\s&|]+)"
        )
        
        def wrapper(match):
            return f"({match.group(0)})"
        
        result = comparison_pattern.sub(wrapper, result)
        
        return result
    
    def _handle_pattern_operator(self, condition: str, op: str, py_op: str) -> str:
        """Handle LIKE and MATCHES operators."""
        # Pattern: $col$ LIKE "pattern"
        pattern = rf"({self.df_var}\['[^']+'\])\s+{op}\s+(['\"][^'\"]+['\"])"
        
        def replacer(match):
            col = match.group(1)
            pat = match.group(2)
            return f"{col}{py_op}({pat})"
        
        return re.sub(pattern, replacer, condition, flags=re.IGNORECASE)
    
    def _handle_in_operator(self, condition: str) -> str:
        """Handle IN operator: $col$ IN (val1, val2, ...)"""
        pattern = rf"({self.df_var}\['[^']+'\])\s+IN\s+\(([^)]+)\)"
        
        def replacer(match):
            col = match.group(1)
            values = match.group(2)
            return f"{col}.isin([{values}])"
        
        return re.sub(pattern, replacer, condition, flags=re.IGNORECASE)
    
    def _format_result(self, result: str) -> str:
        """
        Format the result value.
        
        Args:
            result: Result string from rule
            
        Returns:
            Properly formatted Python value
        """
        result = result.strip()
        
        # Check if it's a column reference
        if self.COLUMN_PATTERN.match(result):
            return self.replace_column_refs(result)
        
        # Check if it's already a string
        if (result.startswith('"') and result.endswith('"')) or \
           (result.startswith("'") and result.endswith("'")):
            return result
        
        # Try to parse as number
        try:
            float(result)
            return result
        except ValueError:
            pass
        
        # Check for boolean
        if result.upper() in ("TRUE", "FALSE"):
            return result.capitalize()
        
        # Assume it's a string that needs quotes
        return f"'{result}'"
    
    def _generate_select_code(
        self, 
        conditions: List[str], 
        choices: List[str], 
        default: str, 
        output_column: str
    ) -> str:
        """
        Generate np.select() code.
        
        Args:
            conditions: List of Python condition strings
            choices: List of result values
            default: Default value for no match
            output_column: Output column name
            
        Returns:
            Complete Python code
        """
        if not conditions:
            # Only default rule
            return f"{self.df_var}['{output_column}'] = {default}"
        
        # Format conditions list
        conditions_str = ",\n    ".join(conditions)
        choices_str = ", ".join(choices)
        
        code = f"""conditions = [
    {conditions_str}
]
choices = [{choices_str}]
{self.df_var}['{output_column}'] = np.select(conditions, choices, default={default})"""
        
        return code
    
    def parse_single_rule(self, condition: str, result: str) -> Tuple[str, str]:
        """
        Parse a single rule for use in conditionals.
        
        Args:
            condition: KNIME condition
            result: Result value
            
        Returns:
            Tuple of (python_condition, python_result)
        """
        py_condition = self._parse_condition(condition)
        py_result = self._format_result(result)
        return py_condition, py_result


# Singleton instance
rule_parser = RuleEngineParser()
