"""
String Manipulation Parser

Parses KNIME String Manipulation expressions into Python/Pandas code.
"""
import re
from typing import Dict, List, Optional
from app.services.parsers.base_parser import BaseExpressionParser


class StringManipulationParser(BaseExpressionParser):
    """
    Parser for KNIME String Manipulation node expressions.
    
    Converts expressions like:
    - join($FirstName$, " ", $LastName$) → df['FirstName'].str.cat(df['LastName'], sep=' ')
    - upperCase($Name$) → df['Name'].str.upper()
    - substr($Text$, 0, 5) → df['Text'].str[0:5]
    """
    
    def __init__(self, df_var: str = "df"):
        super().__init__(df_var)
    
    def parse(self, expression: str, output_column: str = "result", **kwargs) -> str:
        """
        Parse String Manipulation expression into Python code.
        """
        # Validate
        is_valid, error = self.validate_expression(expression)
        if not is_valid:
            return f"# Error: {error}"
        
        # Find the outermost function
        func_match = re.match(r'^(\w+)\s*\((.+)\)$', expression.strip(), re.DOTALL)
        
        if not func_match:
            # Simple concatenation or column reference
            parsed = self.replace_column_refs(expression)
            return f"{self.df_var}['{output_column}'] = {parsed}"
        
        func_name = func_match.group(1).lower()
        args_str = func_match.group(2)
        
        # Parse arguments
        args = self._split_args(args_str)
        
        # Generate code
        code = self._generate_code(func_name, args)
        
        return f"{self.df_var}['{output_column}'] = {code}"
    
    def _split_args(self, args_str: str) -> List[str]:
        """Split arguments, respecting nesting and quotes."""
        args = []
        current = ""
        depth = 0
        in_string = False
        quote_char = None
        
        for char in args_str:
            if char in ('"', "'") and not in_string:
                in_string = True
                quote_char = char
            elif char == quote_char and in_string:
                in_string = False
            elif char == '(' and not in_string:
                depth += 1
            elif char == ')' and not in_string:
                depth -= 1
            elif char == ',' and depth == 0 and not in_string:
                args.append(current.strip())
                current = ""
                continue
            current += char
        
        if current.strip():
            args.append(current.strip())
        
        return args
    
    def _generate_code(self, func_name: str, args: List[str]) -> str:
        """Generate Python code based on function name."""
        
        # Get the first column argument
        first_col = self._extract_column(args[0]) if args else None
        
        if func_name in ("uppercase", "uppercase"):
            if first_col:
                return f"{self.df_var}['{first_col}'].str.upper()"
            return f"# No column found in: {args}"
        
        elif func_name in ("lowercase", "lowercase"):
            if first_col:
                return f"{self.df_var}['{first_col}'].str.lower()"
            return f"# No column found"
        
        elif func_name == "capitalize":
            if first_col:
                return f"{self.df_var}['{first_col}'].str.capitalize()"
            return f"# No column found"
        
        elif func_name == "trim":
            if first_col:
                return f"{self.df_var}['{first_col}'].str.strip()"
            return f"# No column found"
        
        elif func_name == "length":
            if first_col:
                return f"{self.df_var}['{first_col}'].str.len()"
            return f"# No column found"
        
        elif func_name == "substr":
            if len(args) >= 2 and first_col:
                start = args[1].strip()
                if len(args) >= 3:
                    length = args[2].strip()
                    return f"{self.df_var}['{first_col}'].str[{start}:{start}+{length}]"
                return f"{self.df_var}['{first_col}'].str[{start}:]"
        
        elif func_name == "join":
            return self._handle_join(args)
        
        elif func_name == "replace":
            if len(args) >= 3 and first_col:
                old = args[1].strip()
                new = args[2].strip()
                return f"{self.df_var}['{first_col}'].str.replace({old}, {new})"
        
        elif func_name in ("regexreplace", "regexreplace"):
            if len(args) >= 3 and first_col:
                pattern = args[1].strip()
                repl = args[2].strip()
                return f"{self.df_var}['{first_col}'].str.replace({pattern}, {repl}, regex=True)"
        
        elif func_name == "contains":
            if len(args) >= 2 and first_col:
                pattern = args[1].strip()
                return f"{self.df_var}['{first_col}'].str.contains({pattern})"
        
        elif func_name == "split":
            if len(args) >= 2 and first_col:
                sep = args[1].strip()
                return f"{self.df_var}['{first_col}'].str.split({sep})"
        
        elif func_name in ("indexof", "indexof"):
            if len(args) >= 2 and first_col:
                pattern = args[1].strip()
                return f"{self.df_var}['{first_col}'].str.find({pattern})"
        
        elif func_name in ("toint", "toint"):
            if first_col:
                return f"{self.df_var}['{first_col}'].astype(int)"
        
        elif func_name in ("todouble", "todouble"):
            if first_col:
                return f"{self.df_var}['{first_col}'].astype(float)"
        
        elif func_name in ("tostring", "tostring"):
            if first_col:
                return f"{self.df_var}['{first_col}'].astype(str)"
        
        return f"# Unknown function: {func_name}"
    
    def _extract_column(self, arg: str) -> Optional[str]:
        """Extract column name from $column$ pattern."""
        match = self.COLUMN_PATTERN.search(arg)
        if match:
            return match.group(1)
        return None
    
    def _handle_join(self, args: List[str]) -> str:
        """Handle join function."""
        if len(args) < 2:
            return "# join requires at least 2 arguments"
        
        # Collect columns and separator
        columns = []
        sep = ""
        
        for arg in args:
            col = self._extract_column(arg)
            if col:
                columns.append(col)
            else:
                # It's likely the separator
                sep = arg.strip().strip("'\"")
        
        if len(columns) == 0:
            return "# No columns found in join"
        
        if len(columns) == 1:
            return f"{self.df_var}['{columns[0]}']"
        
        # Build chain
        result = f"{self.df_var}['{columns[0]}'].str.cat({self.df_var}['{columns[1]}'], sep='{sep}')"
        for col in columns[2:]:
            result = f"{result}.str.cat({self.df_var}['{col}'], sep='{sep}')"
        
        return result


# Singleton instance
string_parser = StringManipulationParser()
