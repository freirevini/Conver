"""
KNIME Rule Engine to Python Converter.

Converts KNIME Rule Engine syntax to Python conditionals:
- Rule Engine → np.select() or pd.case_when()
- Simple rules → if/elif/else
"""
import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """Single rule from Rule Engine."""
    condition: str
    result: str
    is_default: bool = False


@dataclass
class RuleEngineResult:
    """Result of Rule Engine conversion."""
    python_code: str
    imports: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RuleEngineParser:
    """
    Parser for KNIME Rule Engine syntax.
    
    KNIME Rule Engine format:
        $column$ > 50 => "High"
        $column$ > 25 => "Medium"
        TRUE => "Low"
    """
    
    # Patterns for parsing
    RULE_PATTERN = re.compile(
        r'^\s*(.+?)\s*=>\s*(.+?)\s*$',
        re.MULTILINE
    )
    
    COLUMN_PATTERN = re.compile(r'\$([^$]+)\$')
    FLOW_VAR_PATTERN = re.compile(r'\$\$([^$]+)\$\$')
    
    def __init__(self, rules_text: str):
        self.rules_text = rules_text
        self.rules: List[Rule] = []
        self.errors: List[str] = []
    
    def parse(self) -> List[Rule]:
        """Parse Rule Engine text into Rule objects."""
        self.rules = []
        self.errors = []
        
        for line in self.rules_text.strip().split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            
            match = self.RULE_PATTERN.match(line)
            if match:
                condition = match.group(1).strip()
                result = match.group(2).strip()
                
                # Check for default rule
                is_default = condition.upper() in ('TRUE', 'ELSE', 'DEFAULT', '1=1')
                
                self.rules.append(Rule(
                    condition=condition,
                    result=result,
                    is_default=is_default
                ))
            else:
                self.errors.append(f"Cannot parse rule: {line}")
        
        return self.rules
    
    def _convert_condition(self, condition: str) -> str:
        """Convert KNIME condition to Python."""
        result = condition
        
        # Convert column references: $col$ → df["col"]
        result = self.COLUMN_PATTERN.sub(r'df["\1"]', result)
        
        # Convert flow variables: $$var$$ → flow_vars["var"]
        result = self.FLOW_VAR_PATTERN.sub(r'flow_vars["\1"]', result)
        
        # Convert operators
        result = result.replace(' AND ', ' & ')
        result = result.replace(' OR ', ' | ')
        result = result.replace(' NOT ', ' ~')
        result = result.replace('NOT ', '~')
        
        # Convert string comparisons
        result = result.replace(' LIKE ', '.str.contains(')
        
        # Handle MISSING check
        result = re.sub(
            r'MISSING\s*\(([^)]+)\)',
            r'pd.isna(\1)',
            result
        )
        
        # Handle IN operator
        result = re.sub(
            r'(\w+)\s+IN\s*\(([^)]+)\)',
            r'\1.isin([\2])',
            result
        )
        
        return result
    
    def _convert_result(self, result: str) -> str:
        """Convert KNIME result value to Python."""
        # Already quoted string
        if result.startswith('"') and result.endswith('"'):
            return result
        
        # Number - keep as is
        if re.match(r'^-?\d+\.?\d*$', result):
            return result
        
        # Boolean
        if result.upper() in ('TRUE', 'FALSE'):
            return result.capitalize()
        
        # Column reference in result
        if result.startswith('$') and result.endswith('$'):
            col = result[1:-1]
            return f'df["{col}"]'
        
        # Otherwise treat as string
        return f'"{result}"'


class RuleEngineConverter:
    """
    Convert parsed rules to Python code.
    
    Strategies:
    1. np.select() - For vectorized DataFrame operations
    2. if/elif/else - For simple scalar values
    3. pd.case_when() - Alternative pandas approach
    """
    
    def __init__(self, parser: RuleEngineParser):
        self.parser = parser
    
    def to_np_select(self, df_name: str = "df", output_col: str = "result") -> RuleEngineResult:
        """
        Convert rules to np.select() format.
        
        Output:
            conditions = [
                df["col"] > 50,
                df["col"] > 25,
            ]
            choices = ["High", "Medium"]
            df["result"] = np.select(conditions, choices, default="Low")
        """
        rules = self.parser.parse()
        
        if not rules:
            return RuleEngineResult(
                python_code=f'{df_name}["{output_col}"] = None',
                warnings=["No rules found"]
            )
        
        conditions = []
        choices = []
        default = "None"
        
        for rule in rules:
            if rule.is_default:
                default = self.parser._convert_result(rule.result)
            else:
                cond = self.parser._convert_condition(rule.condition)
                result = self.parser._convert_result(rule.result)
                conditions.append(cond)
                choices.append(result)
        
        # Build code
        lines = [
            "# Rule Engine conversion",
            "conditions = ["
        ]
        
        for cond in conditions:
            lines.append(f"    {cond},")
        
        lines.append("]")
        lines.append("choices = [")
        
        for choice in choices:
            lines.append(f"    {choice},")
        
        lines.append("]")
        lines.append(
            f'{df_name}["{output_col}"] = np.select(conditions, choices, default={default})'
        )
        
        return RuleEngineResult(
            python_code="\n".join(lines),
            imports=["import numpy as np"],
            warnings=self.parser.errors
        )
    
    def to_lambda(self, row_name: str = "row") -> RuleEngineResult:
        """
        Convert rules to lambda function for apply().
        
        Output:
            def apply_rules(row):
                if row["col"] > 50:
                    return "High"
                elif row["col"] > 25:
                    return "Medium"
                else:
                    return "Low"
        """
        rules = self.parser.parse()
        
        if not rules:
            return RuleEngineResult(
                python_code=f"def apply_rules({row_name}):\n    return None",
                warnings=["No rules found"]
            )
        
        lines = [f"def apply_rules({row_name}):"]
        
        first = True
        default_result = "None"
        
        for rule in rules:
            if rule.is_default:
                default_result = self.parser._convert_result(rule.result)
            else:
                # Convert for row access
                cond = self.parser._convert_condition(rule.condition)
                cond = cond.replace('df["', f'{row_name}["')
                result = self.parser._convert_result(rule.result)
                result = result.replace('df["', f'{row_name}["')
                
                keyword = "if" if first else "elif"
                lines.append(f"    {keyword} {cond}:")
                lines.append(f"        return {result}")
                first = False
        
        lines.append(f"    else:")
        lines.append(f"        return {default_result}")
        
        return RuleEngineResult(
            python_code="\n".join(lines),
            imports=[],
            warnings=self.parser.errors
        )
    
    def to_case_statement(self, df_name: str = "df", output_col: str = "result") -> RuleEngineResult:
        """
        Convert rules to nested np.where() (case-like).
        
        Output:
            df["result"] = np.where(
                df["col"] > 50, "High",
                np.where(df["col"] > 25, "Medium", "Low")
            )
        """
        rules = self.parser.parse()
        
        if not rules:
            return RuleEngineResult(
                python_code=f'{df_name}["{output_col}"] = None',
                warnings=["No rules found"]
            )
        
        # Find default
        default = "None"
        non_default_rules = []
        
        for rule in rules:
            if rule.is_default:
                default = self.parser._convert_result(rule.result)
            else:
                non_default_rules.append(rule)
        
        # Build nested np.where
        if not non_default_rules:
            return RuleEngineResult(
                python_code=f'{df_name}["{output_col}"] = {default}',
                imports=[]
            )
        
        # Start from last rule and work backwards
        code = default
        
        for rule in reversed(non_default_rules):
            cond = self.parser._convert_condition(rule.condition)
            result = self.parser._convert_result(rule.result)
            code = f"np.where({cond}, {result}, {code})"
        
        return RuleEngineResult(
            python_code=f'{df_name}["{output_col}"] = {code}',
            imports=["import numpy as np"],
            warnings=self.parser.errors
        )


# ==================== Public API ====================

def convert_rule_engine(
    rules_text: str,
    strategy: str = "np_select",
    df_name: str = "df",
    output_col: str = "result"
) -> RuleEngineResult:
    """
    Convert KNIME Rule Engine text to Python code.
    
    Args:
        rules_text: KNIME Rule Engine text
        strategy: "np_select", "lambda", or "case"
        df_name: DataFrame variable name
        output_col: Output column name
        
    Returns:
        RuleEngineResult with Python code and imports
    """
    parser = RuleEngineParser(rules_text)
    converter = RuleEngineConverter(parser)
    
    if strategy == "lambda":
        return converter.to_lambda()
    elif strategy == "case":
        return converter.to_case_statement(df_name, output_col)
    else:
        return converter.to_np_select(df_name, output_col)


def parse_rules(rules_text: str) -> List[Rule]:
    """Parse Rule Engine text into Rule objects."""
    parser = RuleEngineParser(rules_text)
    return parser.parse()
