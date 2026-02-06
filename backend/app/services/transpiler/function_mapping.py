"""
KNIME Function Mapping to Python.

Maps 100+ KNIME functions to Python/Pandas/NumPy equivalents.
Used by the formula transformer to generate valid Python code.
"""
from typing import Dict, Callable, List, Optional, Any
from dataclasses import dataclass


@dataclass
class FunctionMapping:
    """Mapping from KNIME function to Python."""
    python_template: str       # Template with {0}, {1} for args
    python_import: str = ""    # Required import
    min_args: int = 0
    max_args: int = 99
    description: str = ""


# ==================== String Functions ====================

STRING_FUNCTIONS: Dict[str, FunctionMapping] = {
    # Basic String
    "length": FunctionMapping(
        python_template="len({0})",
        description="Returns string length"
    ),
    "strlen": FunctionMapping(
        python_template="len({0})",
        description="Returns string length (alias)"
    ),
    "upper": FunctionMapping(
        python_template="str({0}).upper()",
        description="Convert to uppercase"
    ),
    "upperCase": FunctionMapping(
        python_template="str({0}).upper()",
        description="Convert to uppercase"
    ),
    "lower": FunctionMapping(
        python_template="str({0}).lower()",
        description="Convert to lowercase"
    ),
    "lowerCase": FunctionMapping(
        python_template="str({0}).lower()",
        description="Convert to lowercase"
    ),
    "capitalize": FunctionMapping(
        python_template="str({0}).capitalize()",
        description="Capitalize first letter"
    ),
    "trim": FunctionMapping(
        python_template="str({0}).strip()",
        description="Remove leading/trailing whitespace"
    ),
    "strip": FunctionMapping(
        python_template="str({0}).strip()",
        description="Remove leading/trailing whitespace"
    ),
    "ltrim": FunctionMapping(
        python_template="str({0}).lstrip()",
        description="Remove leading whitespace"
    ),
    "rtrim": FunctionMapping(
        python_template="str({0}).rstrip()",
        description="Remove trailing whitespace"
    ),
    
    # Substring
    "substr": FunctionMapping(
        python_template="str({0})[{1}:{2}]",
        min_args=2, max_args=3,
        description="Extract substring"
    ),
    "substring": FunctionMapping(
        python_template="str({0})[{1}:{2}]",
        min_args=2, max_args=3,
        description="Extract substring"
    ),
    "left": FunctionMapping(
        python_template="str({0})[:{1}]",
        min_args=2, max_args=2,
        description="Left N characters"
    ),
    "right": FunctionMapping(
        python_template="str({0})[-{1}:]",
        min_args=2, max_args=2,
        description="Right N characters"
    ),
    
    # Search
    "indexOf": FunctionMapping(
        python_template="str({0}).find({1})",
        min_args=2,
        description="Find index of substring"
    ),
    "lastIndexOf": FunctionMapping(
        python_template="str({0}).rfind({1})",
        min_args=2,
        description="Find last index of substring"
    ),
    "contains": FunctionMapping(
        python_template="({1} in str({0}))",
        min_args=2, max_args=2,
        description="Check if contains substring"
    ),
    "startsWith": FunctionMapping(
        python_template="str({0}).startswith({1})",
        min_args=2, max_args=2,
        description="Check if starts with"
    ),
    "endsWith": FunctionMapping(
        python_template="str({0}).endswith({1})",
        min_args=2, max_args=2,
        description="Check if ends with"
    ),
    
    # Replace
    "replace": FunctionMapping(
        python_template="str({0}).replace({1}, {2})",
        min_args=3, max_args=3,
        description="Replace substring"
    ),
    "replaceChars": FunctionMapping(
        python_template="str({0}).replace({1}, {2})",
        min_args=3, max_args=3,
        description="Replace characters"
    ),
    "regexReplace": FunctionMapping(
        python_template="re.sub({1}, {2}, str({0}))",
        python_import="import re",
        min_args=3, max_args=3,
        description="Regex replace"
    ),
    
    # Join/Split
    "join": FunctionMapping(
        python_template="{0}.join([{1}])",
        min_args=2,
        description="Join strings with separator"
    ),
    "split": FunctionMapping(
        python_template="str({0}).split({1})",
        min_args=2,
        description="Split string"
    ),
    
    # Concat
    "concat": FunctionMapping(
        python_template="str({0}) + str({1})",
        min_args=2,
        description="Concatenate strings"
    ),
    "string": FunctionMapping(
        python_template="str({0})",
        min_args=1, max_args=1,
        description="Convert to string"
    ),
    
    # Padding
    "padLeft": FunctionMapping(
        python_template="str({0}).rjust({1}, {2})",
        min_args=2, max_args=3,
        description="Pad left"
    ),
    "padRight": FunctionMapping(
        python_template="str({0}).ljust({1}, {2})",
        min_args=2, max_args=3,
        description="Pad right"
    ),
}


# ==================== Math Functions ====================

MATH_FUNCTIONS: Dict[str, FunctionMapping] = {
    # Basic Math
    "abs": FunctionMapping(
        python_template="abs({0})",
        description="Absolute value"
    ),
    "round": FunctionMapping(
        python_template="round({0}, {1})" if "{1}" else "round({0})",
        min_args=1, max_args=2,
        description="Round to N decimals"
    ),
    "floor": FunctionMapping(
        python_template="math.floor({0})",
        python_import="import math",
        description="Floor value"
    ),
    "ceil": FunctionMapping(
        python_template="math.ceil({0})",
        python_import="import math",
        description="Ceiling value"
    ),
    "ceiling": FunctionMapping(
        python_template="math.ceil({0})",
        python_import="import math",
        description="Ceiling value"
    ),
    "trunc": FunctionMapping(
        python_template="math.trunc({0})",
        python_import="import math",
        description="Truncate to integer"
    ),
    
    # Powers/Roots
    "sqrt": FunctionMapping(
        python_template="math.sqrt({0})",
        python_import="import math",
        description="Square root"
    ),
    "pow": FunctionMapping(
        python_template="pow({0}, {1})",
        min_args=2, max_args=2,
        description="Power"
    ),
    "exp": FunctionMapping(
        python_template="math.exp({0})",
        python_import="import math",
        description="Exponential"
    ),
    "log": FunctionMapping(
        python_template="math.log({0})",
        python_import="import math",
        description="Natural logarithm"
    ),
    "log10": FunctionMapping(
        python_template="math.log10({0})",
        python_import="import math",
        description="Base-10 logarithm"
    ),
    "ln": FunctionMapping(
        python_template="math.log({0})",
        python_import="import math",
        description="Natural logarithm"
    ),
    
    # Trigonometry
    "sin": FunctionMapping(
        python_template="math.sin({0})",
        python_import="import math",
        description="Sine"
    ),
    "cos": FunctionMapping(
        python_template="math.cos({0})",
        python_import="import math",
        description="Cosine"
    ),
    "tan": FunctionMapping(
        python_template="math.tan({0})",
        python_import="import math",
        description="Tangent"
    ),
    "asin": FunctionMapping(
        python_template="math.asin({0})",
        python_import="import math",
        description="Arc sine"
    ),
    "acos": FunctionMapping(
        python_template="math.acos({0})",
        python_import="import math",
        description="Arc cosine"
    ),
    "atan": FunctionMapping(
        python_template="math.atan({0})",
        python_import="import math",
        description="Arc tangent"
    ),
    "atan2": FunctionMapping(
        python_template="math.atan2({0}, {1})",
        python_import="import math",
        min_args=2, max_args=2,
        description="Two-argument arc tangent"
    ),
    
    # Aggregation (for single values)
    "min": FunctionMapping(
        python_template="min({0}, {1})",
        min_args=2,
        description="Minimum"
    ),
    "max": FunctionMapping(
        python_template="max({0}, {1})",
        min_args=2,
        description="Maximum"
    ),
    "mod": FunctionMapping(
        python_template="({0} % {1})",
        min_args=2, max_args=2,
        description="Modulo"
    ),
    
    # Sign
    "sign": FunctionMapping(
        python_template="(1 if {0} > 0 else (-1 if {0} < 0 else 0))",
        description="Sign of number"
    ),
    "signum": FunctionMapping(
        python_template="(1 if {0} > 0 else (-1 if {0} < 0 else 0))",
        description="Sign of number"
    ),
    
    # Random
    "rand": FunctionMapping(
        python_template="random.random()",
        python_import="import random",
        min_args=0, max_args=0,
        description="Random 0-1"
    ),
    "randInt": FunctionMapping(
        python_template="random.randint({0}, {1})",
        python_import="import random",
        min_args=2, max_args=2,
        description="Random integer in range"
    ),
    
    # Type conversion
    "toInt": FunctionMapping(
        python_template="int({0})",
        description="Convert to integer"
    ),
    "toDouble": FunctionMapping(
        python_template="float({0})",
        description="Convert to float"
    ),
    "toFloat": FunctionMapping(
        python_template="float({0})",
        description="Convert to float"
    ),
}


# ==================== Date/Time Functions ====================

DATE_FUNCTIONS: Dict[str, FunctionMapping] = {
    "now": FunctionMapping(
        python_template="datetime.datetime.now()",
        python_import="import datetime",
        min_args=0, max_args=0,
        description="Current datetime"
    ),
    "today": FunctionMapping(
        python_template="datetime.date.today()",
        python_import="import datetime",
        min_args=0, max_args=0,
        description="Current date"
    ),
    "year": FunctionMapping(
        python_template="{0}.year",
        description="Extract year"
    ),
    "month": FunctionMapping(
        python_template="{0}.month",
        description="Extract month"
    ),
    "day": FunctionMapping(
        python_template="{0}.day",
        description="Extract day"
    ),
    "hour": FunctionMapping(
        python_template="{0}.hour",
        description="Extract hour"
    ),
    "minute": FunctionMapping(
        python_template="{0}.minute",
        description="Extract minute"
    ),
    "second": FunctionMapping(
        python_template="{0}.second",
        description="Extract second"
    ),
    "dayOfWeek": FunctionMapping(
        python_template="{0}.weekday()",
        description="Day of week (0=Monday)"
    ),
    "dayOfYear": FunctionMapping(
        python_template="{0}.timetuple().tm_yday",
        description="Day of year"
    ),
    "weekOfYear": FunctionMapping(
        python_template="{0}.isocalendar()[1]",
        description="Week of year"
    ),
    "dateAdd": FunctionMapping(
        python_template="{0} + datetime.timedelta({1}={2})",
        python_import="import datetime",
        min_args=3,
        description="Add to date"
    ),
    "dateDiff": FunctionMapping(
        python_template="({0} - {1}).days",
        min_args=2,
        description="Difference in days"
    ),
    "toDate": FunctionMapping(
        python_template="pd.to_datetime({0})",
        python_import="import pandas as pd",
        description="Convert to date"
    ),
    "formatDate": FunctionMapping(
        python_template="{0}.strftime({1})",
        min_args=2,
        description="Format date"
    ),
    "parseDate": FunctionMapping(
        python_template="datetime.datetime.strptime({0}, {1})",
        python_import="import datetime",
        min_args=2,
        description="Parse date string"
    ),
}


# ==================== Logical/Conditional Functions ====================

LOGICAL_FUNCTIONS: Dict[str, FunctionMapping] = {
    "if": FunctionMapping(
        python_template="({1} if {0} else {2})",
        min_args=3, max_args=3,
        description="Conditional"
    ),
    "ifelse": FunctionMapping(
        python_template="({1} if {0} else {2})",
        min_args=3, max_args=3,
        description="Conditional"
    ),
    "iif": FunctionMapping(
        python_template="({1} if {0} else {2})",
        min_args=3, max_args=3,
        description="Inline if"
    ),
    "isNaN": FunctionMapping(
        python_template="pd.isna({0})",
        python_import="import pandas as pd",
        description="Check if NaN"
    ),
    "isMissing": FunctionMapping(
        python_template="pd.isna({0})",
        python_import="import pandas as pd",
        description="Check if missing"
    ),
    "isNull": FunctionMapping(
        python_template="pd.isna({0})",
        python_import="import pandas as pd",
        description="Check if null"
    ),
    "coalesce": FunctionMapping(
        python_template="next((x for x in [{0}, {1}] if pd.notna(x)), None)",
        python_import="import pandas as pd",
        min_args=2,
        description="First non-null value"
    ),
    "nvl": FunctionMapping(
        python_template="({0} if pd.notna({0}) else {1})",
        python_import="import pandas as pd",
        min_args=2, max_args=2,
        description="Null value replacement"
    ),
    "not": FunctionMapping(
        python_template="(not {0})",
        description="Logical NOT"
    ),
    "and": FunctionMapping(
        python_template="({0} and {1})",
        min_args=2,
        description="Logical AND"
    ),
    "or": FunctionMapping(
        python_template="({0} or {1})",
        min_args=2,
        description="Logical OR"
    ),
}


# ==================== Aggregation Functions (for Pandas) ====================

AGGREGATION_FUNCTIONS: Dict[str, FunctionMapping] = {
    "sum": FunctionMapping(
        python_template="{0}.sum()",
        description="Sum values"
    ),
    "mean": FunctionMapping(
        python_template="{0}.mean()",
        description="Mean value"
    ),
    "avg": FunctionMapping(
        python_template="{0}.mean()",
        description="Average value"
    ),
    "median": FunctionMapping(
        python_template="{0}.median()",
        description="Median value"
    ),
    "std": FunctionMapping(
        python_template="{0}.std()",
        description="Standard deviation"
    ),
    "var": FunctionMapping(
        python_template="{0}.var()",
        description="Variance"
    ),
    "count": FunctionMapping(
        python_template="{0}.count()",
        description="Count values"
    ),
    "countDistinct": FunctionMapping(
        python_template="{0}.nunique()",
        description="Count distinct"
    ),
    "first": FunctionMapping(
        python_template="{0}.iloc[0]",
        description="First value"
    ),
    "last": FunctionMapping(
        python_template="{0}.iloc[-1]",
        description="Last value"
    ),
}


# ==================== Master Function Mapping ====================

ALL_FUNCTIONS: Dict[str, FunctionMapping] = {
    **STRING_FUNCTIONS,
    **MATH_FUNCTIONS,
    **DATE_FUNCTIONS,
    **LOGICAL_FUNCTIONS,
    **AGGREGATION_FUNCTIONS,
}


def get_function_mapping(name: str) -> Optional[FunctionMapping]:
    """Get function mapping by name (case-insensitive)."""
    return ALL_FUNCTIONS.get(name) or ALL_FUNCTIONS.get(name.lower())


def get_all_function_names() -> List[str]:
    """Get all supported function names."""
    return list(ALL_FUNCTIONS.keys())


def get_required_imports(function_names: List[str]) -> List[str]:
    """Get required imports for a list of functions."""
    imports = set()
    for name in function_names:
        mapping = get_function_mapping(name)
        if mapping and mapping.python_import:
            imports.add(mapping.python_import)
    return sorted(imports)


# Count for documentation
FUNCTION_COUNT = len(ALL_FUNCTIONS)
