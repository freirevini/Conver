"""Translate KNIME String Manipulation expressions to pandas using Gemini 2.5 Pro.

This module provides a focused LLM-based translator that converts KNIME
StringManipulation expressions (Java-like syntax) to equivalent pandas
one-liners. It validates the output via AST parsing before returning.

Config: Uses GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION env vars.
Model: gemini-2.5-pro (immutable per ADR 04-llm-config).
"""
import ast
import os
import re
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Model is IMMUTABLE per ADR 04
_MODEL_ID = "gemini-2.5-pro"
_AVAILABLE = None  # Lazy init
_CLIENT = None


def _init_client():
    """Lazy-init the Gemini client. Returns True if available."""
    global _AVAILABLE, _CLIENT
    if _AVAILABLE is not None:
        return _AVAILABLE
    
    try:
        from google import genai
        
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if project:
            _CLIENT = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if api_key:
                _CLIENT = genai.Client(api_key=api_key)
            else:
                logger.warning("No GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY set")
                _AVAILABLE = False
                return False
        
        _AVAILABLE = True
        return True
    except Exception as e:
        logger.warning(f"Failed to init Gemini client: {e}")
        _AVAILABLE = False
        return False


_SYSTEM_PROMPT = """You translate KNIME StringManipulation expressions to pandas Python.

CRITICAL OUTPUT RULES:
- Output ONLY the Python code, no markdown, no backticks, no explanation
- Output a SINGLE LINE: df["col"] = <expression>
- Use the exact DataFrame variable name and target column provided

KNIME → pandas mapping:
- $Col$ → df["Col"]
- substr($Col$, start, len) → df["Col"].str.slice(start, start+len)
- indexOfChars($Col$, "x") or indexOf($Col$, "x") → df["Col"].str.find("x")
- toInt(expr) → (expr).astype(int) or pd.to_numeric(expr, errors="coerce").astype("Int64")
- toDouble(expr) → pd.to_numeric(expr, errors="coerce")
- string(expr) → (expr).astype(str)
- length($Col$) → df["Col"].str.len()
- upperCase($Col$) → df["Col"].str.upper()
- lowerCase($Col$) → df["Col"].str.lower()
- strip($Col$) or trim($Col$) → df["Col"].str.strip()
- replace($Col$, "old", "new") → df["Col"].str.replace("old", "new", regex=False)
- join("sep", $A$, $B$) → df["A"].astype(str) + "sep" + df["B"].astype(str)
- if(cond, then, else) → np.where(cond, then, else)
- missing($Col$) → df["Col"].isna()
- For nested functions like toInt(substr(...)), compose: df["Col"].str.slice(a,b).astype(int)"""


def translate_expression(knime_expr: str, target_col: str,
                         df_var: str = "df") -> str:
    """Translate a KNIME StringManipulation expression to pandas.
    
    Returns the pandas code line(s), or empty string if translation fails.
    """
    if not _init_client():
        return ""
    
    prompt = (
        f"Convert this KNIME expression to pandas Python.\n"
        f"Use variable: {df_var}\n"
        f"Target column: {target_col}\n"
        f"KNIME: {knime_expr}\n\n"
        f"Output ONLY the Python code. No markdown. No explanation.\n"
        f"The result should assign to {df_var}[\"{target_col}\"]."
    )
    
    try:
        response = _CLIENT.models.generate_content(
            model=_MODEL_ID,
            contents=prompt,
            config={
                "system_instruction": _SYSTEM_PROMPT,
                "temperature": 0.0,
                "max_output_tokens": 512,
            },
        )
        
        code = _extract_code(response.text, df_var)
        if not code:
            return ""
        
        # Validate: must be valid Python
        try:
            ast.parse(code)
        except SyntaxError:
            logger.warning(f"LLM produced invalid Python: {code[:100]}")
            return ""
        
        # Validate: must reference the df_var
        if df_var not in code:
            logger.warning(f"LLM output missing df_var: {code[:80]}")
            return ""
        
        return code
        
    except Exception as e:
        logger.warning(f"LLM translation failed: {e}")
        return ""


def _extract_code(text: str, df_var: str) -> str:
    """Extract code from LLM response, handling fences and multi-line."""
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove markdown code fences
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    
    # Remove inline backticks
    text = text.replace("`", "")
    
    # Collect all code lines (skip comments, imports, blank lines)
    code_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("import"):
            continue
        code_lines.append(stripped)
    
    if not code_lines:
        return ""
    
    # Try all lines together as a block
    block = "\n".join(code_lines)
    try:
        ast.parse(block)
        return block
    except SyntaxError:
        pass
    
    # Fallback: try first line only
    try:
        ast.parse(code_lines[0])
        return code_lines[0]
    except SyntaxError:
        pass
    
    # Fallback: find line containing assignment to df_var
    for line in code_lines:
        if f'{df_var}["' in line and "=" in line:
            try:
                ast.parse(line)
                return line
            except SyntaxError:
                continue
    
    return ""


@lru_cache(maxsize=64)
def translate_cached(knime_expr: str, target_col: str,
                     df_var: str = "df") -> str:
    """Cached version to avoid repeat API calls for identical expressions."""
    return translate_expression(knime_expr, target_col, df_var)


# ── Java Snippet → pandas translator ──

_JAVA_SYSTEM_PROMPT = """You translate KNIME Java Snippet code to pandas Python.

CONTEXT:
- KNIME Java Snippets run row-by-row, but pandas operates on entire columns
- Input columns are mapped as: c_XYZ → df["OriginalColName"]
- Output columns are mapped as: out_XYZ → df["OutputColName"]
- Simple if/else → np.where(condition, if_value, else_value)
- String concatenation with + → Python f-string or .astype(str) + "..."
- .toString() → .astype(str)
- Multiple output assignments → multiple df["col"] = ... lines

CRITICAL OUTPUT RULES:
- Output ONLY the Python code, no markdown, no backticks, no explanation
- Each output column gets one assignment: df["col"] = expression
- Use np.where() for if/else conditional logic
- Use pd.notna() for null checks
- String concatenation: "text " + df["col"].astype(str) + " more text"
- Integer comparisons: df["col"] > 0

EXAMPLES:
Java: if (c_count > 0) { out_flag = 1; } else { out_flag = 0; }
Python: df["Flag"] = np.where(df["Count"] > 0, 1, 0)

Java: if (c_num > 0) { out_comment = "Found " + c_num.toString() + " items"; } else { out_comment = "None found"; }
Python: df["Comment"] = np.where(df["Num"] > 0, "Found " + df["Num"].astype(str) + " items", "None found")

Java: out_login = userName;
Python: df["Login"] = "system_user"  # flow variable, use placeholder"""


def translate_java_snippet(java_code: str, in_cols: list, out_cols: list,
                            df_var: str = "df") -> str:
    """Translate KNIME Java Snippet code to pandas Python.
    
    Args:
        java_code: Decoded Java source code from scriptBody
        in_cols: List of (KnimeName, JavaName, JavaType) tuples
        out_cols: List of (KnimeName, JavaName, JavaType) tuples
        df_var: DataFrame variable name
    
    Returns:
        Valid Python code string, or empty string if translation fails.
    """
    if not _init_client():
        return ""
    
    if not java_code or not java_code.strip():
        return ""
    
    # Build column mapping context
    mapping_lines = ["Column mappings:"]
    for knime_name, java_name, java_type in in_cols:
        mapping_lines.append(f"  Input: {java_name} → {df_var}[\"{knime_name}\"] ({java_type})")
    for knime_name, java_name, java_type in out_cols:
        mapping_lines.append(f"  Output: {java_name} → {df_var}[\"{knime_name}\"] ({java_type})")
    mapping_ctx = "\n".join(mapping_lines)
    
    prompt = (
        f"Convert this KNIME Java Snippet to pandas Python.\n"
        f"DataFrame variable: {df_var}\n\n"
        f"{mapping_ctx}\n\n"
        f"Java code:\n{java_code}\n\n"
        f"Output ONLY valid Python code. No markdown. No explanation.\n"
        f"Use np.where() for if/else. Each output column gets one line."
    )
    
    try:
        response = _CLIENT.models.generate_content(
            model=_MODEL_ID,
            contents=prompt,
            config={
                "system_instruction": _JAVA_SYSTEM_PROMPT,
                "temperature": 0.0,
                "max_output_tokens": 512,
            },
        )
        
        code = _extract_code(response.text, df_var)
        if not code:
            return ""
        
        # Validate: must be valid Python
        try:
            ast.parse(code)
        except SyntaxError:
            logger.warning(f"Java LLM produced invalid Python: {code[:100]}")
            return ""
        
        # Validate: must reference the df_var
        if df_var not in code:
            logger.warning(f"Java LLM output missing df_var: {code[:80]}")
            return ""
        
        return code
        
    except Exception as e:
        logger.warning(f"Java LLM translation failed: {e}")
        return ""

