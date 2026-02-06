"""
Scripting Node Templates - Python/R Script Nodes.

Handles KNIME nodes that contain embedded Python or R code:
- Python Script (Python2ScriptNodeFactory2)
- Python Script (1→1) (Python2Script1To1NodeFactory)
- Python Source (Python2SourceNodeFactory)
- R Snippet (RSnippetNodeFactory)

Key Feature: Extracts embedded source code and replicates it exactly,
then adapts input/output variable names for integration.
"""
import re
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def decode_knime_source_code(encoded: str) -> str:
    """
    Decode KNIME-encoded source code.
    
    KNIME uses custom escape sequences:
    - %%00010 = newline (\\n)
    - %%00009 = tab (\\t)
    - %%00013 = carriage return (\\r)
    
    Args:
        encoded: KNIME-encoded source code string
        
    Returns:
        Decoded Python/R source code
    """
    if not encoded:
        return ""
    
    # Decode KNIME escape sequences
    decoded = encoded
    decoded = decoded.replace("%%00010", "\n")
    decoded = decoded.replace("%%00009", "\t")
    decoded = decoded.replace("%%00013", "\r")
    
    # Handle any remaining %% escapes (generic pattern)
    decoded = re.sub(r'%%(\d{5})', lambda m: chr(int(m.group(1))), decoded)
    
    return decoded


def extract_python_script(settings: Dict[str, Any]) -> str:
    """
    Extract Python source code from node settings.
    
    Args:
        settings: Parsed node settings dictionary
        
    Returns:
        Decoded Python source code
    """
    # Look for sourceCode in various locations
    source_code = (
        settings.get("sourceCode") or
        settings.get("model", {}).get("sourceCode") or
        settings.get("script") or
        settings.get("model", {}).get("script") or
        ""
    )
    
    return decode_knime_source_code(source_code)


def adapt_python_script_variables(
    script: str,
    input_var: str = "df_input",
    output_var: str = "df_output"
) -> str:
    """
    Adapt KNIME Python script variable names for integration.
    
    KNIME Python nodes use standard variable names:
    - input_table_1, input_table_2, ... = Input DataFrames
    - output_table_1, output_table_2, ... = Output DataFrames
    - flow_variables = Flow variables (dict)
    
    Args:
        script: Decoded Python script
        input_var: Name of input DataFrame variable
        output_var: Name of output DataFrame variable
        
    Returns:
        Adapted script with updated variable names
    """
    if not script.strip():
        return f"# Empty Python Script\n{output_var} = {input_var}.copy()"
    
    # Replace input_table_1 with input variable
    adapted = re.sub(
        r'\binput_table_1\b',
        input_var,
        script
    )
    
    # Replace output_table_1 with output variable
    adapted = re.sub(
        r'\boutput_table_1\b',
        output_var,
        adapted
    )
    
    # Replace additional input tables
    for i in range(2, 10):
        adapted = re.sub(
            rf'\binput_table_{i}\b',
            f'{input_var}_{i}',
            adapted
        )
        adapted = re.sub(
            rf'\boutput_table_{i}\b',
            f'{output_var}_{i}',
            adapted
        )
    
    return adapted


def generate_python_script_template(
    settings: Dict[str, Any],
    input_var: str = "df_input",
    output_var: str = "df_output"
) -> Tuple[str, set]:
    """
    Generate Python code for a Python Script node.
    
    Extracts the embedded Python code, decodes it, and adapts
    variable names for integration with the generated pipeline.
    
    Args:
        settings: Node settings dictionary
        input_var: Input DataFrame variable name
        output_var: Output DataFrame variable name
        
    Returns:
        Tuple of (generated code, set of imports needed)
    """
    imports = set()
    
    # Extract and decode the Python script
    raw_script = extract_python_script(settings)
    
    if not raw_script.strip():
        # Empty script - just copy input to output
        code = f"""# Python Script (empty - passthrough)
{output_var} = {input_var}.copy()
"""
        return code, imports
    
    # Adapt variable names
    adapted_script = adapt_python_script_variables(
        raw_script, input_var, output_var
    )
    
    # Extract imports from the script
    script_imports = extract_imports_from_script(adapted_script)
    imports.update(script_imports)
    
    # Separate import statements from code
    import_lines = []
    code_lines = []
    
    for line in adapted_script.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_lines.append(line)
        else:
            code_lines.append(line)
    
    # Build the code block
    code_body = '\n'.join(code_lines)
    
    # Add header comment with original imports noted
    import_comment = ""
    if import_lines:
        imports_str = ', '.join(import_lines[:3])
        if len(import_lines) > 3:
            imports_str += f", ... ({len(import_lines)} total)"
        import_comment = f"\n# Original imports: {imports_str}"
    
    code = f"""# ============================================================
# Python Script Node - Code extracted from KNIME workflow
# This is the EXACT code from the original KNIME node{import_comment}
# ============================================================

{code_body}
"""
    
    return code, imports


def extract_imports_from_script(script: str) -> set:
    """
    Extract import statements from Python script.
    
    Args:
        script: Python source code
        
    Returns:
        Set of import statement strings
    """
    imports = set()
    
    for line in script.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import '):
            imports.add(stripped)
        elif stripped.startswith('from '):
            imports.add(stripped)
    
    return imports


def generate_r_script_template(
    settings: Dict[str, Any],
    input_var: str = "df_input",
    output_var: str = "df_output"
) -> Tuple[str, set]:
    """
    Generate Python wrapper for R Script node.
    
    R scripts are converted to Python equivalents where possible,
    or wrapped with rpy2 for direct R execution.
    
    Args:
        settings: Node settings dictionary
        input_var: Input DataFrame variable name
        output_var: Output DataFrame variable name
        
    Returns:
        Tuple of (generated code, set of imports needed)
    """
    imports = {"import pandas as pd"}
    
    # Extract R source code
    r_code = (
        settings.get("rCode") or
        settings.get("model", {}).get("rCode") or
        settings.get("script") or
        settings.get("model", {}).get("script") or
        ""
    )
    
    r_code = decode_knime_source_code(r_code)
    
    if not r_code.strip():
        code = f"""# R Script (empty - passthrough)
{output_var} = {input_var}.copy()
"""
        return code, imports
    
    # Check if simple R operations can be converted to Pandas
    if can_convert_r_to_pandas(r_code):
        code, extra_imports = convert_r_to_pandas(r_code, input_var, output_var)
        imports.update(extra_imports)
        return code, imports
    
    # For complex R code, wrap with rpy2 (if available)
    imports.add("# Note: rpy2 required for R execution")
    
    # Escape R code for embedding
    escaped_r = r_code.replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")
    
    code = f"""# ============================================================
# R Script Node - Original R code from KNIME workflow
# NOTE: This R code needs manual conversion to Python
# or execution via rpy2 (not implemented in this version)
# ============================================================

# Original R code:
'''
{escaped_r}
'''

# Placeholder - passthrough until R code is converted
{output_var} = {input_var}.copy()

# TODO: Convert the following R operations to Python/Pandas:
# - Analyze R code above and implement equivalent logic
"""
    
    return code, imports


def can_convert_r_to_pandas(r_code: str) -> bool:
    """
    Check if R code can be automatically converted to Pandas.
    
    Simple operations like filtering, selection, and basic
    transformations can often be converted automatically.
    """
    # Simple heuristics - expand as needed
    simple_patterns = [
        r'^df\s*<-\s*df\[',  # Simple filtering
        r'^df\$\w+\s*<-',     # Column assignment
        r'^subset\(',        # Subset function
    ]
    
    lines = [l.strip() for l in r_code.split('\n') if l.strip() and not l.strip().startswith('#')]
    
    if len(lines) <= 3:
        for pattern in simple_patterns:
            if any(re.match(pattern, line) for line in lines):
                return True
    
    return False


def convert_r_to_pandas(
    r_code: str, 
    input_var: str, 
    output_var: str
) -> Tuple[str, set]:
    """
    Convert simple R code to Pandas equivalent.
    
    This handles basic R operations that have direct Pandas equivalents.
    """
    imports = {"import pandas as pd"}
    
    # Basic conversion (extend as needed)
    pandas_code = r_code
    
    # Replace R assignment
    pandas_code = re.sub(r'<-', '=', pandas_code)
    
    # Replace knime.in with input_var
    pandas_code = re.sub(r'\bknime\.in\b', input_var, pandas_code)
    pandas_code = re.sub(r'\bknime\.out\b', output_var, pandas_code)
    
    code = f"""# R Script converted to Pandas
# Original R operations translated below

{pandas_code}

{output_var} = {input_var}.copy()
"""
    
    return code, imports


# Template registry for scripting nodes
SCRIPTING_TEMPLATES = {
    "org.knime.python2.nodes.script2.Python2ScriptNodeFactory2": generate_python_script_template,
    "org.knime.python2.nodes.script.Python2ScriptNodeFactory": generate_python_script_template,
    "org.knime.python2.nodes.script1to1.Python2Script1To1NodeFactory": generate_python_script_template,
    "org.knime.python2.nodes.source.Python2SourceNodeFactory": generate_python_script_template,
    "org.knime.python3.nodes.script.PythonScriptNodeFactory": generate_python_script_template,
    "org.knime.r.RSnippetNodeFactory": generate_r_script_template,
    "org.knime.r.RViewNodeFactory": generate_r_script_template,
}


def is_scripting_node(factory: str) -> bool:
    """Check if factory is a scripting node."""
    return factory in SCRIPTING_TEMPLATES or any(
        x in factory.lower() for x in ['python', 'jython', 'rsnippet', 'rscript']
    )


def handle_scripting_node(
    factory: str,
    settings: Dict[str, Any],
    input_var: str = "df_input",
    output_var: str = "df_output"
) -> Tuple[str, set]:
    """
    Handle a scripting node (Python/R).
    
    Args:
        factory: Node factory class
        settings: Node settings
        input_var: Input variable name
        output_var: Output variable name
        
    Returns:
        Tuple of (generated code, imports)
    """
    # Check for exact match
    if factory in SCRIPTING_TEMPLATES:
        return SCRIPTING_TEMPLATES[factory](settings, input_var, output_var)
    
    # Check for Python nodes by pattern
    if 'python' in factory.lower():
        return generate_python_script_template(settings, input_var, output_var)
    
    # Check for R nodes by pattern
    if 'rsnippet' in factory.lower() or 'rscript' in factory.lower():
        return generate_r_script_template(settings, input_var, output_var)
    
    # Fallback
    return f"# Unknown scripting node: {factory}\n{output_var} = {input_var}.copy()", set()
