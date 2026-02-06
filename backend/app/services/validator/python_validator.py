"""
Python Code Validator

Validates generated Python code for syntax and import availability.
"""
import ast
import logging
import importlib.util
from typing import Tuple, List, Set, Dict, Any
import re

logger = logging.getLogger(__name__)


class PythonValidator:
    """
    Validates Python code syntax and checks imports.
    
    Provides static analysis without executing the code.
    """
    
    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Python code syntax.
        
        Args:
            code: Python code string
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            ast.parse(code)
            logger.debug("Syntax validation passed")
            return True, []
            
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f" near: {e.text.strip()}"
            errors.append(error_msg)
            logger.warning(f"Syntax error: {error_msg}")
            return False, errors
            
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")
            logger.error(f"Validation error: {e}")
            return False, errors
    
    def check_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check if all imports in the code are available.
        
        Args:
            code: Python code string
            
        Returns:
            Tuple of (all_available, list_of_missing_packages)
        """
        imports = self.extract_imports(code)
        missing = []
        
        for module_name in imports:
            if not self._is_module_available(module_name):
                missing.append(module_name)
        
        if missing:
            logger.warning(f"Missing packages: {missing}")
        
        return len(missing) == 0, missing
    
    def extract_imports(self, code: str) -> Set[str]:
        """
        Extract all imported module names from code.
        
        Args:
            code: Python code string
            
        Returns:
            Set of root module names
        """
        imports = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get root module (e.g., 'pandas' from 'pandas.DataFrame')
                        root_module = alias.name.split('.')[0]
                        imports.add(root_module)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root_module = node.module.split('.')[0]
                        imports.add(root_module)
                        
        except SyntaxError:
            # If parsing fails, use regex fallback
            import_pattern = r'^(?:from|import)\s+(\w+)'
            matches = re.findall(import_pattern, code, re.MULTILINE)
            imports.update(matches)
        
        return imports
    
    def _is_module_available(self, module_name: str) -> bool:
        """Check if a module is available for import."""
        # Standard library modules that are always available
        stdlib_modules = {
            'os', 'sys', 'json', 're', 'ast', 'math', 'random', 'datetime',
            'collections', 'itertools', 'functools', 'typing', 'logging',
            'pathlib', 'uuid', 'copy', 'time', 'io', 'csv', 'shutil',
            'inspect', 'subprocess', 'threading', 'multiprocessing',
        }
        
        if module_name in stdlib_modules:
            return True
        
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    
    def validate_complete(self, code: str) -> Dict[str, Any]:
        """
        Perform complete validation.
        
        Returns:
            Dictionary with:
            - syntax_valid: bool
            - syntax_errors: List[str]
            - imports_available: bool
            - missing_imports: List[str]
            - line_count: int
            - has_main: bool
        """
        syntax_valid, syntax_errors = self.validate_syntax(code)
        imports_available, missing_imports = self.check_imports(code)
        
        lines = code.split('\n')
        line_count = len(lines)
        
        # Check for main execution block
        has_main = 'if __name__' in code
        
        return {
            'syntax_valid': syntax_valid,
            'syntax_errors': syntax_errors,
            'imports_available': imports_available,
            'missing_imports': missing_imports,
            'line_count': line_count,
            'has_main': has_main,
            'overall_valid': syntax_valid  # Imports being missing doesn't invalidate code
        }
    
    def fix_common_issues(self, code: str) -> str:
        """
        Attempt to fix common syntax issues.
        
        Note: This is a best-effort fix, not guaranteed to work.
        
        Args:
            code: Python code with potential issues
            
        Returns:
            Fixed code (or original if no fixes applied)
        """
        fixed = code
        
        # Fix unclosed string quotes
        lines = fixed.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip comments
            if line.strip().startswith('#'):
                fixed_lines.append(line)
                continue
            
            # Try to balance quotes
            single_quotes = line.count("'") - line.count("\\'")
            double_quotes = line.count('"') - line.count('\\"')
            
            if single_quotes % 2 != 0:
                line += "'"
            if double_quotes % 2 != 0:
                line += '"'
            
            fixed_lines.append(line)
        
        fixed = '\n'.join(fixed_lines)
        
        # Validate the fix
        is_valid, _ = self.validate_syntax(fixed)
        
        if is_valid:
            logger.info("Fixed syntax issues")
            return fixed
        
        # Return original if fix didn't work
        return code
    
    def get_defined_variables(self, code: str) -> Set[str]:
        """
        Extract all variable names defined in the code.
        
        Useful for checking if all expected outputs are generated.
        """
        variables = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.add(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    variables.add(elt.id)
                                    
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        variables.add(node.target.id)
                        
        except SyntaxError:
            # Use regex fallback
            var_pattern = r'^(\w+)\s*='
            matches = re.findall(var_pattern, code, re.MULTILINE)
            variables.update(matches)
        
        return variables
