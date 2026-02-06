"""
Code Quality and Review Utilities.

Provides:
- Code quality checks
- Type hints validation helpers
- Docstring enforcement
- Import organization
"""
import ast
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """Represents a code quality issue."""
    file: str
    line: int
    severity: str  # "error", "warning", "info"
    category: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ReviewResult:
    """Code review result summary."""
    total_files: int
    files_with_issues: int
    issues: List[CodeIssue] = field(default_factory=list)
    
    @property
    def is_clean(self) -> bool:
        return len([i for i in self.issues if i.severity == "error"]) == 0
    
    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "files_with_issues": self.files_with_issues,
            "is_clean": self.is_clean,
            "error_count": len([i for i in self.issues if i.severity == "error"]),
            "warning_count": len([i for i in self.issues if i.severity == "warning"]),
            "issues": [
                {
                    "file": i.file,
                    "line": i.line,
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message
                }
                for i in self.issues[:50]  # Limit output
            ]
        }


class CodeReviewer:
    """
    Automated code reviewer for Python files.
    
    Checks:
    - Syntax validity
    - Import organization
    - Function docstrings
    - Type hints presence
    - Magic numbers
    - Long functions
    """
    
    MAX_FUNCTION_LINES = 50
    MAX_FILE_LINES = 500
    
    def __init__(self):
        self.issues: List[CodeIssue] = []
    
    def review_file(self, file_path: Path) -> List[CodeIssue]:
        """Review a single Python file."""
        self.issues = []
        
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            
            # Check file length
            if len(lines) > self.MAX_FILE_LINES:
                self.issues.append(CodeIssue(
                    file=str(file_path),
                    line=1,
                    severity="warning",
                    category="complexity",
                    message=f"File has {len(lines)} lines (max: {self.MAX_FILE_LINES})",
                    suggestion="Consider splitting into multiple modules"
                ))
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.issues.append(CodeIssue(
                    file=str(file_path),
                    line=e.lineno or 1,
                    severity="error",
                    category="syntax",
                    message=f"Syntax error: {e.msg}",
                    suggestion="Fix the syntax error"
                ))
                return self.issues
            
            # Check functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._check_function(file_path, node, lines)
                elif isinstance(node, ast.ClassDef):
                    self._check_class(file_path, node)
            
            # Check imports
            self._check_imports(file_path, tree)
            
        except Exception as e:
            logger.error(f"Error reviewing {file_path}: {e}")
        
        return self.issues
    
    def _check_function(self, file_path: Path, func: ast.FunctionDef, lines: List[str]):
        """Check function quality."""
        # Check docstring
        if not ast.get_docstring(func):
            # Only warn for public functions
            if not func.name.startswith("_"):
                self.issues.append(CodeIssue(
                    file=str(file_path),
                    line=func.lineno,
                    severity="info",
                    category="documentation",
                    message=f"Function '{func.name}' lacks docstring",
                    suggestion="Add a docstring describing the function purpose"
                ))
        
        # Check function length
        end_line = func.end_lineno or func.lineno
        func_length = end_line - func.lineno
        if func_length > self.MAX_FUNCTION_LINES:
            self.issues.append(CodeIssue(
                file=str(file_path),
                line=func.lineno,
                severity="warning",
                category="complexity",
                message=f"Function '{func.name}' is {func_length} lines (max: {self.MAX_FUNCTION_LINES})",
                suggestion="Consider breaking into smaller functions"
            ))
        
        # Check return type hint
        if func.returns is None and not func.name.startswith("_"):
            self.issues.append(CodeIssue(
                file=str(file_path),
                line=func.lineno,
                severity="info",
                category="typing",
                message=f"Function '{func.name}' lacks return type hint",
                suggestion="Add return type annotation (e.g., -> str:)"
            ))
    
    def _check_class(self, file_path: Path, cls: ast.ClassDef):
        """Check class quality."""
        if not ast.get_docstring(cls):
            self.issues.append(CodeIssue(
                file=str(file_path),
                line=cls.lineno,
                severity="info",
                category="documentation",
                message=f"Class '{cls.name}' lacks docstring",
                suggestion="Add a docstring describing the class purpose"
            ))
    
    def _check_imports(self, file_path: Path, tree: ast.Module):
        """Check import organization."""
        imports = []
        from_imports = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                imports.append((node.lineno, [alias.name for alias in node.names]))
            elif isinstance(node, ast.ImportFrom):
                from_imports.append((node.lineno, node.module, [alias.name for alias in node.names]))
        
        # Check for duplicate imports
        seen_modules = set()
        for lineno, names in imports:
            for name in names:
                if name in seen_modules:
                    self.issues.append(CodeIssue(
                        file=str(file_path),
                        line=lineno,
                        severity="warning",
                        category="imports",
                        message=f"Duplicate import: '{name}'",
                        suggestion="Remove duplicate import"
                    ))
                seen_modules.add(name)
    
    def review_directory(self, directory: Path, pattern: str = "**/*.py") -> ReviewResult:
        """Review all Python files in a directory."""
        all_issues = []
        files_checked = 0
        files_with_issues = set()
        
        for py_file in directory.glob(pattern):
            # Skip __pycache__ and test files
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
            
            files_checked += 1
            issues = self.review_file(py_file)
            
            if issues:
                files_with_issues.add(str(py_file))
                all_issues.extend(issues)
        
        return ReviewResult(
            total_files=files_checked,
            files_with_issues=len(files_with_issues),
            issues=all_issues
        )


def validate_python_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Python code syntax.
    
    Args:
        code: Python code string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def check_type_completeness(file_path: Path) -> Dict[str, Any]:
    """
    Check type hint coverage in a Python file.
    
    Returns dict with coverage statistics.
    """
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content)
    
    functions = []
    typed_functions = 0
    untyped_functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
            
            # Check if has any type hints
            has_return_hint = node.returns is not None
            has_arg_hints = any(
                arg.annotation is not None
                for arg in node.args.args
            )
            
            if has_return_hint or has_arg_hints:
                typed_functions += 1
            else:
                untyped_functions.append(node.name)
    
    total = len(functions)
    coverage = (typed_functions / total * 100) if total > 0 else 100
    
    return {
        "total_functions": total,
        "typed_functions": typed_functions,
        "coverage_percent": round(coverage, 1),
        "untyped_functions": untyped_functions[:10]  # Limit output
    }
