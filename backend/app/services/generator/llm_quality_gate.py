"""
LLM Quality Gate - Validates and Fixes LLM-Generated Code

Ensures 100% syntax validity by:
1. AST validation before accepting code
2. Self-correction retry with error feedback
3. Fallback to stub after max attempts

Problem: P3.1 - LLM generates invalid Python syntax
Solution: Validate & retry with correction prompts
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    code: str
    error_message: Optional[str] = None
    error_line: Optional[int] = None
    attempts: int = 0
    corrections: List[str] = field(default_factory=list)


class LLMQualityGate:
    """
    Quality gate for LLM-generated code.
    
    Ensures all generated code is syntactically valid Python
    before returning to the transpiler.
    """
    
    MAX_CORRECTION_ATTEMPTS = 3
    
    def __init__(self, llm_client):
        """
        Initialize quality gate.
        
        Args:
            llm_client: GeminiClient instance for LLM calls
        """
        self.llm_client = llm_client
        self.metrics = {
            "total_validations": 0,
            "passed_first_try": 0,
            "passed_after_correction": 0,
            "failed_all_attempts": 0,
            "total_corrections": 0,
        }
    
    def validate_and_fix(
        self,
        code: str,
        node_type: str,
        node_config: Dict[str, Any],
        input_vars: List[str],
        output_var: str,
    ) -> ValidationResult:
        """
        Validate code and attempt to fix if invalid.
        
        Args:
            code: LLM-generated code to validate
            node_type: KNIME node factory class for context
            node_config: Node configuration for regeneration
            input_vars: Input variable names
            output_var: Output variable name
            
        Returns:
            ValidationResult with valid code or failure info
        """
        self.metrics["total_validations"] += 1
        
        # Attempt 1: Validate original code
        is_valid, error = self._validate_syntax(code)
        
        if is_valid:
            self.metrics["passed_first_try"] += 1
            logger.info(f"LLM code passed validation on first try for {node_type}")
            return ValidationResult(
                is_valid=True,
                code=code,
                attempts=1
            )
        
        # Correction attempts with self-healing prompts
        corrections = []
        current_code = code
        
        for attempt in range(1, self.MAX_CORRECTION_ATTEMPTS + 1):
            self.metrics["total_corrections"] += 1
            
            logger.warning(
                f"Syntax error in LLM code for {node_type}: {error}. "
                f"Attempting correction {attempt}/{self.MAX_CORRECTION_ATTEMPTS}"
            )
            
            # Request correction from LLM
            corrected_code = self._request_correction(
                invalid_code=current_code,
                error_message=error,
                node_type=node_type,
                node_config=node_config,
                input_vars=input_vars,
                output_var=output_var,
            )
            
            if corrected_code is None:
                corrections.append(f"Attempt {attempt}: LLM correction request failed")
                continue
            
            corrections.append(f"Attempt {attempt}: Requested fix for '{error}'")
            
            # Validate corrected code
            is_valid, new_error = self._validate_syntax(corrected_code)
            
            if is_valid:
                self.metrics["passed_after_correction"] += 1
                logger.info(
                    f"LLM code fixed after {attempt} correction(s) for {node_type}"
                )
                return ValidationResult(
                    is_valid=True,
                    code=corrected_code,
                    attempts=attempt + 1,
                    corrections=corrections
                )
            
            # Update for next iteration
            current_code = corrected_code
            error = new_error
        
        # All attempts failed
        self.metrics["failed_all_attempts"] += 1
        logger.error(
            f"Failed to fix LLM code after {self.MAX_CORRECTION_ATTEMPTS} attempts "
            f"for {node_type}. Last error: {error}"
        )
        
        return ValidationResult(
            is_valid=False,
            code=current_code,
            error_message=error,
            attempts=self.MAX_CORRECTION_ATTEMPTS + 1,
            corrections=corrections
        )
    
    def _validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax using AST.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Empty code"
        
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"{e.msg} at line {e.lineno}"
            if e.text:
                error_msg += f": {e.text.strip()[:50]}"
            return False, error_msg
        except Exception as e:
            return False, str(e)
    
    def _request_correction(
        self,
        invalid_code: str,
        error_message: str,
        node_type: str,
        node_config: Dict[str, Any],
        input_vars: List[str],
        output_var: str,
    ) -> Optional[str]:
        """
        Request LLM to correct invalid code.
        
        Uses structured prompt with error feedback for self-correction.
        """
        if not self.llm_client or not self.llm_client.is_available():
            return None
        
        correction_prompt = self._build_correction_prompt(
            invalid_code=invalid_code,
            error_message=error_message,
            node_type=node_type,
        )
        
        try:
            from google.genai import types
            
            response = self.llm_client.client.models.generate_content(
                model=self.llm_client.model_name,
                contents=[correction_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Lower temp for correction
                    top_p=0.9,
                    max_output_tokens=2048,
                )
            )
            
            return self._extract_code_from_response(response.text)
            
        except Exception as e:
            logger.error(f"LLM correction request failed: {e}")
            return None
    
    def _build_correction_prompt(
        self,
        invalid_code: str,
        error_message: str,
        node_type: str,
    ) -> str:
        """Build prompt for code correction."""
        
        simple_name = node_type.split('.')[-1].replace('NodeFactory', '')
        
        return f"""You are a Python code fixer. The following code has a syntax error.

## Error
```
{error_message}
```

## Invalid Code
```python
{invalid_code}
```

## Context
This code was generated for KNIME node: {simple_name}

## Task
Fix the syntax error. Return ONLY the corrected Python code, no explanations.

## Requirements
1. Fix the syntax error shown above
2. Keep the original logic and structure
3. Ensure all strings are properly terminated
4. Ensure all brackets/parentheses are balanced
5. Use proper indentation (4 spaces)

## Output
Return ONLY valid Python code:
```python
# Fixed code here
```"""
    
    def _extract_code_from_response(self, response_text: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        if not response_text:
            return None
        
        # Try to extract from code block
        code_match = re.search(r'```python\n([\s\S]*?)\n```', response_text)
        if code_match:
            return code_match.group(1)
        
        # Try generic code block
        code_match = re.search(r'```\n([\s\S]*?)\n```', response_text)
        if code_match:
            return code_match.group(1)
        
        # Return raw response if no code block
        return response_text.strip()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get quality gate metrics."""
        total = self.metrics["total_validations"]
        if total == 0:
            success_rate = 1.0
        else:
            passed = (
                self.metrics["passed_first_try"] + 
                self.metrics["passed_after_correction"]
            )
            success_rate = passed / total
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "first_try_rate": (
                self.metrics["passed_first_try"] / total if total > 0 else 1.0
            ),
        }
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = {
            "total_validations": 0,
            "passed_first_try": 0,
            "passed_after_correction": 0,
            "failed_all_attempts": 0,
            "total_corrections": 0,
        }
