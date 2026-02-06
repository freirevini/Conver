"""
Unit Tests for LLM Quality Gate.

Tests the validation and self-correction loop for LLM-generated code.
"""
import ast
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from app.services.generator.llm_quality_gate import LLMQualityGate, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_valid_result_creation(self):
        """Test creating a valid result."""
        result = ValidationResult(
            is_valid=True,
            code="x = 1",
            error_message=None,
            attempts=1
        )
        assert result.is_valid
        assert result.code == "x = 1"
        assert result.error_message is None
    
    def test_invalid_result_creation(self):
        """Test creating an invalid result with error info."""
        result = ValidationResult(
            is_valid=False,
            code="x = ",
            error_message="unexpected EOF",
            error_line=1,
            attempts=2,
            corrections=["Tried adding pass"]
        )
        assert not result.is_valid
        assert result.error_line == 1
        assert len(result.corrections) == 1


class TestLLMQualityGate:
    """Tests for LLMQualityGate class."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        client.generate = Mock(return_value="```python\nx = 1 + 1\n```")
        return client
    
    @pytest.fixture
    def quality_gate(self, mock_llm_client):
        """Create LLMQualityGate instance with mock client."""
        return LLMQualityGate(llm_client=mock_llm_client)
    
    # ============= _validate_syntax tests =============
    
    def test_validate_syntax_valid_code(self, quality_gate):
        """Test validation of valid Python code."""
        is_valid, error = quality_gate._validate_syntax("x = 1 + 2\nprint(x)")
        assert is_valid is True
        assert error is None
    
    def test_validate_syntax_invalid_code(self, quality_gate):
        """Test validation of invalid Python code."""
        is_valid, error = quality_gate._validate_syntax("x = ")
        assert is_valid is False
        assert error is not None
        assert "SyntaxError" in error or "unexpected" in error.lower()
    
    def test_validate_syntax_empty_code(self, quality_gate):
        """Test validation of empty code."""
        is_valid, error = quality_gate._validate_syntax("")
        # Empty code is technically valid Python
        assert is_valid is True
    
    def test_validate_syntax_multiline_valid(self, quality_gate):
        """Test validation of multiline valid code."""
        code = """
import pandas as pd

def process_data(df):
    result = df.copy()
    result['new_col'] = result['old_col'] * 2
    return result
"""
        is_valid, error = quality_gate._validate_syntax(code)
        assert is_valid is True
    
    def test_validate_syntax_indentation_error(self, quality_gate):
        """Test validation catches indentation errors."""
        code = """
def foo():
x = 1  # Missing indentation
"""
        is_valid, error = quality_gate._validate_syntax(code)
        assert is_valid is False
    
    # ============= validate_and_fix tests =============
    
    def test_validate_and_fix_valid_code_first_try(self, quality_gate):
        """Test that valid code passes on first attempt."""
        code = "output_df = input_df.copy()"
        
        result = quality_gate.validate_and_fix(
            code=code,
            node_type="org.knime.test.NodeFactory",
            node_config={},
            input_vars=["input_df"],
            output_var="output_df"
        )
        
        assert result.is_valid
        assert result.attempts == 1
        assert code in result.code
    
    def test_validate_and_fix_with_correction(self, quality_gate, mock_llm_client):
        """Test correction loop for invalid code."""
        # First call returns corrected code
        mock_llm_client.generate.return_value = "```python\nx = 1 + 1\n```"
        
        invalid_code = "x = "  # Invalid
        
        result = quality_gate.validate_and_fix(
            code=invalid_code,
            node_type="org.knime.test.NodeFactory",
            node_config={},
            input_vars=["df"],
            output_var="result"
        )
        
        # Should attempt correction
        assert result.attempts >= 1
    
    # ============= _extract_code_from_response tests =============
    
    def test_extract_code_with_markdown_block(self, quality_gate):
        """Test extracting code from markdown code block."""
        response = """Here is the code:
```python
x = 1 + 1
result = x * 2
```
Hope this helps!"""
        
        extracted = quality_gate._extract_code_from_response(response)
        assert "x = 1 + 1" in extracted
        assert "result = x * 2" in extracted
        assert "Hope this helps" not in extracted
    
    def test_extract_code_without_markdown(self, quality_gate):
        """Test extracting code without markdown blocks."""
        response = "x = 1 + 1\nresult = x * 2"
        
        extracted = quality_gate._extract_code_from_response(response)
        assert "x = 1 + 1" in extracted
    
    def test_extract_code_empty_response(self, quality_gate):
        """Test extracting code from empty response."""
        extracted = quality_gate._extract_code_from_response("")
        assert extracted == "" or extracted.strip() == ""
    
    # ============= Metrics tests =============
    
    def test_get_metrics_initial(self, quality_gate):
        """Test initial metrics are zero."""
        metrics = quality_gate.get_metrics()
        
        assert metrics["total_validations"] == 0
        assert metrics["valid_first_attempt"] == 0
        assert metrics["corrections_needed"] == 0
    
    def test_reset_metrics(self, quality_gate):
        """Test metrics reset works."""
        # Simulate some activity
        quality_gate._metrics_total = 5
        quality_gate._metrics_valid_first = 3
        
        quality_gate.reset_metrics()
        metrics = quality_gate.get_metrics()
        
        assert metrics["total_validations"] == 0


class TestQualityGateEdgeCases:
    """Edge case tests for LLM Quality Gate."""
    
    @pytest.fixture
    def mock_llm_client(self):
        client = Mock()
        return client
    
    @pytest.fixture
    def quality_gate(self, mock_llm_client):
        return LLMQualityGate(llm_client=mock_llm_client)
    
    def test_max_correction_attempts_reached(self, quality_gate, mock_llm_client):
        """Test behavior when max correction attempts reached."""
        # Always return invalid code
        mock_llm_client.generate.return_value = "x = "  # Still invalid
        
        result = quality_gate.validate_and_fix(
            code="x = ",
            node_type="test.Node",
            node_config={},
            input_vars=["df"],
            output_var="result"
        )
        
        # Should stop after MAX_CORRECTION_ATTEMPTS
        assert result.attempts <= LLMQualityGate.MAX_CORRECTION_ATTEMPTS + 1
    
    def test_unicode_in_code(self, quality_gate):
        """Test handling of Unicode characters in code."""
        code = "comment = '日本語コメント'\nx = 1"
        
        is_valid, error = quality_gate._validate_syntax(code)
        assert is_valid is True
    
    def test_code_with_type_hints(self, quality_gate):
        """Test validation handles type hints correctly."""
        code = """
from typing import List, Dict

def process(data: Dict[str, List[int]]) -> List[str]:
    return [str(v) for v in data.values()]
"""
        is_valid, error = quality_gate._validate_syntax(code)
        assert is_valid is True
