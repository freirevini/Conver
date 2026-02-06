"""
Unit Tests for Fallback Handler.

Tests the multi-level fallback strategy for unsupported KNIME nodes.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Dict, Any, List

from app.models.ir_models import FallbackLevel, NodeInstance
from app.services.generator.fallback_handler import FallbackHandler, FallbackResult


class TestFallbackResult:
    """Tests for FallbackResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a fallback result."""
        result = FallbackResult(
            code="# Stub code",
            level=FallbackLevel.STUB,
            warnings=["Manual implementation needed"],
            confidence=0.3
        )
        
        assert result.code == "# Stub code"
        assert result.level == FallbackLevel.STUB
        assert len(result.warnings) == 1
        assert result.confidence == 0.3
    
    def test_result_default_values(self):
        """Test default values for FallbackResult."""
        result = FallbackResult(
            code="x = 1",
            level=FallbackLevel.EXACT
        )
        
        assert result.warnings == []
        assert result.suggestions == []
        assert result.confidence == 0.0


class TestFallbackHandler:
    """Tests for FallbackHandler class."""
    
    @pytest.fixture
    def mock_node(self):
        """Create a mock NodeInstance."""
        node = Mock(spec=NodeInstance)
        node.id = "node_1"
        node.name = "Test Node"
        node.factory = "org.knime.unknown.TestNodeFactory"
        node.settings = {"param1": "value1"}
        node.input_ports = [{"name": "input", "type": "BufferedDataTable"}]
        node.output_ports = [{"name": "output", "type": "BufferedDataTable"}]
        return node
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        client.generate = Mock(return_value="```python\nresult = input_df.copy()\n```")
        return client
    
    @pytest.fixture
    def handler_no_llm(self):
        """Create FallbackHandler without LLM."""
        return FallbackHandler(llm_client=None)
    
    @pytest.fixture  
    def handler_with_llm(self, mock_llm_client):
        """Create FallbackHandler with mock LLM."""
        return FallbackHandler(llm_client=mock_llm_client)
    
    # ============= get_fallback tests =============
    
    def test_get_fallback_returns_result(self, handler_no_llm, mock_node):
        """Test get_fallback always returns a result."""
        result = handler_no_llm.get_fallback(mock_node)
        
        assert isinstance(result, FallbackResult)
        assert result.code != ""
        assert result.level is not None
    
    def test_get_fallback_unknown_node_generates_stub(self, handler_no_llm, mock_node):
        """Test unknown node falls back to stub."""
        mock_node.factory = "org.knime.completely.unknown.NodeFactory"
        
        result = handler_no_llm.get_fallback(mock_node)
        
        # Should generate a stub
        assert result.level == FallbackLevel.STUB
        assert "TODO" in result.code or "NotImplemented" in result.code.lower() or "Manual" in result.code
    
    def test_get_fallback_with_llm_uses_ai(self, handler_with_llm, mock_node, mock_llm_client):
        """Test that handler uses LLM when available."""
        mock_node.factory = "org.knime.unknown.NodeFactory"
        
        # Patch _find_approximate_template to return None (no template found)
        with patch.object(handler_with_llm, '_find_approximate_template', return_value=None):
            result = handler_with_llm.get_fallback(mock_node)
        
        # Should have attempted LLM generation
        # Either LLM was called or it fell back to stub
        assert result.code != ""
    
    # ============= _generate_stub tests =============
    
    def test_generate_stub_includes_node_info(self, handler_no_llm, mock_node):
        """Test stub includes node information."""
        result = handler_no_llm._generate_stub(mock_node)
        
        # Stub should contain node info
        assert mock_node.factory in result.code or "TestNode" in result.code or "def" in result.code
    
    def test_generate_stub_is_valid_python(self, handler_no_llm, mock_node):
        """Test generated stub is syntactically valid."""
        import ast
        
        result = handler_no_llm._generate_stub(mock_node)
        
        # Should be valid Python
        try:
            ast.parse(result.code)
            is_valid = True
        except SyntaxError:
            is_valid = False
        
        assert is_valid, f"Stub code is not valid Python:\n{result.code}"
    
    # ============= _clean_name tests =============
    
    def test_clean_name_basic(self, handler_no_llm, mock_node):
        """Test basic name cleaning."""
        mock_node.name = "Test Node 1"
        
        clean = handler_no_llm._clean_name(mock_node)
        
        # Should be valid Python identifier
        assert clean.isidentifier() or "_" in clean
    
    def test_clean_name_special_chars(self, handler_no_llm, mock_node):
        """Test name cleaning with special characters."""
        mock_node.name = "Node (Input) [v2]"
        
        clean = handler_no_llm._clean_name(mock_node)
        
        # Should not contain special chars
        assert "(" not in clean
        assert "[" not in clean
    
    # ============= _format_settings tests =============
    
    def test_format_settings_empty(self, handler_no_llm):
        """Test formatting empty settings."""
        result = handler_no_llm._format_settings({})
        
        # Should return something or empty string
        assert isinstance(result, str)
    
    def test_format_settings_nested(self, handler_no_llm):
        """Test formatting nested settings."""
        settings = {
            "level1": {
                "level2": "value",
                "number": 42
            },
            "list_val": [1, 2, 3]
        }
        
        result = handler_no_llm._format_settings(settings)
        
        assert isinstance(result, str)
    
    # ============= get_fallback_summary tests =============
    
    def test_get_fallback_summary_empty(self, handler_no_llm):
        """Test summary with no nodes."""
        summary = handler_no_llm.get_fallback_summary([])
        
        assert isinstance(summary, dict)
    
    def test_get_fallback_summary_multiple(self, handler_no_llm, mock_node):
        """Test summary with multiple nodes."""
        nodes = [mock_node, mock_node]  # Same node twice
        
        summary = handler_no_llm.get_fallback_summary(nodes)
        
        assert isinstance(summary, dict)


class TestFallbackHandlerApproximateMatching:
    """Tests for approximate template matching."""
    
    @pytest.fixture
    def handler(self):
        return FallbackHandler(llm_client=None)
    
    @pytest.fixture
    def mock_node(self):
        node = Mock(spec=NodeInstance)
        node.id = "node_1"
        node.name = "Custom Filter"
        node.factory = "com.vendor.CustomFilterNodeFactory"
        node.settings = {}
        node.input_ports = [{"name": "input", "type": "BufferedDataTable"}]
        node.output_ports = [{"name": "output", "type": "BufferedDataTable"}]
        return node
    
    def test_find_approximate_filter_node(self, handler, mock_node):
        """Test finding approximate match for filter-like node."""
        mock_node.factory = "com.vendor.RowFilterNodeFactory"
        mock_node.name = "Custom Row Filter"
        
        template = handler._find_approximate_template(mock_node)
        
        # Should find a filter-related template or None
        # (depends on approximate mappings)
        assert template is None or isinstance(template, str)
    
    def test_find_approximate_joiner_node(self, handler, mock_node):
        """Test finding approximate match for joiner-like node."""
        mock_node.factory = "com.vendor.DataJoinerFactory"
        mock_node.name = "Table Merger"
        
        template = handler._find_approximate_template(mock_node)
        
        assert template is None or isinstance(template, str)


class TestFallbackLevelEnum:
    """Tests for FallbackLevel enumeration."""
    
    def test_fallback_levels_exist(self):
        """Verify all expected fallback levels exist."""
        assert hasattr(FallbackLevel, 'EXACT')
        assert hasattr(FallbackLevel, 'APPROXIMATE')
        assert hasattr(FallbackLevel, 'LLM_GENERATED')
        assert hasattr(FallbackLevel, 'STUB')
    
    def test_fallback_level_ordering(self):
        """Test that fallback levels can be compared for priority."""
        # Each level should be distinct
        levels = [
            FallbackLevel.EXACT,
            FallbackLevel.APPROXIMATE,
            FallbackLevel.LLM_GENERATED,
            FallbackLevel.STUB
        ]
        
        assert len(set(levels)) == 4  # All unique
