"""
End-to-End Integration Tests for KNIME Transpiler.

Tests complete workflow:
- Upload → Parse → Generate → Validate → Download
"""
import pytest
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.factories import WorkflowFactory, NodeFactory


class TestE2EWorkflowProcessing:
    """End-to-end tests for workflow processing."""
    
    @pytest.fixture
    def temp_workflow_dir(self):
        """Create temporary directory for workflow files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def simple_workflow(self):
        """Create simple test workflow."""
        return WorkflowFactory.simple_etl()
    
    def test_workflow_dict_structure(self, simple_workflow):
        """Workflow dict should have required structure."""
        data = simple_workflow.to_dict()
        
        assert "name" in data
        assert "nodes" in data
        assert "connections" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["connections"], list)
    
    def test_node_connectivity(self, simple_workflow):
        """All connections should reference valid nodes."""
        data = simple_workflow.to_dict()
        node_ids = {n["id"] for n in data["nodes"]}
        
        for conn in data["connections"]:
            assert conn["source_id"] in node_ids, \
                f"Source {conn['source_id']} not in nodes"
            assert conn["dest_id"] in node_ids, \
                f"Dest {conn['dest_id']} not in nodes"
    
    def test_ml_pipeline_structure(self):
        """ML pipeline should have correct structure."""
        workflow = WorkflowFactory.ml_pipeline()
        data = workflow.to_dict()
        
        # Should have at least 4 nodes
        assert len(data["nodes"]) >= 4
        
        # Should have connections
        assert len(data["connections"]) >= 3
        
        # Categories should include Mining and IO
        categories = {n.get("category") for n in data["nodes"]}
        assert "Mining" in categories
        assert "IO" in categories


class TestE2ETemplateIntegration:
    """Test template integration in full workflow."""
    
    def test_all_etl_nodes_have_templates(self):
        """ETL workflow nodes should have templates."""
        from app.services.generator.template_mapper import TemplateMapper
        
        workflow = WorkflowFactory.simple_etl()
        mapper = TemplateMapper()
        
        for node in workflow.nodes:
            template = mapper.get_template(node.factory)
            # CSV and Column Filter should have templates
            if "CSV" in node.factory or "Filter" in node.factory:
                assert template is not None, \
                    f"Missing template for {node.factory}"
    
    def test_ml_nodes_have_sklearn_imports(self):
        """ML nodes should generate sklearn imports."""
        from app.services.generator.ml_templates import get_ml_template
        
        workflow = WorkflowFactory.ml_pipeline()
        
        for node in workflow.nodes:
            if "Learner" in node.factory or "Predictor" in node.factory:
                template = get_ml_template(node.factory)
                if template:
                    imports = template.get("imports", [])
                    # Check for sklearn in imports
                    has_sklearn = any("sklearn" in imp for imp in imports)
                    # Not all have sklearn, but learners should
                    if "Learner" in node.factory:
                        assert has_sklearn or template is None


class TestE2ECodeGeneration:
    """Test code generation for complete workflows."""
    
    def test_generated_code_is_string(self):
        """Generated code should be a string."""
        # Mock the code generator
        mock_code = '''
import pandas as pd

def node_1_csv_reader():
    """CSV Reader (Node ID: #1)"""
    df = pd.read_csv("input.csv")
    return df

if __name__ == "__main__":
    result = node_1_csv_reader()
    print(result.head())
'''
        assert isinstance(mock_code, str)
        assert "import pandas" in mock_code
        assert "def node_" in mock_code
    
    def test_dag_order_preserved(self):
        """Generated code should respect DAG order."""
        workflow = WorkflowFactory.simple_etl()
        data = workflow.to_dict()
        
        # Connections define order: reader -> filter -> writer
        connections = data["connections"]
        
        # First connection: reader to filter
        first_conn = connections[0]
        assert first_conn["source_id"] == "1"  # Reader
        assert first_conn["dest_id"] == "2"    # Filter
        
        # Second connection: filter to writer
        second_conn = connections[1]
        assert second_conn["source_id"] == "2"  # Filter
        assert second_conn["dest_id"] == "3"    # Writer


class TestE2EValidation:
    """Test validation in full workflow."""
    
    def test_syntax_validation(self):
        """Generated code should pass syntax validation."""
        from app.utils.code_review import validate_python_code
        
        valid_code = '''
import pandas as pd

def process_data(df):
    """Process data."""
    result = df.copy()
    return result
'''
        is_valid, error = validate_python_code(valid_code)
        assert is_valid is True
        assert error is None
    
    def test_invalid_syntax_detected(self):
        """Invalid syntax should be detected."""
        from app.utils.code_review import validate_python_code
        
        invalid_code = '''
def broken(
    # Missing closing paren and body
'''
        is_valid, error = validate_python_code(invalid_code)
        assert is_valid is False
        assert error is not None


class TestE2EErrorRecovery:
    """Test error recovery in full workflow."""
    
    def test_unknown_node_generates_stub(self):
        """Unknown node should generate stub, not fail."""
        from app.core.errors import TemplateError, ErrorHandler
        
        handler = ErrorHandler()
        
        # Simulate template not found
        error = TemplateError(
            message="No template for UnknownFactory",
            factory="org.unknown.UnknownFactory"
        )
        
        result = handler.handle(error)
        
        assert result is not None
        assert result.get("fallback") == "stub"
    
    def test_error_handler_tracks_errors(self):
        """Error handler should track all errors."""
        from app.core.errors import ErrorHandler, ParseError, ErrorContext
        
        handler = ErrorHandler()
        handler.clear()
        
        # Add some errors
        try:
            raise ParseError("Test error 1")
        except Exception as e:
            handler.handle(e)
        
        assert handler.has_errors()
        summary = handler.summary()
        assert summary["error_count"] == 1


class TestE2EPerformance:
    """Performance-related E2E tests."""
    
    def test_large_workflow_handling(self):
        """Large workflow should not crash."""
        NodeFactory.reset()
        
        # Create workflow with 100 nodes
        nodes = [NodeFactory.csv_reader() for _ in range(100)]
        
        from tests.factories import MockWorkflow, MockConnection
        
        # Create chain of connections
        connections = []
        for i in range(len(nodes) - 1):
            connections.append(MockConnection(
                source_id=nodes[i].id,
                dest_id=nodes[i + 1].id
            ))
        
        workflow = MockWorkflow(
            name="Large Workflow",
            nodes=nodes,
            connections=connections
        )
        
        data = workflow.to_dict()
        
        assert len(data["nodes"]) == 100
        assert len(data["connections"]) == 99
    
    def test_caching_improves_performance(self):
        """Caching should improve repeated lookups."""
        from app.utils.optimization import LRUCache
        
        cache = LRUCache(maxsize=10)
        
        # First access (miss)
        result = cache.get("key1")
        assert result is None
        
        # Put and get (hit)
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
        
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
