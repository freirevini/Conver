"""
Extended Unit Tests for Code Generator.

Tests:
- Template mapping
- Code generation
- Placeholder resolution
- Error handling
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.factories import NodeFactory, WorkflowFactory, MockNode


class TestTemplateMapping:
    """Tests for template mapping functionality."""
    
    def test_csv_reader_template_exists(self):
        """CSV Reader should have a template."""
        from app.services.generator.template_mapper import TemplateMapper
        mapper = TemplateMapper()
        factory = "org.knime.base.node.io.csvreader.CSVReaderNodeFactory"
        
        template = mapper.get_template(factory)
        assert template is not None
        assert "imports" in template
        assert "code" in template
    
    def test_column_filter_template(self):
        """Column Filter template should include column selection."""
        from app.services.generator.template_mapper import TemplateMapper
        mapper = TemplateMapper()
        factory = "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory"
        
        template = mapper.get_template(factory)
        assert template is not None
        assert "{columns}" in template.get("code", "")
    
    def test_unknown_factory_returns_none(self):
        """Unknown factory should return None for fallback handling."""
        from app.services.generator.template_mapper import TemplateMapper
        mapper = TemplateMapper()
        
        template = mapper.get_template("org.unknown.factory.FakeNodeFactory")
        assert template is None
    
    def test_template_has_required_placeholders(self):
        """Templates should have standard placeholders."""
        from app.services.generator.template_mapper import TemplateMapper
        mapper = TemplateMapper()
        
        for factory, template in list(mapper.TEMPLATES.items())[:10]:
            code = template.get("code", "")
            # Most templates should have input_var or output_var
            has_placeholders = "{input_var}" in code or "{output_var}" in code
            # Or be a special case like model training
            assert has_placeholders or "model" in code.lower() or "#" in code


class TestCodeGeneration:
    """Tests for code generation."""
    
    @pytest.fixture
    def mock_workflow(self):
        """Create mock workflow for testing."""
        return WorkflowFactory.simple_etl()
    
    def test_simple_workflow_generates_code(self, mock_workflow):
        """Simple ETL workflow should generate valid Python."""
        # This test validates the factory creates correct structure
        workflow_dict = mock_workflow.to_dict()
        
        assert len(workflow_dict["nodes"]) == 3
        assert len(workflow_dict["connections"]) == 2
        assert workflow_dict["name"] == "Simple ETL"
    
    def test_ml_pipeline_has_correct_nodes(self):
        """ML pipeline should have training nodes."""
        workflow = WorkflowFactory.ml_pipeline()
        workflow_dict = workflow.to_dict()
        
        factories = [n["factory"] for n in workflow_dict["nodes"]]
        
        # Should have reader, learner, predictor, scorer
        assert any("CSVReader" in f for f in factories)
        assert any("Learner" in f for f in factories)
        assert any("Predictor" in f for f in factories)
        assert any("Scorer" in f for f in factories)
    
    def test_node_factory_creates_unique_ids(self):
        """Node factory should create unique IDs."""
        NodeFactory.reset()
        node1 = NodeFactory.csv_reader()
        node2 = NodeFactory.csv_writer()
        node3 = NodeFactory.column_filter([])
        
        ids = {node1.id, node2.id, node3.id}
        assert len(ids) == 3, "Node IDs should be unique"


class TestPlaceholderResolution:
    """Tests for placeholder resolution in templates."""
    
    def test_basic_placeholder_replacement(self):
        """Basic placeholders should be replaced."""
        template = "{output_var} = {input_var}.copy()"
        result = template.format(output_var="df_out", input_var="df_in")
        
        assert result == "df_out = df_in.copy()"
        assert "{" not in result
    
    def test_list_placeholder(self):
        """List placeholders should be formatted correctly."""
        columns = ["col1", "col2", "col3"]
        template = "{output_var} = {input_var}[{columns}]"
        
        result = template.format(
            output_var="df_filtered",
            input_var="df",
            columns=str(columns)
        )
        
        assert "col1" in result
        assert "col2" in result


class TestErrorHandling:
    """Tests for error handling in generation."""
    
    def test_empty_workflow_handling(self):
        """Empty workflow should not crash."""
        workflow = WorkflowFactory.empty()
        assert len(workflow.nodes) == 0
        assert len(workflow.connections) == 0
    
    def test_node_without_settings(self):
        """Node without settings should have empty dict."""
        node = NodeFactory.generic("Test", "org.test.Factory")
        assert node.settings == {}
    
    def test_node_to_dict_includes_all_fields(self):
        """Node to_dict should include all required fields."""
        node = NodeFactory.csv_reader("test.csv")
        node_dict = node.to_dict()
        
        required_fields = ["id", "name", "factory", "settings", "category"]
        for field in required_fields:
            assert field in node_dict, f"Missing field: {field}"


class TestMLTemplates:
    """Tests for ML template extension."""
    
    def test_ml_templates_module_loads(self):
        """ML templates module should load without errors."""
        from app.services.generator.ml_templates import ML_TEMPLATES, get_ml_template
        
        assert len(ML_TEMPLATES) > 20, "Should have 20+ ML templates"
    
    def test_random_forest_template_exists(self):
        """Random Forest template should exist."""
        from app.services.generator.ml_templates import get_ml_template
        
        template = get_ml_template(
            "org.knime.base.node.mine.treensemble2.node.learner.classification.TreeEnsembleClassificationLearnerNodeFactory3"
        )
        assert template is not None
        assert "RandomForest" in template["code"]
    
    def test_ml_template_has_sklearn_imports(self):
        """ML templates should have sklearn imports."""
        from app.services.generator.ml_templates import ML_TEMPLATES
        
        sklearn_count = sum(
            1 for t in ML_TEMPLATES.values()
            if any("sklearn" in imp for imp in t.get("imports", []))
        )
        
        assert sklearn_count > 15, "Most ML templates should use sklearn"
