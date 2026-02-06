"""
Golden Tests for Template Mapper and Code Generation.

These tests validate that templates produce syntactically correct and 
semantically equivalent Python code for KNIME node transpilation.

Following TDD Workflow:
- RED: Define expected outputs first
- GREEN: Verify templates produce correct code
- REFACTOR: Improve templates based on failures
"""
import ast
import pytest
from typing import Dict, Any, List
from pathlib import Path

from app.services.generator.template_mapper import TemplateMapper


class TestTemplateGoldenMaster:
    """Golden master tests for template code generation."""
    
    @pytest.fixture
    def template_mapper(self) -> TemplateMapper:
        """Fresh TemplateMapper instance."""
        return TemplateMapper()
    
    # ============= Category: Data I/O =============
    
    @pytest.mark.parametrize("factory_class,expected_imports", [
        ("org.knime.base.node.io.csvreader.CSVReaderNodeFactory", ["import pandas as pd"]),
        ("org.knime.base.node.io.csvwriter.CSVWriterNodeFactory", ["import pandas as pd"]),
        ("org.knime.base.node.io.table.read.read.TableReaderNodeFactory2", ["import pandas as pd"]),
    ])
    def test_io_templates_have_pandas_import(
        self, 
        template_mapper: TemplateMapper,
        factory_class: str,
        expected_imports: List[str]
    ):
        """Test that I/O templates include pandas import."""
        template = template_mapper.get_template(factory_class)
        
        assert template is not None, f"No template for {factory_class}"
        assert "imports" in template, f"No imports in template for {factory_class}"
        
        for expected in expected_imports:
            assert expected in template["imports"], \
                f"Missing import '{expected}' in {factory_class}"
    
    # ============= Category: Syntax Validity =============
    
    @pytest.mark.parametrize("factory_class", [
        "org.knime.base.node.io.csvreader.CSVReaderNodeFactory",
        "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory",
        "org.knime.base.node.preproc.groupby.GroupByNodeFactory",
        "org.knime.base.node.preproc.joiner.JoinerNodeFactory",
        "org.knime.base.node.preproc.sorter.SorterNodeFactory",
        "org.knime.base.node.preproc.pmml.numbertostring.NumberToStringNodeFactory",
        "org.knime.base.node.preproc.pmml.stringtonumber.StringToNumberNodeFactory",
        "org.knime.base.node.preproc.rounddouble.RoundDoubleNodeFactory",
        "org.knime.time.node.convert.datetimetostring.DateTimeToStringNodeFactory",
        "org.knime.base.node.switches.ifswitch.IFSwitchNodeFactory",
        "org.knime.base.node.switches.caseswitch.CaseSwitchNodeFactory",
        "org.knime.base.node.meta.looper.loopstart.LoopStartNodeFactory",
        "org.knime.base.node.util.trycatch.TryCatchNodeFactory",
        "org.knime.database.node.reader.DBReaderNodeFactory",
        "org.knime.database.connectors.MySQLConnectorNodeFactory",
    ])
    def test_template_code_is_valid_python(
        self, 
        template_mapper: TemplateMapper,
        factory_class: str
    ):
        """Test that template code templates are syntactically valid Python."""
        template = template_mapper.get_template(factory_class)
        
        if template is None:
            pytest.skip(f"No template for {factory_class}")
        
        code = template.get("code", "")
        
        # Replace placeholders with valid Python identifiers
        code_for_validation = code.replace("{input_var}", "input_df")
        code_for_validation = code_for_validation.replace("{output_var}", "output_df")
        code_for_validation = code_for_validation.replace("{columns}", "['col1', 'col2']")
        code_for_validation = code_for_validation.replace("{file_path}", "'test.csv'")
        code_for_validation = code_for_validation.replace("{query}", "'SELECT * FROM table'")
        code_for_validation = code_for_validation.replace("{connection_string}", "'sqlite:///test.db'")
        code_for_validation = code_for_validation.replace("{table_name}", "'test_table'")
        code_for_validation = code_for_validation.replace("{condition}", "True")
        code_for_validation = code_for_validation.replace("{host}", "'localhost'")
        code_for_validation = code_for_validation.replace("{database}", "'testdb'")
        code_for_validation = code_for_validation.replace("{user}", "'user'")
        code_for_validation = code_for_validation.replace("{password}", "'pass'")
        code_for_validation = code_for_validation.replace("{port}", "3306")
        
        # More placeholders
        import re
        code_for_validation = re.sub(r'\{[a-z_]+\}', "'placeholder'", code_for_validation)
        code_for_validation = re.sub(r'\{[a-z_]+\[0\]\}', "'value1'", code_for_validation)
        code_for_validation = re.sub(r'\{[a-z_]+\[1\]\}', "'value2'", code_for_validation)
        
        try:
            ast.parse(code_for_validation)
        except SyntaxError as e:
            pytest.fail(
                f"Template for {factory_class} has invalid Python syntax:\n"
                f"Error: {e}\n"
                f"Code:\n{code_for_validation}"
            )
    
    # ============= Category: Template Coverage =============
    
    def test_minimum_template_count(self, template_mapper: TemplateMapper):
        """Verify minimum number of templates exists."""
        assert len(template_mapper.TEMPLATES) >= 70, \
            f"Expected at least 70 templates, got {len(template_mapper.TEMPLATES)}"
    
    def test_minimum_pattern_count(self, template_mapper: TemplateMapper):
        """Verify minimum number of factory patterns exists."""
        assert len(template_mapper.FACTORY_PATTERNS) >= 40, \
            f"Expected at least 40 patterns, got {len(template_mapper.FACTORY_PATTERNS)}"
    
    # ============= Category: Template Structure =============
    
    def test_all_templates_have_required_fields(self, template_mapper: TemplateMapper):
        """Verify all templates have required fields."""
        required_fields = ["imports", "code", "description"]
        
        for factory_class, template in template_mapper.TEMPLATES.items():
            for field in required_fields:
                assert field in template, \
                    f"Template {factory_class} missing required field: {field}"
    
    def test_all_templates_have_non_empty_code(self, template_mapper: TemplateMapper):
        """Verify all templates have non-empty code."""
        for factory_class, template in template_mapper.TEMPLATES.items():
            code = template.get("code", "")
            assert len(code.strip()) > 10, \
                f"Template {factory_class} has empty or minimal code"
    
    # ============= Category: Conversion Nodes =============
    
    @pytest.mark.parametrize("factory_class,expected_function", [
        ("org.knime.base.node.preproc.pmml.numbertostring.NumberToStringNodeFactory", "astype(str)"),
        ("org.knime.base.node.preproc.pmml.stringtonumber.StringToNumberNodeFactory", "to_numeric"),
        ("org.knime.base.node.preproc.rounddouble.RoundDoubleNodeFactory", "round"),
    ])
    def test_conversion_templates_use_correct_functions(
        self,
        template_mapper: TemplateMapper,
        factory_class: str,
        expected_function: str
    ):
        """Verify conversion templates use appropriate Python/Pandas functions."""
        template = template_mapper.get_template(factory_class)
        
        assert template is not None, f"No template for {factory_class}"
        
        code = template.get("code", "")
        assert expected_function in code, \
            f"Template {factory_class} should use '{expected_function}'"
    
    # ============= Category: Flow Control Nodes =============
    
    @pytest.mark.parametrize("factory_class,expected_pattern", [
        ("org.knime.base.node.switches.ifswitch.IFSwitchNodeFactory", "if"),
        ("org.knime.base.node.switches.caseswitch.CaseSwitchNodeFactory", "match"),
        ("org.knime.base.node.meta.looper.loopstart.LoopStartNodeFactory", "for"),
        ("org.knime.base.node.util.trycatch.TryCatchNodeFactory", "try:"),
    ])
    def test_flow_control_templates_use_correct_patterns(
        self,
        template_mapper: TemplateMapper,
        factory_class: str,
        expected_pattern: str
    ):
        """Verify flow control templates use correct Python constructs."""
        template = template_mapper.get_template(factory_class)
        
        assert template is not None, f"No template for {factory_class}"
        
        code = template.get("code", "")
        assert expected_pattern in code, \
            f"Template {factory_class} should contain '{expected_pattern}'"
    
    # ============= Category: Database Nodes =============
    
    @pytest.mark.parametrize("factory_class,expected_import", [
        ("org.knime.database.node.reader.DBReaderNodeFactory", "sqlalchemy"),
        ("org.knime.database.connectors.MySQLConnectorNodeFactory", "mysql.connector"),
        ("org.knime.database.connectors.PostgreSQLConnectorNodeFactory", "psycopg2"),
        ("org.knime.database.connectors.SQLiteConnectorNodeFactory", "sqlite3"),
    ])
    def test_database_templates_have_correct_imports(
        self,
        template_mapper: TemplateMapper,
        factory_class: str,
        expected_import: str
    ):
        """Verify database templates include correct driver imports."""
        template = template_mapper.get_template(factory_class)
        
        assert template is not None, f"No template for {factory_class}"
        
        imports_str = " ".join(template.get("imports", []))
        assert expected_import in imports_str, \
            f"Template {factory_class} should import '{expected_import}'"
    
    # ============= Category: Variable Nodes =============
    
    @pytest.mark.parametrize("factory_class", [
        "org.knime.base.node.preproc.table.rowtovar.TableRowToVariableNodeFactory2",
        "org.knime.base.node.flowvariable.variabletotablerow.VariableToTableRowNodeFactory2",
        "org.knime.base.node.flowvariable.loop.CountingLoopStartNodeFactory",
        "org.knime.base.node.flowvariable.createflowvariable.CreateFlowVariableNodeFactory",
    ])
    def test_variable_templates_exist(
        self,
        template_mapper: TemplateMapper,
        factory_class: str
    ):
        """Verify variable handling templates exist and are valid."""
        template = template_mapper.get_template(factory_class)
        
        assert template is not None, f"Missing template for variable node: {factory_class}"
        assert "code" in template, f"Template {factory_class} missing code"
        assert len(template["code"]) > 20, f"Template {factory_class} code too short"


class TestPatternMatching:
    """Tests for factory pattern matching."""
    
    @pytest.fixture
    def template_mapper(self) -> TemplateMapper:
        return TemplateMapper()
    
    @pytest.mark.parametrize("keyword,expected_factory", [
        ("csvreader", "org.knime.base.node.io.csvreader.CSVReaderNodeFactory"),
        ("groupby", "org.knime.base.node.preproc.groupby.GroupByNodeFactory"),
        ("joiner", "org.knime.base.node.preproc.joiner.JoinerNodeFactory"),
        ("ifswitch", "org.knime.base.node.switches.ifswitch.IFSwitchNodeFactory"),
        ("dbreader", "org.knime.database.node.reader.DBReaderNodeFactory"),
        ("mysql", "org.knime.database.connectors.MySQLConnectorNodeFactory"),
    ])
    def test_pattern_resolves_to_factory(
        self,
        template_mapper: TemplateMapper,
        keyword: str,
        expected_factory: str
    ):
        """Verify pattern keywords resolve to correct factory classes."""
        assert keyword in template_mapper.FACTORY_PATTERNS, \
            f"Pattern '{keyword}' not in FACTORY_PATTERNS"
        
        actual_factory = template_mapper.FACTORY_PATTERNS[keyword]
        assert actual_factory == expected_factory, \
            f"Pattern '{keyword}' maps to wrong factory: {actual_factory}"


class TestCodeGenerationIntegration:
    """Integration tests for code generation pipeline."""
    
    @pytest.fixture
    def template_mapper(self) -> TemplateMapper:
        return TemplateMapper()
    
    def test_generate_code_for_simple_workflow(self, template_mapper: TemplateMapper):
        """Test code generation for a simple 3-node workflow."""
        # Simulated workflow: CSV Reader -> Column Filter -> CSV Writer
        nodes = [
            {
                "factory": "org.knime.base.node.io.csvreader.CSVReaderNodeFactory",
                "input_var": None,
                "output_var": "df_1"
            },
            {
                "factory": "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory",
                "input_var": "df_1",
                "output_var": "df_2"
            },
            {
                "factory": "org.knime.base.node.io.csvwriter.CSVWriterNodeFactory",
                "input_var": "df_2",
                "output_var": "df_3"
            }
        ]
        
        generated_imports = set()
        generated_code = []
        
        for node in nodes:
            template = template_mapper.get_template(node["factory"])
            
            if template:
                generated_imports.update(template.get("imports", []))
                code = template.get("code", "")
                
                # Replace placeholders
                if node["input_var"]:
                    code = code.replace("{input_var}", node["input_var"])
                code = code.replace("{output_var}", node["output_var"])
                
                generated_code.append(code)
        
        # Verify all templates found
        assert len(generated_code) == 3, "Should generate code for all 3 nodes"
        
        # Verify imports collected
        assert "import pandas as pd" in generated_imports, \
            "Should include pandas import"
        
        # Verify code references correct variables
        full_code = "\n".join(generated_code)
        assert "df_1" in full_code, "Should reference df_1"
        assert "df_2" in full_code, "Should reference df_2"
    
    def test_generate_code_with_database_nodes(self, template_mapper: TemplateMapper):
        """Test code generation for database workflow."""
        # Simulated: MySQL Reader -> Transform -> DB Writer
        nodes = [
            ("org.knime.database.connectors.MySQLConnectorNodeFactory", "df_mysql"),
            ("org.knime.base.node.preproc.filter.row.RowFilterNodeFactory", "df_filtered"),
            ("org.knime.database.node.writer.DBWriterNodeFactory", "df_output"),
        ]
        
        all_imports = set()
        for factory, output in nodes:
            template = template_mapper.get_template(factory)
            if template:
                all_imports.update(template.get("imports", []))
        
        # Verify database-specific imports
        imports_str = " ".join(all_imports)
        assert "mysql.connector" in imports_str, "Should include MySQL connector"
        assert "sqlalchemy" in imports_str, "Should include SQLAlchemy for writer"
