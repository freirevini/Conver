"""
Characterization tests for WorkflowParser.

These tests capture the CURRENT behavior to detect regressions.
Run with --update-golden to regenerate snapshots.
"""
import pytest
from pathlib import Path

from app.services.parser.workflow_parser import WorkflowParser
from tests.conftest import compare_with_golden_master


class TestWorkflowParserCharacterization:
    """Golden Master tests for WorkflowParser."""
    
    def test_parse_workflow_structure(self, sample_workflow_dir: Path, golden_master_dir: Path):
        """Capture current workflow parsing output structure."""
        parser = WorkflowParser()
        result = parser.parse_workflow(str(sample_workflow_dir))
        
        # Verify basic structure exists
        assert "name" in result
        assert "nodes" in result
        assert "connections" in result
        assert "metadata" in result
        
        # Create summary for golden master (not full data to avoid path issues)
        summary = {
            "workflow_name": result["name"],
            "node_count": len(result["nodes"]),
            "connection_count": len(result["connections"]),
            "has_metadata": bool(result["metadata"]),
            "node_types": list(set(n.get("type", "unknown") for n in result["nodes"])),
        }
        
        assert compare_with_golden_master(summary, "workflow_structure", golden_master_dir)
    
    def test_node_extraction_fields(self, sample_workflow_dir: Path, golden_master_dir: Path):
        """Verify all expected fields are extracted from nodes."""
        parser = WorkflowParser()
        result = parser.parse_workflow(str(sample_workflow_dir))
        
        if not result["nodes"]:
            pytest.skip("No nodes in workflow")
        
        # Check first node has expected fields
        first_node = result["nodes"][0]
        expected_fields = ["id", "name", "type", "factory_class", "position", "is_metanode", "settings_file"]
        
        fields_present = {f: f in first_node for f in expected_fields}
        
        assert compare_with_golden_master(
            {"fields_present": fields_present, "sample_node_keys": list(first_node.keys())},
            "node_fields",
            golden_master_dir
        )
    
    def test_connection_extraction(self, sample_workflow_dir: Path, golden_master_dir: Path):
        """Verify connection data structure."""
        parser = WorkflowParser()
        result = parser.parse_workflow(str(sample_workflow_dir))
        
        if not result["connections"]:
            pytest.skip("No connections in workflow")
        
        first_conn = result["connections"][0]
        expected_fields = ["source_id", "dest_id", "source_port", "dest_port"]
        
        fields_present = {f: f in first_conn for f in expected_fields}
        
        # Check for ID=-1 connections (boundary)
        boundary_sources = [c for c in result["connections"] if c.get("source_id") == -1]
        boundary_dests = [c for c in result["connections"] if c.get("dest_id") == -1]
        
        summary = {
            "fields_present": fields_present,
            "total_connections": len(result["connections"]),
            "boundary_source_count": len(boundary_sources),
            "boundary_dest_count": len(boundary_dests),
        }
        
        assert compare_with_golden_master(summary, "connection_structure", golden_master_dir)


class TestWorkflowParserEdgeCases:
    """Edge case tests for WorkflowParser."""
    
    def test_missing_workflow_file(self, tmp_path: Path):
        """Verify error handling for missing workflow.knime."""
        parser = WorkflowParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_workflow(str(tmp_path))
    
    def test_empty_workflow_dir(self, tmp_path: Path):
        """Parse directory with only workflow.knime but no nodes."""
        # Create minimal workflow.knime
        workflow_file = tmp_path / "workflow.knime"
        workflow_file.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<config xmlns="http://www.knime.org/2008/09/XMLConfig">
    <entry key="name" type="xstring" value="EmptyWorkflow"/>
    <config key="nodes"/>
    <config key="connections"/>
</config>''')
        
        parser = WorkflowParser()
        result = parser.parse_workflow(str(tmp_path))
        
        # Parser fallbacks to directory name when XML name extraction fails
        # This is expected behavior - we're testing it doesn't crash
        assert result["nodes"] == []
        assert result["connections"] == []
