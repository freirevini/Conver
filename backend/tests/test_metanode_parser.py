"""
Characterization tests for MetanodeParser.

These tests capture the CURRENT behavior to detect regressions.
"""
import pytest
from pathlib import Path

from app.services.parser.knwf_extractor import KnwfExtractor
from app.services.parser.metanode_parser import MetanodeParser
from tests.conftest import compare_with_golden_master


class TestMetanodeParserCharacterization:
    """Golden Master tests for MetanodeParser."""
    
    @pytest.fixture
    def extractor(self, sample_workflow_dir: Path) -> KnwfExtractor:
        """Create extractor for sample workflow."""
        # Find the knwf file or use extracted directory
        knwf_path = sample_workflow_dir.parent.parent / "fluxo_knime_exemplo.knwf"
        if knwf_path.exists():
            extractor = KnwfExtractor()
            extractor.extract(str(knwf_path))
            return extractor
        
        # Create mock extractor for extracted directory
        return MockExtractor(sample_workflow_dir)
    
    def test_metanode_identification(self, sample_workflow_dir: Path, golden_master_dir: Path):
        """Verify metanode detection in workflow."""
        # Check for metanode directories (contain workflow.knime)
        metanodes = []
        for item in sample_workflow_dir.iterdir():
            if item.is_dir():
                workflow_file = item / "workflow.knime"
                settings_file = item / "settings.xml"
                if workflow_file.exists():
                    metanodes.append({
                        "name": item.name,
                        "has_workflow_knime": True,
                        "has_settings_xml": settings_file.exists(),
                    })
        
        summary = {
            "metanode_count": len(metanodes),
            "metanode_names": [m["name"] for m in metanodes],
        }
        
        assert compare_with_golden_master(summary, "metanode_identification", golden_master_dir)
    
    def test_metanode_structure(self, sample_metanode_dir: Path, golden_master_dir: Path):
        """Verify metanode internal structure parsing."""
        # Check what's inside the metanode
        workflow_file = sample_metanode_dir / "workflow.knime"
        settings_file = sample_metanode_dir / "settings.xml"
        
        child_nodes = []
        for item in sample_metanode_dir.iterdir():
            if item.is_dir() and "(#" in item.name:
                child_nodes.append(item.name)
        
        summary = {
            "metanode_name": sample_metanode_dir.name,
            "has_workflow_knime": workflow_file.exists(),
            "has_settings_xml": settings_file.exists(),
            "child_node_count": len(child_nodes),
            "child_node_samples": child_nodes[:5],
        }
        
        assert compare_with_golden_master(summary, "metanode_structure", golden_master_dir)
    
    def test_nested_metanode_detection(self, sample_metanode_dir: Path, golden_master_dir: Path):
        """Verify nested metanode detection."""
        nested_metanodes = []
        
        for item in sample_metanode_dir.iterdir():
            if item.is_dir():
                nested_workflow = item / "workflow.knime"
                if nested_workflow.exists():
                    nested_metanodes.append(item.name)
        
        summary = {
            "parent_metanode": sample_metanode_dir.name,
            "nested_metanode_count": len(nested_metanodes),
            "nested_names": nested_metanodes,
        }
        
        assert compare_with_golden_master(summary, "nested_metanodes", golden_master_dir)


class MockExtractor:
    """Mock extractor for testing with extracted directories."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._files = {}
        self._load_files()
    
    def _load_files(self):
        """Load all files from directory."""
        for path in self.base_path.rglob("*"):
            if path.is_file():
                rel_path = str(path.relative_to(self.base_path.parent)).replace("\\", "/")
                self._files[rel_path] = path
    
    def get_file_text(self, path: str) -> str:
        """Get file content as text."""
        normalized = path.replace("\\", "/")
        for key, file_path in self._files.items():
            if key.endswith(normalized) or normalized.endswith(key):
                return file_path.read_text(encoding="utf-8")
        return ""
    
    def get_all_files(self):
        """Get all files."""
        return self._files
