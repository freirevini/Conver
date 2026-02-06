"""
Schema Extractor for KNIME node output specifications.

Parses port_X/spec.xml files from executed KNIME workflows to extract
column metadata (names, types, counts) for validation purposes.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from zipfile import ZipFile, BadZipFile

logger = logging.getLogger(__name__)


@dataclass
class ColumnSpec:
    """Specification for a single column."""
    name: str
    knime_type: str
    pandas_dtype: str
    index: int


@dataclass
class NodeOutputSchema:
    """Complete output schema for a node port."""
    node_id: str
    node_name: str
    port_index: int
    column_count: int
    columns: List[ColumnSpec] = field(default_factory=list)
    
    @property
    def column_names(self) -> List[str]:
        """List of column names in order."""
        return [col.name for col in self.columns]
    
    @property
    def column_types(self) -> Dict[str, str]:
        """Mapping of column name to pandas dtype."""
        return {col.name: col.pandas_dtype for col in self.columns}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "port_index": self.port_index,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "column_types": self.column_types,
        }


class SchemaExtractor:
    """
    Extracts output schema from KNIME workflow node folders.
    
    Parses spec.xml files in port_X directories to retrieve
    column metadata saved during KNIME workflow execution.
    """
    
    # KNIME cell class to pandas dtype mapping
    TYPE_MAPPING: Dict[str, str] = {
        # Numeric types
        "org.knime.core.data.def.DoubleCell": "float64",
        "org.knime.core.data.def.IntCell": "int64",
        "org.knime.core.data.def.LongCell": "int64",
        
        # String types
        "org.knime.core.data.def.StringCell": "object",
        
        # Boolean
        "org.knime.core.data.def.BooleanCell": "bool",
        
        # Date/Time types
        "org.knime.core.data.time.localdate.LocalDateCell": "datetime64[ns]",
        "org.knime.core.data.time.localdatetime.LocalDateTimeCell": "datetime64[ns]",
        "org.knime.core.data.time.localtime.LocalTimeCell": "object",
        "org.knime.core.data.date.DateAndTimeCell": "datetime64[ns]",
        
        # Binary/Blob
        "org.knime.core.data.def.BinaryObjectDataCell": "object",
        
        # Missing value
        "org.knime.core.data.MissingCell": "object",
    }
    
    def __init__(self, workflow_path: Optional[Path] = None):
        """
        Initialize schema extractor.
        
        Args:
            workflow_path: Path to extracted workflow directory
        """
        self.workflow_path = workflow_path
    
    def extract_from_knwf(
        self,
        knwf_path: Path,
        extract_dir: Optional[Path] = None
    ) -> Dict[str, List[NodeOutputSchema]]:
        """
        Extract schemas from a .knwf file.
        
        Args:
            knwf_path: Path to .knwf file
            extract_dir: Directory to extract to (temp if not provided)
            
        Returns:
            Dict mapping node_id to list of output schemas
        """
        import tempfile
        
        if extract_dir is None:
            extract_dir = Path(tempfile.mkdtemp())
        
        try:
            with ZipFile(knwf_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Find workflow directory
            workflow_dir = self._find_workflow_dir(extract_dir)
            if workflow_dir:
                return self.extract_all_schemas(workflow_dir)
            
            logger.warning(f"Could not find workflow directory in {knwf_path}")
            return {}
            
        except BadZipFile:
            logger.error(f"Invalid .knwf file: {knwf_path}")
            return {}
    
    def _find_workflow_dir(self, extract_dir: Path) -> Optional[Path]:
        """Find the workflow directory containing workflow.knime."""
        for path in extract_dir.rglob("workflow.knime"):
            return path.parent
        return None
    
    def extract_all_schemas(
        self,
        workflow_dir: Path
    ) -> Dict[str, List[NodeOutputSchema]]:
        """
        Extract schemas from all nodes in a workflow directory.
        
        Args:
            workflow_dir: Path to workflow directory (contains workflow.knime)
            
        Returns:
            Dict mapping node_id to list of output schemas
        """
        schemas: Dict[str, List[NodeOutputSchema]] = {}
        
        # Check if workflow was saved with data
        saved_with_data = (workflow_dir / ".savedWithData").exists()
        if not saved_with_data:
            logger.warning("Workflow not saved with data - schemas may be unavailable")
        
        # Find all node directories
        for node_dir in workflow_dir.iterdir():
            if not node_dir.is_dir():
                continue
            
            # Skip internal directories
            if node_dir.name.startswith("."):
                continue
            
            # Extract node ID and name from folder name
            node_id, node_name = self._parse_node_folder(node_dir.name)
            if node_id:
                node_schemas = self.extract_node_schemas(node_dir, node_id, node_name)
                if node_schemas:
                    schemas[node_id] = node_schemas
        
        return schemas
    
    def _parse_node_folder(self, folder_name: str) -> tuple[Optional[str], str]:
        """
        Parse node folder name to extract ID and name.
        
        Examples:
            "Database Reader (legacy) (#1435)" -> ("1435", "Database Reader (legacy)")
            "Column Filter (#1604)" -> ("1604", "Column Filter")
        """
        import re
        
        match = re.search(r"\(#(\d+)\)$", folder_name)
        if match:
            node_id = match.group(1)
            node_name = folder_name[:match.start()].strip()
            return node_id, node_name
        
        return None, folder_name
    
    def extract_node_schemas(
        self,
        node_dir: Path,
        node_id: str,
        node_name: str
    ) -> List[NodeOutputSchema]:
        """
        Extract schemas from all output ports of a node.
        
        Args:
            node_dir: Path to node directory
            node_id: Node ID
            node_name: Node display name
            
        Returns:
            List of NodeOutputSchema for each output port
        """
        schemas = []
        
        # Find all port_X directories
        port_dirs = sorted(
            [d for d in node_dir.iterdir() if d.is_dir() and d.name.startswith("port_")],
            key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0
        )
        
        for port_dir in port_dirs:
            # Try multiple locations for spec.xml
            spec_file = self._find_spec_xml(port_dir)
            
            if spec_file and spec_file.exists():
                try:
                    port_index = int(port_dir.name.split("_")[1])
                    schema = self._parse_spec_xml(spec_file, node_id, node_name, port_index)
                    if schema:
                        schemas.append(schema)
                except Exception as e:
                    logger.warning(f"Failed to parse {spec_file}: {e}")
        
        return schemas
    
    def _find_spec_xml(self, port_dir: Path) -> Optional[Path]:
        """
        Find spec.xml in various possible locations.
        
        KNIME stores spec.xml in different places depending on node type:
        - Direct: port_X/spec.xml
        - Subdirectory: port_X/spec/spec.xml (need to extract from spec.zip)
        """
        # Direct location (most common)
        direct = port_dir / "spec.xml"
        if direct.exists():
            return direct
        
        # Some nodes have spec subdirectory with spec.xml
        subdir = port_dir / "spec" / "spec.xml"
        if subdir.exists():
            return subdir
        
        # TODO: Handle spec.zip extraction for nodes like DB Loader
        # This is a future enhancement
        
        return None
    
    def _parse_spec_xml(
        self,
        spec_file: Path,
        node_id: str,
        node_name: str,
        port_index: int
    ) -> Optional[NodeOutputSchema]:
        """
        Parse a spec.xml file to extract column metadata.
        
        Args:
            spec_file: Path to spec.xml
            node_id: Node ID
            node_name: Node display name
            port_index: Output port index
            
        Returns:
            NodeOutputSchema or None if parsing fails
        """
        try:
            tree = ET.parse(spec_file)
            root = tree.getroot()
            
            # KNIME uses namespace - iterate to find elements regardless of namespace
            column_count = 0
            for elem in root.iter():
                if elem.attrib.get("key") == "number_columns":
                    column_count = int(elem.attrib.get("value", "0"))
                    break
            
            if column_count == 0:
                return None
            
            # Parse each column spec
            columns = []
            for i in range(column_count):
                col_spec = self._parse_column_spec(root, i)
                if col_spec:
                    columns.append(col_spec)
            
            return NodeOutputSchema(
                node_id=node_id,
                node_name=node_name,
                port_index=port_index,
                column_count=column_count,
                columns=columns,
            )
            
        except ET.ParseError as e:
            logger.error(f"XML parse error in {spec_file}: {e}")
            return None
    
    def _parse_column_spec(
        self,
        root: ET.Element,
        index: int
    ) -> Optional[ColumnSpec]:
        """Parse a single column specification."""
        config_key = f"column_spec_{index}"
        
        # Find the config element for this column using iteration (namespace-agnostic)
        config = None
        for elem in root.iter():
            if elem.attrib.get("key") == config_key:
                config = elem
                break
        
        if config is None:
            return None
        
        # Get column name
        column_name = None
        for elem in config.iter():
            if elem.attrib.get("key") == "column_name":
                column_name = elem.attrib.get("value", "")
                break
        
        if not column_name:
            return None
        
        # Get column type from nested config
        knime_type = "unknown"
        type_config = None
        for elem in config.iter():
            if elem.attrib.get("key") == "column_type":
                type_config = elem
                break
        
        if type_config is not None:
            for elem in type_config.iter():
                if elem.attrib.get("key") == "cell_class":
                    knime_type = elem.attrib.get("value", "unknown")
                    break
        
        # Map to pandas dtype
        pandas_dtype = self.TYPE_MAPPING.get(knime_type, "object")
        
        return ColumnSpec(
            name=column_name,
            knime_type=knime_type,
            pandas_dtype=pandas_dtype,
            index=index,
        )
    
    
    def map_knime_type(self, knime_type: str) -> str:
        """Map KNIME cell type to pandas dtype."""
        return self.TYPE_MAPPING.get(knime_type, "object")
