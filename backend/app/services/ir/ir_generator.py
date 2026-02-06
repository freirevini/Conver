"""
IR Generator - Converts parsed KNIME workflow to Intermediate Representation.

Handles:
- Combining parser outputs into unified IR
- Execution order calculation
- Parallel group identification
- Dependency resolution
- JSON/YAML serialization
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from app.models.ir_models import (
    Connection,
    ExecutionLayer,
    FlowVariable,
    NodeInstance,
    ParallelGroup,
    PythonEnvironment,
    WorkflowIR,
    WorkflowMetadata,
)
from app.services.catalog.node_registry import get_node_registry, FallbackLevel
from app.services.graph.graph_builder import ExecutionGraphBuilder, GraphAnalysis
from app.services.parser.knwf_extractor import KnwfExtractor
from app.services.parser.metanode_parser import MetanodeParser

logger = logging.getLogger(__name__)


class IRGenerator:
    """
    Generates Intermediate Representation from KNIME workflow files.
    
    The IR captures the complete semantics of the workflow in a format
    suitable for code generation.
    """
    
    def __init__(self):
        self.registry = get_node_registry()
        self.graph_builder = ExecutionGraphBuilder()
    
    def generate(
        self,
        knwf_path: Path | str,
        workflow_name: Optional[str] = None,
    ) -> WorkflowIR:
        """
        Generate IR from a KNWF file.
        
        Args:
            knwf_path: Path to the .knwf file
            workflow_name: Optional name for the workflow
            
        Returns:
            Complete WorkflowIR
        """
        knwf_path = Path(knwf_path)
        
        # Extract and parse
        with KnwfExtractor(knwf_path) as extractor:
            # Parse metanodes recursively
            metanode_parser = MetanodeParser(extractor)
            
            # Get workflow root and parse from there
            root_path = extractor.workflow_root
            
            # Parse all nodes
            nodes = self._parse_all_nodes(extractor, metanode_parser)
            
            # Parse all connections
            connections = self._parse_all_connections(extractor)
            
            # Get flow variables
            flow_variables = self._parse_flow_variables(extractor)
            
            # Analyze graph
            analysis = self.graph_builder.analyze(nodes, connections)
            
            # Determine dependencies
            dependencies = self._calculate_dependencies(nodes)
            
            # Create metadata
            metadata = WorkflowMetadata(
                name=workflow_name or knwf_path.stem,
                knime_version=extractor.knime_version,
                source_file=str(knwf_path),
                modified_date=datetime.now(),
            )
            
            # Create environment
            environment = PythonEnvironment(
                dependencies=dependencies,
            )
            
            # Build IR
            ir = WorkflowIR(
                version="2.0",
                metadata=metadata,
                environment=environment,
                nodes=nodes,
                connections=connections,
                flow_variables=flow_variables,
                execution_order=analysis.execution_order,
                execution_layers=analysis.execution_layers,
                parallel_groups=analysis.parallel_groups,
                loops=analysis.loops,
                switches=analysis.switches,
            )
            
            # Identify unsupported nodes
            ir.unsupported_nodes = self._find_unsupported_nodes(nodes)
            
            # Add warnings
            ir.warnings = analysis.errors.copy()
            if ir.unsupported_nodes:
                ir.warnings.append(
                    f"{len(ir.unsupported_nodes)} nodes require manual implementation"
                )
            
            logger.info(
                f"Generated IR: {len(nodes)} nodes, {len(connections)} connections, "
                f"{len(ir.unsupported_nodes)} unsupported"
            )
            
            return ir
    
    def _parse_all_nodes(
        self,
        extractor: KnwfExtractor,
        metanode_parser: MetanodeParser,
    ) -> List[NodeInstance]:
        """Parse all nodes from the workflow."""
        nodes = []
        
        # Get all node folders
        for folder_path, node_name, node_id in extractor.get_node_folders():
            # Check if this is a metanode
            is_metanode = any(
                m[0] == folder_path
                for m in extractor.get_metanode_folders()
            )
            
            if is_metanode:
                # Parse metanode recursively
                metanode_def = metanode_parser.parse_metanode(folder_path)
                
                # Convert to NodeInstance
                node = NodeInstance(
                    node_id=f"node_{node_id}",
                    node_type="Metanode",
                    factory_class="org.knime.core.node.workflow.SubNodeContainer",
                    name=node_name,
                    is_metanode=True,
                    children=metanode_def.children,
                    internal_connections=metanode_def.connections,
                )
            else:
                # Parse regular node
                node = self._parse_node(extractor, folder_path, node_name, node_id)
            
            # Determine fallback level
            registry_node = self.registry.get_node(node.factory_class)
            if registry_node:
                node.fallback_level = registry_node.mapping.fallback_level
                node.category = registry_node.category
            else:
                # Try to match by name
                registry_node = self.registry.get_node(node_name)
                if registry_node:
                    node.fallback_level = registry_node.mapping.fallback_level
                    node.category = registry_node.category
                else:
                    node.fallback_level = FallbackLevel.STUB
            
            nodes.append(node)
        
        return nodes
    
    def _parse_node(
        self,
        extractor: KnwfExtractor,
        folder_path: str,
        node_name: str,
        node_id: int,
    ) -> NodeInstance:
        """Parse a single regular node."""
        from xml.etree import ElementTree as ET
        
        settings_path = folder_path + "settings.xml"
        settings_content = extractor.get_file_text(settings_path)
        
        settings = {}
        factory_class = "unknown"
        
        if settings_content:
            try:
                root = ET.fromstring(settings_content)
                settings = self._parse_settings_xml(root)
                
                # Extract factory class
                factory_class = self._extract_factory(root)
                
            except ET.ParseError as e:
                logger.warning(f"Failed to parse {settings_path}: {e}")
        
        return NodeInstance(
            node_id=f"node_{node_id}",
            node_type=node_name,
            factory_class=factory_class,
            name=node_name,
            settings=settings,
        )
    
    def _parse_settings_xml(self, root) -> Dict[str, Any]:
        """Parse settings.xml into a flat dictionary."""
        settings = {}
        
        # Recursively extract all config entries
        def extract_entries(element, prefix=""):
            for child in element:
                key = child.get("key", child.tag)
                full_key = f"{prefix}.{key}" if prefix else key
                
                if child.tag == "entry":
                    value_type = child.get("type", "xstring")
                    value = child.get("value", "")
                    
                    if value_type == "xint":
                        settings[full_key] = int(value) if value else 0
                    elif value_type == "xdouble":
                        settings[full_key] = float(value) if value else 0.0
                    elif value_type == "xboolean":
                        settings[full_key] = value.lower() == "true"
                    else:
                        settings[full_key] = value
                else:
                    extract_entries(child, full_key)
        
        extract_entries(root)
        return settings
    
    def _extract_factory(self, root) -> str:
        """Extract factory class from settings.xml."""
        for entry in root.iter("entry"):
            if entry.get("key") == "factory":
                return entry.get("value", "unknown")
        return "unknown"
    
    def _parse_all_connections(
        self,
        extractor: KnwfExtractor,
    ) -> List[Connection]:
        """Parse all connections from workflow.knime files."""
        from xml.etree import ElementTree as ET
        
        connections = []
        
        # Find all workflow.knime files
        for path in extractor.get_all_files().keys():
            if not path.endswith("workflow.knime"):
                continue
            
            content = extractor.get_file_text(path)
            if not content:
                continue
            
            try:
                root = ET.fromstring(content)
                
                # Find connection entries
                for config in root.iter("config"):
                    if config.get("key") == "connections":
                        # Parse individual connections
                        for conn_config in config:
                            conn = self._parse_connection(conn_config)
                            if conn:
                                connections.append(conn)
                                
            except ET.ParseError as e:
                logger.warning(f"Failed to parse {path}: {e}")
        
        return connections
    
    def _parse_connection(self, config) -> Optional[Connection]:
        """Parse a single connection config."""
        source_id = None
        source_port = 0
        dest_id = None
        dest_port = 0
        
        for entry in config.iter("entry"):
            key = entry.get("key", "")
            value = entry.get("value", "")
            
            if key == "sourceID":
                source_id = f"node_{value}"
            elif key == "sourcePort":
                source_port = int(value) if value else 0
            elif key == "destID":
                dest_id = f"node_{value}"
            elif key == "destPort":
                dest_port = int(value) if value else 0
        
        if source_id and dest_id:
            return Connection(
                source_node_id=source_id,
                source_port=source_port,
                dest_node_id=dest_id,
                dest_port=dest_port,
            )
        
        return None
    
    def _parse_flow_variables(
        self,
        extractor: KnwfExtractor,
    ) -> List[FlowVariable]:
        """Parse global flow variables."""
        # Flow variables are typically in the root workflow settings
        # This is a simplified implementation
        return []
    
    def _calculate_dependencies(
        self,
        nodes: List[NodeInstance],
    ) -> List[str]:
        """Calculate Python package dependencies based on nodes."""
        deps: Set[str] = {"pandas>=2.0"}  # Always needed
        
        for node in nodes:
            registry_node = self.registry.get_node(node.factory_class)
            if registry_node and registry_node.mapping.pip_packages:
                deps.update(registry_node.mapping.pip_packages)
        
        return sorted(deps)
    
    def _find_unsupported_nodes(
        self,
        nodes: List[NodeInstance],
    ) -> List[str]:
        """Find nodes that don't have template support."""
        unsupported = []
        
        for node in nodes:
            if node.fallback_level == FallbackLevel.STUB:
                unsupported.append(f"{node.name} ({node.factory_class})")
        
        return unsupported
    
    def to_json(self, ir: WorkflowIR, path: Optional[Path] = None) -> str:
        """Serialize IR to JSON."""
        json_str = ir.to_json()
        
        if path:
            path.write_text(json_str, encoding="utf-8")
        
        return json_str
    
    def to_yaml(self, ir: WorkflowIR, path: Optional[Path] = None) -> str:
        """Serialize IR to YAML."""
        yaml_str = ir.to_yaml()
        
        if path:
            path.write_text(yaml_str, encoding="utf-8")
        
        return yaml_str
    
    def from_json(self, json_str: str) -> WorkflowIR:
        """Deserialize IR from JSON."""
        data = json.loads(json_str)
        return self._dict_to_ir(data)
    
    def from_yaml(self, yaml_str: str) -> WorkflowIR:
        """Deserialize IR from YAML."""
        data = yaml.safe_load(yaml_str)
        return self._dict_to_ir(data)
    
    def _dict_to_ir(self, data: Dict[str, Any]) -> WorkflowIR:
        """Convert dictionary back to WorkflowIR."""
        # This is a simplified implementation
        # Full implementation would reconstruct all nested objects
        ir = WorkflowIR(version=data.get("version", "2.0"))
        
        if "metadata" in data:
            ir.metadata = WorkflowMetadata(
                name=data["metadata"].get("name", "Unknown"),
                knime_version=data["metadata"].get("knime_version"),
                source_file=data["metadata"].get("source_file"),
            )
        
        ir.execution_order = data.get("execution_order", [])
        ir.unsupported_nodes = data.get("unsupported_nodes", [])
        ir.warnings = data.get("warnings", [])
        
        return ir
