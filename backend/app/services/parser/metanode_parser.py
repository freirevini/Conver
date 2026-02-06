"""
Metanode Parser - Recursive parsing of KNIME metanodes and components.

Handles:
- Unlimited depth metanode recursion
- Boundary node (input/output port) detection
- Flow variable resolution between levels
- Component support (KNIME 4.x+)
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

from app.models.ir_models import (
    Connection,
    ConnectionType,
    FlowVariable,
    NodeCategory,
    NodeInstance,
    NodePort,
    PortType,
)
from app.services.parser.knwf_extractor import KnwfExtractor

logger = logging.getLogger(__name__)

# Feature flag - controlled by environment variable for safe rollback
ENABLE_METANODE_WORKFLOW_PARSING = os.environ.get(
    "ENABLE_METANODE_WORKFLOW_PARSING", "true"
).lower() == "true"


@dataclass
class BoundaryNodes:
    """Input and output boundary nodes for a metanode."""
    input_nodes: List[NodePort] = field(default_factory=list)
    output_nodes: List[NodePort] = field(default_factory=list)


@dataclass
class MetanodeDefinition:
    """Complete definition of a metanode including internal structure."""
    node_id: str
    name: str
    path: str
    depth: int
    children: List[NodeInstance] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    boundary_nodes: BoundaryNodes = field(default_factory=BoundaryNodes)
    flow_variables: List[FlowVariable] = field(default_factory=list)
    is_component: bool = False


class MetanodeParser:
    """
    Parser for KNIME metanodes with recursive depth handling.
    
    Metanodes are "workflows within workflows" that encapsulate
    multiple nodes into a single reusable unit. They can be nested
    to arbitrary depth.
    """
    
    # XML namespaces used in KNIME files
    KNIME_NS = {"knime": "http://www.knime.org/2008/09/XMLConfig"}
    
    # Node folder pattern: "NodeName (#123)"
    NODE_FOLDER_PATTERN = re.compile(r'^(.+) \(#(\d+)\)$')
    
    def __init__(self, extractor: KnwfExtractor):
        """
        Initialize parser with a KNWF extractor.
        
        Args:
            extractor: Loaded KnwfExtractor instance
        """
        self.extractor = extractor
        self._parsed_metanodes: Dict[str, MetanodeDefinition] = {}
        self._max_depth = 0
    
    def parse_metanode(
        self, 
        path: str, 
        depth: int = 0,
        parent_id: Optional[str] = None
    ) -> MetanodeDefinition:
        """
        Parse a metanode at the given path.
        
        Args:
            path: Path to the metanode folder
            depth: Current nesting depth (0 = top level)
            parent_id: ID of parent metanode if any
            
        Returns:
            Complete MetanodeDefinition with children
        """
        if depth > 100:
            raise RecursionError(f"Metanode depth exceeded 100 at {path}")
        
        self._max_depth = max(self._max_depth, depth)
        
        # Check cache
        if path in self._parsed_metanodes:
            return self._parsed_metanodes[path]
        
        # Extract node info from path
        folder_name = path.rstrip('/').split('/')[-1]
        match = self.NODE_FOLDER_PATTERN.match(folder_name)
        
        if match:
            node_name = match.group(1)
            node_id = f"node_{match.group(2)}"
        else:
            node_name = folder_name
            node_id = f"metanode_{depth}_{hash(path) % 10000}"
        
        logger.debug(f"Parsing metanode: {node_name} (depth={depth})")
        
        # Parse settings.xml for this metanode
        settings_path = path + "settings.xml"
        node_settings = self._parse_settings_xml(settings_path)
        
        # [NEW] Parse workflow.knime for metanode-specific data (meta ports, connections)
        # Controlled by feature flag for safe rollback
        workflow_data = {}
        if ENABLE_METANODE_WORKFLOW_PARSING:
            workflow_data = self._parse_metanode_workflow(path)
        
        # Check if this is a Component (KNIME 4.x+)
        is_component = self._is_component(node_settings)
        
        # Parse child nodes
        children = self._parse_children(path, depth)
        
        # Parse internal connections (now with boundary support from workflow.knime)
        connections = self._parse_connections(path, node_settings)
        
        # [NEW] If workflow_data has connections, prefer those (they include ID=-1 boundaries)
        if workflow_data.get('connections'):
            connections = workflow_data['connections']
        
        # Get boundary nodes - now enhanced with meta_ports from workflow.knime
        boundary_nodes = self._get_boundary_nodes(path, node_settings, children)
        
        # [NEW] Enhance boundary nodes with meta_ports info if available
        if workflow_data.get('meta_in_ports'):
            boundary_nodes.input_nodes = workflow_data['meta_in_ports']
        if workflow_data.get('meta_out_ports'):
            boundary_nodes.output_nodes = workflow_data['meta_out_ports']
        
        # Get flow variables
        flow_variables = self._parse_flow_variables(node_settings)
        
        metanode = MetanodeDefinition(
            node_id=node_id,
            name=node_name,
            path=path,
            depth=depth,
            children=children,
            connections=connections,
            boundary_nodes=boundary_nodes,
            flow_variables=flow_variables,
            is_component=is_component,
        )
        
        self._parsed_metanodes[path] = metanode
        return metanode
    
    def _parse_settings_xml(self, path: str) -> Dict[str, Any]:
        """Parse a settings.xml file into a dictionary."""
        content = self.extractor.get_file_text(path)
        if not content:
            return {}
        
        try:
            root = ET.fromstring(content)
            return self._xml_to_dict(root)
        except ET.ParseError as e:
            logger.warning(f"Failed to parse settings.xml at {path}: {e}")
            return {}
    
    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert KNIME XML config to dictionary."""
        result: Dict[str, Any] = {}
        
        # Handle entry elements (leaf nodes with values)
        if element.tag == "entry":
            key = element.get("key", "")
            value_type = element.get("type", "xstring")
            value = element.get("value", "")
            
            # Convert value based on type
            if value_type == "xint":
                return {key: int(value) if value else 0}
            elif value_type == "xdouble":
                return {key: float(value) if value else 0.0}
            elif value_type == "xboolean":
                return {key: value.lower() == "true"}
            else:
                return {key: value}
        
        # Handle config elements (containers)
        key = element.get("key", element.tag)
        
        children_dict: Dict[str, Any] = {}
        for child in element:
            child_dict = self._xml_to_dict(child)
            children_dict.update(child_dict)
        
        if children_dict:
            result[key] = children_dict
        
        return result
    
    def _is_component(self, settings: Dict[str, Any]) -> bool:
        """Check if this metanode is a KNIME Component."""
        # Components have specific markers in their settings
        if "settings.xml" in settings:
            inner = settings["settings.xml"]
            if isinstance(inner, dict):
                return inner.get("isComponent", False)
        return False
    
    def _parse_children(
        self, 
        parent_path: str, 
        parent_depth: int
    ) -> List[NodeInstance]:
        """Parse all child nodes within a metanode."""
        children = []
        
        # Get all files in this metanode folder
        all_files = list(self.extractor.get_all_files().keys())
        
        # Find direct child node folders
        seen_folders: Set[str] = set()
        
        for file_path in all_files:
            if not file_path.startswith(parent_path):
                continue
            
            relative = file_path[len(parent_path):]
            if '/' not in relative:
                continue
            
            child_folder = relative.split('/')[0]
            match = self.NODE_FOLDER_PATTERN.match(child_folder)
            
            if match and child_folder not in seen_folders:
                seen_folders.add(child_folder)
                child_path = parent_path + child_folder + '/'
                
                # Check if this child is itself a metanode
                is_nested_metanode = self._is_metanode(child_path)
                
                if is_nested_metanode:
                    # Recursive parse
                    nested = self.parse_metanode(
                        child_path, 
                        depth=parent_depth + 1
                    )
                    
                    # Convert to NodeInstance
                    node = NodeInstance(
                        node_id=nested.node_id,
                        node_type="Metanode",
                        factory_class="org.knime.core.node.workflow.SubNodeContainer",
                        name=nested.name,
                        is_metanode=True,
                        is_component=nested.is_component,
                        children=[n for n in nested.children],
                        internal_connections=nested.connections,
                        category=NodeCategory.METANODE if not nested.is_component else NodeCategory.COMPONENT,
                    )
                else:
                    # Regular node
                    node = self._parse_node(child_path, match.group(1), match.group(2))
                
                children.append(node)
        
        return children
    
    def _is_metanode(self, path: str) -> bool:
        """Check if a folder contains a metanode (has child node folders)."""
        all_files = list(self.extractor.get_all_files().keys())
        
        for file_path in all_files:
            if not file_path.startswith(path):
                continue
            
            relative = file_path[len(path):]
            if '/' not in relative:
                continue
            
            child_folder = relative.split('/')[0]
            if self.NODE_FOLDER_PATTERN.match(child_folder):
                return True
        
        return False
    
    def _parse_node(
        self, 
        path: str, 
        node_name: str, 
        node_id_str: str
    ) -> NodeInstance:
        """Parse a regular (non-metanode) node."""
        settings_path = path + "settings.xml"
        settings = self._parse_settings_xml(settings_path)
        
        # Extract node type and factory from settings
        factory_class = "unknown"
        node_type = node_name
        
        if "settings.xml" in settings:
            inner = settings["settings.xml"]
            if isinstance(inner, dict):
                factory_class = inner.get("factory", factory_class)
                node_type = inner.get("node-name", node_type)
        
        # Determine category from factory class
        category = self._categorize_node(factory_class, node_name)
        
        # Parse ports
        input_ports = self._parse_ports(settings, "inPorts")
        output_ports = self._parse_ports(settings, "outPorts")
        
        # Parse flow variables
        flow_vars = self._parse_flow_variables(settings)
        
        return NodeInstance(
            node_id=f"node_{node_id_str}",
            node_type=node_type,
            factory_class=factory_class,
            name=node_name,
            settings=settings,
            input_ports=input_ports,
            output_ports=output_ports,
            flow_variables=flow_vars,
            category=category,
        )
    
    def _parse_ports(
        self, 
        settings: Dict[str, Any], 
        port_key: str
    ) -> List[NodePort]:
        """Parse port definitions from settings."""
        ports = []
        
        if "settings.xml" not in settings:
            return ports
        
        inner = settings["settings.xml"]
        if not isinstance(inner, dict):
            return ports
        
        port_config = inner.get(port_key, {})
        if not isinstance(port_config, dict):
            return ports
        
        # Ports are typically numbered: port_0, port_1, etc.
        for key, value in port_config.items():
            if key.startswith("port_") and isinstance(value, dict):
                try:
                    port_index = int(key.split("_")[1])
                    port_type_str = value.get("port-type", "data")
                    
                    # Map to PortType enum
                    port_type = PortType.DATA
                    if "model" in port_type_str.lower():
                        port_type = PortType.MODEL
                    elif "database" in port_type_str.lower():
                        port_type = PortType.DATABASE
                    elif "flow" in port_type_str.lower():
                        port_type = PortType.FLOW_VARIABLE
                    
                    ports.append(NodePort(
                        index=port_index,
                        port_type=port_type,
                        name=value.get("name"),
                        spec=value.get("spec"),
                    ))
                except (ValueError, IndexError):
                    continue
        
        return sorted(ports, key=lambda p: p.index)
    
    def _parse_connections(
        self, 
        path: str, 
        settings: Dict[str, Any]
    ) -> List[Connection]:
        """Parse internal connections within a metanode."""
        connections = []
        
        if "settings.xml" not in settings:
            return connections
        
        inner = settings["settings.xml"]
        if not isinstance(inner, dict):
            return connections
        
        conn_config = inner.get("connections", {})
        if not isinstance(conn_config, dict):
            return connections
        
        for key, value in conn_config.items():
            if not isinstance(value, dict):
                continue
            
            try:
                source_id = value.get("sourceID", "")
                source_port = value.get("sourcePort", 0)
                dest_id = value.get("destID", "")
                dest_port = value.get("destPort", 0)
                
                if source_id and dest_id:
                    connections.append(Connection(
                        source_node_id=f"node_{source_id}",
                        source_port=int(source_port) if source_port else 0,
                        dest_node_id=f"node_{dest_id}",
                        dest_port=int(dest_port) if dest_port else 0,
                        connection_type=ConnectionType.DATA,
                    ))
            except (ValueError, TypeError):
                continue
        
        return connections
    
    def _get_boundary_nodes(
        self, 
        path: str, 
        settings: Dict[str, Any],
        children: List[NodeInstance]
    ) -> BoundaryNodes:
        """Get input/output boundary ports for a metanode."""
        boundary = BoundaryNodes()
        
        # Boundary nodes are typically indicated in the metanode settings
        # or can be inferred from nodes with external connections
        
        if "settings.xml" in settings:
            inner = settings["settings.xml"]
            if isinstance(inner, dict):
                # Parse input ports
                in_ports = inner.get("inPorts", {})
                if isinstance(in_ports, dict):
                    for key, value in in_ports.items():
                        if isinstance(value, dict):
                            boundary.input_nodes.append(NodePort(
                                index=len(boundary.input_nodes),
                                port_type=PortType.DATA,
                                name=value.get("name"),
                            ))
                
                # Parse output ports
                out_ports = inner.get("outPorts", {})
                if isinstance(out_ports, dict):
                    for key, value in out_ports.items():
                        if isinstance(value, dict):
                            boundary.output_nodes.append(NodePort(
                                index=len(boundary.output_nodes),
                                port_type=PortType.DATA,
                                name=value.get("name"),
                            ))
        
        return boundary
    
    def _parse_flow_variables(
        self, 
        settings: Dict[str, Any]
    ) -> List[FlowVariable]:
        """Parse flow variables from settings."""
        flow_vars = []
        
        if "settings.xml" not in settings:
            return flow_vars
        
        inner = settings["settings.xml"]
        if not isinstance(inner, dict):
            return flow_vars
        
        var_config = inner.get("flowVariables", {})
        if not isinstance(var_config, dict):
            return flow_vars
        
        for key, value in var_config.items():
            if isinstance(value, dict):
                var_type = value.get("type", "string")
                var_value = value.get("value", "")
                
                flow_vars.append(FlowVariable(
                    name=key,
                    var_type=var_type if var_type in ["string", "int", "double", "boolean"] else "string",
                    value=var_value,
                ))
        
        return flow_vars
    
    def _categorize_node(self, factory_class: str, node_name: str) -> NodeCategory:
        """Categorize a node based on its factory class and name."""
        factory_lower = factory_class.lower()
        name_lower = node_name.lower()
        
        # I/O nodes
        if any(k in factory_lower for k in ["reader", "writer", "csv", "excel", "parquet", "json"]):
            return NodeCategory.IO
        
        # Database nodes
        if any(k in factory_lower for k in ["database", "db", "sql", "jdbc"]):
            return NodeCategory.DATABASE
        
        # Transformation nodes
        if any(k in factory_lower for k in ["column", "row", "filter", "rename", "resorter"]):
            return NodeCategory.TRANSFORM
        
        # Aggregation
        if any(k in factory_lower for k in ["groupby", "pivot", "aggregate", "statistics"]):
            return NodeCategory.AGGREGATION
        
        # Joining
        if any(k in factory_lower for k in ["joiner", "concatenate", "merge"]):
            return NodeCategory.JOINING
        
        # Machine Learning
        if any(k in factory_lower for k in ["learner", "predictor", "cluster", "regression", "classifier"]):
            return NodeCategory.MACHINE_LEARNING
        
        # Flow Control
        if any(k in factory_lower for k in ["loop", "switch", "if", "case", "end"]):
            return NodeCategory.FLOW_CONTROL
        
        # DateTime
        if any(k in factory_lower for k in ["date", "time", "datetime"]):
            return NodeCategory.DATETIME
        
        # Text
        if any(k in factory_lower for k in ["string", "text", "regex"]):
            return NodeCategory.TEXT
        
        return NodeCategory.UNKNOWN
    
    def get_max_depth(self) -> int:
        """Get the maximum metanode nesting depth encountered."""
        return self._max_depth
    
    def flatten_hierarchy(
        self, 
        metanode: MetanodeDefinition
    ) -> List[NodeInstance]:
        """
        Flatten a metanode hierarchy into a single list of nodes.
        
        Args:
            metanode: Root metanode to flatten
            
        Returns:
            Flat list of all nodes, including those in nested metanodes
        """
        flat_nodes = []
        
        def _flatten(nodes: List[NodeInstance], prefix: str = "") -> None:
            for node in nodes:
                # Create a unique ID including the hierarchy path
                if prefix:
                    node.node_id = f"{prefix}_{node.node_id}"
                
                flat_nodes.append(node)
                
                if node.children:
                    _flatten(node.children, node.node_id)
        
        _flatten(metanode.children)
        return flat_nodes
    
    def resolve_flow_variables(
        self, 
        metanode: MetanodeDefinition
    ) -> Dict[str, FlowVariable]:
        """
        Resolve all flow variables across the metanode hierarchy.
        
        Variables from parent metanodes are inherited by children,
        but can be overridden at any level.
        
        Args:
            metanode: Root metanode
            
        Returns:
            Dictionary of resolved flow variables by name
        """
        resolved: Dict[str, FlowVariable] = {}
        
        def _resolve(node: MetanodeDefinition, parent_vars: Dict[str, FlowVariable]) -> None:
            # Start with parent variables
            current_vars = parent_vars.copy()
            
            # Override with this node's variables
            for fv in node.flow_variables:
                current_vars[fv.name] = fv
            
            # Update global resolved dict
            resolved.update(current_vars)
            
            # Process children
            for child in node.children:
                if child.is_metanode and child.node_id in self._parsed_metanodes:
                    child_meta = self._parsed_metanodes[child.node_id]
                    _resolve(child_meta, current_vars)
        
        _resolve(metanode, {})
        return resolved
    
    # ============================================================
    # [NEW] Workflow.knime parsing for metanodes
    # ============================================================
    
    def _parse_metanode_workflow(self, path: str) -> Dict[str, Any]:
        """
        Parse workflow.knime inside a metanode directory.
        
        This extracts:
        - meta_in_ports: Input port definitions for the metanode
        - meta_out_ports: Output port definitions for the metanode
        - connections: Internal connections (including ID=-1 boundaries)
        
        Args:
            path: Path to metanode directory (ends with /)
            
        Returns:
            Dict with meta_in_ports, meta_out_ports, and connections
        """
        workflow_path = path + "workflow.knime"
        content = self.extractor.get_file_text(workflow_path)
        
        if not content:
            logger.debug(f"No workflow.knime found at {workflow_path}")
            return {}
        
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.warning(f"Failed to parse workflow.knime at {workflow_path}: {e}")
            return {}
        
        result = {
            'meta_in_ports': self._parse_meta_ports(root, 'meta_in_ports'),
            'meta_out_ports': self._parse_meta_ports(root, 'meta_out_ports'),
            'connections': self._parse_workflow_connections(root),
        }
        
        logger.debug(
            f"Parsed metanode workflow: "
            f"{len(result['meta_in_ports'])} inputs, "
            f"{len(result['meta_out_ports'])} outputs, "
            f"{len(result['connections'])} connections"
        )
        
        return result
    
    def _parse_meta_ports(self, root: ET.Element, port_key: str) -> List[NodePort]:
        """
        Parse meta_in_ports or meta_out_ports from workflow.knime.
        
        Structure in workflow.knime:
        ```xml
        <config key="meta_in_ports">
          <entry key="array-size" type="xint" value="2"/>
          <config key="port_0">
            <entry key="index" type="xint" value="0"/>
            <entry key="name" type="xstring" value="Input Table"/>
          </config>
        </config>
        ```
        """
        ports = []
        
        # Find the meta ports config (namespace-agnostic)
        for config in root.iter():
            if config.attrib.get('key') == port_key:
                # Found the ports config, now parse each port
                for port_config in config:
                    if port_config.attrib.get('key', '').startswith('port_'):
                        port = self._parse_single_meta_port(port_config)
                        if port:
                            ports.append(port)
                break
        
        return sorted(ports, key=lambda p: p.index)
    
    def _parse_single_meta_port(self, port_config: ET.Element) -> Optional[NodePort]:
        """Parse a single meta port entry."""
        port_index = None
        port_name = None
        port_type_str = 'data'
        
        for entry in port_config:
            key = entry.attrib.get('key', '')
            value = entry.attrib.get('value', '')
            
            if key == 'index':
                try:
                    port_index = int(value)
                except ValueError:
                    pass
            elif key == 'name':
                port_name = value
            elif key == 'type':
                port_type_str = value
        
        if port_index is None:
            return None
        
        # Map type string to PortType
        port_type = PortType.DATA
        if 'model' in port_type_str.lower():
            port_type = PortType.MODEL
        elif 'database' in port_type_str.lower():
            port_type = PortType.DATABASE
        elif 'flow' in port_type_str.lower():
            port_type = PortType.FLOW_VARIABLE
        
        return NodePort(
            index=port_index,
            port_type=port_type,
            name=port_name,
        )
    
    def _parse_workflow_connections(self, root: ET.Element) -> List[Connection]:
        """
        Parse connections from workflow.knime.
        
        This handles ID=-1 specially - these are boundary connections.
        """
        connections = []
        
        for config in root.iter():
            if config.attrib.get('key') == 'connections':
                for conn_config in config:
                    if conn_config.attrib.get('key', '').startswith('connection_'):
                        conn = self._parse_single_connection(conn_config)
                        if conn:
                            connections.append(conn)
                break
        
        return connections
    
    def _parse_single_connection(self, conn_config: ET.Element) -> Optional[Connection]:
        """Parse a single connection entry."""
        source_id = None
        dest_id = None
        source_port = 0
        dest_port = 0
        
        for entry in conn_config:
            key = entry.attrib.get('key', '')
            value = entry.attrib.get('value', '')
            
            try:
                if key == 'sourceID':
                    source_id = int(value)
                elif key == 'destID':
                    dest_id = int(value)
                elif key == 'sourcePort':
                    source_port = int(value)
                elif key == 'destPort':
                    dest_port = int(value)
            except ValueError:
                continue
        
        if source_id is None or dest_id is None:
            return None
        
        # Determine connection type based on ID=-1 (boundary)
        conn_type = ConnectionType.DATA
        if source_id == -1 or dest_id == -1:
            conn_type = ConnectionType.DATA  # Still data, but marked as boundary
        
        return Connection(
            source_node_id=f"node_{source_id}" if source_id != -1 else "boundary_input",
            source_port=source_port,
            dest_node_id=f"node_{dest_id}" if dest_id != -1 else "boundary_output",
            dest_port=dest_port,
            connection_type=conn_type,
        )
