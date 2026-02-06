"""
KNIME Workflow Parser

Parses workflow.knime XML files to extract:
- Workflow name and metadata
- Node list (ID, name, type, factory class, position)
- Connections between nodes
"""
import os
import logging
from typing import Dict, List, Optional, Any
from lxml import etree

logger = logging.getLogger(__name__)


class WorkflowParser:
    """
    Parser for KNIME workflow.knime XML files.
    
    KNIME workflow.knime structure:
    <config xmlns="http://www.knime.org/2008/09/XMLConfig">
        <entry key="name" type="xstring" value="WorkflowName"/>
        <config key="nodes">
            <config key="node_1">
                <entry key="id" type="xint" value="1"/>
                ...
            </config>
        </config>
        <config key="connections">
            <config key="connection_0">
                <entry key="sourceID" type="xint" value="1"/>
                ...
            </config>
        </config>
    </config>
    """
    
    def parse_workflow(self, workflow_dir: str) -> Dict[str, Any]:
        """
        Parse a KNIME workflow directory.
        
        Args:
            workflow_dir: Path to directory containing workflow.knime
            
        Returns:
            Dictionary with workflow data:
            {
                'name': str,
                'nodes': List[Dict],
                'connections': List[Dict],
                'metadata': Dict
            }
        """
        workflow_xml_path = os.path.join(workflow_dir, 'workflow.knime')
        
        if not os.path.exists(workflow_xml_path):
            # Try to find workflow.knime in subdirectories
            workflow_xml_path = self._find_workflow_xml(workflow_dir)
            if workflow_xml_path is None:
                raise FileNotFoundError(
                    f"workflow.knime not found in {workflow_dir}"
                )
        
        logger.info(f"Parsing workflow: {workflow_xml_path}")
        
        return self.parse_workflow_knime(workflow_xml_path)
    
    def parse_workflow_knime(self, xml_path: str) -> Dict[str, Any]:
        """
        Parse workflow.knime XML file.
        
        Args:
            xml_path: Path to workflow.knime file
            
        Returns:
            Dictionary with workflow data
        """
        try:
            tree = etree.parse(xml_path)
            root = tree.getroot()
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Invalid XML in {xml_path}: {e}")
        
        workflow_dir = os.path.dirname(xml_path)
        
        workflow_data = {
            'name': self._get_workflow_name(root, workflow_dir),
            'nodes': self._extract_nodes(root, workflow_dir),
            'connections': self._extract_connections(root),
            'metadata': self._extract_metadata(root, xml_path)
        }
        
        logger.info(
            f"Parsed workflow '{workflow_data['name']}': "
            f"{len(workflow_data['nodes'])} nodes, "
            f"{len(workflow_data['connections'])} connections"
        )
        
        return workflow_data
    
    def _find_workflow_xml(self, search_dir: str) -> Optional[str]:
        """Find workflow.knime file in directory tree."""
        for root, dirs, files in os.walk(search_dir):
            if 'workflow.knime' in files:
                return os.path.join(root, 'workflow.knime')
        return None
    
    def _get_workflow_name(self, root: etree._Element, workflow_dir: str) -> str:
        """Extract workflow name from XML or directory."""
        # Try to get from XML
        name_entry = root.find(".//entry[@key='name']")
        if name_entry is not None:
            name = name_entry.get('value', '')
            if name:
                return name
        
        # Fallback to directory name
        return os.path.basename(workflow_dir)
    
    def _extract_nodes(self, root: etree._Element, workflow_dir: str) -> List[Dict]:
        """
        Extract all nodes from the workflow.
        
        Each node has:
        - id: Node ID (int)
        - name: Display name
        - type: Node type (NativeNode, MetaNode, etc.)
        - factory_class: Full factory class name
        - position: (x, y) coordinates
        - settings_path: Path to settings.xml
        """
        nodes = []
        nodes_config = root.find(".//config[@key='nodes']")
        
        if nodes_config is None:
            logger.warning("No nodes config found in workflow.knime")
            return nodes
        
        for node_config in nodes_config.findall('config'):
            try:
                node = self._parse_node_config(node_config, workflow_dir)
                if node:
                    nodes.append(node)
            except Exception as e:
                logger.warning(f"Error parsing node: {e}")
                continue
        
        return nodes
    
    def _parse_node_config(self, node_config: etree._Element, workflow_dir: str) -> Optional[Dict]:
        """Parse a single node configuration."""
        # Get node ID
        id_entry = node_config.find("entry[@key='id']")
        if id_entry is None:
            return None
        
        node_id = int(id_entry.get('value', 0))
        
        # Get node name from settings file path
        name_entry = node_config.find("entry[@key='node_settings_file']")
        node_name = name_entry.get('value', f'Node_{node_id}') if name_entry is not None else f'Node_{node_id}'
        
        # Get node type
        type_entry = node_config.find("entry[@key='node_type']")
        node_type = type_entry.get('value', 'NativeNode') if type_entry is not None else 'NativeNode'
        
        # [NEW] Get node_is_meta flag for metanode detection
        is_meta_entry = node_config.find("entry[@key='node_is_meta']")
        is_metanode = (
            is_meta_entry is not None and 
            is_meta_entry.get('value', 'false').lower() == 'true'
        )
        
        # Get factory class (if available at this level)
        factory_entry = node_config.find("entry[@key='factory']")
        factory_class = factory_entry.get('value', '') if factory_entry is not None else ''
        
        # Get position
        position = self._extract_position(node_config)
        
        # [NEW] Get settings file path (for metanodes, this points to workflow.knime)
        settings_file = node_name  # This is the relative path from workflow dir
        
        # Determine settings path
        # Node directories are named like "Node Name (#123)" or just "settings.xml" in node dir
        settings_path = self._find_node_settings(workflow_dir, node_name, node_id)
        
        return {
            'id': node_id,
            'name': node_name,
            'type': node_type,
            'factory_class': factory_class,
            'position': position,
            'settings_path': settings_path,
            # [NEW] Added fields for metanode support
            'is_metanode': is_metanode,
            'settings_file': settings_file,
        }
    
    def _extract_position(self, node_config: etree._Element) -> Dict[str, int]:
        """Extract node position from UI settings."""
        position = {'x': 0, 'y': 0}
        
        # Look for UI bounds
        ui_settings = node_config.find(".//config[@key='ui_settings']")
        if ui_settings is None:
            return position
        
        bounds = ui_settings.find(".//config[@key='extrainfo.node.bounds']")
        if bounds is None:
            return position
        
        # Bounds are stored as entries with keys 0, 1, 2, 3 (x, y, width, height)
        x_entry = bounds.find("entry[@key='0']")
        y_entry = bounds.find("entry[@key='1']")
        
        if x_entry is not None:
            position['x'] = int(x_entry.get('value', 0))
        if y_entry is not None:
            position['y'] = int(y_entry.get('value', 0))
        
        return position
    
    def _find_node_settings(self, workflow_dir: str, node_name: str, node_id: int) -> Optional[str]:
        """
        Find the settings.xml file for a node.
        
        KNIME stores node settings in directories named like:
        - "Column Filter (#123)/settings.xml"
        - "CSV Reader (#1)/settings.xml"
        """
        # Try exact match first
        node_dir = os.path.join(workflow_dir, node_name)
        if os.path.isdir(node_dir):
            settings_path = os.path.join(node_dir, 'settings.xml')
            if os.path.exists(settings_path):
                return settings_path
        
        # Try to find by node ID pattern
        for item in os.listdir(workflow_dir):
            item_path = os.path.join(workflow_dir, item)
            if os.path.isdir(item_path):
                # Check if directory name contains the node ID
                if f"(#{node_id})" in item or f"#{node_id})" in item:
                    settings_path = os.path.join(item_path, 'settings.xml')
                    if os.path.exists(settings_path):
                        return settings_path
        
        return None
    
    def _extract_connections(self, root: etree._Element) -> List[Dict]:
        """
        Extract connections between nodes.
        
        Each connection has:
        - source_id: Source node ID
        - dest_id: Destination node ID
        - source_port: Source output port
        - dest_port: Destination input port
        """
        connections = []
        conn_config = root.find(".//config[@key='connections']")
        
        if conn_config is None:
            logger.warning("No connections config found in workflow.knime")
            return connections
        
        for conn in conn_config.findall('config'):
            try:
                connection = self._parse_connection(conn)
                if connection:
                    connections.append(connection)
            except Exception as e:
                logger.warning(f"Error parsing connection: {e}")
                continue
        
        return connections
    
    def _parse_connection(self, conn: etree._Element) -> Optional[Dict]:
        """Parse a single connection."""
        source_id = conn.find("entry[@key='sourceID']")
        dest_id = conn.find("entry[@key='destID']")
        
        if source_id is None or dest_id is None:
            return None
        
        source_port = conn.find("entry[@key='sourcePort']")
        dest_port = conn.find("entry[@key='destPort']")
        
        return {
            'source_id': int(source_id.get('value', 0)),
            'dest_id': int(dest_id.get('value', 0)),
            'source_port': int(source_port.get('value', 0)) if source_port is not None else 0,
            'dest_port': int(dest_port.get('value', 0)) if dest_port is not None else 0
        }
    
    def _extract_metadata(self, root: etree._Element, xml_path: str) -> Dict:
        """Extract workflow metadata."""
        metadata = {
            'xml_path': xml_path,
            'has_annotations': False,
            'knime_version': None
        }
        
        # Check for annotations
        annotations = root.find(".//config[@key='annotations']")
        metadata['has_annotations'] = annotations is not None
        
        # Try to get KNIME version
        version_entry = root.find("entry[@key='version']")
        if version_entry is not None:
            metadata['knime_version'] = version_entry.get('value')
        
        return metadata
