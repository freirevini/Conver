"""
KNIME Node Parser

Parses individual node settings.xml files to extract:
- Factory class (node type identifier)
- Configuration parameters
- Column selections, filters, etc.
"""
import os
import logging
from typing import Dict, List, Optional, Any
from lxml import etree

logger = logging.getLogger(__name__)


class NodeParser:
    """
    Parser for KNIME node settings.xml files.
    
    Each node directory contains a settings.xml with structure:
    <config xmlns="http://www.knime.org/2008/09/XMLConfig">
        <config key="nodeAnnotation">...</config>
        <entry key="customDescription" type="xstring" value=""/>
        <entry key="factory" type="xstring" value="org.knime...Factory"/>
        <entry key="bundle-name" type="xstring" value="KNIME Base Nodes"/>
        <config key="model">
            <!-- Node-specific settings -->
        </config>
    </config>
    """
    
    def parse_node_settings(self, settings_path: Optional[str]) -> Dict[str, Any]:
        """
        Parse settings.xml for a single node.
        
        Args:
            settings_path: Path to settings.xml file
            
        Returns:
            Dictionary with node settings:
            {
                'factory': str,
                'bundle_name': str,
                'configuration': Dict,
                'model_info': Dict
            }
        """
        if settings_path is None or not os.path.exists(settings_path):
            logger.debug(f"No settings file found: {settings_path}")
            return self._empty_settings()
        
        try:
            tree = etree.parse(settings_path)
            root = tree.getroot()
        except etree.XMLSyntaxError as e:
            logger.warning(f"Invalid XML in {settings_path}: {e}")
            return self._empty_settings()
        
        settings = {
            'factory': self._get_factory_class(root),
            'bundle_name': self._get_bundle_name(root),
            'configuration': self._extract_configuration(root),
            'model_info': self._extract_model_info(root)
        }
        
        logger.debug(f"Parsed node settings: {settings['factory']}")
        
        return settings
    
    def _empty_settings(self) -> Dict[str, Any]:
        """Return empty settings structure."""
        return {
            'factory': '',
            'bundle_name': '',
            'configuration': {},
            'model_info': {}
        }
    
    def _get_factory_class(self, root: etree._Element) -> str:
        """
        Extract the factory class from settings.
        
        The factory class identifies the node type, e.g.:
        - org.knime.base.node.io.csvreader.CSVReaderNodeFactory
        - org.knime.base.node.preproc.filter.row.RowFilterNodeFactory
        """
        factory_entry = root.find("entry[@key='factory']")
        if factory_entry is not None:
            return factory_entry.get('value', '')
        
        # Try alternative locations
        factory_settings = root.find(".//entry[@key='factory']")
        if factory_settings is not None:
            return factory_settings.get('value', '')
        
        return ''
    
    def _get_bundle_name(self, root: etree._Element) -> str:
        """Extract bundle name (KNIME extension identifier)."""
        bundle_entry = root.find("entry[@key='bundle-name']")
        if bundle_entry is not None:
            return bundle_entry.get('value', '')
        return ''
    
    def _extract_configuration(self, root: etree._Element) -> Dict[str, Any]:
        """
        Extract configuration from 'model' section.
        
        The model section contains node-specific parameters like:
        - File paths for readers/writers
        - Column selections
        - Filter conditions
        - Algorithm parameters
        """
        config = {}
        model_section = root.find(".//config[@key='model']")
        
        if model_section is None:
            return config
        
        # Parse all entries in model section
        config = self._parse_config_section(model_section)
        
        return config
    
    def _parse_config_section(self, section: etree._Element, max_depth: int = 5) -> Dict[str, Any]:
        """
        Recursively parse a config section.
        
        Handles nested configs and various entry types.
        """
        if max_depth <= 0:
            return {}
        
        result = {}
        
        # Parse entries
        for entry in section.findall('entry'):
            key = entry.get('key', '')
            value = self._parse_entry_value(entry)
            if key:
                result[key] = value
        
        # Parse nested configs
        for config in section.findall('config'):
            key = config.get('key', '')
            if key:
                result[key] = self._parse_config_section(config, max_depth - 1)
        
        return result
    
    def _parse_entry_value(self, entry: etree._Element) -> Any:
        """
        Parse entry value based on type attribute.
        
        KNIME entry types:
        - xstring: String
        - xint: Integer
        - xlong: Long integer
        - xdouble: Float
        - xboolean: Boolean
        - xchar: Character
        """
        value_type = entry.get('type', 'xstring')
        value_str = entry.get('value', '')
        
        try:
            if value_type in ('xint', 'xlong'):
                return int(value_str) if value_str else 0
            elif value_type == 'xdouble':
                return float(value_str) if value_str else 0.0
            elif value_type == 'xboolean':
                return value_str.lower() == 'true'
            elif value_type == 'xchar':
                return value_str[0] if value_str else ''
            else:
                return value_str
        except (ValueError, IndexError):
            return value_str
    
    def _extract_model_info(self, root: etree._Element) -> Dict[str, Any]:
        """Extract additional model information if available."""
        info = {}
        
        # Check for internal settings
        internal = root.find(".//config[@key='internal']")
        if internal is not None:
            info['internal'] = self._parse_config_section(internal, max_depth=2)
        
        # Check for flow variables
        flow_vars = root.find(".//config[@key='flowVariables']")
        if flow_vars is not None:
            info['flow_variables'] = self._parse_config_section(flow_vars, max_depth=2)
        
        return info
    
    def get_node_type_simple(self, factory_class: str) -> str:
        """
        Extract simple node type from factory class.
        
        Example:
        'org.knime.base.node.io.csvreader.CSVReaderNodeFactory' -> 'CSV Reader'
        """
        if not factory_class:
            return 'Unknown'
        
        # Extract the last part before 'NodeFactory'
        parts = factory_class.split('.')
        if parts:
            last_part = parts[-1]
            # Remove 'NodeFactory' suffix
            if last_part.endswith('NodeFactory'):
                last_part = last_part[:-len('NodeFactory')]
            # Add spaces before capitals (CamelCase to spaces)
            import re
            return re.sub(r'(?<!^)(?=[A-Z])', ' ', last_part)
        
        return factory_class
    
    def extract_columns(self, configuration: Dict) -> List[str]:
        """
        Extract column names from node configuration.
        
        Many nodes have column selections stored in various formats.
        """
        columns = []
        
        # Common patterns for column storage
        column_keys = [
            'included_names', 'include_list', 'column_names',
            'selected_columns', 'filter_columns', 'columns'
        ]
        
        for key in column_keys:
            if key in configuration:
                col_data = configuration[key]
                if isinstance(col_data, dict):
                    # Nested structure with array entries
                    for k, v in col_data.items():
                        if isinstance(v, str) and v:
                            columns.append(v)
                elif isinstance(col_data, list):
                    columns.extend(col_data)
                elif isinstance(col_data, str):
                    columns.append(col_data)
        
        return columns
