"""
Boundary Resolver - Handle metanode boundary connections (ID=-1).

In KNIME workflow.knime files, connections with sourceID=-1 or destID=-1
represent connections to/from metanode boundary ports (inputs/outputs).

This module resolves these special connections to actual port references.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResolvedConnection:
    """A connection with resolved boundary information."""
    source_id: int
    dest_id: int
    source_port: int
    dest_port: int
    is_boundary_source: bool = False
    is_boundary_dest: bool = False
    boundary_source_port_name: Optional[str] = None
    boundary_dest_port_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_id': self.source_id,
            'dest_id': self.dest_id,
            'source_port': self.source_port,
            'dest_port': self.dest_port,
            'is_boundary_source': self.is_boundary_source,
            'is_boundary_dest': self.is_boundary_dest,
            'boundary_source_port_name': self.boundary_source_port_name,
            'boundary_dest_port_name': self.boundary_dest_port_name,
        }


class BoundaryResolver:
    """
    Resolves boundary connections in KNIME metanodes.
    
    Boundary connections use ID=-1 to indicate:
    - sourceID=-1: Connection FROM a metanode input port
    - destID=-1: Connection TO a metanode output port
    
    Example in workflow.knime:
    ```xml
    <config key="connection_0">
      <entry key="sourceID" type="xint" value="-1"/>  <!-- From metanode input -->
      <entry key="destID" type="xint" value="1594"/>
      <entry key="sourcePort" type="xint" value="0"/> <!-- meta_in_ports index -->
    </config>
    ```
    """
    
    BOUNDARY_ID = -1
    
    def __init__(
        self, 
        meta_in_ports: Optional[List[Dict]] = None,
        meta_out_ports: Optional[List[Dict]] = None
    ):
        """
        Initialize resolver with metanode port definitions.
        
        Args:
            meta_in_ports: List of input port definitions with 'name' and 'index'
            meta_out_ports: List of output port definitions with 'name' and 'index'
        """
        self.meta_in_ports = meta_in_ports or []
        self.meta_out_ports = meta_out_ports or []
    
    def is_boundary_connection(self, connection: Dict[str, Any]) -> bool:
        """Check if a connection involves boundary ports."""
        source_id = connection.get('source_id', 0)
        dest_id = connection.get('dest_id', 0)
        return source_id == self.BOUNDARY_ID or dest_id == self.BOUNDARY_ID
    
    def resolve_connection(self, connection: Dict[str, Any]) -> ResolvedConnection:
        """
        Resolve a connection, identifying boundary ports.
        
        Args:
            connection: Raw connection dict with source_id, dest_id, etc.
            
        Returns:
            ResolvedConnection with boundary information filled in
        """
        source_id = connection.get('source_id', 0)
        dest_id = connection.get('dest_id', 0)
        source_port = connection.get('source_port', 0)
        dest_port = connection.get('dest_port', 0)
        
        is_boundary_source = source_id == self.BOUNDARY_ID
        is_boundary_dest = dest_id == self.BOUNDARY_ID
        
        # Get boundary port names if available
        boundary_source_name = None
        boundary_dest_name = None
        
        if is_boundary_source and self.meta_in_ports:
            port_info = self._get_port_by_index(self.meta_in_ports, source_port)
            if port_info:
                boundary_source_name = port_info.get('name')
        
        if is_boundary_dest and self.meta_out_ports:
            port_info = self._get_port_by_index(self.meta_out_ports, dest_port)
            if port_info:
                boundary_dest_name = port_info.get('name')
        
        return ResolvedConnection(
            source_id=source_id,
            dest_id=dest_id,
            source_port=source_port,
            dest_port=dest_port,
            is_boundary_source=is_boundary_source,
            is_boundary_dest=is_boundary_dest,
            boundary_source_port_name=boundary_source_name,
            boundary_dest_port_name=boundary_dest_name,
        )
    
    def resolve_all(self, connections: List[Dict[str, Any]]) -> List[ResolvedConnection]:
        """Resolve all connections in a list."""
        return [self.resolve_connection(c) for c in connections]
    
    def get_boundary_connections(
        self, 
        connections: List[Dict[str, Any]]
    ) -> Dict[str, List[ResolvedConnection]]:
        """
        Separate boundary connections from internal connections.
        
        Returns:
            Dict with 'input_boundaries', 'output_boundaries', 'internal' lists
        """
        result = {
            'input_boundaries': [],   # sourceID=-1 (from metanode input)
            'output_boundaries': [],  # destID=-1 (to metanode output)
            'internal': [],           # Normal node-to-node connections
        }
        
        for conn in connections:
            resolved = self.resolve_connection(conn)
            
            if resolved.is_boundary_source:
                result['input_boundaries'].append(resolved)
            elif resolved.is_boundary_dest:
                result['output_boundaries'].append(resolved)
            else:
                result['internal'].append(resolved)
        
        return result
    
    def _get_port_by_index(
        self, 
        ports: List[Dict[str, Any]], 
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Get port definition by index."""
        for port in ports:
            if port.get('index') == index:
                return port
        
        # Fallback: treat list index as port index
        if 0 <= index < len(ports):
            return ports[index]
        
        return None
