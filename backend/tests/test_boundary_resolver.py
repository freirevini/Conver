"""
Tests for BoundaryResolver.

Tests the ID=-1 boundary connection resolution logic.
"""
import pytest

from app.services.parser.boundary_resolver import BoundaryResolver, ResolvedConnection


class TestBoundaryResolver:
    """Tests for BoundaryResolver class."""
    
    def test_is_boundary_connection_source(self):
        """Detect boundary source (from metanode input)."""
        resolver = BoundaryResolver()
        
        conn = {'source_id': -1, 'dest_id': 123, 'source_port': 0, 'dest_port': 0}
        assert resolver.is_boundary_connection(conn) is True
    
    def test_is_boundary_connection_dest(self):
        """Detect boundary dest (to metanode output)."""
        resolver = BoundaryResolver()
        
        conn = {'source_id': 123, 'dest_id': -1, 'source_port': 0, 'dest_port': 0}
        assert resolver.is_boundary_connection(conn) is True
    
    def test_is_not_boundary_connection(self):
        """Normal internal connection is not boundary."""
        resolver = BoundaryResolver()
        
        conn = {'source_id': 123, 'dest_id': 456, 'source_port': 0, 'dest_port': 0}
        assert resolver.is_boundary_connection(conn) is False
    
    def test_resolve_boundary_source(self):
        """Resolve connection from metanode input."""
        meta_in_ports = [
            {'index': 0, 'name': 'Input Table 1'},
            {'index': 1, 'name': 'Input Table 2'},
        ]
        resolver = BoundaryResolver(meta_in_ports=meta_in_ports)
        
        conn = {'source_id': -1, 'dest_id': 123, 'source_port': 0, 'dest_port': 1}
        resolved = resolver.resolve_connection(conn)
        
        assert resolved.is_boundary_source is True
        assert resolved.is_boundary_dest is False
        assert resolved.boundary_source_port_name == 'Input Table 1'
        assert resolved.source_port == 0
        assert resolved.dest_id == 123
    
    def test_resolve_boundary_dest(self):
        """Resolve connection to metanode output."""
        meta_out_ports = [
            {'index': 0, 'name': 'Output Table'},
        ]
        resolver = BoundaryResolver(meta_out_ports=meta_out_ports)
        
        conn = {'source_id': 123, 'dest_id': -1, 'source_port': 0, 'dest_port': 0}
        resolved = resolver.resolve_connection(conn)
        
        assert resolved.is_boundary_source is False
        assert resolved.is_boundary_dest is True
        assert resolved.boundary_dest_port_name == 'Output Table'
        assert resolved.dest_port == 0
        assert resolved.source_id == 123
    
    def test_resolve_internal_connection(self):
        """Resolve normal internal connection."""
        resolver = BoundaryResolver()
        
        conn = {'source_id': 100, 'dest_id': 200, 'source_port': 0, 'dest_port': 1}
        resolved = resolver.resolve_connection(conn)
        
        assert resolved.is_boundary_source is False
        assert resolved.is_boundary_dest is False
        assert resolved.source_id == 100
        assert resolved.dest_id == 200
    
    def test_get_boundary_connections_separation(self):
        """Separate connections by type."""
        resolver = BoundaryResolver()
        
        connections = [
            {'source_id': -1, 'dest_id': 100, 'source_port': 0, 'dest_port': 0},
            {'source_id': 100, 'dest_id': 200, 'source_port': 0, 'dest_port': 0},
            {'source_id': 200, 'dest_id': -1, 'source_port': 0, 'dest_port': 0},
            {'source_id': 150, 'dest_id': 250, 'source_port': 1, 'dest_port': 0},
        ]
        
        result = resolver.get_boundary_connections(connections)
        
        assert len(result['input_boundaries']) == 1
        assert len(result['output_boundaries']) == 1
        assert len(result['internal']) == 2
    
    def test_resolve_all(self):
        """Resolve all connections at once."""
        resolver = BoundaryResolver()
        
        connections = [
            {'source_id': -1, 'dest_id': 100, 'source_port': 0, 'dest_port': 0},
            {'source_id': 100, 'dest_id': 200, 'source_port': 0, 'dest_port': 0},
        ]
        
        resolved = resolver.resolve_all(connections)
        
        assert len(resolved) == 2
        assert all(isinstance(r, ResolvedConnection) for r in resolved)
    
    def test_to_dict(self):
        """ResolvedConnection serializes to dict."""
        conn = ResolvedConnection(
            source_id=-1,
            dest_id=100,
            source_port=0,
            dest_port=1,
            is_boundary_source=True,
            boundary_source_port_name='Input'
        )
        
        d = conn.to_dict()
        
        assert d['source_id'] == -1
        assert d['is_boundary_source'] is True
        assert d['boundary_source_port_name'] == 'Input'
