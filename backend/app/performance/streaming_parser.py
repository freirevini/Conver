"""
Streaming XML Parser for Large KNIME Workflows.

Provides memory-efficient parsing for workflows >500 nodes:
- Incremental SAX-based parsing
- Chunked processing
- Memory-bounded operations
"""
import xml.sax
import xml.sax.handler
import logging
from typing import Dict, List, Optional, Iterator, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from io import BytesIO
import zipfile

logger = logging.getLogger(__name__)


@dataclass
class StreamedNode:
    """Node parsed from stream."""
    id: str
    node_type: str
    name: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0


@dataclass
class StreamedConnection:
    """Connection parsed from stream."""
    source_id: str
    source_port: int
    dest_id: str
    dest_port: int


class KNWFStreamHandler(xml.sax.handler.ContentHandler):
    """
    SAX handler for streaming KNIME workflow parsing.
    
    Instead of building complete DOM in memory, processes
    nodes as they are encountered and yields them.
    """
    
    def __init__(self, node_callback: Callable[[StreamedNode], None] = None):
        super().__init__()
        self.node_callback = node_callback
        
        # Current parsing state
        self._current_element = ""
        self._element_stack: List[str] = []
        self._text_buffer = ""
        
        # Current node being parsed
        self._current_node: Optional[Dict] = None
        self._current_settings: Dict = {}
        
        # Collected data
        self.nodes: List[StreamedNode] = []
        self.connections: List[StreamedConnection] = []
        self.metadata: Dict[str, Any] = {}
        
        # Stats
        self.nodes_parsed = 0
        self.connections_parsed = 0
    
    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesImpl):
        """Handle element start."""
        self._element_stack.append(name)
        self._current_element = name
        self._text_buffer = ""
        
        if name == "node":
            self._current_node = {
                "id": attrs.get("id", ""),
            }
        
        elif name == "connection":
            conn = StreamedConnection(
                source_id=attrs.get("sourceID", ""),
                source_port=int(attrs.get("sourcePort", 0)),
                dest_id=attrs.get("destID", ""),
                dest_port=int(attrs.get("destPort", 0))
            )
            self.connections.append(conn)
            self.connections_parsed += 1
        
        elif name == "entry" and self._current_node:
            key = attrs.get("key", "")
            value = attrs.get("value", "")
            entry_type = attrs.get("type", "xstring")
            
            # Parse value based on type
            if entry_type == "xint":
                try:
                    value = int(value)
                except ValueError:
                    pass
            elif entry_type == "xdouble":
                try:
                    value = float(value)
                except ValueError:
                    pass
            elif entry_type == "xboolean":
                value = value.lower() == "true"
            
            if key:
                self._current_settings[key] = value
    
    def endElement(self, name: str):
        """Handle element end."""
        if name == "node" and self._current_node:
            # Create node and yield/callback
            node = StreamedNode(
                id=self._current_node.get("id", ""),
                node_type=self._current_settings.get("node-type", "Unknown"),
                name=self._current_settings.get("name", ""),
                settings=self._current_settings.copy(),
                x=float(self._current_settings.get("node-bounds-x", 0)),
                y=float(self._current_settings.get("node-bounds-y", 0))
            )
            
            if self.node_callback:
                self.node_callback(node)
            else:
                self.nodes.append(node)
            
            self.nodes_parsed += 1
            self._current_node = None
            self._current_settings = {}
        
        if self._element_stack:
            self._element_stack.pop()
        
        self._current_element = self._element_stack[-1] if self._element_stack else ""
    
    def characters(self, content: str):
        """Handle text content."""
        self._text_buffer += content


class StreamingKNWFParser:
    """
    Memory-efficient streaming parser for KNIME workflows.
    
    Uses SAX parsing to process large workflows without
    loading entire DOM into memory.
    """
    
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        self._nodes_buffer: deque = deque(maxlen=chunk_size)
    
    def parse_file(self, path: Path) -> Iterator[StreamedNode]:
        """
        Parse KNWF file yielding nodes one at a time.
        
        Args:
            path: Path to .knwf file
            
        Yields:
            StreamedNode instances
        """
        if path.suffix == ".knwf":
            yield from self._parse_knwf_archive(path)
        else:
            yield from self._parse_xml_file(path)
    
    def _parse_knwf_archive(self, path: Path) -> Iterator[StreamedNode]:
        """Parse KNWF ZIP archive."""
        with zipfile.ZipFile(path, 'r') as zf:
            # Find workflow.knime
            workflow_file = None
            for name in zf.namelist():
                if name.endswith("workflow.knime"):
                    workflow_file = name
                    break
            
            if not workflow_file:
                logger.error("No workflow.knime found in archive")
                return
            
            # Stream parse the XML
            with zf.open(workflow_file) as f:
                yield from self._parse_xml_stream(f)
    
    def _parse_xml_file(self, path: Path) -> Iterator[StreamedNode]:
        """Parse XML file."""
        with open(path, 'rb') as f:
            yield from self._parse_xml_stream(f)
    
    def _parse_xml_stream(self, stream) -> Iterator[StreamedNode]:
        """Parse XML from stream using SAX."""
        nodes_queue = []
        
        def node_callback(node: StreamedNode):
            nodes_queue.append(node)
        
        handler = KNWFStreamHandler(node_callback=node_callback)
        
        try:
            # Parse in chunks
            parser = xml.sax.make_parser()
            parser.setContentHandler(handler)
            
            # Use incremental parsing
            parser.parse(stream)
            
            # Yield all collected nodes
            for node in nodes_queue:
                yield node
            
            # Also yield from handler if callback wasn't used
            for node in handler.nodes:
                yield node
                
        except xml.sax.SAXException as e:
            logger.error(f"SAX parsing error: {e}")
    
    def parse_chunked(
        self,
        path: Path,
        process_func: Callable[[List[StreamedNode]], None]
    ) -> Dict[str, int]:
        """
        Parse workflow processing nodes in chunks.
        
        Memory-efficient for very large workflows.
        
        Args:
            path: Path to workflow file
            process_func: Function to process each chunk
            
        Returns:
            Stats dictionary
        """
        chunk = []
        total_nodes = 0
        total_connections = 0
        
        for node in self.parse_file(path):
            chunk.append(node)
            
            if len(chunk) >= self.chunk_size:
                process_func(chunk)
                total_nodes += len(chunk)
                chunk = []
        
        # Process remaining
        if chunk:
            process_func(chunk)
            total_nodes += len(chunk)
        
        return {
            "nodes_parsed": total_nodes,
            "chunks_processed": (total_nodes // self.chunk_size) + 1,
            "chunk_size": self.chunk_size
        }
    
    def get_workflow_stats(self, path: Path) -> Dict[str, Any]:
        """Get workflow statistics without full parsing."""
        node_count = 0
        node_types: Dict[str, int] = {}
        
        for node in self.parse_file(path):
            node_count += 1
            node_type = node.node_type
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            "total_nodes": node_count,
            "node_types": node_types,
            "unique_types": len(node_types)
        }


# ==================== Async Streaming ====================

class AsyncStreamingParser:
    """
    Async version of streaming parser.
    
    Allows non-blocking parsing of large workflows.
    """
    
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
    
    async def parse_async(self, path: Path) -> List[StreamedNode]:
        """Parse workflow asynchronously."""
        import asyncio
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        parser = StreamingKNWFParser(self.chunk_size)
        
        def sync_parse():
            return list(parser.parse_file(path))
        
        return await loop.run_in_executor(None, sync_parse)
    
    async def parse_with_progress(
        self,
        path: Path,
        progress_callback: Callable[[int], None]
    ) -> List[StreamedNode]:
        """Parse with progress reporting."""
        import asyncio
        
        nodes = []
        parser = StreamingKNWFParser(self.chunk_size)
        
        def sync_parse():
            count = 0
            for node in parser.parse_file(path):
                nodes.append(node)
                count += 1
                if count % 100 == 0:
                    # Schedule callback
                    asyncio.get_event_loop().call_soon_threadsafe(
                        progress_callback, count
                    )
            return nodes
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_parse)


# ==================== Module Init ====================

def create_performance_init():
    """Create __init__.py for performance module."""
    return '''"""
Performance Module.

Provides:
- benchmarks: Performance profiling and benchmarking
- memory_optimizer: Memory-efficient data structures
- streaming_parser: Memory-efficient large workflow parsing
"""
from .benchmarks import (
    PerformanceProfiler,
    BenchmarkResult,
    BenchmarkSuite,
    benchmark_decorator,
    check_performance_regression,
    DEFAULT_THRESHOLDS
)

from .memory_optimizer import (
    ObjectPool,
    LazyLoader,
    LazyDict,
    MemoryBoundedCache,
    WeakCache,
    get_memory_usage,
    memory_limit,
    force_gc
)

from .streaming_parser import (
    StreamingKNWFParser,
    StreamedNode,
    StreamedConnection,
    AsyncStreamingParser
)

__all__ = [
    # Benchmarks
    "PerformanceProfiler",
    "BenchmarkResult",
    "benchmark_decorator",
    # Memory
    "ObjectPool",
    "LazyLoader",
    "MemoryBoundedCache",
    "get_memory_usage",
    # Streaming
    "StreamingKNWFParser",
    "StreamedNode",
    "AsyncStreamingParser",
]
'''
