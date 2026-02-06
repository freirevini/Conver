"""
IR (Intermediate Representation) Models for KNIME Workflow Transpiler.

Dataclass-based models representing the complete semantic structure of KNIME workflows,
enabling serialization to JSON/YAML and serving as input for code generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime


class PortType(str, Enum):
    """Types of ports in KNIME nodes."""
    DATA = "data"
    MODEL = "model"
    FLOW_VARIABLE = "flow_variable"
    DATABASE = "database"
    PMML = "pmml"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ConnectionType(str, Enum):
    """Types of connections between nodes."""
    DATA = "data"
    MODEL = "model"
    FLOW_VARIABLE = "flow_variable"


class NodeCategory(str, Enum):
    """Categories of KNIME nodes."""
    IO = "io"
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATION = "aggregation"
    JOINING = "joining"
    MACHINE_LEARNING = "ml"
    FLOW_CONTROL = "flow_control"
    DATABASE = "database"
    TEXT = "text"
    DATETIME = "datetime"
    STATISTICS = "statistics"
    VISUALIZATION = "visualization"
    METANODE = "metanode"
    COMPONENT = "component"
    UNKNOWN = "unknown"


class FallbackLevel(str, Enum):
    """Levels of fallback for code generation."""
    TEMPLATE_EXACT = "template_exact"             # 100% functional template
    TEMPLATE_APPROXIMATE = "template_approximate"  # Similar template with adaptations
    LLM_GENERATED = "llm_generated"               # AI-generated code
    STUB = "stub"                                 # TODO stub for manual implementation
    
    # Aliases for backward compatibility
    DETERMINISTIC = "template_exact"
    LLM = "llm_generated"


@dataclass
class NodePort:
    """Represents an input or output port of a node."""
    index: int
    port_type: PortType
    name: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "port_type": self.port_type.value,
            "name": self.name,
            "spec": self.spec,
        }


@dataclass
class Connection:
    """Represents a connection between two nodes."""
    source_node_id: str
    source_port: int
    dest_node_id: str
    dest_port: int
    connection_type: ConnectionType = ConnectionType.DATA
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "source_port": self.source_port,
            "dest_node_id": self.dest_node_id,
            "dest_port": self.dest_port,
            "connection_type": self.connection_type.value,
        }


@dataclass
class FlowVariable:
    """Represents a KNIME flow variable."""
    name: str
    var_type: Literal["string", "int", "double", "boolean"]
    value: Any
    scope: Literal["global", "node", "loop"] = "global"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.var_type,
            "value": self.value,
            "scope": self.scope,
        }


@dataclass
class NodeInstance:
    """Represents a single node instance in a KNIME workflow."""
    node_id: str
    node_type: str
    factory_class: str
    name: str
    settings: Dict[str, Any] = field(default_factory=dict)
    input_ports: List[NodePort] = field(default_factory=list)
    output_ports: List[NodePort] = field(default_factory=list)
    position: Tuple[int, int] = (0, 0)
    is_metanode: bool = False
    is_component: bool = False
    children: Optional[List["NodeInstance"]] = None
    internal_connections: Optional[List[Connection]] = None
    flow_variables: List[FlowVariable] = field(default_factory=list)
    category: NodeCategory = NodeCategory.UNKNOWN
    fallback_level: FallbackLevel = FallbackLevel.TEMPLATE_EXACT
    custom_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "factory_class": self.factory_class,
            "name": self.name,
            "settings": self.settings,
            "input_ports": [p.to_dict() for p in self.input_ports],
            "output_ports": [p.to_dict() for p in self.output_ports],
            "position": list(self.position),
            "is_metanode": self.is_metanode,
            "is_component": self.is_component,
            "category": self.category.value,
            "fallback_level": self.fallback_level.value,
        }
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        if self.internal_connections:
            result["internal_connections"] = [c.to_dict() for c in self.internal_connections]
        if self.flow_variables:
            result["flow_variables"] = [fv.to_dict() for fv in self.flow_variables]
        if self.custom_description:
            result["custom_description"] = self.custom_description
        return result


@dataclass
class LoopStructure:
    """Represents a loop structure (LoopStart to LoopEnd)."""
    loop_id: str
    start_node_id: str
    end_node_id: str
    loop_type: Literal["chunk", "counting", "table_row", "column", "recursive", "generic"]
    body_nodes: List[str] = field(default_factory=list)
    iteration_variable: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "loop_type": self.loop_type,
            "body_nodes": self.body_nodes,
            "iteration_variable": self.iteration_variable,
        }


@dataclass
class SwitchBranch:
    """Represents a branch in an IF/Switch structure."""
    branch_id: str
    condition: Optional[str] = None
    nodes: List[str] = field(default_factory=list)
    is_default: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "condition": self.condition,
            "nodes": self.nodes,
            "is_default": self.is_default,
        }


@dataclass
class SwitchStructure:
    """Represents a switch/case or IF structure."""
    switch_id: str
    start_node_id: str
    end_node_id: str
    branches: List[SwitchBranch] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "switch_id": self.switch_id,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "branches": [b.to_dict() for b in self.branches],
        }


@dataclass
class ParallelGroup:
    """Represents a group of nodes that can execute in parallel."""
    group_id: int
    nodes: List[str]
    estimated_load: float = 1.0  # Relative computational load
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "nodes": self.nodes,
            "estimated_load": self.estimated_load,
        }


@dataclass
class ExecutionLayer:
    """Represents a layer of nodes in topological order."""
    layer_index: int
    node_ids: List[str]
    can_parallelize: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "node_ids": self.node_ids,
            "can_parallelize": self.can_parallelize,
        }


@dataclass
class WorkflowMetadata:
    """Metadata about the workflow."""
    name: str
    description: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    knime_version: Optional[str] = None
    source_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "modified_date": self.modified_date.isoformat() if self.modified_date else None,
            "knime_version": self.knime_version,
            "source_file": self.source_file,
        }


@dataclass
class PythonEnvironment:
    """Python environment requirements for generated code."""
    python_version: str = "3.12"
    dependencies: List[str] = field(default_factory=list)
    env_variables: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "dependencies": self.dependencies,
            "env_variables": self.env_variables,
        }


@dataclass
class WorkflowIR:
    """
    Complete Intermediate Representation of a KNIME workflow.
    
    This is the main data structure that captures the full semantics of a workflow
    and serves as the input for Python code generation.
    """
    version: str = "2.0"
    metadata: WorkflowMetadata = field(default_factory=lambda: WorkflowMetadata(name="Unnamed"))
    environment: PythonEnvironment = field(default_factory=PythonEnvironment)
    
    # Core workflow structure
    nodes: List[NodeInstance] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    
    # Flow variables
    flow_variables: List[FlowVariable] = field(default_factory=list)
    
    # Execution analysis
    execution_order: List[str] = field(default_factory=list)
    execution_layers: List[ExecutionLayer] = field(default_factory=list)
    parallel_groups: List[ParallelGroup] = field(default_factory=list)
    
    # Control flow structures
    loops: List[LoopStructure] = field(default_factory=list)
    switches: List[SwitchStructure] = field(default_factory=list)
    
    # Generation metadata
    unsupported_nodes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire IR to a dictionary for JSON/YAML serialization."""
        return {
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "environment": self.environment.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
            "flow_variables": [fv.to_dict() for fv in self.flow_variables],
            "execution_order": self.execution_order,
            "execution_layers": [l.to_dict() for l in self.execution_layers],
            "parallel_groups": [g.to_dict() for g in self.parallel_groups],
            "loops": [l.to_dict() for l in self.loops],
            "switches": [s.to_dict() for s in self.switches],
            "unsupported_nodes": self.unsupported_nodes,
            "warnings": self.warnings,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeInstance]:
        """Find a node by its ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
            if node.children:
                for child in node.children:
                    if child.node_id == node_id:
                        return child
        return None
    
    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get all nodes that this node depends on."""
        deps = []
        for conn in self.connections:
            if conn.dest_node_id == node_id:
                deps.append(conn.source_node_id)
        return deps
    
    def get_node_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on this node."""
        deps = []
        for conn in self.connections:
            if conn.source_node_id == node_id:
                deps.append(conn.dest_node_id)
        return deps


@dataclass
class ValidationResult:
    """Result of validating generated code against KNIME output."""
    is_valid: bool
    node_id: Optional[str] = None
    knime_output_hash: Optional[str] = None
    python_output_hash: Optional[str] = None
    discrepancies: List[str] = field(default_factory=list)
    tolerance_used: float = 1e-5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "node_id": self.node_id,
            "knime_output_hash": self.knime_output_hash,
            "python_output_hash": self.python_output_hash,
            "discrepancies": self.discrepancies,
            "tolerance_used": self.tolerance_used,
        }


@dataclass
class GeneratedCode:
    """Container for generated Python code."""
    script: str
    imports: List[str] = field(default_factory=list)
    functions: Dict[str, str] = field(default_factory=dict)
    main_block: str = ""
    requirements: List[str] = field(default_factory=list)
    source_ir: Optional[WorkflowIR] = None
    generation_time: Optional[datetime] = None
    
    def to_file_content(self) -> str:
        """Generate complete Python file content."""
        return self.script
