"""
Execution Graph Builder - DAG construction and analysis for KNIME workflows.

Handles:
- NetworkX DiGraph construction from workflow connections
- Cycle detection for controlled loops
- Parallel component identification
- Topological execution ordering
- Multi-port connection handling
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from app.models.ir_models import (
    Connection,
    ConnectionType,
    ExecutionLayer,
    LoopStructure,
    NodeInstance,
    ParallelGroup,
    SwitchBranch,
    SwitchStructure,
    WorkflowIR,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphAnalysis:
    """Complete analysis results of the execution graph."""
    dag: nx.DiGraph
    execution_order: List[str]
    execution_layers: List[ExecutionLayer]
    parallel_groups: List[ParallelGroup]
    loops: List[LoopStructure]
    switches: List[SwitchStructure]
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)


class ExecutionGraphBuilder:
    """
    Builds and analyzes the execution graph for a KNIME workflow.
    
    The execution graph represents dependencies between nodes,
    enabling topological sorting for correct execution order
    and identification of parallelizable sections.
    """
    
    # Known loop node patterns
    LOOP_START_PATTERNS = [
        "loopstart", "chunk loop start", "counting loop start",
        "table row to variable loop start", "column list loop start",
        "recursive loop start", "generic loop start"
    ]
    
    LOOP_END_PATTERNS = [
        "loopend", "loop end", "variable loop end",
        "recursive loop end", "generic loop end"
    ]
    
    # Known switch/IF patterns
    SWITCH_START_PATTERNS = ["if switch", "case switch start", "switch"]
    SWITCH_END_PATTERNS = ["end if", "case switch end", "end switch"]
    
    def __init__(self):
        self._node_map: Dict[str, NodeInstance] = {}
    
    def build_dag(
        self, 
        nodes: List[NodeInstance],
        connections: List[Connection]
    ) -> nx.DiGraph:
        """
        Build a directed acyclic graph from nodes and connections.
        
        Args:
            nodes: List of node instances
            connections: List of connections between nodes
            
        Returns:
            NetworkX DiGraph representing the workflow
        """
        dag = nx.DiGraph()
        
        # Add nodes with attributes
        for node in nodes:
            self._node_map[node.node_id] = node
            dag.add_node(
                node.node_id,
                node_type=node.node_type,
                factory=node.factory_class,
                name=node.name,
                category=node.category.value,
                is_metanode=node.is_metanode,
            )
        
        # Add edges (connections)
        for conn in connections:
            # Skip if nodes don't exist (might be from nested metanodes)
            if conn.source_node_id not in dag.nodes:
                logger.warning(f"Source node not found: {conn.source_node_id}")
                continue
            if conn.dest_node_id not in dag.nodes:
                logger.warning(f"Dest node not found: {conn.dest_node_id}")
                continue
            
            dag.add_edge(
                conn.source_node_id,
                conn.dest_node_id,
                source_port=conn.source_port,
                dest_port=conn.dest_port,
                connection_type=conn.connection_type.value,
            )
        
        logger.info(f"Built DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
        return dag
    
    def analyze(
        self,
        nodes: List[NodeInstance],
        connections: List[Connection]
    ) -> GraphAnalysis:
        """
        Perform complete analysis of the workflow graph.
        
        Args:
            nodes: List of node instances
            connections: List of connections
            
        Returns:
            Complete GraphAnalysis with execution order, parallelization, etc.
        """
        dag = self.build_dag(nodes, connections)
        
        errors = []
        
        # Check for cycles (excluding controlled loops)
        loops = self.detect_loops(dag)
        
        # Remove loop back-edges temporarily for topological sort
        temp_dag = dag.copy()
        for loop in loops:
            # Find back-edge from loop end to loop start
            if temp_dag.has_edge(loop.end_node_id, loop.start_node_id):
                temp_dag.remove_edge(loop.end_node_id, loop.start_node_id)
        
        # Check if DAG is still cyclic after removing known loops
        try:
            if not nx.is_directed_acyclic_graph(temp_dag):
                cycles = list(nx.simple_cycles(temp_dag))
                errors.append(f"Workflow contains {len(cycles)} unexpected cycle(s)")
        except Exception as e:
            logger.warning(f"Cycle detection error: {e}")
        
        # Get topological order
        try:
            execution_order = list(nx.topological_sort(temp_dag))
        except nx.NetworkXUnfeasible:
            execution_order = list(dag.nodes())
            errors.append("Could not determine topological order due to cycles")
        
        # Get execution layers
        execution_layers = self.get_execution_layers(temp_dag)
        
        # Identify parallel groups
        parallel_groups = self.identify_parallel_groups(temp_dag, execution_layers)
        
        # Detect switch structures
        switches = self.detect_switches(dag)
        
        return GraphAnalysis(
            dag=dag,
            execution_order=execution_order,
            execution_layers=execution_layers,
            parallel_groups=parallel_groups,
            loops=loops,
            switches=switches,
            is_valid=len(errors) == 0,
            errors=errors,
        )
    
    def get_execution_layers(self, dag: nx.DiGraph) -> List[ExecutionLayer]:
        """
        Partition nodes into execution layers based on dependencies.
        
        Nodes in the same layer have no dependencies on each other
        and can potentially run in parallel.
        
        Args:
            dag: The workflow DAG
            
        Returns:
            List of execution layers in order
        """
        if not dag.nodes():
            return []
        
        layers = []
        remaining_nodes = set(dag.nodes())
        processed_nodes: Set[str] = set()
        layer_index = 0
        
        while remaining_nodes:
            # Find nodes with all dependencies satisfied
            current_layer_nodes = []
            
            for node in remaining_nodes:
                predecessors = set(dag.predecessors(node))
                if predecessors.issubset(processed_nodes):
                    current_layer_nodes.append(node)
            
            if not current_layer_nodes:
                # Cycle or disconnected - add remaining nodes
                current_layer_nodes = list(remaining_nodes)
            
            # Determine if this layer can be parallelized
            can_parallelize = len(current_layer_nodes) > 1
            
            layers.append(ExecutionLayer(
                layer_index=layer_index,
                node_ids=current_layer_nodes,
                can_parallelize=can_parallelize,
            ))
            
            processed_nodes.update(current_layer_nodes)
            remaining_nodes -= set(current_layer_nodes)
            layer_index += 1
        
        return layers
    
    def identify_parallel_groups(
        self, 
        dag: nx.DiGraph,
        layers: List[ExecutionLayer]
    ) -> List[ParallelGroup]:
        """
        Identify groups of nodes that can execute in parallel.
        
        Args:
            dag: The workflow DAG
            layers: Pre-computed execution layers
            
        Returns:
            List of parallel groups with load estimates
        """
        parallel_groups = []
        group_id = 0
        
        for layer in layers:
            if not layer.can_parallelize:
                continue
            
            if len(layer.node_ids) < 2:
                continue
            
            # Estimate computational load for each node
            estimated_loads = []
            for node_id in layer.node_ids:
                load = self._estimate_node_load(node_id)
                estimated_loads.append(load)
            
            total_load = sum(estimated_loads)
            
            parallel_groups.append(ParallelGroup(
                group_id=group_id,
                nodes=layer.node_ids,
                estimated_load=total_load,
            ))
            group_id += 1
        
        return parallel_groups
    
    def _estimate_node_load(self, node_id: str) -> float:
        """
        Estimate the computational load of a node.
        
        This is a heuristic based on node type.
        """
        if node_id not in self._node_map:
            return 1.0
        
        node = self._node_map[node_id]
        
        # High load nodes
        if node.category.value in ["ml", "statistics"]:
            return 5.0
        
        # Medium load nodes
        if node.category.value in ["aggregation", "joining"]:
            return 3.0
        
        # Low load nodes (simple transformations)
        return 1.0
    
    def detect_loops(self, dag: nx.DiGraph) -> List[LoopStructure]:
        """
        Detect KNIME loop structures in the graph.
        
        KNIME loops use explicit LoopStart/LoopEnd node pairs.
        """
        loops = []
        loop_id = 0
        
        # Find all loop start nodes
        loop_starts = []
        for node_id in dag.nodes():
            node_name = dag.nodes[node_id].get("name", "").lower()
            if any(p in node_name for p in self.LOOP_START_PATTERNS):
                loop_starts.append(node_id)
        
        # For each loop start, find the corresponding end
        for start_id in loop_starts:
            start_name = dag.nodes[start_id].get("name", "").lower()
            
            # Determine loop type
            loop_type = "generic"
            if "chunk" in start_name:
                loop_type = "chunk"
            elif "counting" in start_name:
                loop_type = "counting"
            elif "table row" in start_name:
                loop_type = "table_row"
            elif "column" in start_name:
                loop_type = "column"
            elif "recursive" in start_name:
                loop_type = "recursive"
            
            # Find connected loop end
            end_id = None
            body_nodes = []
            
            # BFS to find loop end
            visited = set()
            queue = [start_id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                node_name = dag.nodes[current].get("name", "").lower()
                
                if current != start_id and any(p in node_name for p in self.LOOP_END_PATTERNS):
                    end_id = current
                    break
                
                if current != start_id:
                    body_nodes.append(current)
                
                for successor in dag.successors(current):
                    queue.append(successor)
            
            if end_id:
                loops.append(LoopStructure(
                    loop_id=f"loop_{loop_id}",
                    start_node_id=start_id,
                    end_node_id=end_id,
                    loop_type=loop_type,
                    body_nodes=body_nodes,
                ))
                loop_id += 1
        
        return loops
    
    def detect_switches(self, dag: nx.DiGraph) -> List[SwitchStructure]:
        """
        Detect KNIME IF/Switch structures in the graph.
        """
        switches = []
        switch_id = 0
        
        # Find all switch start nodes
        switch_starts = []
        for node_id in dag.nodes():
            node_name = dag.nodes[node_id].get("name", "").lower()
            if any(p in node_name for p in self.SWITCH_START_PATTERNS):
                switch_starts.append(node_id)
        
        for start_id in switch_starts:
            # Find branches (multiple successors)
            successors = list(dag.successors(start_id))
            
            if len(successors) < 2:
                continue
            
            branches = []
            end_id = None
            
            # Each successor represents a branch
            for i, succ_id in enumerate(successors):
                branch_nodes = []
                
                # Follow the branch until we hit a switch end or merge point
                visited = set()
                queue = [succ_id]
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    node_name = dag.nodes[current].get("name", "").lower()
                    
                    if any(p in node_name for p in self.SWITCH_END_PATTERNS):
                        end_id = current
                        break
                    
                    branch_nodes.append(current)
                    
                    for successor in dag.successors(current):
                        if successor not in visited:
                            queue.append(successor)
                
                branches.append(SwitchBranch(
                    branch_id=f"branch_{i}",
                    nodes=branch_nodes,
                    is_default=(i == len(successors) - 1),
                ))
            
            switches.append(SwitchStructure(
                switch_id=f"switch_{switch_id}",
                start_node_id=start_id,
                end_node_id=end_id or "",
                branches=branches,
            ))
            switch_id += 1
        
        return switches
    
    def get_subgraph(
        self, 
        dag: nx.DiGraph, 
        node_ids: List[str]
    ) -> nx.DiGraph:
        """Extract a subgraph containing only the specified nodes."""
        return dag.subgraph(node_ids).copy()
    
    def get_ancestors(self, dag: nx.DiGraph, node_id: str) -> Set[str]:
        """Get all ancestor nodes of a given node."""
        return nx.ancestors(dag, node_id)
    
    def get_descendants(self, dag: nx.DiGraph, node_id: str) -> Set[str]:
        """Get all descendant nodes of a given node."""
        return nx.descendants(dag, node_id)
