"""
KNIME Topology Builder

Builds a Directed Acyclic Graph (DAG) from workflow nodes and connections.
Provides topological ordering for code generation.
"""
import logging
from typing import Dict, List, Tuple, Any
import networkx as nx

logger = logging.getLogger(__name__)


class TopologyBuilder:
    """
    Builds execution graph from KNIME workflow structure.
    
    Uses NetworkX to create a DAG and determine proper execution order.
    """
    
    def build_dag(self, nodes: List[Dict], connections: List[Dict]) -> nx.DiGraph:
        """
        Build a Directed Acyclic Graph from nodes and connections.
        
        Args:
            nodes: List of node dictionaries with 'id' and other attributes
            connections: List of connection dictionaries with source/dest IDs
            
        Returns:
            NetworkX DiGraph with nodes and edges
        """
        G = nx.DiGraph()
        
        # Add all nodes with their attributes
        for node in nodes:
            node_id = node['id']
            G.add_node(node_id, **node)
            logger.debug(f"Added node {node_id}: {node.get('name', 'Unknown')}")
        
        # Add edges (connections)
        for conn in connections:
            source_id = conn['source_id']
            dest_id = conn['dest_id']
            
            # Only add edge if both nodes exist
            if G.has_node(source_id) and G.has_node(dest_id):
                G.add_edge(
                    source_id,
                    dest_id,
                    source_port=conn.get('source_port', 0),
                    dest_port=conn.get('dest_port', 0)
                )
                logger.debug(f"Added edge: {source_id} -> {dest_id}")
            else:
                logger.warning(
                    f"Skipping connection {source_id} -> {dest_id}: "
                    f"node(s) not found"
                )
        
        # Validate DAG (no cycles)
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            logger.error(f"Workflow contains cycles: {cycles}")
            # Remove cycles by breaking at first edge
            for cycle in cycles:
                if len(cycle) >= 2:
                    G.remove_edge(cycle[0], cycle[1])
        
        logger.info(
            f"Built DAG: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        
        return G
    
    def get_execution_order(self, dag: nx.DiGraph) -> List[int]:
        """
        Get topological order for node execution.
        
        Returns nodes in an order where all dependencies are
        processed before dependent nodes.
        
        Args:
            dag: NetworkX DiGraph
            
        Returns:
            List of node IDs in execution order
        """
        try:
            order = list(nx.topological_sort(dag))
            logger.info(f"Execution order: {order}")
            return order
        except nx.NetworkXUnfeasible:
            logger.error("Cannot determine execution order: graph has cycles")
            # Fallback: return nodes in ID order
            return sorted(dag.nodes())
    
    def get_node_dependencies(self, dag: nx.DiGraph, node_id: int) -> List[int]:
        """
        Get all predecessor nodes (dependencies) for a node.
        
        Args:
            dag: NetworkX DiGraph
            node_id: Node to get dependencies for
            
        Returns:
            List of predecessor node IDs
        """
        if not dag.has_node(node_id):
            return []
        return list(dag.predecessors(node_id))
    
    def get_node_dependents(self, dag: nx.DiGraph, node_id: int) -> List[int]:
        """
        Get all successor nodes (nodes that depend on this one).
        
        Args:
            dag: NetworkX DiGraph
            node_id: Node to get dependents for
            
        Returns:
            List of successor node IDs
        """
        if not dag.has_node(node_id):
            return []
        return list(dag.successors(node_id))
    
    def get_source_nodes(self, dag: nx.DiGraph) -> List[int]:
        """
        Get nodes with no incoming edges (data sources).
        
        These are typically Reader nodes (CSV Reader, DB Reader, etc.)
        """
        sources = [n for n in dag.nodes() if dag.in_degree(n) == 0]
        logger.debug(f"Source nodes: {sources}")
        return sources
    
    def get_sink_nodes(self, dag: nx.DiGraph) -> List[int]:
        """
        Get nodes with no outgoing edges (data destinations).
        
        These are typically Writer nodes (CSV Writer, DB Writer, etc.)
        """
        sinks = [n for n in dag.nodes() if dag.out_degree(n) == 0]
        logger.debug(f"Sink nodes: {sinks}")
        return sinks
    
    def get_input_variable_name(self, dag: nx.DiGraph, node_id: int, port: int = 0) -> str:
        """
        Get the variable name for input data to a node.
        
        Uses predecessor node ID to construct variable name.
        
        Args:
            dag: NetworkX DiGraph
            node_id: Node receiving input
            port: Input port number (for nodes with multiple inputs)
            
        Returns:
            Variable name like 'df_node_5'
        """
        predecessors = list(dag.predecessors(node_id))
        
        if not predecessors:
            return "df_input"
        
        # For multiple inputs, use port to select
        if port < len(predecessors):
            pred_id = predecessors[port]
        else:
            pred_id = predecessors[0]
        
        return f"df_node_{pred_id}"
    
    def get_output_variable_name(self, node_id: int) -> str:
        """
        Get the variable name for output data from a node.
        
        Args:
            node_id: Node producing output
            
        Returns:
            Variable name like 'df_node_5'
        """
        return f"df_node_{node_id}"
    
    def get_subgraphs(self, dag: nx.DiGraph) -> List[nx.DiGraph]:
        """
        Get independent subgraphs (disconnected components).
        
        Useful for parallel processing of independent workflow branches.
        
        Args:
            dag: NetworkX DiGraph
            
        Returns:
            List of subgraph DiGraphs
        """
        # Get weakly connected components (ignoring edge direction)
        components = list(nx.weakly_connected_components(dag))
        
        subgraphs = []
        for component in components:
            subgraph = dag.subgraph(component).copy()
            subgraphs.append(subgraph)
        
        logger.info(f"Found {len(subgraphs)} independent subgraph(s)")
        return subgraphs
    
    def visualize_dag(self, dag: nx.DiGraph) -> str:
        """
        Create a simple text visualization of the DAG.
        
        Returns:
            ASCII representation of the workflow
        """
        lines = ["Workflow DAG:", "=" * 40]
        
        order = self.get_execution_order(dag)
        
        for node_id in order:
            node_data = dag.nodes[node_id]
            name = node_data.get('name', f'Node_{node_id}')
            
            predecessors = list(dag.predecessors(node_id))
            successors = list(dag.successors(node_id))
            
            pred_str = f"<- {predecessors}" if predecessors else "(source)"
            succ_str = f"-> {successors}" if successors else "(sink)"
            
            lines.append(f"  [{node_id}] {name}")
            lines.append(f"        {pred_str} {succ_str}")
        
        lines.append("=" * 40)
        
        return "\n".join(lines)
