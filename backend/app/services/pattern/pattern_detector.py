"""
Pattern Detector - Bridge between Workflow Analysis and Pattern Registry.

This module analyzes parsed KNIME workflows to detect semantic patterns
(DateGenerator, Loop, etc.) and provides optimized Python code generation.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from app.services.pattern import (
    get_pattern_registry,
    DetectedPattern,
    PatternType
)

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Analyzes KNIME workflows to detect semantic patterns.
    
    Integrates with the existing transpilation pipeline to provide
    pattern-based code generation before falling back to node-by-node templates.
    """
    
    def __init__(self):
        self.registry = get_pattern_registry()
        self.detected_patterns: List[DetectedPattern] = []
        self._pattern_nodes: set = set()  # Node IDs already handled by patterns
    
    def analyze_metanode(
        self,
        metanode_path: str,
        nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> Optional[DetectedPattern]:
        """
        Analyze a metanode for semantic patterns.
        
        Args:
            metanode_path: Path to the metanode directory
            nodes: List of node definitions within the metanode
            connections: List of connections between nodes
            
        Returns:
            DetectedPattern if a pattern is found, None otherwise
        """
        # Extract metanode info from path
        metanode_name = Path(metanode_path).name if metanode_path else "Unknown"
        
        # Detect output type from node configurations
        output_type = self._detect_output_type(nodes, connections)
        
        metanode_info = {
            'name': metanode_name,
            'path': metanode_path,
            'output_type': output_type,
            'node_count': len(nodes)
        }
        
        # Run pattern detection
        detected_list = self.registry.detect_all(nodes, connections, metanode_info)
        
        if detected_list:
            pattern = detected_list[0]  # Take highest confidence
            self.detected_patterns.append(pattern)
            
            # Mark nodes as handled by this pattern
            for node_id in pattern.node_ids:
                self._pattern_nodes.add(node_id)
            
            logger.info(
                f"Pattern detected: {pattern.pattern_name} in '{metanode_name}' "
                f"(confidence: {pattern.confidence:.0%})"
            )
            return pattern
        
        return None
    
    def _detect_output_type(
        self,
        nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Detect the output port type of the metanode."""
        # Check for TableRowToVariable at output
        for node in nodes:
            factory = node.get('factory', '').lower()
            if 'tablerowtovariable' in factory:
                return 'FlowVariablePortObject'
        return None
    
    def is_node_in_pattern(self, node_id: str) -> bool:
        """Check if a node is already handled by a detected pattern."""
        return node_id in self._pattern_nodes
    
    def generate_pattern_code(
        self,
        pattern: DetectedPattern,
        input_var: str = "df",
        output_var: str = "result"
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Generate Python code for a detected pattern.
        
        Returns:
            Tuple of (code, imports) or None if generation fails
        """
        return self.registry.generate_code(pattern, input_var, output_var)
    
    def get_all_detected_patterns(self) -> List[DetectedPattern]:
        """Get all patterns detected so far."""
        return self.detected_patterns.copy()
    
    def clear(self):
        """Clear all detected patterns."""
        self.detected_patterns.clear()
        self._pattern_nodes.clear()


class WorkflowPatternAnalyzer:
    """
    High-level analyzer that processes entire workflows for patterns.
    
    Walks through the workflow structure, identifies metanodes,
    and runs pattern detection on each.
    """
    
    def __init__(self):
        self.detector = PatternDetector()
        self.pattern_code_blocks: List[Dict[str, Any]] = []
    
    def analyze_workflow(
        self,
        workflow_data: Dict[str, Any],
        base_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a complete workflow for semantic patterns.
        
        Args:
            workflow_data: Parsed workflow structure
            base_path: Base path for the workflow files
            
        Returns:
            Analysis results with detected patterns and generated code
        """
        results = {
            'total_nodes': 0,
            'metanodes_analyzed': 0,
            'patterns_detected': [],
            'pattern_code': [],
            'nodes_in_patterns': 0
        }
        
        nodes = workflow_data.get('nodes', [])
        connections = workflow_data.get('connections', [])
        results['total_nodes'] = len(nodes)
        
        # Identify metanodes
        metanodes = [n for n in nodes if n.get('node_is_meta', False)]
        
        for metanode in metanodes:
            metanode_name = metanode.get('node_name', 'Unknown')
            metanode_path = metanode.get('path', '')
            metanode_nodes = metanode.get('internal_nodes', [])
            metanode_connections = metanode.get('internal_connections', [])
            
            if not metanode_nodes:
                continue
            
            results['metanodes_analyzed'] += 1
            
            # Analyze for patterns
            pattern = self.detector.analyze_metanode(
                metanode_path,
                metanode_nodes,
                metanode_connections
            )
            
            if pattern:
                results['patterns_detected'].append({
                    'metanode': metanode_name,
                    'pattern': pattern.pattern_name,
                    'confidence': pattern.confidence,
                    'nodes': len(pattern.node_ids)
                })
                
                # Generate code
                code_result = self.detector.generate_pattern_code(
                    pattern,
                    input_var=f"df_{metanode_name.lower().replace(' ', '_')}_in",
                    output_var=f"df_{metanode_name.lower().replace(' ', '_')}_out"
                )
                
                if code_result:
                    code, imports = code_result
                    results['pattern_code'].append({
                        'metanode': metanode_name,
                        'pattern': pattern.pattern_name,
                        'code': code,
                        'imports': imports
                    })
                    results['nodes_in_patterns'] += len(pattern.node_ids)
        
        return results
    
    def get_pattern_code_for_node(self, node_id: str) -> Optional[str]:
        """
        Get pattern-generated code that covers a specific node.
        
        Returns the code block if the node is part of a detected pattern.
        """
        if self.detector.is_node_in_pattern(node_id):
            for pattern in self.detector.detected_patterns:
                if node_id in pattern.node_ids:
                    result = self.detector.generate_pattern_code(pattern)
                    if result:
                        return result[0]
        return None
    
    def should_skip_node(self, node_id: str) -> bool:
        """Check if a node should be skipped (already handled by pattern)."""
        return self.detector.is_node_in_pattern(node_id)


def analyze_and_generate_patterns(
    workflow_data: Dict[str, Any],
    base_path: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Convenience function to analyze workflow and get pattern code.
    
    Args:
        workflow_data: Parsed workflow structure
        base_path: Base path for workflow files
        
    Returns:
        Tuple of (code_blocks, all_imports)
    """
    analyzer = WorkflowPatternAnalyzer()
    results = analyzer.analyze_workflow(workflow_data, base_path)
    
    code_blocks = []
    all_imports = set()
    
    for item in results.get('pattern_code', []):
        code_blocks.append(item['code'])
        all_imports.update(item['imports'])
    
    logger.info(
        f"Pattern analysis complete: {len(results['patterns_detected'])} patterns detected, "
        f"{results['nodes_in_patterns']} nodes covered"
    )
    
    return code_blocks, list(all_imports)
