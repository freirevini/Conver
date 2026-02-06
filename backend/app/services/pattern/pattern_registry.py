"""
Semantic Pattern Registry for KNIME Workflow Transpilation.

This module implements pattern detection for high-level business logic
in KNIME workflows, enabling semantic transpilation instead of node-by-node conversion.

Patterns:
- DateGeneratorPattern: Detects date calculation metanodes
- LoopPattern: Detects iteration/loop structures
"""
from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of semantic patterns."""
    DATE_GENERATOR = "date_generator"
    LOOP_ITERATION = "loop_iteration"
    VARIABLE_FLOW = "variable_flow"
    CONDITIONAL_BRANCH = "conditional_branch"


@dataclass
class PatternSignature:
    """
    Defines the structural signature that identifies a pattern.
    
    Used to match node combinations within metanodes or workflow segments.
    """
    required_factories: List[str] = field(default_factory=list)
    optional_factories: List[str] = field(default_factory=list)
    output_port_type: Optional[str] = None
    min_nodes: int = 2
    max_nodes: int = 50
    name_patterns: List[str] = field(default_factory=list)
    
    def matches_factory(self, factory: str) -> bool:
        """Check if a factory matches any required pattern."""
        factory_lower = factory.lower()
        for required in self.required_factories:
            if required.lower() in factory_lower:
                return True
        return False


@dataclass
class DetectedPattern:
    """Result of pattern detection."""
    pattern_type: PatternType
    pattern_name: str
    node_ids: List[str]
    metanode_name: Optional[str] = None
    confidence: float = 1.0
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    python_code: Optional[str] = None


class BasePattern(ABC):
    """
    Abstract base class for all semantic patterns.
    
    Each pattern implements detection logic and Python code generation.
    """
    
    def __init__(self):
        self.pattern_type: PatternType = PatternType.DATE_GENERATOR
        self.name: str = "BasePattern"
        self.description: str = ""
        self.signature: PatternSignature = PatternSignature()
    
    @abstractmethod
    def detect(
        self, 
        nodes: List[Dict[str, Any]], 
        connections: List[Dict[str, Any]],
        metanode_info: Optional[Dict[str, Any]] = None
    ) -> Optional[DetectedPattern]:
        """
        Detect if nodes match this pattern.
        
        Args:
            nodes: List of node definitions
            connections: List of node connections
            metanode_info: Optional metanode context
            
        Returns:
            DetectedPattern if matched, None otherwise
        """
        pass
    
    @abstractmethod
    def generate_python(
        self, 
        detected: DetectedPattern,
        input_var: str = "df",
        output_var: str = "result"
    ) -> Tuple[str, List[str]]:
        """
        Generate Python code for the detected pattern.
        
        Args:
            detected: The detected pattern instance
            input_var: Input variable name
            output_var: Output variable name
            
        Returns:
            Tuple of (code, imports)
        """
        pass


class DateGeneratorPattern(BasePattern):
    """
    Detects Date Generator metanodes.
    
    Signature:
    - Contains DateTime creation/manipulation nodes
    - Ends with TableRowToVariable
    - Output is FlowVariablePortObject
    
    Python equivalent: datetime calculations with variable assignment.
    """
    
    def __init__(self):
        super().__init__()
        self.pattern_type = PatternType.DATE_GENERATOR
        self.name = "DateGenerator"
        self.description = "Generates date variables for SQL queries and flow control"
        self.signature = PatternSignature(
            required_factories=[
                "CreateDateTimeNodeFactory",
                "TableRowToVariableNodeFactory"
            ],
            optional_factories=[
                "DateTimeShiftNodeFactory",
                "ExtractDateTimeFieldsNodeFactory",
                "GroupByNodeFactory",
                "MathFormulaNodeFactory"
            ],
            output_port_type="FlowVariablePortObject",
            name_patterns=["DATA", "DATE", "REFERÊNCIA", "REFERENCIA"]
        )
    
    def detect(
        self, 
        nodes: List[Dict[str, Any]], 
        connections: List[Dict[str, Any]],
        metanode_info: Optional[Dict[str, Any]] = None
    ) -> Optional[DetectedPattern]:
        """Detect Date Generator pattern in nodes."""
        
        if len(nodes) < self.signature.min_nodes:
            return None
        
        # Check for required factories
        factories = [n.get('factory', '') for n in nodes]
        
        has_datetime_create = any(
            'createdatetime' in f.lower() or 'create date' in f.lower()
            for f in factories
        )
        has_row_to_variable = any(
            'tablerowtovariable' in f.lower() or 'row to variable' in f.lower()
            for f in factories
        )
        
        if not (has_datetime_create or has_row_to_variable):
            return None
        
        # Check metanode name patterns
        metanode_name = metanode_info.get('name', '') if metanode_info else ''
        name_match = any(
            pattern.lower() in metanode_name.lower()
            for pattern in self.signature.name_patterns
        )
        
        # Calculate confidence
        confidence = 0.0
        if has_datetime_create:
            confidence += 0.3
        if has_row_to_variable:
            confidence += 0.3
        if name_match:
            confidence += 0.3
        if metanode_info and metanode_info.get('output_type') == 'FlowVariablePortObject':
            confidence += 0.1
        
        if confidence < 0.5:
            return None
        
        # Extract date-related settings
        extracted_data = self._extract_date_settings(nodes)
        
        logger.info(f"DateGeneratorPattern detected in '{metanode_name}' (confidence: {confidence:.0%})")
        
        return DetectedPattern(
            pattern_type=self.pattern_type,
            pattern_name=self.name,
            node_ids=[n.get('id', '') for n in nodes],
            metanode_name=metanode_name,
            confidence=confidence,
            extracted_data=extracted_data
        )
    
    def _extract_date_settings(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract date-related settings from nodes."""
        settings = {
            'date_columns': [],
            'shift_operations': [],
            'output_variables': []
        }
        
        for node in nodes:
            node_settings = node.get('settings', {})
            factory = node.get('factory', '').lower()
            
            # Extract from DateTimeShift
            if 'datetimeshift' in factory:
                shift_value = node_settings.get('shiftvalue', node_settings.get('shift_value'))
                shift_unit = node_settings.get('granularity', node_settings.get('shift_unit'))
                if shift_value:
                    settings['shift_operations'].append({
                        'value': shift_value,
                        'unit': shift_unit
                    })
            
            # Extract from TableRowToVariable
            if 'tablerowtovariable' in factory:
                var_name = node_settings.get('variableName', node_settings.get('variable_name'))
                if var_name:
                    settings['output_variables'].append(var_name)
        
        return settings
    
    def generate_python(
        self, 
        detected: DetectedPattern,
        input_var: str = "df",
        output_var: str = "date_vars"
    ) -> Tuple[str, List[str]]:
        """Generate Python code for Date Generator pattern."""
        
        metanode_name = detected.metanode_name or "Date Generator"
        variables = detected.extracted_data.get('output_variables', [])
        
        code = f'''# =============================================================================
# Date Generator: {metanode_name}
# Pattern: DateGeneratorPattern (Semantic Transpilation)
# =============================================================================

from datetime import datetime
from dateutil.relativedelta import relativedelta

# Current execution date
execution_date = datetime.now()

# Processing date (first day of current month)
processing_date = execution_date.replace(day=1)

# Reference period (previous month)
start_date = (execution_date - relativedelta(months=1)).replace(day=1)
end_date = execution_date.replace(day=1) - relativedelta(days=1)

# Date variables for SQL queries
{output_var} = {{
    'processing_date': processing_date.strftime('%Y-%m-%d'),
    'start_date': start_date.strftime('%Y-%m-%d'),
    'end_date': end_date.strftime('%Y-%m-%d'),
    'year_month': processing_date.strftime('%Y%m'),
}}

# Log generated dates
print(f"Date Generator: {{processing_date.strftime('%Y-%m-%d')}}")
print(f"Reference Period: {{start_date.strftime('%Y-%m-%d')}} to {{end_date.strftime('%Y-%m-%d')}}")
'''
        
        imports = [
            'from datetime import datetime',
            'from dateutil.relativedelta import relativedelta'
        ]
        
        return code, imports


class LoopPattern(BasePattern):
    """
    Detects Loop iteration patterns.
    
    Signature:
    - Contains LoopStart and LoopEnd nodes
    - Iterates over table rows or chunks
    
    Python equivalent: for loop with pandas iterrows/apply.
    """
    
    def __init__(self):
        super().__init__()
        self.pattern_type = PatternType.LOOP_ITERATION
        self.name = "LoopIteration"
        self.description = "Iterates over rows or chunks, consolidating results"
        self.signature = PatternSignature(
            required_factories=[
                "LoopStart",
                "LoopEnd"
            ],
            optional_factories=[
                "ChunkLoopStart",
                "ChunkLoopEnd",
                "TableRowLoopStart",
                "TableRowLoopEnd"
            ],
            name_patterns=["LOOP", "ITERATE", "CALCULA"]
        )
    
    def detect(
        self, 
        nodes: List[Dict[str, Any]], 
        connections: List[Dict[str, Any]],
        metanode_info: Optional[Dict[str, Any]] = None
    ) -> Optional[DetectedPattern]:
        """Detect Loop pattern in nodes."""
        
        factories = [n.get('factory', '').lower() for n in nodes]
        
        # Check for loop start/end pairs
        has_loop_start = any('loopstart' in f for f in factories)
        has_loop_end = any('loopend' in f for f in factories)
        
        if not (has_loop_start and has_loop_end):
            return None
        
        metanode_name = metanode_info.get('name', '') if metanode_info else ''
        
        # Identify loop type
        loop_type = 'row'
        if any('chunk' in f for f in factories):
            loop_type = 'chunk'
        elif any('recursive' in f for f in factories):
            loop_type = 'recursive'
        
        confidence = 0.9 if (has_loop_start and has_loop_end) else 0.5
        
        logger.info(f"LoopPattern ({loop_type}) detected in '{metanode_name}'")
        
        return DetectedPattern(
            pattern_type=self.pattern_type,
            pattern_name=self.name,
            node_ids=[n.get('id', '') for n in nodes],
            metanode_name=metanode_name,
            confidence=confidence,
            extracted_data={'loop_type': loop_type}
        )
    
    def generate_python(
        self, 
        detected: DetectedPattern,
        input_var: str = "df",
        output_var: str = "result"
    ) -> Tuple[str, List[str]]:
        """Generate Python code for Loop pattern."""
        
        metanode_name = detected.metanode_name or "Loop"
        loop_type = detected.extracted_data.get('loop_type', 'row')
        
        if loop_type == 'chunk':
            code = f'''# =============================================================================
# Chunk Loop: {metanode_name}
# Pattern: LoopPattern (Semantic Transpilation)
# =============================================================================

chunk_size = 1000
results = []

for chunk_start in range(0, len({input_var}), chunk_size):
    chunk = {input_var}.iloc[chunk_start:chunk_start + chunk_size].copy()
    
    # Process chunk
    # TODO: Add loop body operations
    processed_chunk = chunk
    
    results.append(processed_chunk)

{output_var} = pd.concat(results, ignore_index=True)
print(f"Processed {{len({output_var})}} rows in {{len(results)}} chunks")
'''
        else:  # row iteration
            code = f'''# =============================================================================
# Row Loop: {metanode_name}
# Pattern: LoopPattern (Semantic Transpilation)
# =============================================================================

results = []

for idx, row in {input_var}.iterrows():
    # Process each row
    # TODO: Add loop body operations
    result_row = row.copy()
    
    results.append(result_row)

{output_var} = pd.DataFrame(results)
print(f"Processed {{len({output_var})}} rows")
'''
        
        imports = ['import pandas as pd']
        
        return code, imports


class PatternRegistry:
    """
    Central registry for all semantic patterns.
    
    Manages pattern registration, detection, and code generation.
    """
    
    def __init__(self):
        self.patterns: Dict[PatternType, BasePattern] = {}
        self._register_default_patterns()
    
    def _register_default_patterns(self):
        """Register built-in patterns."""
        self.register(DateGeneratorPattern())
        self.register(LoopPattern())
        logger.info(f"PatternRegistry initialized with {len(self.patterns)} patterns")
    
    def register(self, pattern: BasePattern):
        """Register a new pattern."""
        self.patterns[pattern.pattern_type] = pattern
        logger.debug(f"Registered pattern: {pattern.name}")
    
    def detect_all(
        self,
        nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        metanode_info: Optional[Dict[str, Any]] = None
    ) -> List[DetectedPattern]:
        """
        Attempt to detect all registered patterns.
        
        Returns list of detected patterns, sorted by confidence.
        """
        detected = []
        
        for pattern in self.patterns.values():
            try:
                result = pattern.detect(nodes, connections, metanode_info)
                if result:
                    detected.append(result)
            except Exception as e:
                logger.warning(f"Pattern detection failed for {pattern.name}: {e}")
        
        # Sort by confidence (highest first)
        detected.sort(key=lambda x: x.confidence, reverse=True)
        
        return detected
    
    def generate_code(
        self,
        detected: DetectedPattern,
        input_var: str = "df",
        output_var: str = "result"
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Generate Python code for a detected pattern.
        
        Returns (code, imports) or None if pattern not found.
        """
        pattern = self.patterns.get(detected.pattern_type)
        if not pattern:
            logger.warning(f"No pattern handler for {detected.pattern_type}")
            return None
        
        return pattern.generate_python(detected, input_var, output_var)
    
    def get_pattern(self, pattern_type: PatternType) -> Optional[BasePattern]:
        """Get a pattern by type."""
        return self.patterns.get(pattern_type)
    
    def list_patterns(self) -> List[Dict[str, str]]:
        """List all registered patterns."""
        return [
            {
                'type': p.pattern_type.value,
                'name': p.name,
                'description': p.description
            }
            for p in self.patterns.values()
        ]


# Singleton instance for global access
_registry: Optional[PatternRegistry] = None


def get_pattern_registry() -> PatternRegistry:
    """Get the global pattern registry instance."""
    global _registry
    if _registry is None:
        _registry = PatternRegistry()
    return _registry
