"""
Semantic Pattern Detection for KNIME Workflows.

This module provides pattern recognition for high-level business logic
in KNIME workflows, enabling semantic transpilation.
"""
from .pattern_registry import (
    PatternType,
    PatternSignature,
    DetectedPattern,
    BasePattern,
    DateGeneratorPattern,
    LoopPattern,
    PatternRegistry,
    get_pattern_registry,
)
from .pattern_detector import (
    PatternDetector,
    WorkflowPatternAnalyzer,
    analyze_and_generate_patterns,
)

__all__ = [
    'PatternType',
    'PatternSignature',
    'DetectedPattern',
    'BasePattern',
    'DateGeneratorPattern',
    'LoopPattern',
    'PatternRegistry',
    'get_pattern_registry',
    'PatternDetector',
    'WorkflowPatternAnalyzer',
    'analyze_and_generate_patterns',
]

