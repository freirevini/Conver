"""
KNIME Node Catalog Service

Provides O(1) lookup for node metadata, templates, and function mappings.
Used for deterministic KNIME → Python translation.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeCatalogService:
    """
    Service for querying the KNIME node catalog.
    
    Features:
    - O(1) node lookup by factory_class
    - Operator mapping (KNIME → Python)
    - Function mapping (String/Math)
    - Type conversion (KNIME → Pandas)
    """
    
    _instance = None
    _catalog: Dict[str, Any] = None
    
    def __new__(cls):
        """Singleton pattern for catalog instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_catalog()
        return cls._instance
    
    def _load_catalog(self) -> None:
        """Load catalog from JSON file."""
        # Try multiple paths
        possible_paths = [
            Path(__file__).parent.parent / "data" / "knime_catalog.json",
            Path(__file__).parent.parent.parent / "data" / "knime_catalog.json",
            Path(__file__).parent.parent.parent.parent / "data" / "knime_catalog.json",
        ]
        
        catalog_path = None
        for path in possible_paths:
            if path.exists():
                catalog_path = path
                break
        
        if catalog_path is None:
            logger.warning(f"Catalog not found in any of: {possible_paths}")
            self._catalog = {"nodes": {}, "operator_mappings": {}}
            return
        
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                self._catalog = json.load(f)
            logger.info(f"Loaded catalog with {len(self._catalog.get('nodes', {}))} nodes")
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            self._catalog = {"nodes": {}, "operator_mappings": {}}
    
    def get_node(self, factory_class: str) -> Optional[Dict[str, Any]]:
        """
        Get node metadata by factory class.
        
        Args:
            factory_class: Full KNIME factory class name
            
        Returns:
            Node metadata dict or None if not found
        """
        return self._catalog.get("nodes", {}).get(factory_class)
    
    def get_template(self, factory_class: str) -> Optional[str]:
        """
        Get Python template for a node.
        
        Args:
            factory_class: Full KNIME factory class name
            
        Returns:
            Template string or None
        """
        node = self.get_node(factory_class)
        if node:
            return node.get("template")
        return None
    
    def get_imports(self, factory_class: str) -> List[str]:
        """
        Get required imports for a node.
        
        Args:
            factory_class: Full KNIME factory class name
            
        Returns:
            List of import statements
        """
        node = self.get_node(factory_class)
        if node:
            return node.get("imports", [])
        return []
    
    def get_complexity(self, factory_class: str) -> str:
        """
        Get complexity level for a node.
        
        Args:
            factory_class: Full KNIME factory class name
            
        Returns:
            Complexity level: LOW, MEDIUM, HIGH
        """
        node = self.get_node(factory_class)
        if node:
            return node.get("complexity", "HIGH")
        return "HIGH"
    
    def is_deterministic(self, factory_class: str) -> bool:
        """
        Check if node can be translated deterministically.
        
        Args:
            factory_class: Full KNIME factory class name
            
        Returns:
            True if node has LOW or MEDIUM complexity (no LLM needed)
        """
        complexity = self.get_complexity(factory_class)
        return complexity in ("LOW", "MEDIUM")
    
    def requires_parser(self, factory_class: str) -> Optional[str]:
        """
        Check if node requires a special parser.
        
        Args:
            factory_class: Full KNIME factory class name
            
        Returns:
            Parser name (string_parser, math_parser, etc.) or None
        """
        node = self.get_node(factory_class)
        if node:
            return node.get("requires_parser")
        return None
    
    def map_operator(self, knime_operator: str) -> str:
        """
        Map KNIME operator to Python operator.
        
        Args:
            knime_operator: KNIME operator (e.g., GREATER_THAN)
            
        Returns:
            Python operator (e.g., ">")
        """
        mappings = self._catalog.get("operator_mappings", {})
        return mappings.get(knime_operator, knime_operator)
    
    def get_string_function(self, func_name: str) -> Optional[Dict[str, str]]:
        """
        Get Python equivalent for KNIME string function.
        
        Args:
            func_name: KNIME function name (e.g., "join", "substr")
            
        Returns:
            Dict with 'python' template and 'args'
        """
        return self._catalog.get("string_functions", {}).get(func_name)
    
    def get_math_function(self, func_name: str) -> Optional[Dict[str, str]]:
        """
        Get Python equivalent for KNIME math function.
        
        Args:
            func_name: KNIME function name (e.g., "ABS", "ROUND")
            
        Returns:
            Dict with 'python' function and 'args'
        """
        return self._catalog.get("math_functions", {}).get(func_name)
    
    def map_type(self, knime_type: str) -> Dict[str, str]:
        """
        Map KNIME cell type to pandas/Python type.
        
        Args:
            knime_type: KNIME type (e.g., "IntCell", "StringCell")
            
        Returns:
            Dict with 'pandas' and 'python' type strings
        """
        mappings = self._catalog.get("type_mappings", {})
        return mappings.get(knime_type, {"pandas": "object", "python": "str"})
    
    def get_db_connector(self, db_type: str) -> Optional[Dict[str, Any]]:
        """
        Get DB connector template.
        
        Args:
            db_type: Database type (mysql, postgresql, oracle, h2)
            
        Returns:
            Connector template dict
        """
        return self._catalog.get("db_connectors", {}).get(db_type.lower())
    
    def get_aggregation_map(self) -> Dict[str, str]:
        """
        Get KNIME → Pandas aggregation function mapping.
        
        Returns:
            Dict mapping KNIME aggregations to pandas
        """
        groupby_node = self.get_node("org.knime.base.node.preproc.groupby.GroupByNodeFactory")
        if groupby_node:
            return groupby_node.get("aggregation_map", {})
        return {}
    
    def list_supported_nodes(self) -> List[str]:
        """
        List all supported factory classes.
        
        Returns:
            List of factory class names
        """
        return list(self._catalog.get("nodes", {}).keys())
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get catalog statistics.
        
        Returns:
            Dict with counts
        """
        nodes = self._catalog.get("nodes", {})
        return {
            "total_nodes": len(nodes),
            "low_complexity": sum(1 for n in nodes.values() if n.get("complexity") == "LOW"),
            "medium_complexity": sum(1 for n in nodes.values() if n.get("complexity") == "MEDIUM"),
            "high_complexity": sum(1 for n in nodes.values() if n.get("complexity") == "HIGH"),
            "operators": len(self._catalog.get("operator_mappings", {})),
            "string_functions": len(self._catalog.get("string_functions", {})),
            "math_functions": len(self._catalog.get("math_functions", {})),
            "type_mappings": len(self._catalog.get("type_mappings", {})),
        }


# Singleton instance
catalog_service = NodeCatalogService()
