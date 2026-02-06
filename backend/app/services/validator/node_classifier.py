"""
Strategic Node Classifier for KNIME workflow validation.

Classifies KNIME nodes into strategic categories that require
schema validation during transpilation.

Categories:
- db_query: Nodes that query databases (SQL, BigQuery, etc.)
- output: Nodes that write data (DB Loader, CSV Writer, etc.)
- file_loader: Nodes that read external files (CSV, Excel, etc.)
- python_embedded: Python nodes containing SQL/Excel operations
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class NodeCategory(Enum):
    """Strategic node categories requiring validation."""
    DB_QUERY = "db_query"
    OUTPUT = "output"
    FILE_LOADER = "file_loader"
    PYTHON_EMBEDDED = "python_embedded"


@dataclass
class ClassificationResult:
    """Result of node classification."""
    node_id: str
    node_name: str
    factory_class: str
    category: Optional[NodeCategory]
    matched_pattern: Optional[str]
    is_strategic: bool
    
    @property
    def category_label(self) -> str:
        """Human-readable category label."""
        labels = {
            NodeCategory.DB_QUERY: "🗃️ DB Query",
            NodeCategory.OUTPUT: "📤 Output/Sink",
            NodeCategory.FILE_LOADER: "📂 File Loader",
            NodeCategory.PYTHON_EMBEDDED: "🐍 Python SQL/Excel",
        }
        return labels.get(self.category, "❓ Unknown") if self.category else "➖ Not Strategic"


class StrategicNodeClassifier:
    """
    Classifies KNIME nodes into strategic categories.
    
    Strategic nodes are those that interact with external systems
    (databases, files) and require schema validation.
    """
    
    # DB Query patterns - nodes that read from databases
    DB_QUERY_PATTERNS: List[str] = [
        r"Database.*Reader",
        r"Database.*Connector",
        r"DB.*Query.*Reader",
        r"DB.*Reader",
        r"BigQuery",
        r"Snowflake.*Connector",
        r"PostgreSQL|MySQL|Oracle|SQLite|SQL\s*Server",
        r"JDBC.*Reader",
        r"H2.*Connector",
        r"Amazon.*Redshift",
        r"Teradata",
        r"Vertica",
    ]
    
    # Output patterns - nodes that write data
    OUTPUT_PATTERNS: List[str] = [
        r"DB.*Loader",
        r"DB.*Writer",
        r"Database.*Writer",
        r"CSV.*Writer",
        r"Excel.*Writer",
        r"Table.*Writer",
        r"File.*Writer",
        r"Parquet.*Writer",
        r"JSON.*Writer",
        r"XML.*Writer",
        r"BigQuery.*Loader",
        r"Snowflake.*Loader",
    ]
    
    # File Loader patterns - nodes that read external files
    FILE_LOADER_PATTERNS: List[str] = [
        r"CSV.*Reader",
        r"Excel.*Reader",
        r"File.*Reader",
        r"Table.*Reader",
        r"PDF.*Parser",
        r"PDF.*Extractor",
        r"Parquet.*Reader",
        r"JSON.*Reader",
        r"XML.*Reader",
        r"Line.*Reader",
        r"Fixed.*Width.*Reader",
    ]
    
    # Python node factory patterns
    PYTHON_NODE_PATTERNS: List[str] = [
        r"Python.*Script",
        r"Python.*Source",
        r"Python.*View",
        r"Python.*Learner",
        r"Python.*Predictor",
        r"PythonScript",
    ]
    
    # Patterns to detect SQL/Excel usage in Python code
    PYTHON_SQL_PATTERNS: List[str] = [
        r"pd\.read_sql",
        r"read_sql_query",
        r"read_sql_table",
        r"engine\.execute",
        r"cursor\.execute",
        r"sqlalchemy",
        r"psycopg2",
        r"pymysql",
        r"pyodbc",
        r"cx_Oracle",
    ]
    
    PYTHON_FILE_PATTERNS: List[str] = [
        r"pd\.read_excel",
        r"pd\.read_csv",
        r"openpyxl",
        r"xlrd",
        r"xlwt",
    ]
    
    def __init__(self):
        """Initialize classifier with compiled regex patterns."""
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        self._db_query_re = [re.compile(p, re.IGNORECASE) for p in self.DB_QUERY_PATTERNS]
        self._output_re = [re.compile(p, re.IGNORECASE) for p in self.OUTPUT_PATTERNS]
        self._file_loader_re = [re.compile(p, re.IGNORECASE) for p in self.FILE_LOADER_PATTERNS]
        self._python_node_re = [re.compile(p, re.IGNORECASE) for p in self.PYTHON_NODE_PATTERNS]
        self._python_sql_re = [re.compile(p, re.IGNORECASE) for p in self.PYTHON_SQL_PATTERNS]
        self._python_file_re = [re.compile(p, re.IGNORECASE) for p in self.PYTHON_FILE_PATTERNS]
    
    def classify(
        self,
        node_id: str,
        node_name: str,
        factory_class: str,
        python_source: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a single node into a strategic category.
        
        Args:
            node_id: KNIME node ID
            node_name: Display name of the node
            factory_class: KNIME factory class name
            python_source: Source code if this is a Python node
            
        Returns:
            ClassificationResult with category and metadata
        """
        # Combine name and factory for matching
        match_text = f"{node_name} {factory_class}"
        
        # Check each category in order
        category, pattern = self._match_category(match_text, python_source)
        
        return ClassificationResult(
            node_id=node_id,
            node_name=node_name,
            factory_class=factory_class,
            category=category,
            matched_pattern=pattern,
            is_strategic=category is not None,
        )
    
    def _match_category(
        self,
        match_text: str,
        python_source: Optional[str]
    ) -> tuple[Optional[NodeCategory], Optional[str]]:
        """Match text against all category patterns."""
        
        # 1. Check DB Query
        for pattern in self._db_query_re:
            if pattern.search(match_text):
                return NodeCategory.DB_QUERY, pattern.pattern
        
        # 2. Check Output/Sink
        for pattern in self._output_re:
            if pattern.search(match_text):
                return NodeCategory.OUTPUT, pattern.pattern
        
        # 3. Check File Loader
        for pattern in self._file_loader_re:
            if pattern.search(match_text):
                return NodeCategory.FILE_LOADER, pattern.pattern
        
        # 4. Check Python nodes with embedded SQL/Excel
        is_python_node = any(p.search(match_text) for p in self._python_node_re)
        if is_python_node and python_source:
            embedded = self._detect_python_embedded(python_source)
            if embedded:
                return NodeCategory.PYTHON_EMBEDDED, embedded
        
        return None, None
    
    def _detect_python_embedded(self, source: str) -> Optional[str]:
        """Detect SQL or file operations in Python source code."""
        # Check for SQL patterns
        for pattern in self._python_sql_re:
            if pattern.search(source):
                return f"python_sql:{pattern.pattern}"
        
        # Check for file patterns
        for pattern in self._python_file_re:
            if pattern.search(source):
                return f"python_file:{pattern.pattern}"
        
        return None
    
    def classify_batch(
        self,
        nodes: List[Dict[str, Any]]
    ) -> List[ClassificationResult]:
        """
        Classify multiple nodes.
        
        Args:
            nodes: List of dicts with node_id, node_name, factory_class, python_source
            
        Returns:
            List of ClassificationResult
        """
        results = []
        for node in nodes:
            result = self.classify(
                node_id=node.get("node_id", ""),
                node_name=node.get("node_name", ""),
                factory_class=node.get("factory_class", ""),
                python_source=node.get("python_source"),
            )
            results.append(result)
        
        return results
    
    def get_strategic_nodes(
        self,
        nodes: List[Dict[str, Any]]
    ) -> List[ClassificationResult]:
        """Get only strategic nodes from a batch."""
        all_results = self.classify_batch(nodes)
        return [r for r in all_results if r.is_strategic]
    
    def summary(self, results: List[ClassificationResult]) -> Dict[str, int]:
        """Generate summary statistics."""
        summary = {
            "total": len(results),
            "strategic": sum(1 for r in results if r.is_strategic),
            "db_query": sum(1 for r in results if r.category == NodeCategory.DB_QUERY),
            "output": sum(1 for r in results if r.category == NodeCategory.OUTPUT),
            "file_loader": sum(1 for r in results if r.category == NodeCategory.FILE_LOADER),
            "python_embedded": sum(1 for r in results if r.category == NodeCategory.PYTHON_EMBEDDED),
        }
        return summary
